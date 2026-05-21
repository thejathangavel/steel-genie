import fitz
import re
import math
import time
import base64
import os
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image as PILImage, ImageEnhance, ImageFilter

# ── OpenCV-based raster beam line detector ───────────────────────────────────
try:
    from detection_cv import detect_beam_lines_raster as _detect_beam_lines_raster
    _RASTER_HOUGH_AVAILABLE = True
    print("[INIT] detection_cv.detect_beam_lines_raster loaded")
except Exception as _e:
    _RASTER_HOUGH_AVAILABLE = False
    print(f"[INIT] detection_cv not available — raster Hough disabled: {_e}")

# ── Load Environment ──────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ENV_PATH = os.path.join(BASE_DIR, ".env")
    if os.path.exists(ENV_PATH):
        load_dotenv(dotenv_path=ENV_PATH, override=True)
        print("[INIT] .env loaded")
    else:
        print("[INIT] .env not found")
except ImportError:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Database ──────────────────────────────────────────────────────────────────
supabase_client = None
try:
    from supabase import create_client as _sb_create
    _url = os.getenv("SUPABASE_URL")
    _key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if _url and _key:
        supabase_client = _sb_create(_url, _key)
        print("[INIT] Supabase ready")
except Exception as e:
    print(f"[INIT] Supabase error: {e}")

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MEMBER_COLORS = {
    "beam":   "#EC4899",
    "column": "#3B82F6",
    "brace":  "#F59E0B",
}

STEEL_PATTERNS = [
    # W-sections: depth 1-2 digits, weight 2-3 digits (all real W shapes ≥ W4X13).
    # The (?!\d) lookahead prevents matching partial OCR reads like "W16X3" from
    # "W16X31" or "W16X3128" (garbage from adjacent load annotations).
    r'W\d{1,2}[Xx]\d{2,3}(?!\d)',
    r'HSS[\d.]+[Xx][\d.]+(?:[Xx][\d.]+(?:/[\d.]+)?)?',  # HSS6X6, HSS6X6X1/4, HSS7.00X0.50
    r'L\d+[Xx]\d+',
    r'C\d+[Xx]\d+',
    r'MC\d+[Xx]\d+',
    r'ISA[\dXx]+',
    r'PIPE[\d.]+',
]

_GRID_LETTER = re.compile(r'^[A-Z](\.\d+)?$')
_GRID_NUMBER  = re.compile(r'^\d+(\.\d+)?$')

# Max distance (PDF points) between a profile label and a column symbol.
# Used for the one-to-one greedy symbol→profile matching in build_members.
SYMBOL_ASSOC_RADIUS = 65   # search radius for matching symbol to nearest label
SYMBOL_SNAP_RADIUS  = 110  # snap marker TO symbol position (wider — position only)

# Grid-intersection classification tolerance.
# Reduced from 35 → 20: beam labels that sit close to (but not at) a grid crossing
# were being falsely promoted to columns by TIER 2.  20 pt ≈ 0.28″ — tight enough
# to cover typical label offsets while excluding beams that frame INTO a column.
GRID_TOL = 20
# Grid-intersection SNAP radius (for marker placement — wider than classification)
GRID_SNAP_RADIUS = 50


# ── Scale conversion ──────────────────────────────────────────────────────────
def scale_to_pts_per_foot(scale_ratio: float) -> float:
    """
    Convert the frontend SCALE_OPTIONS ratio to PDF-points per foot.

    The frontend stores:  ratio = 12 / paper_inches_per_foot
    At 72 dpi:            pts_per_foot = 72 × paper_inches_per_foot
                                       = 72 × (12 / ratio)
                                       = 864 / ratio

    Examples
    --------
    1/8"=1'-0"  → ratio= 96 → pts_per_foot =  9.0
    3/16"=1'-0" → ratio= 64 → pts_per_foot = 13.5
    1/4"=1'-0"  → ratio= 48 → pts_per_foot = 18.0
    """
    if not scale_ratio or scale_ratio <= 0:
        return 0.0
    return 864.0 / scale_ratio


def compute_beam_span(cx: float, cy: float,
                      v_grid: list, h_grid: list,
                      pts_per_foot: float,
                      beam_dir: str = "H") -> dict | None:
    """
    Compute the beam span — both the length in feet AND the two physical
    endpoint positions (in raw PDF points) for rendering as a line overlay.

    Returns a dict:
        { "length_ft": float,
          "x1": float, "y1": float,   # start point (PDF pts)
          "x2": float, "y2": float }  # end point   (PDF pts)
    Returns None when the surrounding grid lines cannot be found.

    H-beam:  endpoints are (left_v_grid, cy) → (right_v_grid, cy)
    V-beam:  endpoints are (cx, top_h_grid)  → (cx, bottom_h_grid)
    """
    if beam_dir == "V":
        tops    = [gy for gy in h_grid if gy <= cy]
        bottoms = [gy for gy in h_grid if gy >  cy]
        if tops and bottoms:
            span_pt = min(bottoms) - max(tops)
            return {
                "length_ft": round(span_pt / pts_per_foot, 1) if pts_per_foot > 0 else 0.0,
                "x1": cx,         "y1": max(tops),
                "x2": cx,         "y2": min(bottoms),
            }
    else:  # "H" — default
        lefts  = [gx for gx in v_grid if gx <= cx]
        rights = [gx for gx in v_grid if gx >  cx]
        if lefts and rights:
            span_pt = min(rights) - max(lefts)
            return {
                "length_ft": round(span_pt / pts_per_foot, 1) if pts_per_foot > 0 else 0.0,
                "x1": max(lefts), "y1": cy,
                "x2": min(rights), "y2": cy,
            }

    return None


def detect_beam_lines(page, profiles: list, plan_bounds: tuple) -> dict:
    """
    PRIMARY beam detection: find structural centerlines in the PDF vector drawing
    and match them to steel-section text labels.

    In CAD-exported structural framing plans every beam IS drawn as a line segment
    on its centreline.  The profile label (e.g. "W24X76") sits right on — or very
    close to — that line at mid-span.

    Strategy
    --------
    1. Collect all line segments inside the plan boundary that are:
       • 30–600 pt long  (≈ 3-66 ft at 1/8" scale; excludes ticks and grid lines)
       • Clearly horizontal  (dx > dy × 2)  or vertical  (dy > dx × 2)
    2. For each profile label (cx, cy):
       • H-match: label Y is within LABEL_R of a line's Y, AND
                  label X falls inside the line's X span ± LABEL_R
       • V-match: label X is within LABEL_R of a line's X, AND
                  label Y falls inside the line's Y span ± LABEL_R
       • Accept the closest single match (H or V).
    3. Return a dict  { profile_idx: {"x1","y1","x2","y2","dir","length_pt"} }
       for every profile that was successfully matched to a drawn line.
       Unmatched profiles fall back to the grid-based span calculation.

    This gives us EXACT endpoints and EXACT length from the drawing geometry —
    no grid guessing needed.
    """
    bx0, by0, bx1, by1 = plan_bounds
    MIN_LEN  = 30    # ignore dimension ticks (<3 ft at 1/8")
    MAX_LEN  = 1200  # raised: covers very long bays and 1/16" scale drawings
    LABEL_R  = 70    # raised: labels on dense drawings can sit further from centreline

    h_lines: list[tuple] = []   # (lx1, ly, lx2, ly, length)
    v_lines: list[tuple] = []   # (lx, ly1, lx, ly2, length)

    try:
        for d in page.get_drawings():
            for item in d.get("items", []):
                if item[0] != "l":
                    continue
                try:
                    p1, p2 = item[1], item[2]
                    dx = abs(p2.x - p1.x)
                    dy = abs(p2.y - p1.y)
                    ln = math.hypot(dx, dy)
                    if ln < MIN_LEN or ln > MAX_LEN:
                        continue
                    mx = (p1.x + p2.x) / 2
                    my = (p1.y + p2.y) / 2
                    # Midpoint must be inside plan boundary
                    if not (bx0 <= mx <= bx1 and by0 <= my <= by1):
                        continue
                    # FIX: BOTH endpoints must also stay within plan bounds
                    # (with a small tolerance).  This prevents long annotation/
                    # dimension lines whose midpoint barely falls inside the plan
                    # from extending far into the notes or title block area.
                    _EP_TOL = 80   # ≈ 1.1" — proportional to widened plan boundary buffer
                    if (min(p1.x, p2.x) < bx0 - _EP_TOL or
                            max(p1.x, p2.x) > bx1 + _EP_TOL):
                        continue
                    if (min(p1.y, p2.y) < by0 - _EP_TOL or
                            max(p1.y, p2.y) > by1 + _EP_TOL):
                        continue
                    if dx > dy * 2:
                        # Horizontal line: store as (x_start, y, x_end, y, len)
                        h_lines.append((min(p1.x, p2.x), my,
                                        max(p1.x, p2.x), my, ln))
                    elif dy > dx * 2:
                        # Vertical line: store as (x, y_start, x, y_end, len)
                        v_lines.append((mx, min(p1.y, p2.y),
                                        mx, max(p1.y, p2.y), ln))
                except Exception:
                    continue
    except Exception:
        pass

    print(f"[BEAM_LINES] H structural lines: {len(h_lines)}  "
          f"V structural lines: {len(v_lines)}")

    result: dict[int, dict] = {}

    for p_idx, p in enumerate(profiles):
        pcx, pcy = p["cx"], p["cy"]
        best:   tuple | None = None
        best_d: float        = float("inf")
        best_dir: str        = "H"

        # ── Try to match a horizontal line ────────────────────────────────
        for (lx1, ly, lx2, _, ln) in h_lines:
            dy_label = abs(pcy - ly)
            if dy_label > LABEL_R:
                continue
            if pcx < lx1 - LABEL_R or pcx > lx2 + LABEL_R:
                continue
            # Score = perpendicular distance from label to line axis
            if dy_label < best_d:
                best_d   = dy_label
                best     = (lx1, ly, lx2, ly, ln)
                best_dir = "H"

        # ── Try to match a vertical line ──────────────────────────────────
        for (lx, ly1, _, ly2, ln) in v_lines:
            dx_label = abs(pcx - lx)
            if dx_label > LABEL_R:
                continue
            if pcy < ly1 - LABEL_R or pcy > ly2 + LABEL_R:
                continue
            if dx_label < best_d:
                best_d   = dx_label
                best     = (lx, ly1, lx, ly2, ln)
                best_dir = "V"

        if best:
            result[p_idx] = {
                "x1": best[0], "y1": best[1],
                "x2": best[2], "y2": best[3],
                "dir":       best_dir,
                "length_pt": best[4],
            }

    matched = len(result)
    print(f"[BEAM_LINES] {matched}/{len(profiles)} profiles matched "
          f"to drawn beam lines")
    return result


def detect_beam_directions(page, profiles: list, plan_bounds: tuple) -> dict:
    """
    Determine whether each beam profile label lies on a horizontal or vertical
    structural member by finding the nearest significant vector line segment.

    Strategy
    --------
    • Collect all line segments from page vector drawings that are:
        – Inside the plan boundary
        – Between MIN_LEN and MAX_LEN pts long (ignores ticks and grid lines)
        – Clearly horizontal (dx/dy > 2) or vertical (dy/dx > 2)
    • For each profile, whichever orientation class has a member closer than
      SEARCH_R pts wins.  Ties (or nothing within range) default to "H".

    Returns
    -------
    dict  profile_idx → "H" | "V"
    """
    bx0, by0, bx1, by1 = plan_bounds

    MIN_LEN  = 40    # ignore ticks, hatching, and annotation lines
    MAX_LEN  = 600   # ignore full-width grid lines and sheet border
    DIR_RATIO = 2.0  # dx/dy (or dy/dx) must exceed this to be "clearly" H or V
    SEARCH_R  = 60   # max distance from label to candidate line midpoint

    h_lines: list[tuple[float, float]] = []
    v_lines: list[tuple[float, float]] = []

    try:
        for d in page.get_drawings():
            for item in d.get("items", []):
                if item[0] != "l":
                    continue
                try:
                    p1, p2 = item[1], item[2]
                    dx = abs(p2.x - p1.x)
                    dy = abs(p2.y - p1.y)
                    ln = math.hypot(dx, dy)
                    if ln < MIN_LEN or ln > MAX_LEN:
                        continue
                    lx = (p1.x + p2.x) / 2
                    ly = (p1.y + p2.y) / 2
                    if not (bx0 <= lx <= bx1 and by0 <= ly <= by1):
                        continue
                    if dx > dy * DIR_RATIO:
                        h_lines.append((lx, ly))
                    elif dy > dx * DIR_RATIO:
                        v_lines.append((lx, ly))
                except Exception:
                    continue
    except Exception:
        pass

    print(f"[BEAM_DIR] H-lines: {len(h_lines)}  V-lines: {len(v_lines)}")

    directions: dict[int, str] = {}
    for p_idx, p in enumerate(profiles):
        pcx, pcy = p["cx"], p["cy"]
        d_h = min((math.hypot(pcx - lx, pcy - ly) for lx, ly in h_lines),
                  default=float("inf"))
        d_v = min((math.hypot(pcx - lx, pcy - ly) for lx, ly in v_lines),
                  default=float("inf"))

        if d_h <= SEARCH_R or d_v <= SEARCH_R:
            directions[p_idx] = "H" if d_h <= d_v else "V"
        else:
            directions[p_idx] = "H"  # default — horizontal beam

    return directions


# ── Profile helpers ───────────────────────────────────────────────────────────
def normalize_profile(text: str) -> str:
    t = text.upper().strip()
    t = re.sub(r'[^A-Z0-9X/]', '', t)
    t = re.sub(r'^VV', "W", t)
    t = re.sub(r'^V(?=\d)', "W", t)
    t = t.rstrip('/')  # strip trailing slash OCR artifact (e.g. "HSS6X6X3/8/")
    return t


_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp'}


def _pytesseract_to_dict(data: dict, page_w: float, page_h: float,
                         pix_w: int, pix_h: int) -> dict:
    """Convert pytesseract image_to_data output to a fitz-compatible text dict."""
    sx = page_w / max(pix_w, 1)
    sy = page_h / max(pix_h, 1)
    blocks = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        try:
            conf = int(data["conf"][i])
        except (ValueError, TypeError):
            conf = 0
        if conf < 30:        # skip very low-confidence words
            continue
        lx = data["left"][i]  * sx
        ty = data["top"][i]   * sy
        rx = (data["left"][i] + data["width"][i])  * sx
        by = (data["top"][i]  + data["height"][i]) * sy
        h  = max(by - ty, 1)
        blocks.append({
            "type": 0,
            "bbox": (lx, ty, rx, by),
            "lines": [{
                "bbox": (lx, ty, rx, by),
                "spans": [{"text": text, "bbox": (lx, ty, rx, by),
                            "size": h * 0.75}],
            }],
        })
    return {"blocks": blocks}


_easyocr_reader = None   # lazy-loaded singleton

def _get_easyocr_reader():
    """Return a cached EasyOCR reader (loaded once per process)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            print("[OCR] Loading EasyOCR model (first-run download may take a moment)...")
            _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            print("[OCR] EasyOCR ready")
        except Exception as e:
            print(f"[OCR] EasyOCR load failed: {e}")
    return _easyocr_reader


def _easyocr_to_dict(results: list, page_w: float, page_h: float,
                     pix_w: int, pix_h: int) -> dict:
    """
    Convert EasyOCR result list to fitz-compatible text dict.

    Each item in results may be a 3-tuple (bbox_pts, text, conf) or a
    4-tuple (bbox_pts, text, conf, rot_pass) where rot_pass is:
      0 = original orientation (horizontal text)
      1 = 90° CCW rotation (reads CW-rotated vertical text in original)
      2 = 90° CW rotation  (reads CCW-rotated vertical text in original)

    rot_pass is stored in each span so that extract_profiles can use it
    to infer beam direction for raster images (pass 1/2 → vertical member).

    All coordinates are in PIXEL space of the rendered image; we scale to
    fitz page-point space.
    """
    sx = page_w / max(pix_w, 1)
    sy = page_h / max(pix_h, 1)
    blocks = []
    for item in results:
        if len(item) == 4:
            bbox_pts, text, conf, rot_pass = item
        else:
            bbox_pts, text, conf = item
            rot_pass = 0
        text = text.strip()
        if not text or conf < 0.20:          # lowered from 0.25 for better recall
            continue
        xs = [p[0] for p in bbox_pts]
        ys = [p[1] for p in bbox_pts]
        lx = min(xs) * sx;  rx = max(xs) * sx
        ty = min(ys) * sy;  by = max(ys) * sy
        h  = max(by - ty, 1)
        blocks.append({
            "type": 0,
            "bbox": (lx, ty, rx, by),
            "lines": [{
                "bbox": (lx, ty, rx, by),
                "spans": [{"text": text, "bbox": (lx, ty, rx, by),
                            "size": h * 0.75,
                            "rot_pass": rot_pass}],
            }],
        })
    return {"blocks": blocks}


def _get_text_dict(page, page_w: float = None, page_h: float = None
                   ) -> tuple[dict, bool]:
    """
    Return (text_dict, is_raster) for a fitz page.

    Attempt order
    ─────────────
    1. Embedded vector text (zero cost, always correct for proper PDFs)
    2. EasyOCR (pure-Python, no Tesseract required, good on engineering drawings)
    3. PyMuPDF built-in OCR  — requires Tesseract + tessdata
    4. pytesseract           — requires pytesseract + Tesseract binary

    Returns is_raster=False when embedded text was sufficient, True otherwise.
    """
    # ── 1. Embedded text ─────────────────────────────────────────────────────
    td = page.get_text("dict")
    n_chars = sum(len(sp["text"].strip())
                  for bl in td.get("blocks", [])
                  for ln in bl.get("lines", [])
                  for sp in ln.get("spans", []))
    if n_chars >= 20:
        return td, False

    print(f"[OCR] Embedded text chars={n_chars} — activating OCR")

    pw = page_w or page.rect.width
    ph = page_h or page.rect.height

    # ── 2. EasyOCR — 3-pass (0°, 90° CCW, 90° CW) + preprocessing ──────────────
    #
    # Structural drawings place beam labels PARALLEL to the member axis so
    # vertical member labels are rotated 90° in the image.  A single forward
    # pass only reads horizontal text; we rotate the image and transform
    # coordinates back to catch vertical text as well.
    #
    # Each result is stored as a 4-tuple (bbox, text, conf, rot_pass) where
    # rot_pass encodes which orientation found the text:
    #   0 = original  →  horizontal member label  (dir_hint = H)
    #   1 = 90° CCW   →  CW-rotated label in original  (dir_hint = V)
    #   2 = 90° CW    →  CCW-rotated label in original (dir_hint = V)
    #
    # Preprocessing: convert to greyscale, boost contrast, sharpen.
    # This significantly improves OCR accuracy on dense engineering drawings
    # with thin lines, small text, and low ink-to-paper contrast.
    reader = _get_easyocr_reader()
    if reader is not None:
        try:
            pix     = page.get_pixmap(dpi=300)        # was 200 — higher res for small text
            pil_img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # CLAHE preprocessing: adaptive local contrast enhancement.
            # Better than global contrast because it improves dark/faded regions
            # without over-saturating bright ones, keeping thin strokes like
            # "/" in HSS profiles (e.g. HSS6X6X3/8) readable.
            try:
                import cv2 as _cv2
                _gray = np.array(pil_img.convert("L"))
                # Step 1: CLAHE for local contrast (handles uneven ink/scan)
                _clahe = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                _enhanced = _clahe.apply(_gray)
                # Step 2: Unsharp mask to sharpen blurry scanned text.
                # Scanned drawings often have low-frequency blur from the scanner
                # optics; sharpening recovers thin strokes like "/" in "HSS6X6X3/8".
                _blur    = _cv2.GaussianBlur(_enhanced, (0, 0), sigmaX=1.5)
                _sharp   = _cv2.addWeighted(_enhanced, 1.5, _blur, -0.5, 0)
                pil_img = PILImage.fromarray(
                    _cv2.cvtColor(_sharp, _cv2.COLOR_GRAY2RGB))
            except Exception:
                # Fallback: mild global contrast if cv2 unavailable
                pil_img = PILImage.fromarray(
                    np.stack([np.array(ImageEnhance.Contrast(
                        pil_img.convert("L")).enhance(1.4))] * 3, axis=-1))

            img_arr = np.array(pil_img)
            H_pix, W_pix = img_arr.shape[:2]

            # Characters found in structural steel section labels + grid bubbles
            _ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789./-×X"

            # EasyOCR parameters tuned for engineering drawings.
            # We intentionally keep link_threshold at the default (0.4) to
            # prevent characters from different nearby labels being merged
            # into one span (e.g. "W12X26" + adjacent "1" → "W12X261").
            # text_threshold lowered slightly to catch low-ink labels.
            _OCR_KW = dict(
                detail=1, allowlist=_ALLOW, paragraph=False,
                text_threshold=0.65,   # default 0.7 — slightly more detections
                low_text=0.35,         # default 0.4 — catch edge characters
                link_threshold=0.4,    # default 0.4 — keep to avoid cross-label merging
                contrast_ths=0.1,
                adjust_contrast=0.5,
            )

            # Pass 0 — 0°: horizontal text (most beam labels, grid numbers)
            res = [(bbox, text, conf, 0)
                   for (bbox, text, conf) in reader.readtext(img_arr, **_OCR_KW)]

            # Pass 1 — 90° CCW: reads labels rotated 90° CW in the original.
            #   rot90 CCW transform: orig(x,y) → rot(x_r=y, y_r=W-1-x)
            #   Inverse: rot(x_r,y_r) → orig(x=W-1-y_r, y=x_r)
            img_rot = np.rot90(img_arr, k=1)
            for (bbox, text, conf) in reader.readtext(img_rot, **_OCR_KW):
                orig_bbox = [[W_pix - 1 - float(py), float(px)]
                             for (px, py) in bbox]
                res.append((orig_bbox, text, conf, 1))

            # Pass 2 — 90° CW: reads labels rotated 90° CCW in the original.
            #   rot90 CW (=rot270 CCW): orig(x,y) → rot(x_r=H-1-y, y_r=x)
            #   Inverse: rot(x_r,y_r) → orig(x=y_r, y=H-1-x_r)
            img_rot3 = np.rot90(img_arr, k=3)
            for (bbox, text, conf) in reader.readtext(img_rot3, **_OCR_KW):
                orig_bbox = [[float(py), H_pix - 1 - float(px)]
                             for (px, py) in bbox]
                res.append((orig_bbox, text, conf, 2))

            td2 = _easyocr_to_dict(res, pw, ph, W_pix, H_pix)
            n2  = sum(len(sp["text"].strip())
                      for bl in td2.get("blocks", [])
                      for ln in bl.get("lines", [])
                      for sp in ln.get("spans", []))
            if n2 >= 10:
                print(f"[OCR] EasyOCR 3-pass: {n2} chars / {len(res)} regions  "
                      f"(pass0={sum(1 for r in res if r[3]==0)}  "
                      f"pass1={sum(1 for r in res if r[3]==1)}  "
                      f"pass2={sum(1 for r in res if r[3]==2)})")
                return td2, True
        except Exception as e:
            print(f"[OCR] EasyOCR inference failed: {e}")

    # ── 3. PyMuPDF built-in OCR (Tesseract) ──────────────────────────────────
    try:
        tp  = page.get_textpage_ocr(flags=0, language="eng", dpi=300, full=True)
        td3 = page.get_text("dict", textpage=tp)
        n3  = sum(len(sp["text"].strip())
                  for bl in td3.get("blocks", [])
                  for ln in bl.get("lines", [])
                  for sp in ln.get("spans", []))
        if n3 >= 10:
            print(f"[OCR] PyMuPDF/Tesseract: {n3} chars extracted")
            return td3, True
    except Exception as e:
        print(f"[OCR] PyMuPDF OCR: {e}")

    # ── 4. pytesseract ────────────────────────────────────────────────────────
    try:
        import pytesseract as _pyt
        pix     = page.get_pixmap(dpi=200)
        pil_img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
        data    = _pyt.image_to_data(pil_img, output_type=_pyt.Output.DICT,
                                     lang="eng", config="--psm 11")
        td4  = _pytesseract_to_dict(data, pw, ph, pix.width, pix.height)
        n4   = sum(len(sp["text"].strip())
                   for bl in td4.get("blocks", [])
                   for ln in bl.get("lines", [])
                   for sp in ln.get("spans", []))
        if n4 >= 10:
            print(f"[OCR] pytesseract: {n4} chars extracted")
            return td4, True
    except Exception as e:
        print(f"[OCR] pytesseract: {e}")

    print("[OCR] All OCR methods failed for this raster image.")
    return {"blocks": []}, True


# ── Plan boundary (from grid bubble positions) ────────────────────────────────
def find_plan_boundary(page, page_w, page_h, text_dict=None):
    """
    Derive the structural plan extent from grid bubble positions.

    Grid bubbles (letter labels A/B/C… AND number labels 1/2/3…) always appear
    AROUND the perimeter of the structural plan.  Their collective bounding box
    plus a small buffer gives a reliable plan boundary regardless of drawing
    orientation.

    This handles BOTH common layouts transparently:
      • Letters at top/bottom, numbers at left/right   (e.g. "Structural snaps")
      • Letters at left/right, numbers at top/bottom   (e.g. Calsteel drawings)
      • Any mixed layout

    Why NOT separate letter_xs / number_ys (old approach):
      In a "letters-on-left" drawing all letter X values cluster near the left
      edge, so letter_xs gives a ~0-width X range — the plan boundary collapses
      to a sliver and nothing is extracted.

    Gap detection (Y axis):
      If the sorted Y positions of all grid labels have a gap > 12% of page
      height we drop everything below that gap.  Such a gap signals the end of
      the structural plan and the start of the notes / title block area.
    """
    all_xs: list[float] = []
    all_ys: list[float] = []

    _td = text_dict if text_dict is not None else page.get_text("dict")
    for block in _td["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                t  = span["text"].strip()
                fs = span.get("size", 10)
                if not t or len(t) > 5 or fs < 6:
                    continue
                cx = (span["bbox"][0] + span["bbox"][2]) / 2
                cy = (span["bbox"][1] + span["bbox"][3]) / 2

                if _GRID_LETTER.match(t):
                    all_xs.append(cx)
                    all_ys.append(cy)
                elif _GRID_NUMBER.match(t):
                    try:
                        n = float(t)
                        if 0.5 <= n <= 25:
                            all_xs.append(cx)
                            all_ys.append(cy)
                    except ValueError:
                        pass

    # ── Gap detection on Y: drop labels below the first large vertical gap ─────
    # Large gap  ≡  notes / title block is separated from the structural plan.
    # Threshold raised to 20% so that widely-spaced bottom grid rows (e.g. A.7→A)
    # are NOT incorrectly trimmed — those rows still contain real structural members.
    if len(all_ys) >= 3:
        sorted_ys = sorted(all_ys)
        best_gap   = 0.0
        gap_cutoff = sorted_ys[-1]           # default: keep everything
        for i in range(len(sorted_ys) - 1):
            g = sorted_ys[i + 1] - sorted_ys[i]
            if g > best_gap:
                best_gap   = g
                gap_cutoff = sorted_ys[i]    # last Y before the gap
        if best_gap > page_h * 0.28:         # was 0.12 — raised; only fires for very obvious title-block separation
            old_max = max(all_ys)
            all_xs = [all_xs[i] for i, y in enumerate(all_ys) if y <= gap_cutoff]
            all_ys = [y           for y in all_ys               if y <= gap_cutoff]
            if all_ys:
                print(f"[BOUNDARY] gap={best_gap:.0f}pt — trimmed Y max "
                      f"from {old_max:.0f} to {max(all_ys):.0f}")

    if len(all_xs) >= 2 and len(all_ys) >= 2:
        # 120 pt buffer ≈ 1.7″ — large enough to include member labels that sit
        # below/beyond the outermost grid bubble (e.g. HSS10X8X3/8 FLAT at row A).
        buf = 120
        b = (
            max(0,      min(all_xs) - buf),
            max(0,      min(all_ys) - buf),
            min(page_w, max(all_xs) + buf),
            min(page_h, max(all_ys) + buf),
        )
        print(f"[BOUNDARY] grid-derived x=[{b[0]:.0f},{b[2]:.0f}] "
              f"y=[{b[1]:.0f},{b[3]:.0f}]")
        return b

    print("[BOUNDARY] fallback margins")
    return (page_w * 0.03, page_h * 0.02, page_w * 0.97, page_h * 0.98)


# ── Column symbol detection (I/H cross-section marks in vector drawings) ──────
def detect_column_symbols(page):
    """
    Detect the small I-section / W-section plan-view symbols drawn in the PDF.

    In structural steel framing plans, each column position is marked with a
    small I or H shaped symbol (two short horizontal lines joined by a vertical
    web line) representing the W-beam cross-section viewed from above.

    Strategy:
      1. Collect all drawing paths whose bounding box is small (5–50 PDF pts).
      2. For each small path, inspect its line segments for both horizontal AND
         vertical components — the defining characteristic of an I/H shape.
      3. Also accept rectangular paths (some CAD exports draw the column symbol
         as a filled or outlined small rectangle with flanges).
      4. Cluster nearby hits and deduplicate.
    """
    try:
        drawings = page.get_drawings()
    except Exception:
        return []

    raw = []   # (cx, cy) candidates

    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        w, h = rect.width, rect.height

        # Column symbol bounding box: small in both axes
        # Too large = beam line or wall; too small = arrowhead or dimension tick
        if not (4 < w < 50 and 4 < h < 50):
            continue

        cx = (rect.x0 + rect.x1) / 2
        cy = (rect.y0 + rect.y1) / 2

        # ── Check drawing fill colour up-front ───────────────────────────────
        # Column plan marks are solid BLACK fills.
        # Dimension ticks / annotation boxes are un-filled (fill=None) or light.
        # We read the fill once here and use it inside the item loop.
        drawing_fill       = d.get("fill")
        drawing_brightness = 1.0   # assume light until proven dark
        if drawing_fill is not None and len(drawing_fill) >= 3:
            drawing_brightness = (drawing_fill[0] +
                                  drawing_fill[1] +
                                  drawing_fill[2]) / 3

        items     = d.get("items", [])
        has_h     = False
        has_v     = False
        has_curve = False
        n_lines   = 0

        for item in items:
            kind = item[0]
            if kind == "c":
                has_curve = True   # arc/circle → callout bubble, not a column
                break
            elif kind == "l":
                n_lines += 1
                try:
                    p1, p2 = item[1], item[2]
                    dx = abs(p2.x - p1.x)
                    dy = abs(p2.y - p1.y)
                    if math.hypot(dx, dy) < 2:
                        continue
                    if dx > dy * 1.5:
                        has_h = True
                    elif dy > dx * 1.5:
                        has_v = True
                except Exception:
                    continue
            elif kind == "re":
                # PDF rectangle primitive.
                # ONLY accept if the drawing has a VERY DARK (near-black) fill —
                # this is the signature of a structural column plan mark.
                # Un-filled outlines and light-coloured annotation boxes are ignored.
                if drawing_brightness < 0.25:
                    try:
                        rr = item[1]
                        rw = abs(rr.x1 - rr.x0)
                        rh = abs(rr.y1 - rr.y0)
                        ra = rw / rh if rh > 0 else 0
                        # Must be reasonably square and large enough to be a mark
                        if rw >= 4 and rh >= 4 and 0.25 < ra < 4.0:
                            has_h = True
                            has_v = True
                            n_lines += 4
                    except Exception:
                        pass

        aspect = w / h if h > 0 else 0

        # ── Accept I/H shape: flanges (horizontal) + web (vertical) ──────────
        if not has_curve and has_h and has_v:
            raw.append((cx, cy))
            continue

        # ── Accept small outlined rectangle (some CAD exports use a box) ─────
        if not has_curve and n_lines == 4 and 0.5 < aspect < 2.0 and w < 28 and h < 28:
            raw.append((cx, cy))
            continue

        # ── Accept small CIRCLED I/H symbol ───────────────────────────────────
        # Many CAD drawings ring the column I/H mark with a small circle.
        # Grid bubbles are large (>45pt); section-cut bubbles rarely have both
        # H and V lines inside them. Limit to bbox 10–45pt to stay specific.
        if has_curve and has_h and has_v and 10 < w < 45 and 10 < h < 45:
            raw.append((cx, cy))

    # Deduplicate using EXPANDING cluster (DBSCAN-style, eps=15pt).
    #
    # WHY NOT a fixed radius:
    #   • Some CAD exports draw each column symbol as 6-8 separate filled-rectangle
    #     sub-paths (one per flange/web piece). Sub-paths of the SAME symbol are
    #     typically 8-15pt apart from each other.
    #   • A simple fixed-radius check from the SEED only merges points within that
    #     radius of the first point. If the cluster spans 70pt but each step is
    #     only 10pt, a 12pt seed-radius misses outer sub-paths.
    #   • A 60pt fixed radius merges real adjacent columns when the drawing scale
    #     is small (columns can be 40-50pt apart on some sheets).
    #
    # EXPANDING CLUSTER solution:
    #   eps=15pt — small enough to never bridge two real columns (always >30pt apart
    #   at any typical drawing scale), large enough to chain sub-paths of the same
    #   symbol together step-by-step regardless of total symbol span.
    EPS = 15
    symbols = []
    used = [False] * len(raw)
    for i in range(len(raw)):
        if used[i]:
            continue
        # Seed the cluster with point i
        cluster_idx = [i]
        used[i] = True
        # Expand: keep sweeping until no new neighbours are added
        queue = [i]
        while queue:
            cur_idx = queue.pop()
            cx_cur, cy_cur = raw[cur_idx]
            for j in range(len(raw)):
                if not used[j]:
                    if math.hypot(cx_cur - raw[j][0], cy_cur - raw[j][1]) < EPS:
                        used[j] = True
                        cluster_idx.append(j)
                        queue.append(j)
        xs = [raw[k][0] for k in cluster_idx]
        ys = [raw[k][1] for k in cluster_idx]
        symbols.append({
            "cx": sum(xs) / len(xs),
            "cy": sum(ys) / len(ys),
        })

    print(f"[SYMBOLS] Column symbols detected: {len(symbols)}")
    return symbols


# ── Grid line extraction ──────────────────────────────────────────────────────
def extract_grid_lines(page, page_w, page_h, plan_bounds, text_dict=None):
    """
    Extract the X positions of vertical grid lines and the Y positions of
    horizontal grid lines from grid bubble label text.

    Orientation-agnostic strategy
    ─────────────────────────────
    Rather than assuming "letters are at top/bottom" or "numbers are at
    left/right" (which breaks on Calsteel-style drawings that put letters on
    the left and numbers at the top), we measure the SPREAD of each label
    family within the plan boundary:

      • If the label family spans more in X than in Y → the bubbles form a
        horizontal row at the top or bottom of the plan → they label VERTICAL
        grid lines → contribute their X positions to v_raw.

      • If the family spans more in Y than in X → the bubbles form a vertical
        column at the left or right → they label HORIZONTAL grid lines →
        contribute their Y positions to h_raw.

    This works correctly for both common layouts and any mixed-orientation plan.
    """
    bx0, by0, bx1, by1 = plan_bounds

    # Collect positions of each label family inside the plan boundary
    letter_pts: list[tuple[float, float]] = []
    number_pts: list[tuple[float, float]] = []

    _td = text_dict if text_dict is not None else page.get_text("dict")
    for block in _td["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                t  = span["text"].strip()
                fs = span.get("size", 10)
                if not t or len(t) > 5 or fs < 6:
                    continue
                cx = (span["bbox"][0] + span["bbox"][2]) / 2
                cy = (span["bbox"][1] + span["bbox"][3]) / 2
                # Must be inside (or very close to) the plan boundary
                if not (bx0 - 60 <= cx <= bx1 + 60 and
                        by0 - 60 <= cy <= by1 + 60):
                    continue

                if _GRID_LETTER.match(t):
                    letter_pts.append((cx, cy))
                elif _GRID_NUMBER.match(t):
                    try:
                        if not (0.5 <= float(t) <= 25):
                            continue
                    except ValueError:
                        continue
                    number_pts.append((cx, cy))

    def _classify_family(pts):
        """Return (v_positions, h_positions) for a set of bubble label points."""
        if not pts:
            return [], []
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_spread = max(xs) - min(xs)
        y_spread = max(ys) - min(ys)
        if x_spread >= y_spread:
            # Bubbles form a horizontal row → label vertical grid lines → use X
            return xs, []
        else:
            # Bubbles form a vertical column → label horizontal grid lines → use Y
            return [], ys

    lv, lh = _classify_family(letter_pts)
    nv, nh = _classify_family(number_pts)

    v_raw = lv + nv
    h_raw = lh + nh

    def dedup(vals, tol=18):
        out = []
        for v in sorted(vals):
            if not out or abs(v - out[-1]) > tol:
                out.append(v)
        return out

    v_grid = dedup(v_raw)
    h_grid = dedup(h_raw)
    print(f"[GRID] V({len(v_grid)}): {[round(x) for x in v_grid]}")
    print(f"[GRID] H({len(h_grid)}): {[round(y) for y in h_grid]}")
    return v_grid, h_grid


# ── Member classification ─────────────────────────────────────────────────────
def classify_member(profile: str,
                    cx: float = 0, cy: float = 0,
                    column_symbols: list = None,
                    v_grid: list = None,
                    h_grid: list = None) -> str:
    """
    Classify a steel section label as column / beam / brace.

    Works for ANY drawing — ordered from most reliable to least:

      TIER 0  Profile type   PIPE, square HSS → always column
      TIER 1  Symbol nearby  I/H mark detected close to this label
      TIER 2  Grid cross     label sits near a named grid intersection
      TIER 3  Depth rule     W6/W8/W10 are almost always columns in practice
      TIER 4  Weight rule    heavy section for its depth → column

    Keeping TIER 3 (depth rule) separate ensures short W-sections are caught
    even on drawings where symbol detection or grid detection yields nothing.
    """
    p = profile.upper().strip()

    # ── Angle / brace sections ────────────────────────────────────────────────
    if re.match(r'ISA', p) or re.match(r'L\d', p):
        return "brace"

    # ── TIER 0 — Profile type: always a column regardless of drawing ──────────
    if re.match(r'PIPE', p):
        return "column"

    hss = re.match(r'HSS([\d.]+)[Xx]([\d.]+)', p)
    if hss:
        try:
            d1 = float(hss.group(1))
            d2 = float(hss.group(2))
            # Only perfectly square HSS (e.g. HSS6X6, HSS8X8) are columns;
            # all rectangular HSS spanning between grids are beams.
            return "column" if abs(d1 - d2) < 0.5 else "beam"
        except ValueError:
            return "beam"

    w = re.match(r'W(\d+)[Xx](\d+)', p)
    if w:
        depth  = int(w.group(1))
        weight = int(w.group(2))

        # ── TIER 1: near a detected I/H column symbol ─────────────────────────
        # The symbol drawn on the plan is the strongest signal — use it first.
        if column_symbols:
            for sym in column_symbols:
                if math.hypot(cx - sym["cx"], cy - sym["cy"]) < SYMBOL_ASSOC_RADIUS:
                    return "column"

        # ── TIER 2: at a named grid intersection ──────────────────────────────
        # Columns sit exactly at grid line crossings; beams span between them.
        if v_grid and h_grid:
            near_v = any(abs(cx - gx) < GRID_TOL for gx in v_grid)
            near_h = any(abs(cy - gy) < GRID_TOL for gy in h_grid)
            if near_v and near_h:
                return "column"

        # ── TIER 3: depth rule — short W-sections are columns on every drawing ─
        # W6 and W8 are almost never used as beams in structural framing plans.
        # W10 sections are columns far more often than beams.
        # This rule fires even when no symbols or grid are detected.
        if depth <= 8:
            return "column"
        if depth == 10 and weight >= 22:
            return "column"

        # ── TIER 4: weight-per-depth heuristic ────────────────────────────────
        # For W12/W14 (used both ways) and larger sections:
        # a high weight-to-depth ratio signals a compact column section.
        thresholds = {
            12: 40,   # W12×40+  → column  (W12×26/30 stay beam)
            14: 38,   # W14×38+  → column  (W14×22/26/30 stay beam)
            16: 57,   # W16×57+  (W16×26/31/36/40 stay beam)
            18: 71,   # W18×71+  (W18×35/40/46/50 stay beam)
            21: 83,
            24: 94,
            27: 102,
        }
        if weight >= thresholds.get(depth, 9999):
            return "column"

        return "beam"

    return "beam"


# ── Schedule / legend table exclusion ────────────────────────────────────────
def detect_schedule_zones(page, plan_bounds, text_dict=None):
    """
    Detect zones that are purely schedule/legend tables (not part of the
    structural plan) so we don't extract phantom members from them.

    Approach: look for steel-section labels that are OUTSIDE the plan boundary
    entirely (i.e., in the title block, notes column, or border area).
    The plan boundary already correctly clips the extraction region, so we
    only need to exclude labels that are JUST inside the boundary but clearly
    belong to tabular data — identified by extremely tight x-spread (< 8 pt,
    meaning they share almost the exact same x) AND a large y-span (> 250 pt)
    AND a very high label density (≥ 25 labels).

    Previously this used a 12-hit / 30-pt / 150-pt threshold which falsely
    excluded entire column lines on complex framing plans where many beams
    frame into the same grid line, producing a dense vertical label cluster
    that looks superficially like a schedule table.
    """
    bx0, by0, bx1, by1 = plan_bounds

    hits = []
    _td = text_dict if text_dict is not None else page.get_text("dict")
    for block in _td["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                if not text:
                    continue
                cx = (span["bbox"][0] + span["bbox"][2]) / 2
                cy = (span["bbox"][1] + span["bbox"][3]) / 2
                if not (bx0 <= cx <= bx1 and by0 <= cy <= by1):
                    continue
                for pat in STEEL_PATTERNS:
                    if re.search(pat, text, re.IGNORECASE):
                        hits.append((cx, cy))
                        break

    if len(hits) < 10:
        return []

    excluded = []
    used = [False] * len(hits)

    for i, (ax, ay) in enumerate(hits):
        if used[i]:
            continue
        cluster = [(ax, ay)]
        used[i] = True
        for j, (bx, by) in enumerate(hits):
            if not used[j] and abs(ax - bx) < 8:   # very tight: < 8pt x-spread
                cluster.append((bx, by))
                used[j] = True

        # Require 25+ labels in that tiny x-band — only a true printed schedule
        # has this many entries at virtually the same x position.
        # Structural framing plans never concentrate this many beam labels on a
        # single column line with < 8pt x variation.
        if len(cluster) < 25:
            continue

        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        x_spread = max(xs) - min(xs)
        y_spread = max(ys) - min(ys)

        if x_spread < 8 and y_spread > 250:
            zone = (min(xs) - 60, min(ys) - 40, max(xs) + 60, max(ys) + 40)
            excluded.append(zone)
            print(f"[SCHEDULE] Excluded zone x=[{min(xs):.0f},{max(xs):.0f}] "
                  f"y=[{min(ys):.0f},{max(ys):.0f}] n={len(cluster)}")

    return excluded


# ── Profile extraction (within plan boundary only) ────────────────────────────
def extract_profiles(page, page_w, page_h, plan_bounds, text_dict=None):
    """
    Extract steel section labels from within the plan boundary.

    CAD PDFs frequently split a label like 'W18X40' across two text spans
    ('W18' and 'X40').  We match at both span-level AND full-line-level to
    catch every case, then deduplicate by proximity.
    """
    bx0, by0, bx1, by1 = plan_bounds
    excluded_zones = detect_schedule_zones(page, plan_bounds, text_dict=text_dict)
    profiles, seen = [], []

    def _try_add(text, cx, cy, rot_pass=0):
        if not text:
            return
        if not (bx0 <= cx <= bx1 and by0 <= cy <= by1):
            return
        if any(zx0 <= cx <= zx1 and zy0 <= cy <= zy1
               for zx0, zy0, zx1, zy1 in excluded_zones):
            return
        for pat in STEEL_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                if not any(math.hypot(cx - sx, cy - sy) < 20
                           for sx, sy in seen):
                    seen.append((cx, cy))
                    profiles.append({
                        "profile": normalize_profile(m.group(0)),
                        "cx": cx, "cy": cy,
                        # dir_hint: passes 1 & 2 found rotated (vertical) labels
                        "dir_hint": "V" if rot_pass in (1, 2) else "H",
                    })
                break

    _td = text_dict if text_dict is not None else page.get_text("dict")
    for block in _td["blocks"]:
        for line in block.get("lines", []):
            spans = line.get("spans", [])

            # ── Span-level: precise position per span ─────────────────────
            for span in spans:
                bbox = span["bbox"]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                _try_add(span["text"].strip(), cx, cy,
                         rot_pass=span.get("rot_pass", 0))

            # ── Line-level: join all spans → catches "W18" + "X40" splits ─
            # Use the rotation pass of the first span (all spans in one line
            # come from the same OCR pass in the raster path).
            line_text = "".join(s["text"] for s in spans).strip()
            if line.get("bbox") and line_text:
                lb = line["bbox"]
                lx = (lb[0] + lb[2]) / 2
                ly = (lb[1] + lb[3]) / 2
                first_pass = spans[0].get("rot_pass", 0) if spans else 0
                _try_add(line_text, lx, ly, rot_pass=first_pass)

    print(f"[EXTRACT] {len(profiles)} profiles found in plan")
    return profiles


# ── Build members + summary ───────────────────────────────────────────────────
def _span_valid(bx1f, by1f, bx2f, by2f, lx_frac, ly_frac, beam_dir) -> bool:
    """
    Sanity-check: the label must lie on (or very near) the computed span line.

    Why this matters
    ----------------
    For vector PDFs detect_beam_lines() guarantees this by construction.
    For the grid-based fallback (compute_beam_span), wrong grid-line matches
    can produce span endpoints that are on the opposite side of the plan from
    the label — the line appears visually far from the chip on screen.

    Rules (all tolerances as fractions of the page dimension):
      H-beam  — span Y must be within PERP_TOL of label Y
                 label X must fall inside [bx1−EXT_TOL, bx2+EXT_TOL]
      V-beam  — span X must be within PERP_TOL of label X
                 label Y must fall inside [by1−EXT_TOL, by2+EXT_TOL]
    """
    PERP_TOL = 0.09   # perpendicular axis: 9 % of page length
    EXT_TOL  = 0.12   # allow label up to 12 % beyond an endpoint (skewed labels)

    if beam_dir == "H":
        span_y = (by1f + by2f) / 2
        if abs(ly_frac - span_y) > PERP_TOL:
            return False
        x_lo = min(bx1f, bx2f) - EXT_TOL
        x_hi = max(bx1f, bx2f) + EXT_TOL
        return x_lo <= lx_frac <= x_hi
    else:   # "V"
        span_x = (bx1f + bx2f) / 2
        if abs(lx_frac - span_x) > PERP_TOL:
            return False
        y_lo = min(by1f, by2f) - EXT_TOL
        y_hi = max(by1f, by2f) + EXT_TOL
        return y_lo <= ly_frac <= y_hi


def build_members(profiles, page_w, page_h,
                  column_symbols=None, v_grid=None, h_grid=None,
                  pts_per_foot: float = 0.0,
                  beam_dirs: dict = None,
                  beam_line_map: dict = None,
                  plan_bounds: tuple = None):
    """
    Two-pass pipeline that prevents beam labels near a column symbol from
    being falsely promoted to "column" (the old fan-out problem).

    Pass 1 — One-to-one symbol→profile matching (greedy, closest-first).
              Each column symbol claims the single nearest unclaimed profile
              within SYMBOL_ASSOC_RADIUS.  Those profiles are definitively
              columns; no other profile can be claimed by the same symbol.

    Pass 2 — All remaining profiles are classified by TIER 2 / 3 / 4 only
              (no TIER 1 — that was handled exclusively in Pass 1).

    Position snapping is applied afterwards:
      Snap 1 — snap to nearest grid intersection.
      Snap 2 — snap to nearest unclaimed symbol (fallback).
    """
    members = []

    # ── Pass 1: exclusive symbol → profile matching ───────────────────────────
    symbol_matched_cols: set[int] = set()   # profile indices confirmed as columns
    profile_sym_pos: dict[int, tuple] = {}  # p_idx → (sx_frac, sy_frac) of matched symbol
    if column_symbols:
        # Build all (distance, symbol_idx, profile_idx) pairs within radius
        candidates = []
        for s_idx, sym in enumerate(column_symbols):
            for p_idx, p in enumerate(profiles):
                d = math.hypot(p["cx"] - sym["cx"], p["cy"] - sym["cy"])
                if d < SYMBOL_ASSOC_RADIUS:
                    candidates.append((d, s_idx, p_idx))
        candidates.sort()          # process closest pairs first
        used_s: set[int] = set()
        used_p: set[int] = set()
        for d, s_idx, p_idx in candidates:
            if s_idx not in used_s and p_idx not in used_p:
                symbol_matched_cols.add(p_idx)
                used_s.add(s_idx)
                used_p.add(p_idx)
                # Store the symbol position so the frontend region filter can
                # check it — the user draws around the SYMBOL, not the label
                profile_sym_pos[p_idx] = (
                    round(column_symbols[s_idx]["cx"] / page_w, 4),
                    round(column_symbols[s_idx]["cy"] / page_h, 4),
                )
        print(f"[BUILD] {len(symbol_matched_cols)} profiles matched to column symbols "
              f"(of {len(column_symbols)} symbols, {len(profiles)} profiles)")

    # ── Pass 2: classify + snap ───────────────────────────────────────────────
    claimed_symbols: set[int] = set()

    for p_idx, p in enumerate(profiles):
        # Determine member type
        if p_idx in symbol_matched_cols:
            mtype = "column"
        else:
            # TIER 1 is handled exclusively in Pass 1 — pass column_symbols=None
            # here so classify_member uses TIER 2 / 3 / 4 only.
            mtype = classify_member(
                p["profile"], p["cx"], p["cy"],
                column_symbols=None,
                v_grid=v_grid, h_grid=h_grid,
            )

        render_cx, render_cy = p["cx"], p["cy"]
        snapped = False

        if mtype == "column":
            # Snap 1: snap to nearest grid intersection
            if v_grid and h_grid:
                nearest_vx = min(v_grid, key=lambda gx: abs(p["cx"] - gx))
                nearest_hy = min(h_grid, key=lambda gy: abs(p["cy"] - gy))
                if (abs(p["cx"] - nearest_vx) < GRID_SNAP_RADIUS and
                        abs(p["cy"] - nearest_hy) < GRID_SNAP_RADIUS):
                    render_cx = nearest_vx
                    render_cy = nearest_hy
                    snapped = True

            # Snap 2: snap to nearest unclaimed column symbol (fallback)
            if not snapped and column_symbols:
                best, best_dist, best_idx = None, float("inf"), -1
                for idx, sym in enumerate(column_symbols):
                    if idx in claimed_symbols:
                        continue
                    d = math.hypot(p["cx"] - sym["cx"], p["cy"] - sym["cy"])
                    if d < best_dist:
                        best_dist, best, best_idx = d, sym, idx
                if best and best_dist < SYMBOL_SNAP_RADIUS:
                    render_cx = best["cx"]
                    render_cy = best["cy"]
                    claimed_symbols.add(best_idx)
                    snapped = True  # noqa: F841

        # ── Beam direction, length, and span endpoints ────────────────────────
        beam_dir = None
        length_ft = 0.0
        bx1 = by1 = bx2 = by2 = None

        if mtype == "beam":
            line_hit = (beam_line_map or {}).get(p_idx)

            lx_frac = p["cx"] / page_w
            ly_frac = p["cy"] / page_h

            if line_hit:
                # ── PRIMARY: trust the drawn vector line unconditionally ────
                #
                # detect_beam_lines() extracts PDF vector geometry — the line
                # segment IS the beam centreline as drawn by the structural
                # engineer.  Beam labels in complex drawings are often placed
                # off to the side with a leader line, so the label's (cx, cy)
                # can legitimately be 60–100 pt from the centreline.  Validating
                # against label position causes valid matches to be discarded,
                # leaving the chip at the label position while neighbouring
                # beams' lines are drawn at the centreline — the visual
                # "chip here, line over there" offset the user reported.
                #
                # We only reject a match when the line is too short to be a
                # real structural bay (< MIN_STRUCT_PT).  Short lines are
                # dimension ticks, hatch lines, or leader stubs that occasionally
                # fall within LABEL_R of a label.
                MIN_STRUCT_PT = 60   # ≈ 6 ft at 1/8" scale
                beam_dir  = line_hit["dir"]
                length_pt = line_hit["length_pt"]

                if length_pt >= MIN_STRUCT_PT:
                    length_ft = round(length_pt / pts_per_foot, 1) if pts_per_foot > 0 else 0.0
                    bx1 = round(line_hit["x1"] / page_w, 4)
                    by1 = round(line_hit["y1"] / page_h, 4)
                    bx2 = round(line_hit["x2"] / page_w, 4)
                    by2 = round(line_hit["y2"] / page_h, 4)
                    # Chip renders at the TRUE midpoint of the span — this
                    # guarantees the chip and the SVG line always coincide.
                    render_cx = (line_hit["x1"] + line_hit["x2"]) / 2
                    render_cy = (line_hit["y1"] + line_hit["y2"]) / 2
                else:
                    print(f"[BUILD] Vector match too short ({length_pt:.0f} pt) "
                          f"for {p['profile']} — dropped as annotation line")
            else:
                # ── FALLBACK: grid-based approximation ──────────────────────
                # Used when no vector centreline was found (raster images, or
                # diagonal framing plans where H/V detection misses the beam).
                # Priority: vector-based direction dict → OCR rotation hint → "H"
                beam_dir = ((beam_dirs or {}).get(p_idx)
                            or p.get("dir_hint", "H"))

                # Try both directions and pick whichever yields a geometrically
                # valid span.  This helps pages where direction detection is
                # unreliable (e.g. rotated/angled structural grids).
                def _try_span(direction):
                    if not (v_grid and h_grid):
                        return None
                    sp = compute_beam_span(
                        p["cx"], p["cy"], v_grid, h_grid, pts_per_foot, direction)
                    # Edge-beam extension: if no surrounding grid on one side,
                    # extend with the plan boundary so perimeter beams get a length.
                    if sp is None and plan_bounds:
                        pb0x, pb0y, pb1x, pb1y = plan_bounds
                        sp = compute_beam_span(
                            p["cx"], p["cy"],
                            sorted(set(v_grid) | {pb0x, pb1x}),
                            sorted(set(h_grid) | {pb0y, pb1y}),
                            pts_per_foot, direction)
                    return sp

                span = _try_span(beam_dir)

                # If primary direction failed validation, try the other.
                if span:
                    _bx1 = round(span["x1"] / page_w, 4)
                    _by1 = round(span["y1"] / page_h, 4)
                    _bx2 = round(span["x2"] / page_w, 4)
                    _by2 = round(span["y2"] / page_h, 4)
                    if not _span_valid(_bx1, _by1, _bx2, _by2, lx_frac, ly_frac, beam_dir):
                        print(f"[BUILD] Grid span invalid for {p['profile']} "
                              f"dir={beam_dir} — trying opposite direction")
                        alt_dir = "V" if beam_dir == "H" else "H"
                        alt_span = _try_span(alt_dir)
                        if alt_span:
                            ab1 = round(alt_span["x1"] / page_w, 4)
                            ab2 = round(alt_span["y1"] / page_h, 4)
                            ab3 = round(alt_span["x2"] / page_w, 4)
                            ab4 = round(alt_span["y2"] / page_h, 4)
                            if _span_valid(ab1, ab2, ab3, ab4, lx_frac, ly_frac, alt_dir):
                                span = alt_span
                                _bx1, _by1, _bx2, _by2 = ab1, ab2, ab3, ab4
                                beam_dir = alt_dir
                            else:
                                span = None   # both directions invalid
                        else:
                            span = None

                if span:
                    _bx1 = round(span["x1"] / page_w, 4)
                    _by1 = round(span["y1"] / page_h, 4)
                    _bx2 = round(span["x2"] / page_w, 4)
                    _by2 = round(span["y2"] / page_h, 4)
                    if _span_valid(_bx1, _by1, _bx2, _by2, lx_frac, ly_frac, beam_dir):
                        length_ft = span["length_ft"]
                        bx1, by1, bx2, by2 = _bx1, _by1, _bx2, _by2
                        render_cx = (span["x1"] + span["x2"]) / 2
                        render_cy = (span["y1"] + span["y2"]) / 2
                    else:
                        print(f"[BUILD] Grid span discarded for {p['profile']} "
                              f"(label=({lx_frac:.3f},{ly_frac:.3f}) "
                              f"span=({_bx1:.3f},{_by1:.3f})→({_bx2:.3f},{_by2:.3f})"
                              f" dir={beam_dir})")

        sym = profile_sym_pos.get(p_idx)
        members.append({
            "profile":   p["profile"],
            "type":      mtype,
            "length_ft": length_ft,
            "beam_dir":  beam_dir,   # "H" | "V" for beams; None for columns/braces
            # Beam span endpoints (fractions of page) for the line overlay.
            # bx1/by1 → bx2/by2 is the physical line drawn from column to column.
            "bx1": bx1, "by1": by1,
            "bx2": bx2, "by2": by2,
            # render position — midpoint of span (or snapped to grid/symbol for cols)
            "x":  round(render_cx / page_w, 4),
            "y":  round(render_cy / page_h, 4),
            # original label text position — used by region filter
            "lx": round(p["cx"] / page_w, 4),
            "ly": round(p["cy"] / page_h, 4),
            # matched column-symbol position (only for symbol-matched columns)
            "sx": sym[0] if sym else None,
            "sy": sym[1] if sym else None,
            "w":  0.025, "h": 0.012,
            "color":     MEMBER_COLORS.get(mtype, "#6B7280"),
            "confirmed": True,
            "is_column": mtype == "column",
        })

    return members


def build_summary(members):
    s = {
        "column": 0, "beam": 0,
        "vertical_brace": 0, "horizontal_brace": 0,
        "joists": 0, "moment_connection": 0, "default_connection": 0,
        "bolt": 0, "camber": 0, "anchor": 0,
        "weld_studs": 0, "total_weight_tons": 0,
    }
    for m in members:
        t = m.get("type", "beam")
        if   t == "column":                      s["column"] += 1
        elif t == "beam":                         s["beam"]   += 1
        elif t in ("brace", "vertical_brace"):    s["vertical_brace"] += 1
        elif t == "horizontal_brace":             s["horizontal_brace"] += 1
    return s


# ── Request models ────────────────────────────────────────────────────────────
class AnalysisRequest(BaseModel):
    filename:    str
    page_index:  int   = 0
    scale_ratio: float = None
    ocr_dpi:     int   = 400

class SaveProjectRequest(BaseModel):
    name:        str
    filename:    str
    scale:       str   = None
    scale_ratio: int   = None
    members:     list  = []
    page_count:  int   = 1


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("===== STEELGENIE BACKEND STARTED =====")

@app.get("/health")
def health():
    return {"status": "ok", "supabase": supabase_client is not None}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    content   = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    MAX_PREVIEW_W  = 2800   # full-size preview width cap
    MAX_THUMB_W    = 220    # sidebar thumbnail width cap

    def _render_page_to_b64(pil_img: "PILImage.Image", max_w: int, quality: int, fmt: str = "JPEG") -> str:
        if pil_img.width > max_w:
            ratio = max_w / pil_img.width
            pil_img = pil_img.resize((max_w, int(pil_img.height * ratio)), PILImage.LANCZOS)
        buf = io.BytesIO()
        if fmt == "PNG":
            pil_img.save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        pil_img.save(buf, format="JPEG", quality=quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    ext = os.path.splitext(file.filename)[1].lower()
    if ext == ".pdf":
        doc        = fitz.open(file_path)
        page_count = len(doc)

        # Full-size preview for page 0 (viewed in the drawing window)
        pix0 = doc[0].get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
        pil0 = PILImage.frombytes("RGB", [pix0.width, pix0.height], pix0.samples)

        # Small thumbnails for every page (sidebar strip)
        page_thumbnails = []
        for i in range(page_count):
            pix_t = doc[i].get_pixmap(matrix=fitz.Matrix(0.5, 0.5))
            pil_t = PILImage.frombytes("RGB", [pix_t.width, pix_t.height], pix_t.samples)
            page_thumbnails.append(_render_page_to_b64(pil_t, MAX_THUMB_W, 70))

        doc.close()
        main_b64 = _render_page_to_b64(pil0, MAX_PREVIEW_W, 85, fmt="PNG")
    else:
        pil = PILImage.open(file_path).convert("RGB")
        page_count = 1
        main_b64 = _render_page_to_b64(pil, MAX_PREVIEW_W, 85, fmt="PNG")
        page_thumbnails = [_render_page_to_b64(
            pil.resize((MAX_THUMB_W, int(pil.height * MAX_THUMB_W / pil.width)), PILImage.LANCZOS), MAX_THUMB_W, 70)]

    return {
        "image":           main_b64,
        "page_count":      page_count,
        "filename":        file.filename,
        "page_thumbnails": page_thumbnails,
    }


@app.get("/page-image/{filename}/{page_index}")
async def get_page_image(filename: str, page_index: int):
    """Return full-size preview for a specific page of an already-uploaded PDF."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        doc = fitz.open(file_path)
        if page_index < 0 or page_index >= len(doc):
            doc.close()
            raise HTTPException(404, f"Page {page_index} not found (total {len(doc)})")
        pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
        pil = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
    else:
        if page_index != 0:
            raise HTTPException(404, "Image files have only one page")
        pil = PILImage.open(file_path).convert("RGB")

    MAX_PREVIEW_W = 2800
    if pil.width > MAX_PREVIEW_W:
        r = MAX_PREVIEW_W / pil.width
        pil = pil.resize((MAX_PREVIEW_W, int(pil.height * r)), PILImage.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"image": f"data:image/png;base64,{b64}", "page_index": page_index}


@app.post("/analyse")
async def analyse_pdf(req: AnalysisRequest):
    start = time.time()
    try:
        tmp_path = os.path.join(UPLOAD_DIR, req.filename)
        ext      = os.path.splitext(req.filename)[1].lower()

        doc  = fitz.open(tmp_path)
        page = doc[req.page_index]

        # ── Page dimensions ───────────────────────────────────────────────────
        # For raster image files (JPEG, PNG…) fitz may scale dimensions by the
        # image's embedded DPI metadata (96 DPI screenshot → 75% of pixel size).
        # We always use PIL pixel dimensions so the fractional member positions
        # align with the preview image the frontend received from /upload.
        if ext in _IMAGE_EXTS:
            _pil_tmp = PILImage.open(tmp_path)
            page_w   = float(_pil_tmp.width)
            page_h   = float(_pil_tmp.height)
            _pil_tmp.close()
            print(f"[ANALYSE] Image file — using PIL dims {page_w:.0f}x{page_h:.0f}")
        else:
            page_w, page_h = page.rect.width, page.rect.height

        # ── 0. Text extraction (vector embedded OR OCR for raster images) ─────
        text_dict, is_raster = _get_text_dict(page, page_w, page_h)
        print(f"[ANALYSE] is_raster={is_raster}  text_blocks={len(text_dict.get('blocks', []))}")

        # 1. Plan boundary — excludes schedules, notes, title block
        plan_bounds = find_plan_boundary(page, page_w, page_h, text_dict=text_dict)

        # 2. Detect column symbols (I/H shapes in vector paths) — skip for raster
        column_symbols = [] if is_raster else detect_column_symbols(page)

        # 3. Grid lines — SECONDARY signal
        v_grid, h_grid = extract_grid_lines(page, page_w, page_h, plan_bounds,
                                            text_dict=text_dict)

        # 4. Extract profile labels within the plan
        profiles = extract_profiles(page, page_w, page_h, plan_bounds,
                                    text_dict=text_dict)
        print(f"[ANALYSE] Profiles in plan: {len(profiles)}")

        # ── DIAGNOSTIC DUMP ──────────────────────────────────────────────────
        syms_in_plan = [s for s in column_symbols
                        if plan_bounds[0] <= s["cx"] <= plan_bounds[2]
                        and plan_bounds[1] <= s["cy"] <= plan_bounds[3]]
        print(f"\n{'='*60}")
        print(f"[DIAG] Page size      : {page_w:.0f} x {page_h:.0f} pt")
        print(f"[DIAG] Raster mode    : {is_raster}")
        print(f"[DIAG] Plan boundary  : x=[{plan_bounds[0]:.0f},{plan_bounds[2]:.0f}] "
              f"y=[{plan_bounds[1]:.0f},{plan_bounds[3]:.0f}]")
        print(f"[DIAG] V grid lines   : {len(v_grid)}  -> {[round(x) for x in v_grid]}")
        print(f"[DIAG] H grid lines   : {len(h_grid)}  -> {[round(y) for y in h_grid]}")
        print(f"[DIAG] Symbols total  : {len(column_symbols)}  "
              f"| inside plan: {len(syms_in_plan)}")
        for s in column_symbols:
            tag = "IN " if (plan_bounds[0] <= s["cx"] <= plan_bounds[2]
                            and plan_bounds[1] <= s["cy"] <= plan_bounds[3]) else "OUT"
            print(f"  [{tag}] cx={s['cx']:.0f}  cy={s['cy']:.0f}")
        print(f"[DIAG] Profiles found : {len(profiles)}")
        for i, p in enumerate(profiles):
            print(f"  [{i:02d}] {p['profile']:14s}  cx={p['cx']:.0f}  cy={p['cy']:.0f}")
        print(f"{'='*60}\n")
        # ─────────────────────────────────────────────────────────────────────

        # 5. Scale → pts per foot (from user-selected drawing scale)
        pts_per_foot = scale_to_pts_per_foot(req.scale_ratio) if req.scale_ratio else 0.0
        print(f"[ANALYSE] scale_ratio={req.scale_ratio}  pts_per_foot={pts_per_foot:.2f}")

        # 6. PRIMARY: match beam centerlines to profile labels
        #    • Vector PDF: read page.get_drawings() — exact mathematical geometry.
        #    • Raster image: render to 300-DPI greyscale and run HoughLinesP.
        #      Before this fix the raster path skipped this step entirely
        #      (beam_line_map = {}), forcing 100 % of beams through the
        #      grid-based fallback which only achieves bay-level precision.
        if is_raster and _RASTER_HOUGH_AVAILABLE and profiles:
            _DPI_HOUGH  = 300
            _PX_PER_PT  = _DPI_HOUGH / 72.0
            _pix        = page.get_pixmap(dpi=_DPI_HOUGH, colorspace=fitz.csGRAY)
            _gray_arr   = np.frombuffer(_pix.samples, dtype=np.uint8).reshape(
                              _pix.height, _pix.width)
            beam_line_map = _detect_beam_lines_raster(
                _gray_arr, profiles, plan_bounds, _PX_PER_PT)
            print(f"[ANALYSE] Raster Hough beam_line_map: {len(beam_line_map)} hits")
            del _pix, _gray_arr   # free memory
        else:
            beam_line_map = (
                {} if is_raster
                else detect_beam_lines(page, profiles, plan_bounds)
            )

        # 7. FALLBACK direction detection
        #    • Vector: use adjacent drawn lines via detect_beam_directions().
        #    • Raster:  Hough hits already carry a 'dir' key; OCR rotation hints
        #      (dir_hint) fill in for the remaining unmatched profiles.
        if is_raster:
            # Build beam_dirs from Hough results first, then fill gaps from dir_hint
            beam_dirs: dict[int, str] = {}
            for p_idx, hit in beam_line_map.items():
                beam_dirs[p_idx] = hit["dir"]
            # Any profile not matched by Hough falls back to OCR rotation hint
            for p_idx, p in enumerate(profiles):
                if p_idx not in beam_dirs:
                    beam_dirs[p_idx] = p.get("dir_hint", "H")
            h_hints = sum(1 for d in beam_dirs.values() if d == "H")
            v_hints = sum(1 for d in beam_dirs.values() if d == "V")
            print(f"[ANALYSE] Raster beam_dirs (Hough+hint): H={h_hints}  V={v_hints}")
        else:
            beam_dirs = detect_beam_directions(page, profiles, plan_bounds)

        # 8. Classify and build
        members = build_members(profiles, page_w, page_h,
                                column_symbols=column_symbols,
                                v_grid=v_grid, h_grid=h_grid,
                                pts_per_foot=pts_per_foot,
                                beam_dirs=beam_dirs,
                                beam_line_map=beam_line_map,
                                plan_bounds=plan_bounds)
        summary = build_summary(members)

        counts  = {t: summary[t] for t in ["column", "beam", "vertical_brace"]}
        elapsed = round(time.time() - start, 2)
        method  = ("pymupdf+ocr+hough+grid" if (is_raster and _RASTER_HOUGH_AVAILABLE)
                   else "pymupdf+ocr+grid"   if is_raster
                   else "pymupdf+symbols+grid")
        print(f"[ANALYSE] {len(members)} members in {elapsed}s — {counts}")

        doc.close()
        return {
            "members":         members,
            "summary":         summary,
            "method":          method,
            "elapsed":         elapsed,
            "elapsed_seconds": elapsed,
            "count":           len(members),
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/projects")
async def get_projects():
    if not supabase_client:
        raise HTTPException(503, "Supabase not configured")
    return supabase_client.table("projects").select("*, files(*)").execute().data


_SAVED_FILE = os.path.join(BASE_DIR, "saved_projects.json")


@app.post("/save-project")
async def save_project(req: SaveProjectRequest):
    import json as _json, uuid as _uuid

    projects: list = []
    if os.path.exists(_SAVED_FILE):
        try:
            with open(_SAVED_FILE, "r", encoding="utf-8") as f:
                projects = _json.load(f)
        except Exception:
            projects = []

    entry = {
        "id":           str(_uuid.uuid4()),
        "name":         req.name,
        "filename":     req.filename,
        "scale":        req.scale,
        "scale_ratio":  req.scale_ratio,
        "members":      req.members,
        "member_count": len(req.members),
        "page_count":   req.page_count,
        "created_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    projects.insert(0, entry)

    with open(_SAVED_FILE, "w", encoding="utf-8") as f:
        _json.dump(projects, f, ensure_ascii=False, indent=2)

    return {"status": "ok", "id": entry["id"]}


@app.get("/saved-projects")
async def get_saved_projects():
    import json as _json

    if not os.path.exists(_SAVED_FILE):
        return []
    try:
        with open(_SAVED_FILE, "r", encoding="utf-8") as f:
            projects = _json.load(f)
        return [
            {
                "id":           p.get("id"),
                "name":         p.get("name"),
                "filename":     p.get("filename", ""),
                "scale":        p.get("scale", ""),
                "member_count": p.get("member_count", 0),
                "page_count":   p.get("page_count", 1),
                "created_at":   p.get("created_at", ""),
            }
            for p in projects
        ]
    except Exception:
        return []


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
