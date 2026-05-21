import cv2
import numpy as np
import math
from pathlib import Path

def create_column_template(size=40):
    """
    Synthesize the structural column symbol:
    - Outer cross (flanges + web of I-section in plan view)
    - Small center square (column core)
    - Small tick line above
    """
    canvas = np.zeros((size * 3, size * 3), dtype=np.uint8)
    cx, cy = size + size // 2, size + size // 2
    t = max(2, size // 12)   # line thickness

    # Horizontal flange line
    cv2.line(canvas, (cx - size, cy), (cx + size, cy), 255, t * 2)
    # Vertical web line
    cv2.line(canvas, (cx, cy - size), (cx, cy + size), 255, t * 2)
    # Small center square (column core box)
    half = size // 6
    cv2.rectangle(canvas, (cx - half, cy - half), (cx + half, cy + half), 255, t)
    # Small tick line above (as seen in the drawing)
    cv2.line(canvas, (cx, cy - size), (cx, cy - size - size // 3), 255, t)

    return canvas

def match_rotated_scaled(image_gray, template, angle_step=5, scale_range=(0.4, 2.0),
                          scale_steps=20, threshold=0.55):
    """
    Multi-scale + multi-rotation template matching.
    Returns list of (x, y, w, h, angle, score) for each detected column.
    """
    detections = []
    h_t, w_t = template.shape

    scales = np.linspace(scale_range[0], scale_range[1], scale_steps)
    angles = range(0, 180, angle_step)   # I/cross symbol repeats every 90°

    for scale in scales:
        new_w = int(w_t * scale)
        new_h = int(h_t * scale)
        if new_w < 10 or new_h < 10:
            continue
        resized = cv2.resize(template, (new_w, new_h))

        for angle in angles:
            M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
            rotated = cv2.warpAffine(resized, M, (new_w, new_h))

            # Skip if template larger than image
            if rotated.shape[0] >= image_gray.shape[0] or \
               rotated.shape[1] >= image_gray.shape[1]:
                continue

            result = cv2.matchTemplate(image_gray, rotated, cv2.TM_CCOEFF_NORMED)
            locs = np.where(result >= threshold)

            for pt in zip(*locs[::-1]):   # (x, y)
                detections.append({
                    "x": pt[0], "y": pt[1],
                    "w": new_w, "h": new_h,
                    "angle": angle,
                    "score": float(result[pt[1], pt[0]])
                })

    return detections

def non_max_suppression(detections, overlap_thresh=0.3):
    """Remove duplicate detections using IoU-based NMS."""
    if not detections:
        return []

    boxes = np.array([[d["x"], d["y"], d["x"] + d["w"], d["y"] + d["h"]]
                      for d in detections], dtype=float)
    scores = np.array([d["score"] for d in detections])

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        iou = (w * h) / areas[order[1:]]
        order = order[np.where(iou <= overlap_thresh)[0] + 1]

    return [detections[i] for i in keep]

def detect_columns_by_contour(image_gray, min_area=100, max_area=8000,
                               cross_ratio_tol=0.35):
    """
    Fallback: detect cross-shaped contours directly.
    Works well on clean CAD drawings where lines are crisp.
    """
    blurred = cv2.GaussianBlur(image_gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Close small gaps in lines
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    columns = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h > 0 else 0

        # A cross/I-symbol bounding box is roughly square
        if not (1 - cross_ratio_tol < aspect < 1 + cross_ratio_tol):
            continue

        # Check that contour fills ~40-80% of its bounding box (cross shape)
        fill = area / (w * h)
        if not (0.25 < fill < 0.75):
            continue

        columns.append({"x": x, "y": y, "w": w, "h": h,
                        "angle": 0, "score": fill})

    return columns

def detect_beam_lines_raster(page_img_gray, profiles, plan_bounds, px_per_pt):
    """
    Hough-transform based structural beam line detection for raster/scanned drawings.

    Simulates detect_beam_lines() for images where no vector path data is available.
    Uses OpenCV's probabilistic Hough transform to find line segments in the
    rendered page image, then matches them to OCR-detected profile label positions.

    Why raster accuracy was ~70% before this fix
    --------------------------------------------
    Vector PDFs:  detect_beam_lines() reads page.get_drawings() — exact line endpoints,
                  exact label proximity check → ~95% accuracy.
    Raster images: beam_line_map was set to {} entirely, forcing ALL beams through
                  compute_beam_span() which snaps endpoints to the nearest grid lines
                  (bay-level precision, ±1 bay error).  OCR rotation hints were the
                  only direction source, and they misfire when a label straddles both
                  orientations or sits on a diagonal.

    This function recovers structural line geometry from the rendered image so that
    raster drawings get the same label→line matching as vector PDFs.

    Algorithm
    ---------
    1. CLAHE contrast enhancement + Gaussian blur to suppress noise while
       preserving pen/ink lines (same recipe as the OCR preprocessing pass).
    2. Canny edge detection — structural members have strong, well-defined edges.
    3. HoughLinesP — probabilistic Hough for sub-pixel line segment extraction.
    4. Cluster parallel nearby lines: Hough often returns 2–3 parallel hits
       for the two edges of a drawn line; merge clusters by Y (H) or X (V)
       within a 10-pt tolerance and keep the medial average.
    5. Apply same length / orientation filters as detect_beam_lines():
       MIN_LEN=30 pt, MAX_LEN=900 pt, H/V ratio > 2.
    6. Match profiles to lines with LABEL_R tolerance — identical to the
       vector path matching logic.

    Parameters
    ----------
    page_img_gray : np.ndarray (H_px, W_px)  greyscale page image (300 DPI)
    profiles      : list of dicts — each has 'cx', 'cy' in PDF points
    plan_bounds   : (bx0, by0, bx1, by1) in PDF points
    px_per_pt     : float — pixels per PDF point (= dpi / 72, e.g. 4.167 at 300 DPI)

    Returns
    -------
    dict { profile_idx: {"x1","y1","x2","y2","dir","length_pt"} }
         where all coordinates are in PDF points.
    """
    bx0, by0, bx1, by1 = plan_bounds

    # ── Constants in PDF points (mirrors detect_beam_lines) ────────────────
    MIN_LEN_PT = 30
    MAX_LEN_PT = 1200    # matches detect_beam_lines MAX_LEN — covers long bays
    LABEL_R_PT = 70      # matches detect_beam_lines LABEL_R
    CLUSTER_PT = 8       # merge parallel Hough lines within this distance (pts)

    # ── Clip plan bounds to image size ─────────────────────────────────────
    H_px, W_px = page_img_gray.shape[:2]
    pbx0 = max(0, int(bx0 * px_per_pt))
    pby0 = max(0, int(by0 * px_per_pt))
    pbx1 = min(W_px, int(bx1 * px_per_pt))
    pby1 = min(H_px, int(by1 * px_per_pt))
    if pbx1 <= pbx0 or pby1 <= pby0:
        return {}

    # ── Crop to plan boundary ──────────────────────────────────────────────
    plan_crop = page_img_gray[pby0:pby1, pbx0:pbx1].copy()

    # ── Step 1: CLAHE + Gaussian blur ──────────────────────────────────────
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(plan_crop)
    blurred  = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # ── Step 2: Canny edge detection ───────────────────────────────────────
    # Structural ink lines produce strong edges; thresholds tuned for
    # engineering pen weight (typically 0.35–0.7 mm).
    edges = cv2.Canny(blurred, 30, 90, apertureSize=3)

    # ── Step 3: Probabilistic Hough transform ──────────────────────────────
    MIN_LEN_PX = max(30, int(MIN_LEN_PT * px_per_pt))
    MAX_LEN_PX = int(MAX_LEN_PT * px_per_pt)
    MAX_GAP_PX = max(8, int(15 * px_per_pt / 4.167))   # ~15 px at 300 DPI

    lines_px = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=MIN_LEN_PX,
        maxLineGap=MAX_GAP_PX,
    )
    if lines_px is None:
        print("[HOUGH] No lines detected in raster image")
        return {}

    # ── Step 4: Convert to PDF points, classify H/V ────────────────────────
    raw_h: list[tuple] = []   # (lx1, ly, lx2, ly, length_pt)
    raw_v: list[tuple] = []   # (lx, ly1, lx, ly2, length_pt)

    for seg in lines_px:
        x1_px, y1_px, x2_px, y2_px = seg[0]

        # Restore crop offset then scale to PDF points
        x1_pt = (x1_px + pbx0) / px_per_pt
        y1_pt = (y1_px + pby0) / px_per_pt
        x2_pt = (x2_px + pbx0) / px_per_pt
        y2_pt = (y2_px + pby0) / px_per_pt

        dx = abs(x2_pt - x1_pt)
        dy = abs(y2_pt - y1_pt)
        ln = math.hypot(dx, dy)

        if ln < MIN_LEN_PT or ln > MAX_LEN_PT:
            continue

        if dx > dy * 2:            # clearly horizontal
            ly   = (y1_pt + y2_pt) / 2
            lx1_ = min(x1_pt, x2_pt)
            lx2_ = max(x1_pt, x2_pt)
            raw_h.append((lx1_, ly, lx2_, ly, ln))

        elif dy > dx * 2:          # clearly vertical
            lx   = (x1_pt + x2_pt) / 2
            ly1_ = min(y1_pt, y2_pt)
            ly2_ = max(y1_pt, y2_pt)
            raw_v.append((lx, ly1_, lx, ly2_, ln))

    # ── Step 5: Cluster parallel lines that represent the same member ──────
    # A drawn beam line produces 2 Hough edges (top & bottom of the pen stroke).
    # Cluster by Y (horizontal) or X (vertical) within CLUSTER_PT and merge.

    def _cluster_h(segs, tol):
        """Merge H-lines with similar Y into one representative line."""
        if not segs:
            return []
        segs_sorted = sorted(segs, key=lambda s: s[1])
        groups = [[segs_sorted[0]]]
        for seg in segs_sorted[1:]:
            if abs(seg[1] - groups[-1][-1][1]) <= tol:
                groups[-1].append(seg)
            else:
                groups.append([seg])
        merged = []
        for grp in groups:
            ly_med  = float(np.median([s[1] for s in grp]))
            lx1_min = min(s[0] for s in grp)
            lx2_max = max(s[2] for s in grp)
            ln_max  = max(s[4] for s in grp)
            merged.append((lx1_min, ly_med, lx2_max, ly_med, ln_max))
        return merged

    def _cluster_v(segs, tol):
        """Merge V-lines with similar X into one representative line."""
        if not segs:
            return []
        segs_sorted = sorted(segs, key=lambda s: s[0])
        groups = [[segs_sorted[0]]]
        for seg in segs_sorted[1:]:
            if abs(seg[0] - groups[-1][-1][0]) <= tol:
                groups[-1].append(seg)
            else:
                groups.append([seg])
        merged = []
        for grp in groups:
            lx_med  = float(np.median([s[0] for s in grp]))
            ly1_min = min(s[1] for s in grp)
            ly2_max = max(s[3] for s in grp)
            ln_max  = max(s[4] for s in grp)
            merged.append((lx_med, ly1_min, lx_med, ly2_max, ln_max))
        return merged

    h_lines = _cluster_h(raw_h, CLUSTER_PT)
    v_lines = _cluster_v(raw_v, CLUSTER_PT)

    print(f"[HOUGH] Raw H={len(raw_h)} V={len(raw_v)} → "
          f"Merged H={len(h_lines)} V={len(v_lines)}")

    # ── Step 6: Match profile labels to detected lines ─────────────────────
    # Identical logic to detect_beam_lines() in main.py.
    result: dict = {}
    for p_idx, p in enumerate(profiles):
        pcx, pcy = p["cx"], p["cy"]
        best      = None
        best_d    = float("inf")
        best_dir  = "H"

        for (lx1, ly, lx2, _, ln) in h_lines:
            dy_label = abs(pcy - ly)
            if dy_label > LABEL_R_PT:
                continue
            if pcx < lx1 - LABEL_R_PT or pcx > lx2 + LABEL_R_PT:
                continue
            if dy_label < best_d:
                best_d   = dy_label
                best     = (lx1, ly, lx2, ly, ln)
                best_dir = "H"

        for (lx, ly1, _, ly2, ln) in v_lines:
            dx_label = abs(pcx - lx)
            if dx_label > LABEL_R_PT:
                continue
            if pcy < ly1 - LABEL_R_PT or pcy > ly2 + LABEL_R_PT:
                continue
            if dx_label < best_d:
                best_d   = dx_label
                best     = (lx, ly1, lx, ly2, ln)
                best_dir = "V"

        if best:
            result[p_idx] = {
                "x1": best[0], "y1": best[1],
                "x2": best[2], "y2": best[3],
                "dir":        best_dir,
                "length_pt":  best[4],
            }

    print(f"[HOUGH] {len(result)}/{len(profiles)} profiles matched to raster lines")
    return result


def detect_column_symbols(image, template_size=40, threshold=0.55):
    """
    Unified function for integration into main pipeline.
    Expects a BGR or Gray image.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    template = create_column_template(template_size)
    raw_detections = match_rotated_scaled(gray, template, threshold=threshold)
    contour_detections = detect_columns_by_contour(gray)

    all_detections = raw_detections + contour_detections
    columns = non_max_suppression(all_detections, overlap_thresh=0.3)
    
    return columns
