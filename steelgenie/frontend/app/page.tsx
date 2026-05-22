"use client";
import { useState, useRef, useCallback, useEffect, useMemo } from "react";

// ── TYPES ──────────────────────────────────────────────────────────────────────
interface Member {
  profile: string;
  type: string;
  length_ft: number;
  beam_dir?: string;  // "H" | "V" — horizontal or vertical beam orientation
  // Beam span line endpoints (fractions of page) — used for the SVG line overlay.
  // Null when the beam is at the plan boundary or grid data is unavailable.
  bx1?: number | null; by1?: number | null;
  bx2?: number | null; by2?: number | null;
  x: number;   // midpoint render position (centre of span for beams)
  y: number;
  lx?: number; // original label text position
  ly?: number;
  sx?: number; // column symbol position (only for symbol-matched columns)
  sy?: number;
  w: number;
  h: number;
  color: string;
  overridden?: boolean;
}

interface Summary {
  column: number;
  beam: number;
  vertical_brace: number;
  horizontal_brace: number;
  joists: number;
  moment_connection: number;
  default_connection: number;
  bolt: number;
  camber: number;
  anchor: number;
  weld_studs: number;
  total_weight_tons: number;
  hrs_ton: string;
}

interface Toast {
  type: "blue" | "green" | "red";
  message: string;
}

interface ContextMenu {
  member: Member;
  idx: number;
  x: number;
  y: number;
}

interface CropRect { x0: number; y0: number; x1: number; y1: number; }

// ── CONSTANTS ─────────────────────────────────────────────────────────────────
const MEMBER_COLORS: Record<string, string> = {
  beam:   "#EC4899",
  column: "#3B82F6",
  brace:  "#F59E0B",
};

const ZOOM_STEPS = [0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4];

const SCALE_OPTIONS = [
  { label: '1/32" = 1\'-0"', ratio: 384 },
  { label: '3/64" = 1\'-0"', ratio: 256 },
  { label: '1/16" = 1\'-0"', ratio: 192 },
  { label: '3/32" = 1\'-0"', ratio: 128 },
  { label: '1/8" = 1\'-0"',  ratio: 96  },
  { label: '3/16" = 1\'-0"', ratio: 64  },
  { label: '1/4" = 1\'-0"',  ratio: 48  },
  { label: '3/8" = 1\'-0"',  ratio: 32  },
];

const SUMMARY_ROWS = [
  { key: "column",             label: "Column"             },
  { key: "beam",               label: "Beam"               },
  { key: "vertical_brace",     label: "Vertical Brace"     },
  { key: "horizontal_brace",   label: "Horizontal Brace"   },
  { key: "joists",             label: "Joists"             },
  { key: "moment_connection",  label: "Moment Connection"  },
  { key: "default_connection", label: "Default Connection" },
  { key: "bolt",               label: "Bolt"               },
  { key: "camber",             label: "Camber"             },
  { key: "anchor",             label: "Anchor"             },
  { key: "weld_studs",         label: "Weld Studs"         },
  { key: "total_weight_tons",  label: "Total Weight (tons)"},
];

// ── MAIN COMPONENT ─────────────────────────────────────────────────────────────
export default function Home() {
  const [pdfImage, setPdfImage]           = useState<string | null>(null);
  const [filename, setFilename]           = useState("");
  const [pageCount, setPageCount]         = useState(0);
  const [pageThumbnails, setPageThumbnails] = useState<string[]>([]);
  const [selectedPageIndex, setSelectedPageIndex] = useState(0);
  const [uploading, setUploading]         = useState(false);
  const [uploadStage, setUploadStage]     = useState<"sending"|"processing"|null>(null);
  const [dragging, setDragging]           = useState(false);
  const [selectedScale, setSelectedScale] = useState<string | null>(null);
  const [selectedRatio, setSelectedRatio] = useState<number | null>(null);
  const [scaleOpen, setScaleOpen]         = useState(false);
  const [status, setStatus]               = useState<"not_set"|"estimating"|"built">("not_set");
  const [members, setMembers]             = useState<Member[]>([]);
  const [baseSummary, setBaseSummary]     = useState<Summary | null>(null);
  const [toast, setToast]                 = useState<Toast | null>(null);
  const [hoveredMember, setHoveredMember] = useState<Member | null>(null);
  const [tooltipPos, setTooltipPos]       = useState({ x: 0, y: 0 });
  const [zoomLevel, setZoomLevel]         = useState(1);
  const [contextMenu, setContextMenu]     = useState<ContextMenu | null>(null);
  const [cropMode, setCropMode]           = useState(false);
  const [cropDrag, setCropDrag]           = useState<CropRect | null>(null);
  const [cropRect, setCropRect]           = useState<CropRect | null>(null);
  const [activeTool, setActiveTool]       = useState<"select"|"hand"|"ruler"|"marker">("select");
  const [isPanning, setIsPanning]         = useState(false);
  const [panStart, setPanStart]           = useState({ x: 0, y: 0, scrollLeft: 0, scrollTop: 0 });
  const [rulerStart, setRulerStart]       = useState<{x:number;y:number}|null>(null);
  const [rulerEnd, setRulerEnd]           = useState<{x:number;y:number}|null>(null);
  const [rulerDragging, setRulerDragging] = useState(false);
  const [markerDots, setMarkerDots]       = useState<{x:number;y:number}[]>([]);
  const [rulerLines, setRulerLines]       = useState<{x1:number;y1:number;x2:number;y2:number}[]>([]);
  const [undoStack, setUndoStack]         = useState<Array<"marker"|"ruler">>([]);
  const [showSaveModal, setShowSaveModal]         = useState(false);
  const [saveProjectName, setSaveProjectName]     = useState("");
  const [saveLoading, setSaveLoading]             = useState(false);
  const [showProjectsModal, setShowProjectsModal] = useState(false);
  const [savedProjects, setSavedProjects]         = useState<any[]>([]);
  const [projectsLoading, setProjectsLoading]     = useState(false);

  // ── MULTI-PAGE ────────────────────────────────────────────────────────────
  const [currentPageIdx, setCurrentPageIdx]   = useState(0);
  const [pageImageCache, setPageImageCache]   = useState<Record<number, string>>({});
  const [pageDataCache, setPageDataCache]     = useState<Record<number, {
    members: Member[]; summary: Summary | null; status: "not_set" | "estimating" | "built";
  }>>({});
  const [pageLoading, setPageLoading]         = useState(false);
  const [extracting, setExtracting]           = useState(false);

  const fileInputRef       = useRef<HTMLInputElement>(null);
  const imageWrapperRef    = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // ── DERIVED SUMMARY — recomputes live; filters to crop region when active ──
  const regionMembers = useMemo(() => {
    if (!cropRect) return members;
    // Use 5% tolerance — generous enough to handle any coordinate rounding,
    // zoom drift, or leader-line offsets between symbol, label, and snap position.
    const TOL = 0.05;
    const result = members.filter(m => {
      const lx = m.lx !== undefined ? m.lx : m.x;
      const ly = m.ly !== undefined ? m.ly : m.y;
      const inside = (px: number, py: number) =>
        px >= cropRect.x0 - TOL && px <= cropRect.x1 + TOL &&
        py >= cropRect.y0 - TOL && py <= cropRect.y1 + TOL;
      return inside(lx, ly)
          || inside(m.x, m.y)
          || (m.sx != null && m.sy != null && inside(m.sx, m.sy));
    });
    // Debug: log whenever a region is active to help trace coordinate issues
    console.log(
      `[REGION] box=(${cropRect.x0.toFixed(3)},${cropRect.y0.toFixed(3)})→(${cropRect.x1.toFixed(3)},${cropRect.y1.toFixed(3)})`,
      `| members=${members.length} found=${result.length}`,
      `| first3:`, members.slice(0,3).map(m=>({p:m.profile,x:m.x.toFixed(3),y:m.y.toFixed(3),lx:(m.lx??m.x).toFixed(3),ly:(m.ly??m.y).toFixed(3)}))
    );
    return result;
  }, [members, cropRect]);

  const summary = useMemo<Summary | null>(() => {
    if (!baseSummary) return null;
    // When a region is active use regionMembers, otherwise all members
    const src = cropRect ? regionMembers : members;
    let column = 0, beam = 0, vertical_brace = 0, horizontal_brace = 0;
    for (const m of src) {
      if      (m.type === "column")                               column++;
      else if (m.type === "beam")                                 beam++;
      else if (m.type === "brace" || m.type === "vertical_brace") vertical_brace++;
      else if (m.type === "horizontal_brace")                     horizontal_brace++;
    }
    return { ...baseSummary, column, beam, vertical_brace, horizontal_brace };
  }, [members, baseSummary, cropRect, regionMembers]);

  // ── CLOSE CONTEXT MENU ON OUTSIDE CLICK ───────────────────────────────────
  useEffect(() => {
    if (!contextMenu) return;
    const close = () => setContextMenu(null);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [contextMenu]);

  // ── CORRECTION HANDLERS ───────────────────────────────────────────────────
  function correctMember(idx: number, newType: string) {
    setMembers(prev => prev.map((m, i) =>
      i !== idx ? m : {
        ...m,
        type:       newType,
        color:      MEMBER_COLORS[newType] ?? "#6B7280",
        overridden: true,
      }
    ));
    setContextMenu(null);
  }

  function removeMember(idx: number) {
    setMembers(prev => prev.filter((_, i) => i !== idx));
    setContextMenu(null);
  }

  // ── ZOOM ──────────────────────────────────────────────────────────────────
  function stepZoom(dir: 1 | -1) {
    setZoomLevel(prev => {
      const i = ZOOM_STEPS.indexOf(prev);
      if (i === -1) return dir === 1 ? 1.25 : 0.75;
      const next = i + dir;
      return next < 0 ? ZOOM_STEPS[0] : next >= ZOOM_STEPS.length ? ZOOM_STEPS[ZOOM_STEPS.length - 1] : ZOOM_STEPS[next];
    });
  }

  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    stepZoom(e.deltaY < 0 ? 1 : -1);
  }, []);

  // ── CANVAS INTERACTION HELPERS ────────────────────────────────────────────
  function getImgPct(e: React.MouseEvent): { x: number; y: number } {
    const el = imageWrapperRef.current;
    if (!el) return { x: 0, y: 0 };
    const r = el.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (e.clientX - r.left)  / r.width)),
      y: Math.max(0, Math.min(1, (e.clientY - r.top)   / r.height)),
    };
  }

  function onCanvasMouseDown(e: React.MouseEvent) {
    if (activeTool === "hand") {
      e.preventDefault();
      setIsPanning(true);
      const sc = scrollContainerRef.current;
      setPanStart({ x: e.clientX, y: e.clientY, scrollLeft: sc?.scrollLeft ?? 0, scrollTop: sc?.scrollTop ?? 0 });
      return;
    }
    if (activeTool === "ruler") {
      e.preventDefault();
      const p = getImgPct(e);
      setRulerStart(p); setRulerEnd(p); setRulerDragging(true);
      return;
    }
    if (activeTool === "marker") {
      const p = getImgPct(e);
      setMarkerDots(prev => [...prev, p]);
      setUndoStack(prev => [...prev, "marker"]);
      return;
    }
    if (!cropMode) return;
    e.preventDefault();
    const p = getImgPct(e);
    setCropDrag({ x0: p.x, y0: p.y, x1: p.x, y1: p.y });
  }

  function onCanvasMouseMove(e: React.MouseEvent) {
    if (activeTool === "hand" && isPanning) {
      const sc = scrollContainerRef.current;
      if (!sc) return;
      sc.scrollLeft = panStart.scrollLeft - (e.clientX - panStart.x);
      sc.scrollTop  = panStart.scrollTop  - (e.clientY - panStart.y);
      return;
    }
    if (activeTool === "ruler" && rulerDragging) {
      setRulerEnd(getImgPct(e));
      return;
    }
    if (!cropMode || !cropDrag) return;
    const p = getImgPct(e);
    setCropDrag(prev => prev ? { ...prev, x1: p.x, y1: p.y } : null);
  }

  function onCanvasMouseUp(e: React.MouseEvent) {
    if (activeTool === "hand") { setIsPanning(false); return; }
    if (activeTool === "ruler" && rulerDragging && rulerStart) {
      const p = getImgPct(e);
      setRulerLines(prev => [...prev, { x1: rulerStart.x, y1: rulerStart.y, x2: p.x, y2: p.y }]);
      setUndoStack(prev => [...prev, "ruler"]);
      setRulerStart(null); setRulerEnd(null); setRulerDragging(false);
      return;
    }
    if (!cropMode || !cropDrag) return;
    const p  = getImgPct(e);
    const r: CropRect = {
      x0: Math.min(cropDrag.x0, p.x), y0: Math.min(cropDrag.y0, p.y),
      x1: Math.max(cropDrag.x0, p.x), y1: Math.max(cropDrag.y0, p.y),
    };
    if (r.x1 - r.x0 > 0.01 && r.y1 - r.y0 > 0.01) setCropRect(r);
    setCropDrag(null);
  }

  // ── UPLOAD ────────────────────────────────────────────────────────────────
  async function handleUpload(file: File) {
    const ext = file.name.toLowerCase().split('.').pop();
    if (!['pdf', 'jpg', 'jpeg', 'png'].includes(ext || "")) {
      showToast("red", "Only PDF, JPG, or PNG files are accepted");
      return;
    }
    setUploading(true);
    setUploadStage("sending");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const controller = new AbortController();
      const timeoutId  = setTimeout(() => controller.abort("Timeout"), 300000);
      const stageTimer = setTimeout(() => setUploadStage("processing"), 1500);

      const res = await fetch("http://localhost:8000/upload", {
        method: "POST", body: formData, signal: controller.signal,
      });
      clearTimeout(timeoutId);
      clearTimeout(stageTimer);
      if (!res.ok) throw new Error("Upload failed");
      const data = await res.json();
      setPdfImage(data.image);
      setFilename(data.filename);
      setPageCount(data.page_count);
      setPageThumbnails(data.thumbnails || [data.image]);
      setSelectedPageIndex(0);
      setStatus("not_set");
      setMembers([]);
      setBaseSummary(null);
      setSelectedScale(null);
      setSelectedRatio(null);
      setZoomLevel(1);
      // Multi-page setup
      setCurrentPageIdx(0);
      setPageThumbnails(data.page_thumbnails ?? [data.image]);
      setPageImageCache({ 0: data.image });
      setPageDataCache({});
      showToast("blue", "Drawing uploaded. Select scale to begin.");
    } catch (err: any) {
      showToast("red", err.name === "AbortError" ? "Upload timed out" : "Upload failed. Check if backend is running.");
    } finally {
      setUploading(false);
      setUploadStage(null);
    }
  }

  // ── CORE ANALYSIS (shared by scale-select and extract button) ────────────
  async function runAnalysis(ratio: number, pageIdx: number) {
    setExtracting(true);
    setStatus("estimating");
    setMembers([]);
    showToast("blue", "Building blueprint. Please wait...");
    try {
      const controller = new AbortController();
      const timeoutId  = setTimeout(() => controller.abort("Timeout"), 120000);
      const res = await fetch("http://localhost:8000/analyse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename, scale_ratio: ratio, page_index: pageIdx, ocr_dpi: 400 }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (!res.ok) throw new Error("Analysis failed");
      const data = await res.json();
      setMembers(data.members);
      setBaseSummary(data.summary);
      setStatus("built");
      // Cache per-page so switching back restores results
      setPageDataCache(prev => ({
        ...prev,
        [pageIdx]: { members: data.members, summary: data.summary, status: "built" },
      }));
      showToast("green", `Blueprint built in ${data.elapsed_seconds ?? data.elapsed ?? 0}s`);
      setTimeout(() => setToast(null), 4000);
    } catch (err: any) {
      showToast("red", err.name === "AbortError" ? "Analysis timed out" : "Analysis failed. Please try again.");
      setStatus("not_set");
    } finally {
      setExtracting(false);
    }
  }

  // ── SCALE SELECT → ANALYSIS ───────────────────────────────────────────────
  function handleScaleSelect(label: string, ratio: number) {
    setSelectedScale(label);
    setSelectedRatio(ratio);
    setScaleOpen(false);
    runAnalysis(ratio, currentPageIdx);
  }

  // ── EXTRACT BUTTON — re-runs analysis for current page + scale ────────────
  async function handleExtract() {
    if (!selectedRatio || extracting) return;
    runAnalysis(selectedRatio, currentPageIdx);
  }

  // ── PAGE SWITCH ───────────────────────────────────────────────────────────
  async function switchPage(newIdx: number) {
    if (newIdx === currentPageIdx || pageLoading || extracting) return;

    // Persist current page data so switching back restores it
    setPageDataCache(prev => ({
      ...prev,
      [currentPageIdx]: { members, summary: baseSummary, status },
    }));

    setCurrentPageIdx(newIdx);
    setCropRect(null); setCropMode(false); setCropDrag(null);
    setMarkerDots([]); setRulerLines([]); setUndoStack([]);

    // Restore cached extraction for the new page (if already extracted).
    // pageDataCache is captured from current render — safe to read directly
    // since switchPage is not memoized and always sees the latest cache.
    const cached = pageDataCache[newIdx];
    if (cached) {
      setMembers(cached.members);
      setBaseSummary(cached.summary);
      setStatus(cached.status);
    } else {
      setMembers([]);
      setBaseSummary(null);
      setStatus("not_set");
    }

    // Load full-size image (from cache or backend)
    if (pageImageCache[newIdx]) {
      setPdfImage(pageImageCache[newIdx]);
    } else {
      setPageLoading(true);
      try {
        const res  = await fetch(`http://localhost:8000/page-image/${encodeURIComponent(filename)}/${newIdx}`);
        const data = await res.json();
        setPageImageCache(prev => ({ ...prev, [newIdx]: data.image }));
        setPdfImage(data.image);
      } catch {
        showToast("red", "Failed to load page image.");
      } finally {
        setPageLoading(false);
      }
    }
  }

  function showToast(type: Toast["type"], message: string) {
    setToast({ type, message });
  }

  async function handleSave() {
    if (!saveProjectName.trim() || saveLoading) return;
    setSaveLoading(true);
    try {
      const res = await fetch("http://localhost:8000/save-project", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: saveProjectName.trim(),
          filename,
          scale: selectedScale,
          scale_ratio: selectedRatio,
          members,
          page_count: pageCount,
        }),
      });
      if (!res.ok) throw new Error("Save failed");
      showToast("green", `"${saveProjectName.trim()}" saved!`);
      setTimeout(() => setToast(null), 4000);
      setShowSaveModal(false);
      setSaveProjectName("");
    } catch {
      showToast("red", "Failed to save project. Please try again.");
    } finally {
      setSaveLoading(false);
    }
  }

  function handleUndo() {
    if (undoStack.length === 0) return;
    const last = undoStack[undoStack.length - 1];
    if (last === "marker") setMarkerDots(d => d.slice(0, -1));
    if (last === "ruler")  setRulerLines(l => l.slice(0, -1));
    setUndoStack(prev => prev.slice(0, -1));
  }

  // Ctrl+Z keyboard shortcut for undo
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "z" && !e.shiftKey) {
        e.preventDefault();
        handleUndo();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [undoStack]);

  async function handleLoadProjects() {
    setShowProjectsModal(true);
    setProjectsLoading(true);
    try {
      const res = await fetch("http://localhost:8000/saved-projects");
      if (!res.ok) throw new Error();
      setSavedProjects(await res.json());
    } catch {
      setSavedProjects([]);
    } finally {
      setProjectsLoading(false);
    }
  }

  function handleReset() {
    setPdfImage(null); setFilename(""); setPageCount(0);
    setPageThumbnails([]); setSelectedPageIndex(0);
    setStatus("not_set"); setMembers([]); setBaseSummary(null);
    setSelectedScale(null); setSelectedRatio(null);
    setToast(null); setZoomLevel(1); setContextMenu(null);
    setCropMode(false); setCropDrag(null); setCropRect(null);
    setActiveTool("select"); setIsPanning(false);
    setRulerStart(null); setRulerEnd(null); setRulerDragging(false);
    setMarkerDots([]); setRulerLines([]); setUndoStack([]);
    setCurrentPageIdx(0); setPageThumbnails([]);
    setPageImageCache({}); setPageDataCache({});
    setPageLoading(false); setExtracting(false);
  }

  const onDragOver  = useCallback((e: React.DragEvent) => { e.preventDefault(); setDragging(true); }, []);
  const onDragLeave = useCallback(() => setDragging(false), []);
  const onDrop      = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0]; if (f) handleUpload(f);
  }, [filename]);

  // ── UPLOAD SCREEN ─────────────────────────────────────────────────────────
  if (!pdfImage) {
    return (
      <div style={{ minHeight: "100vh", backgroundColor: "#0F172A", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div style={{ textAlign: "center", padding: "48px", maxWidth: "480px", width: "100%" }}>
          <div style={{ fontSize: "32px", fontWeight: "bold", color: "white", marginBottom: "8px" }}>Calsteel</div>
          <div style={{ color: "#94A3B8", marginBottom: "32px", fontSize: "15px" }}>Upload a structural drawing to begin</div>

          <div
            onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
            style={{
              border: `2px dashed ${dragging ? "#3B82F6" : "#334155"}`,
              borderRadius: "12px", padding: "48px 24px", cursor: "pointer",
              backgroundColor: dragging ? "#1E3A5F" : "#1E293B",
              transition: "all 0.2s", marginBottom: "20px",
            }}
          >
            <div style={{ fontSize: "40px", marginBottom: "12px" }}>📄</div>
            <div style={{ color: "#CBD5E1", fontSize: "15px" }}>Drag & drop your PDF here</div>
            <div style={{ color: "#64748B", fontSize: "13px", marginTop: "6px" }}>or click to browse — PDF only, max 500MB</div>
          </div>

          <input ref={fileInputRef} type="file" accept=".pdf,.jpg,.jpeg,.png" style={{ display: "none" }}
            onChange={e => { const f = e.target.files?.[0]; if (f) handleUpload(f); }} />

          <button onClick={() => fileInputRef.current?.click()} disabled={uploading} style={{
            backgroundColor: uploading ? "#334155" : "#1D4ED8", color: "white", border: "none",
            borderRadius: "8px", padding: "12px 32px", fontSize: "15px", fontWeight: "600",
            cursor: uploading ? "not-allowed" : "pointer", width: "100%",
          }}>
            {uploading ? (uploadStage === "processing" ? "⚙️ Rendering drawing..." : "⏳ Uploading...") : "Upload Drawing"}
          </button>
        </div>
      </div>
    );
  }

  // ── MAIN VIEWER ───────────────────────────────────────────────────────────
  return (
    <div style={{ height: "100vh", backgroundColor: "#0F172A", display: "flex", flexDirection: "column", overflow: "hidden" }}>

      {/* TOP NAV */}
      <div style={{
        height: "48px", backgroundColor: "#0F172A", borderBottom: "1px solid #1E293B",
        display: "flex", alignItems: "center", padding: "0 16px", flexShrink: 0, gap: "16px",
      }}>
        <span style={{ color: "white", fontWeight: "bold", fontSize: "16px" }}>Calsteel</span>
        <div style={{ display: "flex", gap: "4px", flex: 1, justifyContent: "center" }}>
          {["Plans", "Columns", "Braces", "BOM", "Config"].map(tab => (
            <div key={tab} style={{
              padding: "4px 16px", fontSize: "14px",
              color: tab === "Plans" ? "white" : "#64748B",
              borderBottom: tab === "Plans" ? "2px solid #3B82F6" : "2px solid transparent",
              cursor: tab === "Plans" ? "default" : "not-allowed", userSelect: "none",
            }}>{tab}</div>
          ))}
        </div>
        <button onClick={() => setShowSaveModal(true)} style={{
          backgroundColor: "#15803D", border: "1px solid #166534", color: "white",
          borderRadius: "6px", padding: "4px 12px", fontSize: "13px", cursor: "pointer", fontWeight: "600",
        }}>💾 Save</button>
        <button onClick={handleLoadProjects} style={{
          backgroundColor: "transparent", border: "1px solid #334155", color: "#94A3B8",
          borderRadius: "6px", padding: "4px 12px", fontSize: "13px", cursor: "pointer",
        }}>📂 Projects</button>
        <button onClick={handleReset} style={{
          backgroundColor: "transparent", border: "1px solid #334155", color: "#94A3B8",
          borderRadius: "6px", padding: "4px 12px", fontSize: "13px", cursor: "pointer",
        }}>↑ Upload New File</button>
      </div>

      {/* MAIN BODY */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>

        {/* LEFT SIDEBAR */}
        <div style={{
          width: "220px", backgroundColor: "#1E293B", borderRight: "1px solid #0F172A",
          display: "flex", flexDirection: "column", overflow: "hidden", flexShrink: 0,
        }}>
          {/* Header */}
          <div style={{ padding: "10px 12px 6px", color: "#64748B", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.05em", flexShrink: 0 }}>
            Navigation
          </div>
          <div style={{ display: "flex", borderBottom: "1px solid #0F172A", padding: "0 12px", flexShrink: 0 }}>
            {["Pages", "Members"].map(tab => (
              <div key={tab} style={{
                padding: "8px 12px", fontSize: "13px",
                color: tab === "Pages" ? "white" : "#64748B",
                borderBottom: tab === "Pages" ? "2px solid #3B82F6" : "2px solid transparent",
                cursor: "default",
              }}>{tab}</div>
            ))}
          </div>

          {/* ── CONTROLS (above pages) ── */}
          <div style={{ padding: "10px 10px 6px", flexShrink: 0 }}>

            {/* Bottom of Column */}
            <div style={{ display: "flex", justifyContent: "space-between", padding: "0 4px", marginBottom: "8px" }}>
              <span style={{ fontSize: "11px", color: "#64748B" }}>Bottom of Column</span>
              <span style={{ fontSize: "11px", color: "white" }}>-1&apos;-0&quot;</span>
            </div>

            {/* Scale Dropdown */}
            <div style={{ padding: "0 4px", marginBottom: "8px", position: "relative" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                <span style={{ fontSize: "11px", color: "#64748B" }}>Scale</span>
                {selectedRatio && (
                  <span style={{ backgroundColor: "#334155", color: "#94A3B8", fontSize: "10px", padding: "1px 6px", borderRadius: "4px" }}>
                    {selectedRatio}:1
                  </span>
                )}
              </div>
              <button onClick={() => setScaleOpen(p => !p)} style={{
                width: "100%", backgroundColor: "#0F172A", border: "1px solid #334155", borderRadius: "4px",
                padding: "5px 8px", color: selectedScale ? "white" : "#64748B", fontSize: "11px",
                textAlign: "left", cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center",
              }}>
                <span>{selectedScale ?? "Not Set"}</span>
                <span style={{ color: "#64748B" }}>▾</span>
              </button>
              {scaleOpen && (
                <div style={{
                  position: "absolute", top: "100%", left: 0, right: 0,
                  backgroundColor: "#0F172A", border: "1px solid #334155", borderRadius: "6px",
                  zIndex: 100, overflow: "hidden", boxShadow: "0 8px 24px rgba(0,0,0,0.5)",
                }}>
                  {SCALE_OPTIONS.map(opt => (
                    <div key={opt.label} onClick={() => handleScaleSelect(opt.label, opt.ratio)}
                      style={{
                        padding: "7px 10px", fontSize: "12px", cursor: "pointer",
                        color: selectedScale === opt.label ? "white" : "#CBD5E1",
                        backgroundColor: selectedScale === opt.label ? "#1D4ED8" : "transparent",
                      }}
                      onMouseEnter={e => (e.currentTarget.style.backgroundColor = selectedScale === opt.label ? "#1D4ED8" : "#1E293B")}
                      onMouseLeave={e => (e.currentTarget.style.backgroundColor = selectedScale === opt.label ? "#1D4ED8" : "transparent")}
                    >{opt.label}</div>
                  ))}
                </div>
              )}
            </div>

            {/* Status row */}
            <div style={{ display: "flex", justifyContent: "space-between", padding: "0 4px", marginBottom: "10px" }}>
              <span style={{ fontSize: "11px", color: "#64748B" }}>Status</span>
              <span style={{
                fontSize: "11px", fontWeight: "600",
                color: status === "built" ? "#4ADE80" : status === "estimating" ? "#60A5FA" : "#64748B",
              }}>
                {status === "built" ? "Built" : status === "estimating" ? "Estimating" : "Not Set"}
              </span>
            </div>

            {/* Extract button */}
            <div style={{ padding: "0 4px", marginBottom: "8px" }}>
              <button
                onClick={handleExtract}
                disabled={!selectedRatio || extracting}
                style={{
                  width: "100%", borderRadius: "5px", padding: "7px 0",
                  fontSize: "12px", fontWeight: "700", cursor: selectedRatio && !extracting ? "pointer" : "not-allowed",
                  backgroundColor: selectedRatio && !extracting ? "#1D4ED8" : "#1E293B",
                  color: selectedRatio && !extracting ? "white" : "#475569",
                  border: `1px solid ${selectedRatio && !extracting ? "#3B82F6" : "#334155"}`,
                  transition: "all 0.15s",
                }}
              >
                {extracting ? "⏳ Extracting…" : "⚡ Extract"}
              </button>
            </div>

            <div style={{ borderTop: "1px solid #334155", margin: "0 4px 0" }} />
          </div>

          {/* ── PAGE THUMBNAILS (below controls, scrollable) ── */}
          <div style={{ flex: 1, overflowY: "auto", padding: "6px 8px" }}>
            <div style={{ fontSize: "10px", color: "#475569", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: "6px", paddingLeft: "2px" }}>
              {pageCount} page{pageCount !== 1 ? "s" : ""}
            </div>
            {pageThumbnails.map((thumb, idx) => {
              const isActive  = idx === currentPageIdx;
              const pgData    = pageDataCache[idx];
              const pgBuilt   = pgData?.status === "built";
              const pgLoading = pageLoading && idx === currentPageIdx;
              return (
                <div
                  key={idx}
                  onClick={() => switchPage(idx)}
                  style={{
                    marginBottom: "8px", cursor: "pointer", borderRadius: "6px",
                    border: `2px solid ${isActive ? "#3B82F6" : "#1E293B"}`,
                    backgroundColor: isActive ? "#0F2744" : "transparent",
                    overflow: "hidden",
                    transition: "border-color 0.15s",
                    opacity: pgLoading ? 0.5 : 1,
                  }}
                >
                  <img
                    src={thumb}
                    alt={`Page ${idx + 1}`}
                    style={{ width: "100%", display: "block", objectFit: "cover" }}
                  />
                  <div style={{
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                    padding: "4px 6px",
                  }}>
                    <span style={{ fontSize: "11px", color: isActive ? "white" : "#94A3B8", fontWeight: isActive ? "700" : "400" }}>
                      Page {idx + 1}
                    </span>
                    <div style={{ display: "flex", gap: "4px" }}>
                      {pgBuilt && (
                        <span style={{ backgroundColor: "#166534", color: "#86EFAC", fontSize: "9px", padding: "1px 5px", borderRadius: "3px", fontWeight: "600" }}>Built</span>
                      )}
                      {pgBuilt && (
                        <span style={{ backgroundColor: "#1E3A5F", color: "#93C5FD", fontSize: "9px", padding: "1px 5px", borderRadius: "3px", fontWeight: "600" }}>Est.</span>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* TOOLS PANEL */}
        <div style={{
          width: "44px", backgroundColor: "#1E293B", borderRight: "1px solid #0F172A",
          display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
          gap: "4px", flexShrink: 0,
        }}>
          <button title="Select" onClick={() => setActiveTool("select")} style={toolBtnStyle(activeTool === "select")}>↖</button>
          <button title="Hand / Pan" onClick={() => setActiveTool("hand")} style={toolBtnStyle(activeTool === "hand")}>✋</button>
          <div style={{ width: "28px", height: "1px", backgroundColor: "#334155", margin: "4px 0" }} />
          <button title="Zoom In"  onClick={() => stepZoom(1)}  style={toolBtnStyle(false)}>+</button>
          <button title="Zoom Out" onClick={() => stepZoom(-1)} style={toolBtnStyle(false)}>−</button>
          <div style={{ width: "28px", height: "1px", backgroundColor: "#334155", margin: "4px 0" }} />
          <button title="Ruler" onClick={() => setActiveTool("ruler")} style={toolBtnStyle(activeTool === "ruler")}>📏</button>
          <button title="Marker" onClick={() => setActiveTool("marker")} style={toolBtnStyle(activeTool === "marker")}>✏️</button>
          <div style={{ width: "28px", height: "1px", backgroundColor: "#334155", margin: "4px 0" }} />
          <button
            title="Undo (Ctrl+Z)"
            onClick={handleUndo}
            style={{
              ...toolBtnStyle(false),
              opacity: undoStack.length === 0 ? 0.3 : 1,
              cursor: undoStack.length === 0 ? "not-allowed" : "pointer",
            }}
          >↩</button>
        </div>

        {/* CENTER: PDF VIEWER */}
        <div style={{
          flex: 1, minWidth: 0,
          position: "relative",          /* anchor for floating controls */
          backgroundColor: "#1E293B",
          overflow: "hidden",            /* hard-clip so children never push layout */
        }}>

          {/* ── Page loading overlay (shown while fetching a new page image) ── */}
          {pageLoading && (
            <div style={{
              position: "absolute", inset: 0, zIndex: 200,
              backgroundColor: "rgba(15,23,42,0.75)",
              display: "flex", alignItems: "center", justifyContent: "center",
              backdropFilter: "blur(3px)",
            }}>
              <div style={{ color: "#93C5FD", fontSize: "14px", fontWeight: "600" }}>Loading page…</div>
            </div>
          )}

          {/* ── Scrollable image ── */}
          <div
            ref={scrollContainerRef}
            onWheel={onWheel}
            style={{ position: "absolute", inset: 0, overflow: "auto" }}
          >
            {/* Width drives zoom; min-width keeps it filling the pane at ≤100% */}
            <div
              ref={imageWrapperRef}
              onMouseDown={onCanvasMouseDown}
              onMouseMove={onCanvasMouseMove}
              onMouseUp={onCanvasMouseUp}
              style={{
                position: "relative",
                width: `${zoomLevel * 100}%`, minWidth: "100%",
                cursor: activeTool === "hand"
                  ? (isPanning ? "grabbing" : "grab")
                  : (activeTool === "ruler" || activeTool === "marker" || cropMode)
                    ? "crosshair"
                    : "default",
                userSelect: activeTool !== "select" || cropMode ? "none" : "auto",
              }}
            >
              <img
                src={pdfImage}
                alt="Structural Drawing"
                style={{
                  width: "100%", height: "auto", display: "block",
                  imageRendering: zoomLevel > 1 ? "crisp-edges" : "auto",
                  pointerEvents: "none",
                }}
              />

              {/* ── Beam span lines (SVG overlay) ──────────────────────────── */}
              {/* Drawn BELOW the crop dim overlay so they remain visible           */}
              {/* when a region is active.  pointerEvents:none — interaction via    */}
              {/* the midpoint marker divs instead.                                  */}
              <svg
                style={{
                  position: "absolute", inset: 0,
                  width: "100%", height: "100%",
                  overflow: "visible",
                  pointerEvents: "none",
                  zIndex: 13,
                }}
              >
                <defs>
                  {/* Glow filter for selected / highlighted beams */}
                  <filter id="beam-glow" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="2" result="blur"/>
                    <feMerge>
                      <feMergeNode in="blur"/>
                      <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                  </filter>
                </defs>
                {members.map((m, idx) => {
                  if (m.type !== "beam") return null;
                  if (m.bx1 == null || m.by1 == null || m.bx2 == null || m.by2 == null) return null;
                  const isHovered = hoveredMember === m;
                  return (
                    <line
                      key={idx}
                      x1={`${m.bx1 * 100}%`} y1={`${m.by1 * 100}%`}
                      x2={`${m.bx2 * 100}%`} y2={`${m.by2 * 100}%`}
                      stroke={m.overridden ? "#FBBF24" : m.color}
                      strokeWidth={isHovered ? "3.5" : "2.5"}
                      strokeOpacity={isHovered ? "0.92" : "0.68"}
                      strokeLinecap="round"
                      filter={isHovered ? "url(#beam-glow)" : undefined}
                    />
                  );
                })}
                {/* Saved ruler lines */}
                {rulerLines.map((ln, i) => (
                  <g key={`ruler-${i}`}>
                    <line x1={`${ln.x1*100}%`} y1={`${ln.y1*100}%`} x2={`${ln.x2*100}%`} y2={`${ln.y2*100}%`}
                      stroke="#F59E0B" strokeWidth="2" strokeDasharray="6 3" strokeLinecap="round" strokeOpacity="0.9" />
                    <circle cx={`${ln.x1*100}%`} cy={`${ln.y1*100}%`} r="4" fill="#F59E0B" />
                    <circle cx={`${ln.x2*100}%`} cy={`${ln.y2*100}%`} r="4" fill="#F59E0B" />
                  </g>
                ))}
                {/* Live ruler preview while dragging */}
                {rulerStart && rulerEnd && rulerDragging && (
                  <>
                    <line
                      x1={`${rulerStart.x * 100}%`} y1={`${rulerStart.y * 100}%`}
                      x2={`${rulerEnd.x * 100}%`}   y2={`${rulerEnd.y * 100}%`}
                      stroke="#F59E0B" strokeWidth="2" strokeDasharray="6 3"
                      strokeLinecap="round" strokeOpacity="0.9"
                    />
                    <circle cx={`${rulerStart.x * 100}%`} cy={`${rulerStart.y * 100}%`} r="4" fill="#F59E0B" />
                    <circle cx={`${rulerEnd.x * 100}%`}   cy={`${rulerEnd.y * 100}%`}   r="4" fill="#F59E0B" />
                  </>
                )}
              </svg>

              {/* Live drag rectangle while drawing */}
              {cropMode && cropDrag && (() => {
                const sx = Math.min(cropDrag.x0, cropDrag.x1) * 100;
                const sy = Math.min(cropDrag.y0, cropDrag.y1) * 100;
                const sw = Math.abs(cropDrag.x1 - cropDrag.x0) * 100;
                const sh = Math.abs(cropDrag.y1 - cropDrag.y0) * 100;
                return (
                  <div style={{
                    position: "absolute", pointerEvents: "none", zIndex: 40,
                    left: `${sx}%`, top: `${sy}%`, width: `${sw}%`, height: `${sh}%`,
                    border: "2px dashed #3B82F6",
                    backgroundColor: "rgba(59,130,246,0.08)",
                  }} />
                );
              })()}

              {/* Persistent region rectangle after selection */}
              {cropRect && !cropDrag && (() => {
                const sx = cropRect.x0 * 100;
                const sy = cropRect.y0 * 100;
                const sw = (cropRect.x1 - cropRect.x0) * 100;
                const sh = (cropRect.y1 - cropRect.y0) * 100;
                return (
                  <>
                    {/* Dim outside the region */}
                    <div style={{ position: "absolute", inset: 0, backgroundColor: "rgba(0,0,0,0.35)", pointerEvents: "none", zIndex: 25 }} />
                    {/* Clear window inside region */}
                    <div style={{
                      position: "absolute", pointerEvents: "none", zIndex: 26,
                      left: `${sx}%`, top: `${sy}%`, width: `${sw}%`, height: `${sh}%`,
                      border: "2px solid #3B82F6",
                      boxShadow: "0 0 0 9999px rgba(0,0,0,0.35), inset 0 0 0 1px rgba(59,130,246,0.4)",
                      backgroundColor: "transparent",
                    }} />
                    {/* Region label */}
                    <div style={{
                      position: "absolute", pointerEvents: "none", zIndex: 27,
                      left: `${sx}%`, top: `calc(${sy}% - 38px)`,
                      backgroundColor: "#1D4ED8", color: "white",
                      fontSize: "10px", fontWeight: "700", padding: "3px 8px",
                      borderRadius: "3px 3px 0 0", lineHeight: "1.6",
                    }}>
                      <div>✂ {regionMembers.filter(m => m.type === "column").length} col · {regionMembers.filter(m => m.type === "beam").length} beam · {regionMembers.length} total</div>
                      <div style={{ fontWeight: 400, opacity: 0.8, fontSize: "9px" }}>
                        x {cropRect.x0.toFixed(2)}–{cropRect.x1.toFixed(2)} · y {cropRect.y0.toFixed(2)}–{cropRect.y1.toFixed(2)}
                      </div>
                    </div>
                  </>
                );
              })()}

              {/* Markers */}
              <div style={{ position: "absolute", inset: 0, overflow: "hidden" }}>
                {members.map((m, idx) => {
                  const isColumn = m.type === "column";
                  const isBeam   = m.type === "beam";
                  const isHov    = hoveredMember === m;

                  // Orientation-aware midpoint anchoring:
                  // If we have exact span endpoints, always recompute the anchor
                  // from them — this is the geometric truth regardless of what
                  // the backend stored in m.x / m.y.
                  let anchorX = m.x;
                  let anchorY = m.y;
                  if (isBeam && m.bx1 != null && m.bx2 != null && m.by1 != null && m.by2 != null) {
                    if (m.beam_dir === "V") {
                      // Vertical beam: x = beam centre x, y = midpoint(y1,y2)
                      anchorX = (m.bx1 + m.bx2) / 2;
                      anchorY = (m.by1 + m.by2) / 2;
                    } else {
                      // Horizontal beam: x = midpoint(x1,x2), y = beam centre y
                      anchorX = (m.bx1 + m.bx2) / 2;
                      anchorY = (m.by1 + m.by2) / 2;
                    }
                  }

                  return (
                    <div
                      key={idx}
                      style={{
                        position: "absolute",
                        left: `${anchorX * 100}%`,
                        top:  `${anchorY * 100}%`,
                        transform: "translate(-50%, -50%)",
                        zIndex: 20, cursor: "pointer",
                      }}
                      onMouseEnter={e => { setHoveredMember(m); setTooltipPos({ x: e.clientX + 14, y: e.clientY - 36 }); }}
                      onMouseMove={e  => setTooltipPos({ x: e.clientX + 14, y: e.clientY - 36 })}
                      onMouseLeave={() => setHoveredMember(null)}
                      onContextMenu={e => {
                        e.preventDefault(); e.stopPropagation();
                        setHoveredMember(null);
                        setContextMenu({ member: m, idx, x: e.clientX, y: e.clientY });
                      }}
                    >
                      {isColumn ? (
                        /* Column: I-shape symbol + profile chip label */
                        <div style={{ position: "relative", display: "flex", flexDirection: "column", alignItems: "center", gap: "2px" }}>
                          {m.overridden && (
                            <div style={{ position: "absolute", inset: "-4px", border: "1px dashed #FBBF24", borderRadius: "2px", pointerEvents: "none" }} />
                          )}
                          {/* I-shape symbol */}
                          <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                            <div style={{ width: "14px", height: "2px", backgroundColor: m.color, borderRadius: "1px", boxShadow: `0 0 3px ${m.color}` }} />
                            <div style={{ width: "3px", height: "3px", backgroundColor: m.color, borderRadius: "50%", marginTop: "1px" }} />
                          </div>
                          {/* Profile chip */}
                          <div style={{
                            backgroundColor: isHov ? "#1E3A5F" : "#1E3A5F88",
                            border: `1px solid ${m.overridden ? "#FBBF24" : m.color}`,
                            borderRadius: "3px",
                            padding: "0px 3px",
                            fontSize: "7px",
                            fontWeight: "700",
                            color: "white",
                            whiteSpace: "nowrap",
                            letterSpacing: "0.02em",
                            boxShadow: isHov ? `0 0 5px ${m.color}` : "0 1px 3px rgba(0,0,0,0.5)",
                            transition: "all 0.1s",
                          }}>
                            {m.profile}
                          </div>
                        </div>
                      ) : isBeam ? (
                        /* Beam: small profile chip at midspan — the SVG line shows the full span */
                        <div style={{
                          position: "relative",
                          backgroundColor: isHov
                            ? m.overridden ? "#92400E" : "#831843"
                            : m.overridden ? "#78350F88" : "#9D174D99",
                          border: `1px solid ${m.overridden ? "#FBBF24" : m.color}`,
                          borderRadius: "3px",
                          padding: "1px 4px",
                          fontSize: "8px",
                          fontWeight: "700",
                          color: "white",
                          whiteSpace: "nowrap",
                          lineHeight: "1.3",
                          boxShadow: isHov ? `0 0 6px ${m.color}` : "0 1px 3px rgba(0,0,0,0.5)",
                          transition: "all 0.1s",
                          letterSpacing: "0.02em",
                          writingMode: m.beam_dir === "V" ? "vertical-rl" : undefined,
                        }}>
                          {m.profile}
                          {m.length_ft > 0 && (
                            <span style={{
                              color: "#FBD0E8", fontWeight: 400,
                              marginLeft: m.beam_dir === "V" ? "0" : "3px",
                              marginTop:  m.beam_dir === "V" ? "2px" : "0",
                            }}>
                              {formatFt(m.length_ft)}
                            </span>
                          )}
                        </div>
                      ) : (
                        /* Brace: diagonal dash */
                        <div style={{
                          width: "16px", height: "3px",
                          backgroundColor: m.color,
                          transform: "rotate(45deg)",
                          borderRadius: "1px",
                          boxShadow: "0 1px 3px rgba(0,0,0,0.4)",
                        }} />
                      )}
                    </div>
                  );
                })}
                {/* Marker dots */}
                {markerDots.map((dot, i) => (
                  <div key={`dot-${i}`} style={{
                    position: "absolute",
                    left: `${dot.x * 100}%`, top: `${dot.y * 100}%`,
                    transform: "translate(-50%, -50%)",
                    width: "10px", height: "10px",
                    backgroundColor: "#F59E0B", borderRadius: "50%",
                    border: "2px solid white", zIndex: 22,
                    pointerEvents: "none",
                    boxShadow: "0 1px 4px rgba(0,0,0,0.5)",
                  }} />
                ))}
              </div>
            </div>
          </div>

          {/* ── Zoom + Crop controls — floating top-right corner ── */}
          <div style={{
            position: "absolute", top: "12px", right: "12px", zIndex: 50,
            display: "flex", alignItems: "center", gap: "4px",
            backgroundColor: "rgba(15,23,42,0.85)",
            border: "1px solid #334155", borderRadius: "8px",
            padding: "5px 8px",
            boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
            backdropFilter: "blur(4px)",
          }}>
            <button onClick={() => stepZoom(-1)} disabled={zoomLevel === ZOOM_STEPS[0]}
              style={zoomBtnStyle(zoomLevel === ZOOM_STEPS[0])} title="Zoom out (Ctrl+scroll)">−</button>
            <span
              onClick={() => setZoomLevel(1)}
              style={{ color: "#CBD5E1", fontSize: "12px", minWidth: "40px", textAlign: "center", cursor: "pointer", userSelect: "none", fontWeight: "600" }}
              title="Click to reset"
            >{Math.round(zoomLevel * 100)}%</span>
            <button onClick={() => stepZoom(1)} disabled={zoomLevel === ZOOM_STEPS[ZOOM_STEPS.length - 1]}
              style={zoomBtnStyle(zoomLevel === ZOOM_STEPS[ZOOM_STEPS.length - 1])} title="Zoom in (Ctrl+scroll)">+</button>

            {/* Divider */}
            <div style={{ width: "1px", height: "16px", backgroundColor: "#334155", margin: "0 2px" }} />

            {/* Crop / Region select button */}
            <button
              onClick={() => { setCropMode(p => !p); setCropDrag(null); if (cropMode) setCropRect(null); }}
              title={cropMode ? "Exit region select" : "Select region to inspect"}
              style={{
                ...zoomBtnStyle(false),
                width: "auto", padding: "2px 8px", fontSize: "11px",
                backgroundColor: cropMode ? "#1D4ED8" : "transparent",
                color:           cropMode ? "white"   : "#94A3B8",
                borderColor:     cropMode ? "#3B82F6" : "#334155",
                borderRadius: "4px", gap: "4px", display: "flex", alignItems: "center",
              }}
            >
              ✂ {cropMode ? "Cancel" : "Inspect Region"}
            </button>
          </div>

          {/* ── Legend — floating bottom-left corner ── */}
          <div style={{
            position: "absolute", bottom: "16px", left: "16px", zIndex: 10,
            backgroundColor: "rgba(15,23,42,0.85)",
            border: "1px solid #334155", borderRadius: "6px", padding: "10px 12px",
            boxShadow: "0 4px 6px rgba(0,0,0,0.3)", pointerEvents: "none",
            backdropFilter: "blur(4px)",
          }}>
            <div style={{ color: "white", fontSize: "11px", fontWeight: "bold", marginBottom: "7px" }}>Extraction Target</div>
            <div style={{ display: "flex", flexDirection: "column", gap: "7px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                  <div style={{ width: "14px", height: "2px", backgroundColor: "#3B82F6", borderRadius: "1px" }} />
                  <div style={{ width: "3px",  height: "3px", backgroundColor: "#3B82F6", borderRadius: "50%", marginTop: "1px" }} />
                </div>
                <span style={{ color: "white", fontSize: "11px", fontWeight: "600" }}>Column</span>
              </div>
              {/* Beam legend: line + chip */}
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <div style={{ position: "relative", width: "36px", height: "14px" }}>
                  {/* horizontal line */}
                  <div style={{ position: "absolute", top: "6px", left: 0, right: 0, height: "2px", backgroundColor: "#EC4899", opacity: 0.75, borderRadius: "1px" }} />
                  {/* midpoint chip */}
                  <div style={{
                    position: "absolute", top: 0, left: "50%", transform: "translateX(-50%)",
                    backgroundColor: "#9D174D99", border: "1px solid #EC4899",
                    borderRadius: "3px", padding: "0 3px",
                    fontSize: "7px", fontWeight: "700", color: "white", whiteSpace: "nowrap",
                  }}>W24</div>
                </div>
                <span style={{ color: "white", fontSize: "11px", fontWeight: "600" }}>Beam</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <div style={{ width: "10px", height: "10px", border: "1px dashed #FBBF24", borderRadius: "2px" }} />
                <span style={{ color: "#FBBF24", fontSize: "11px" }}>Corrected</span>
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT PANEL */}
        <div style={{
          width: "300px", backgroundColor: "#0F172A", borderLeft: "1px solid #1E293B",
          display: "flex", flexDirection: "column", overflow: "hidden", flexShrink: 0,
        }}>
          {/* Header */}
          <div style={{ padding: "12px 16px", borderBottom: "1px solid #1E293B", flexShrink: 0 }}>
            <div style={{ color: "white", fontWeight: "bold", fontSize: "14px", marginBottom: "8px" }}>Properties</div>
            <div style={{ display: "inline-block", color: "white", fontSize: "13px", borderBottom: "2px solid #3B82F6", paddingBottom: "4px" }}>
              Summary
            </div>
          </div>

          {/* Region filter banner — shows when a region is selected */}
          {cropRect && (
            <div style={{
              backgroundColor: "#0F2744", borderBottom: "1px solid #1D4ED8",
              padding: "8px 14px", flexShrink: 0,
              display: "flex", justifyContent: "space-between", alignItems: "center",
            }}>
              <div>
                <div style={{ color: "#93C5FD", fontSize: "11px", fontWeight: "700", marginBottom: "2px" }}>
                  ✂ Region Filter Active
                </div>
                <div style={{ color: "#64748B", fontSize: "10px" }}>
                  Showing counts for selected area only
                </div>
              </div>
              <button
                onClick={() => { setCropRect(null); setCropMode(false); setCropDrag(null); }}
                style={{
                  backgroundColor: "transparent", border: "1px solid #334155",
                  color: "#94A3B8", borderRadius: "4px", padding: "3px 8px",
                  fontSize: "11px", cursor: "pointer",
                }}
              >Clear ✕</button>
            </div>
          )}

          <div style={{ overflow: "auto", flex: 1 }}>
            <SummarySection
              title={cropRect ? "Region Summary" : "Sheet Summary"}
              summary={summary}
              isFiltered={!!cropRect}
            />
            <SummarySection
              title="Project Summary"
              summary={cropRect ? null : summary}
              showHrsTon
            />
          </div>
        </div>
      </div>

      {/* (no modal — region is shown inline on the image and counted in right panel) */}

      {/* SAVE MODAL */}
      {showSaveModal && (
        <div style={{
          position: "fixed", inset: 0, zIndex: 600,
          backgroundColor: "rgba(0,0,0,0.7)",
          display: "flex", alignItems: "center", justifyContent: "center",
          backdropFilter: "blur(4px)",
        }}>
          <div style={{
            backgroundColor: "#1E293B", border: "1px solid #334155",
            borderRadius: "12px", padding: "24px", minWidth: "340px",
            boxShadow: "0 24px 64px rgba(0,0,0,0.7)",
          }}>
            <div style={{ color: "white", fontWeight: "bold", fontSize: "16px", marginBottom: "6px" }}>Save Project</div>
            <div style={{ color: "#64748B", fontSize: "12px", marginBottom: "16px" }}>
              Enter a name for this project to save it to the database.
            </div>
            <input
              type="text"
              value={saveProjectName}
              onChange={e => setSaveProjectName(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleSave()}
              placeholder="Project name…"
              autoFocus
              style={{
                width: "100%", backgroundColor: "#0F172A",
                border: "1px solid #334155", borderRadius: "6px",
                padding: "9px 12px", color: "white", fontSize: "14px",
                marginBottom: "16px", boxSizing: "border-box", outline: "none",
              }}
            />
            <div style={{ display: "flex", gap: "8px", justifyContent: "flex-end" }}>
              <button
                onClick={() => { setShowSaveModal(false); setSaveProjectName(""); }}
                style={{
                  backgroundColor: "transparent", border: "1px solid #334155",
                  color: "#94A3B8", borderRadius: "6px", padding: "8px 18px",
                  cursor: "pointer", fontSize: "13px",
                }}
              >Cancel</button>
              <button
                onClick={handleSave}
                disabled={!saveProjectName.trim() || saveLoading}
                style={{
                  backgroundColor: saveProjectName.trim() && !saveLoading ? "#1D4ED8" : "#334155",
                  border: "none", color: "white", borderRadius: "6px",
                  padding: "8px 18px", fontSize: "13px", fontWeight: "600",
                  cursor: saveProjectName.trim() && !saveLoading ? "pointer" : "not-allowed",
                }}
              >{saveLoading ? "Saving…" : "OK"}</button>
            </div>
          </div>
        </div>
      )}

      {/* PROJECTS MODAL */}
      {showProjectsModal && (
        <div
          style={{
            position: "fixed", inset: 0, zIndex: 600,
            backgroundColor: "rgba(0,0,0,0.7)",
            display: "flex", alignItems: "center", justifyContent: "center",
            backdropFilter: "blur(4px)",
          }}
          onClick={() => setShowProjectsModal(false)}
        >
          <div
            style={{
              backgroundColor: "#1E293B", border: "1px solid #334155",
              borderRadius: "12px", minWidth: "480px", maxWidth: "620px", width: "90vw",
              boxShadow: "0 24px 64px rgba(0,0,0,0.7)", overflow: "hidden",
            }}
            onClick={e => e.stopPropagation()}
          >
            <div style={{
              padding: "16px 20px", borderBottom: "1px solid #334155",
              display: "flex", justifyContent: "space-between", alignItems: "center",
            }}>
              <span style={{ color: "white", fontWeight: "bold", fontSize: "15px" }}>Saved Projects</span>
              <button onClick={() => setShowProjectsModal(false)} style={{
                backgroundColor: "transparent", border: "1px solid #334155",
                color: "#94A3B8", borderRadius: "4px", padding: "3px 10px",
                cursor: "pointer", fontSize: "13px",
              }}>✕</button>
            </div>
            <div style={{ maxHeight: "60vh", overflowY: "auto" }}>
              {projectsLoading ? (
                <div style={{ padding: "32px", textAlign: "center", color: "#64748B" }}>Loading…</div>
              ) : savedProjects.length === 0 ? (
                <div style={{ padding: "32px", textAlign: "center", color: "#64748B" }}>No saved projects yet.</div>
              ) : savedProjects.map((p: any, i: number) => (
                <div key={p.id ?? i} style={{
                  padding: "14px 20px", borderBottom: "1px solid #0F172A",
                  backgroundColor: i % 2 === 0 ? "transparent" : "#0A0F1A",
                  display: "flex", justifyContent: "space-between", alignItems: "center",
                }}>
                  <div>
                    <div style={{ color: "white", fontWeight: "600", fontSize: "14px", marginBottom: "3px" }}>{p.name}</div>
                    <div style={{ color: "#64748B", fontSize: "12px" }}>
                      {p.filename} · {p.scale ?? "No scale"} · {p.member_count ?? 0} members
                    </div>
                    <div style={{ color: "#475569", fontSize: "11px", marginTop: "2px" }}>
                      {p.created_at ? new Date(p.created_at).toLocaleString() : ""}
                    </div>
                  </div>
                  <span style={{ backgroundColor: "#166534", color: "#86EFAC", fontSize: "10px", padding: "2px 8px", borderRadius: "4px", flexShrink: 0 }}>Saved</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* CONTEXT MENU — right-click correction */}
      {contextMenu && (
        <div
          onMouseDown={e => e.stopPropagation()}
          style={{
            position: "fixed", left: contextMenu.x, top: contextMenu.y,
            backgroundColor: "#1E293B", border: "1px solid #334155",
            borderRadius: "8px", zIndex: 500,
            boxShadow: "0 8px 32px rgba(0,0,0,0.7)", overflow: "hidden", minWidth: "190px",
          }}
        >
          {/* Header */}
          <div style={{ padding: "8px 12px", borderBottom: "1px solid #334155", fontSize: "11px", color: "#94A3B8" }}>
            <span style={{ color: contextMenu.member.color, fontWeight: "bold" }}>{contextMenu.member.profile}</span>
            {" · currently "}
            <span style={{ color: "white" }}>{contextMenu.member.type}</span>
            {contextMenu.member.overridden && (
              <span style={{ color: "#FBBF24", marginLeft: "6px" }}>✎ edited</span>
            )}
          </div>

          {/* Change options */}
          {(["column", "beam", "brace"] as const).filter(t => t !== contextMenu.member.type).map(t => (
            <div
              key={t}
              onClick={() => correctMember(contextMenu.idx, t)}
              style={ctxItemStyle(MEMBER_COLORS[t])}
              onMouseEnter={e => (e.currentTarget.style.backgroundColor = "#1E3A5F")}
              onMouseLeave={e => (e.currentTarget.style.backgroundColor = "transparent")}
            >
              <span style={{ fontSize: "13px" }}>
                {t === "column" ? "⬛" : t === "beam" ? "━" : "╱"}
              </span>
              Mark as {t.charAt(0).toUpperCase() + t.slice(1)}
            </div>
          ))}

          <div style={{ borderTop: "1px solid #334155" }} />

          {/* Remove */}
          <div
            onClick={() => removeMember(contextMenu.idx)}
            style={ctxItemStyle("#F87171")}
            onMouseEnter={e => (e.currentTarget.style.backgroundColor = "#7F1D1D33")}
            onMouseLeave={e => (e.currentTarget.style.backgroundColor = "transparent")}
          >
            <span style={{ fontSize: "13px" }}>✕</span> Remove Marker
          </div>
        </div>
      )}

      {/* TOAST */}
      {toast && (
        <div style={{
          position: "fixed", bottom: "24px", left: "50%", transform: "translateX(-50%)",
          zIndex: 200,
          backgroundColor: toast.type === "blue" ? "#1D4ED8" : toast.type === "green" ? "#15803D" : "#991B1B",
          color: "white", padding: "10px 20px", borderRadius: "8px",
          fontSize: "14px", fontWeight: "500", display: "flex", alignItems: "center", gap: "10px",
          boxShadow: "0 4px 20px rgba(0,0,0,0.4)", whiteSpace: "nowrap",
        }}>
          {toast.type === "blue" ? "⏳" : toast.type === "green" ? "✅" : "❌"}
          {toast.message}
        </div>
      )}

      {/* HOVER TOOLTIP */}
      {hoveredMember && !contextMenu && (
        <div style={{
          position: "fixed", left: tooltipPos.x, top: tooltipPos.y,
          backgroundColor: "#1E293B", border: "1px solid #334155",
          color: "white", padding: "6px 12px", borderRadius: "6px",
          fontSize: "12px", zIndex: 300, pointerEvents: "none", whiteSpace: "nowrap",
        }}>
          <span style={{ color: hoveredMember.color, fontWeight: "bold" }}>{hoveredMember.profile}</span>
          {" — "}{hoveredMember.type}
          {hoveredMember.beam_dir && (
            <span style={{ color: "#94A3B8", marginLeft: "4px", fontSize: "10px" }}>
              ({hoveredMember.beam_dir})
            </span>
          )}
          {hoveredMember.length_ft > 0 && (
            <span style={{ color: "#4ADE80", marginLeft: "6px" }}>
              {formatFt(hoveredMember.length_ft)}
            </span>
          )}
          <span style={{ color: "#475569", marginLeft: "8px", fontSize: "10px" }}>right-click to edit</span>
        </div>
      )}
    </div>
  );
}

// ── FORMAT HELPERS ─────────────────────────────────────────────────────────────
/** Convert decimal feet (e.g. 23.67) to feet-inches string "23'-8\"" */
function formatFt(ft: number): string {
  if (!ft || ft <= 0) return "";
  const feet   = Math.floor(ft);
  const inches = Math.round((ft - feet) * 12);
  if (inches === 0) return `${feet}'-0"`;
  if (inches === 12) return `${feet + 1}'-0"`;
  return `${feet}'-${inches}"`;
}

// ── STYLE HELPERS ──────────────────────────────────────────────────────────────
function zoomBtnStyle(disabled: boolean): React.CSSProperties {
  return {
    backgroundColor: "transparent",
    border: "1px solid #334155",
    color: disabled ? "#334155" : "#94A3B8",
    borderRadius: "4px",
    width: "24px", height: "24px",
    display: "flex", alignItems: "center", justifyContent: "center",
    fontSize: "16px", cursor: disabled ? "not-allowed" : "pointer",
    lineHeight: 1, padding: 0,
  };
}

function ctxItemStyle(color: string): React.CSSProperties {
  return {
    padding: "9px 14px", fontSize: "13px", color, cursor: "pointer",
    display: "flex", alignItems: "center", gap: "8px",
    backgroundColor: "transparent", transition: "background 0.1s",
  };
}

function toolBtnStyle(active: boolean): React.CSSProperties {
  return {
    width: "32px", height: "32px",
    display: "flex", alignItems: "center", justifyContent: "center",
    backgroundColor: active ? "#1D4ED8" : "transparent",
    border: `1px solid ${active ? "#3B82F6" : "transparent"}`,
    borderRadius: "6px",
    color: active ? "white" : "#94A3B8",
    fontSize: "15px",
    cursor: "pointer",
    flexShrink: 0,
    transition: "background 0.1s",
  };
}

// ── SUMMARY SECTION ────────────────────────────────────────────────────────────
function SummarySection({ title, summary, showHrsTon = false, isFiltered = false }:
  { title: string; summary: Summary | null; showHrsTon?: boolean; isFiltered?: boolean }) {
  return (
    <div style={{ borderBottom: "1px solid #1E293B" }}>
      <div style={{ padding: "10px 16px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ color: "white", fontSize: "13px", fontWeight: "600" }}>{title}</span>
        <span style={{ color: "#64748B", fontSize: "12px" }}>▾</span>
      </div>
      {SUMMARY_ROWS.map((row, i) => (
        <div key={row.key} style={{
          display: "flex", justifyContent: "space-between", alignItems: "center",
          padding: "6px 16px", borderTop: "1px solid #1E293B",
          backgroundColor: i % 2 === 0 ? "transparent" : "#0A0F1A",
        }}>
          <span style={{ color: "#94A3B8", fontSize: "12px" }}>{row.label}</span>
          <span style={{ color: "white", fontSize: "12px", fontWeight: "600" }}>
            {summary ? (summary as any)[row.key] ?? 0 : 0}
          </span>
        </div>
      ))}
      {showHrsTon && (
        <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 16px", borderTop: "1px solid #1E293B" }}>
          <span style={{ color: "#94A3B8", fontSize: "12px" }}>Hrs/Ton</span>
          <span style={{ color: "#64748B", fontSize: "12px", fontStyle: "italic" }}>{summary?.hrs_ton ?? "WIP"}</span>
        </div>
      )}
    </div>
  );
}

// ── CROP / REGION INSPECT MODAL ────────────────────────────────────────────────
function CropModal({
  cropRect, pdfImage, members, onCorrect, onRemove, onClose,
}: {
  cropRect: CropRect;
  pdfImage: string;
  members: Member[];
  onCorrect: (idx: number, type: string) => void;
  onRemove:  (idx: number) => void;
  onClose:   () => void;
}) {
  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const [dims, setDims]   = useState({ w: 1, h: 1 });
  const [ctxMenu, setCtxMenu] = useState<{ member: Member; idx: number; x: number; y: number } | null>(null);

  const cropW = cropRect.x1 - cropRect.x0;
  const cropH = cropRect.y1 - cropRect.y0;

  // Draw the cropped region onto the canvas
  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      const MAX_W = Math.min(860, window.innerWidth  * 0.78);
      const MAX_H = Math.min(560, window.innerHeight * 0.68);

      const sx = cropRect.x0 * img.naturalWidth;
      const sy = cropRect.y0 * img.naturalHeight;
      const sw = cropW        * img.naturalWidth;
      const sh = cropH        * img.naturalHeight;

      // Fit within MAX_W × MAX_H preserving crop aspect ratio
      let cw = MAX_W;
      let ch = cw * (sh / sw);
      if (ch > MAX_H) { ch = MAX_H; cw = ch * (sw / sh); }

      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width  = Math.round(cw);
      canvas.height = Math.round(ch);

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
      setDims({ w: canvas.width, h: canvas.height });
    };
    img.src = pdfImage;
  }, [cropRect, pdfImage]);

  // Close context menu on outside click
  useEffect(() => {
    if (!ctxMenu) return;
    const h = () => setCtxMenu(null);
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, [ctxMenu]);

  // Members inside this crop region (use global member index)
  const inside = members
    .map((m, idx) => ({ m, idx }))
    .filter(({ m }) => m.x >= cropRect.x0 && m.x <= cropRect.x1 && m.y >= cropRect.y0 && m.y <= cropRect.y1);

  const colCount  = inside.filter(({ m }) => m.type === "column").length;
  const beamCount = inside.filter(({ m }) => m.type === "beam").length;

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed", inset: 0, zIndex: 600,
        backgroundColor: "rgba(0,0,0,0.7)",
        display: "flex", alignItems: "center", justifyContent: "center",
        backdropFilter: "blur(2px)",
      }}
    >
      {/* Modal box */}
      <div
        onClick={e => e.stopPropagation()}
        style={{
          backgroundColor: "#1E293B", border: "1px solid #334155",
          borderRadius: "12px", overflow: "hidden",
          boxShadow: "0 24px 64px rgba(0,0,0,0.7)",
          display: "flex", flexDirection: "column",
          maxWidth: "90vw",
        }}
      >
        {/* Header */}
        <div style={{
          display: "flex", justifyContent: "space-between", alignItems: "center",
          padding: "10px 16px", borderBottom: "1px solid #334155", flexShrink: 0,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
            <span style={{ color: "white", fontWeight: "700", fontSize: "14px" }}>Region Inspect</span>
            <span style={{ backgroundColor: "#1D4ED8", color: "white", fontSize: "11px", padding: "2px 8px", borderRadius: "4px" }}>
              {colCount} Column{colCount !== 1 ? "s" : ""}
            </span>
            <span style={{ backgroundColor: "#9D174D", color: "white", fontSize: "11px", padding: "2px 8px", borderRadius: "4px" }}>
              {beamCount} Beam{beamCount !== 1 ? "s" : ""}
            </span>
            <span style={{ color: "#64748B", fontSize: "11px" }}>{inside.length} markers total</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span style={{ color: "#475569", fontSize: "11px" }}>Right-click marker to correct</span>
            <button
              onClick={onClose}
              style={{ backgroundColor: "transparent", border: "1px solid #334155", color: "#94A3B8", borderRadius: "4px", padding: "3px 10px", cursor: "pointer", fontSize: "13px" }}
            >✕ Close</button>
          </div>
        </div>

        {/* Canvas + markers */}
        <div style={{ position: "relative", width: dims.w, height: dims.h, flexShrink: 0 }}>
          <canvas ref={canvasRef} style={{ display: "block" }} />

          {inside.map(({ m, idx }) => {
            const relX = (m.x - cropRect.x0) / cropW * 100;
            const relY = (m.y - cropRect.y0) / cropH * 100;
            const isCol = m.type === "column";
            return (
              <div
                key={idx}
                style={{
                  position: "absolute",
                  left: `${relX}%`, top: `${relY}%`,
                  transform: "translate(-50%,-50%)",
                  cursor: "pointer", zIndex: 10,
                  width:  isCol ? "20px" : "28px",
                  height: isCol ? "18px" : "5px",
                }}
                onContextMenu={e => { e.preventDefault(); setCtxMenu({ member: m, idx, x: e.clientX, y: e.clientY }); }}
                title={`${m.profile} — ${m.type}  (right-click to edit)`}
              >
                {m.overridden && (
                  <div style={{ position: "absolute", inset: "-4px", border: "1px dashed #FBBF24", borderRadius: "2px", pointerEvents: "none" }} />
                )}
                {isCol ? (
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                    <div style={{ width: "18px", height: "3px", backgroundColor: m.color, borderRadius: "1px", boxShadow: `0 0 4px ${m.color}` }} />
                    <div style={{ width: "4px",  height: "4px", backgroundColor: m.color, borderRadius: "50%", marginTop: "2px" }} />
                  </div>
                ) : (
                  <div style={{ width: "100%", height: "100%", backgroundColor: m.color, borderRadius: "1px", boxShadow: "0 1px 4px rgba(0,0,0,0.5)" }} />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* In-modal context menu */}
      {ctxMenu && (
        <div
          onMouseDown={e => e.stopPropagation()}
          style={{
            position: "fixed", left: ctxMenu.x, top: ctxMenu.y,
            backgroundColor: "#1E293B", border: "1px solid #334155",
            borderRadius: "8px", zIndex: 700,
            boxShadow: "0 8px 32px rgba(0,0,0,0.7)", overflow: "hidden", minWidth: "190px",
          }}
        >
          <div style={{ padding: "8px 12px", borderBottom: "1px solid #334155", fontSize: "11px", color: "#94A3B8" }}>
            <span style={{ color: ctxMenu.member.color, fontWeight: "bold" }}>{ctxMenu.member.profile}</span>
            {" · "}<span style={{ color: "white" }}>{ctxMenu.member.type}</span>
          </div>
          {(["column","beam","brace"] as const).filter(t => t !== ctxMenu.member.type).map(t => (
            <div key={t}
              onClick={() => { onCorrect(ctxMenu.idx, t); setCtxMenu(null); }}
              style={ctxItemStyle(MEMBER_COLORS[t])}
              onMouseEnter={e => (e.currentTarget.style.backgroundColor = "#1E3A5F")}
              onMouseLeave={e => (e.currentTarget.style.backgroundColor = "transparent")}
            >
              {t === "column" ? "⬛" : t === "beam" ? "━" : "╱"} Mark as {t.charAt(0).toUpperCase() + t.slice(1)}
            </div>
          ))}
          <div style={{ borderTop: "1px solid #334155" }} />
          <div
            onClick={() => { onRemove(ctxMenu.idx); setCtxMenu(null); }}
            style={ctxItemStyle("#F87171")}
            onMouseEnter={e => (e.currentTarget.style.backgroundColor = "#7F1D1D33")}
            onMouseLeave={e => (e.currentTarget.style.backgroundColor = "transparent")}
          >✕ Remove Marker</div>
        </div>
      )}
    </div>
  );
}
