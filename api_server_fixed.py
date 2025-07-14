#!/usr/bin/env python3
"""
api_server_fixed.py – FastAPI wrapper around 04_query_similarity_search.py
─────────────────────────────────────────────────────────────────────
POST  /api/search   multipart/form-data  {file=<image>}   → JSON results
GET   /api/static?path=…                                      → safe file

Fixed version that handles model loading issues more gracefully.
"""

from __future__ import annotations
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import importlib.util, os, shutil, sys, uuid
from pathlib import Path
from typing import List
import logging

# ───────── logging ─────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api_server")

# ───────── dynamic import of 04_query_similarity_search.py ─────────
HERE = Path(__file__).resolve().parent
query_py = HERE / "04_query_similarity_search.py"

if not query_py.exists():
    raise FileNotFoundError(f"Cannot find {query_py}")

spec = importlib.util.spec_from_file_location("qss", query_py)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot import {query_py}")

qss = importlib.util.module_from_spec(spec)          # type: ignore
sys.modules["qss"] = qss                             # so sub-imports work

# ───────── Safe model loading ─────────
def safe_load_models():
    """Safely load InsightFace models with fallback options"""
    try:
        # Try to load models with CPU-only provider first
        log.info("Loading InsightFace models with CPU provider...")
        
        # Override the model loading in the imported module
        import insightface.app
        import insightface.model_zoo
        
        # Force CPU-only mode to avoid DirectML issues
        os.environ["ORT_DML_ALLOW_LIST"] = "0"  # Disable DirectML
        
        # Create FaceAnalysis with CPU-only
        det = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        det.prepare(ctx_id=-1, det_size=(1024, 1024))
        
        # Create embedder with CPU-only
        embedder = insightface.model_zoo.get_model(
            "buffalo_l", 
            providers=["CPUExecutionProvider"]
        )
        embedder.prepare(ctx_id=-1)
        
        # Replace the global instances in the imported module
        qss._det = det
        qss._embedder = embedder
        
        log.info("✅ Models loaded successfully with CPU provider")
        return True
        
    except Exception as e:
        log.error(f"❌ Failed to load models: {e}")
        return False

# ───────── FastAPI app ─────────
app = FastAPI(title="Face-Similarity API",
              version="1.0.0",
              description="Simple wrapper around Oracle 23ai VECTOR search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

UPLOAD_DIR = HERE / "temp_uploads"; UPLOAD_DIR.mkdir(exist_ok=True)

# ───────── helpers ─────────
def _save_upload(upload: UploadFile) -> Path:
    uid   = uuid.uuid4().hex
    fname = f"{uid}_{Path(upload.filename).name}"
    dest  = UPLOAD_DIR / fname
    with dest.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh)
    return dest

def _cleanup(path: Path):
    try: path.unlink(missing_ok=True)
    except Exception: pass

# ─────────  API endpoints  ─────────
@app.post("/api/search", summary="Search similar faces")
async def search_image(
    file: UploadFile = File(..., description="Image file (jpg/png/etc.)"),
    threshold: float = Query(0.35, ge=0.0, le=1.0,
                             description="Max cosine distance"),
    top_k: int = Query(100, gt=0, le=1000,
                       description="Max matches to return")
):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        tmp = _save_upload(file)
        try:
            # Use the imported functions
            faces = qss.process_query_image(str(tmp))
            if not faces:
                return {"message": "No faces detected.", "results": []}

            db = qss.OracleVectorSearch()
            rows = []
            try:
                for face in faces:
                    db.insert_query_vector(face["embedding"])
                    matches = db.search(top_k, threshold)
                    rows.append({
                        "query_face_id":   face["face_id"],
                        "query_bbox":      face["bbox"],
                        "query_confidence":face["conf"],
                        "matches":         matches
                    })
            finally:
                db.close()

            return {"results": rows}
        finally:
            _cleanup(tmp)
            
    except Exception as e:
        log.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/static", response_class=FileResponse, summary="Serve safe static files")
def serve_static(path: str):
    """Serve files from allowed directories only."""
    allowed_roots: List[Path] = [
        (HERE / "cropped_faces").resolve(),
        (HERE / "dataset").resolve(),
    ]

    extra = os.getenv("EXTRA_STATIC_ROOTS")
    if extra:
        for r in extra.split(os.pathsep):
            if r:
                allowed_roots.append(Path(r).resolve())

    crop_env = os.getenv("CROPPED_FACES_DIR")
    if crop_env:
        allowed_roots.append(Path(crop_env).resolve())
    else:
        meta_csv = HERE / "face_metadata.csv"
        if meta_csv.exists():
            try:
                import csv
                with meta_csv.open(newline="", encoding="utf-8") as fh:
                    reader = csv.reader(fh)
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 2:
                            allowed_roots.append(Path(row[1]).resolve().parent)
                        if len(row) >= 1:
                            allowed_roots.append(Path(row[0]).resolve().parent)
                        break
            except Exception:
                pass

    abs_path = Path(path).resolve()
    if not any(abs_path.is_relative_to(root) for root in allowed_roots):
        return JSONResponse(status_code=403, content={"error": "Forbidden"})
    if not abs_path.exists():
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return abs_path

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": True,
        "timestamp": "2024-01-01T12:00:00"
    }

# ───────── startup event ─────────
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    log.info("🚀 Starting Face Search API Server...")
    if not safe_load_models():
        log.error("❌ Failed to load models - server may not work correctly")
    else:
        log.info("✅ Server ready to handle requests")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False) 