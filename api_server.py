#!/usr/bin/env python3
"""
api_server.py – FastAPI wrapper around 04_query_similarity_search.py
─────────────────────────────────────────────────────────────────────
POST  /api/search   multipart/form-data  {file=<image>}   → JSON results
GET   /api/static?path=…                                      → safe file

The module is loaded dynamically so you can keep running `uvicorn --reload`
while tweaking 04_query_similarity_search.py.
"""

from __future__ import annotations
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import importlib.util, os, shutil, sys, uuid
from pathlib import Path
from typing import List

# ───────── dynamic import of 04_query_similarity_search.py ─────────
HERE = Path(__file__).resolve().parent
query_py = HERE / "04_query_similarity_search.py"
spec = importlib.util.spec_from_file_location("qss", query_py)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot import {query_py}")
qss = importlib.util.module_from_spec(spec)          # type: ignore
sys.modules["qss"] = qss                             # so sub-imports work
spec.loader.exec_module(qss)                         # type: ignore

# unified names for convenience
process_query_image = qss.process_query_image
OracleVectorSearch  = qss.OracleVectorSearch
DEFAULT_THR         = qss.SIMILARITY_THRESHOLD
DEFAULT_TOPK        = qss.TOP_K_RESULTS

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
    threshold: float = Query(DEFAULT_THR, ge=0.0, le=1.0,
                             description="Max cosine distance"),
    top_k: int = Query(DEFAULT_TOPK, gt=0, le=1000,
                       description="Max matches to return")
):
    tmp = _save_upload(file)
    try:
        faces = process_query_image(str(tmp))
        if not faces:
            return {"message": "No faces detected.", "results": []}

        db   = OracleVectorSearch()
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

@app.get("/api/static", response_class=FileResponse, summary="Serve safe static files")
def serve_static(path: str):
    """Serve files from a whitelist of directories."""
    allowed_roots: List[Path] = [
        (HERE / "cropped_faces").resolve(),
        (HERE / "dataset").resolve(),
    ]

    extra = os.getenv("EXTRA_STATIC_ROOTS")
    if extra:
        for r in extra.split(os.pathsep):
            if r:
                allowed_roots.append(Path(r).resolve())

    abs_path = Path(path).resolve()
    if not any(abs_path.is_relative_to(root) for root in allowed_roots):
        return JSONResponse(status_code=403, content={"error": "Forbidden"})
    if not abs_path.exists():
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return abs_path
