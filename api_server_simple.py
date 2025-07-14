#!/usr/bin/env python3
"""
api_server_simple.py – FastAPI backend for Oracle 23ai vector face-search
──────────────────────────────────────────────────────────────────────────
* One-time start-up loads RetinaFace (detection-only) + ArcFace on
  DirectML-GPU **or** pure CPU with automatic fallback.
* Query picture → face vector → Oracle VECTOR_DISTANCE search.
"""

from __future__ import annotations
import array, logging, os, shutil, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import oracledb
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ─── Oracle / search settings ──────────────────────────────────────────
ORACLE_DSN        = "localhost:1521/FREEPDB1"
ORACLE_USER       = "system"
ORACLE_PWD        = "1123"
VECTOR_TS         = "VECTOR_TS"          # ASSM tablespace that allows VECTOR
SIM_THR           = 0.35                 # cosine-distance threshold
TOP_K             = 100

ROOT_DIR   = Path(__file__).parent.resolve()
UPLOAD_DIR = ROOT_DIR / "temp_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ─── logging ───────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("api_server")

# ─── FastAPI app shell ────────────────────────────────────────────────
app = FastAPI(
    title="Oracle 23ai Face-Similarity API",
    version="1.4.0",
    description="RetinaFace-aligned / ArcFace-embedded vector search"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# Will hold functions after start-up
pipeline: dict[str, Any] = {"ready": False}

# ═════════════════════════ Oracle helper ══════════════════════════════
class OracleVectorSearch:
    """Insert ONE query vector into QUERY_FACE (VECTOR) and run search."""
    def __init__(self):
        self.con = oracledb.connect(user=ORACLE_USER,
                                    password=ORACLE_PWD,
                                    dsn=ORACLE_DSN)
        self.cur = self.con.cursor()
        self._ensure_query_face()

    def _ensure_query_face(self):
        self.cur.execute("""
            SELECT tablespace_name FROM user_tables
            WHERE table_name = 'QUERY_FACE'""")
        row = self.cur.fetchone()
        if row and row[0] != VECTOR_TS:
            log.warning("QUERY_FACE exists in %s – recreating in %s",
                        row[0], VECTOR_TS)
            self.cur.execute("DROP TABLE query_face PURGE")
            self.con.commit()
            row = None
        if not row:
            self.cur.execute(f"""
                CREATE TABLE query_face (
                  id        NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                  embedding VECTOR(512,FLOAT32)
                ) TABLESPACE {VECTOR_TS}""")
            self.con.commit()

    def insert_query_vector(self, vec: np.ndarray):
        self.cur.execute("TRUNCATE TABLE query_face")
        self.cur.execute("INSERT INTO query_face (embedding) VALUES (:v)",
                         {"v": array.array("f", vec.astype(np.float32))})
        self.con.commit()

    def search(self, k: int, thr: float) -> List[Dict[str, Any]]:
        sql = """
        SELECT f.id,
               f.original_image,
               f.cropped_face_path,
               f.face_id,
               f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2,
               VECTOR_DISTANCE(f.embedding, q.embedding, COSINE) AS dist
        FROM   faces f
        CROSS  JOIN query_face q
        WHERE  VECTOR_DISTANCE(f.embedding, q.embedding, COSINE) <= :thr
        ORDER  BY dist
        FETCH  FIRST :k ROWS ONLY"""
        self.cur.execute(sql, {"thr": thr, "k": k})
        cols = [d[0].lower() for d in self.cur.description]
        rows = [dict(zip(cols, r)) for r in self.cur]
        for r in rows:
            d = float(r["dist"])
            r["similarity"] = round(max(0.0, 1.0 - d) * 100, 2)
        return rows

    def close(self):
        self.cur.close(); self.con.close()

# ═════════════════════ file helpers ═══════════════════════════════════
def _save_upload(up: UploadFile) -> Path:
    fn = f"{uuid.uuid4().hex}_{Path(up.filename).name}"
    dst = UPLOAD_DIR / fn
    with dst.open("wb") as fh:
        shutil.copyfileobj(up.file, fh)
    return dst

def _cleanup(p: Path):
    try: p.unlink(missing_ok=True)
    except Exception: pass

# ═════════════════════ API endpoints ═════════════════════════════════
@app.post("/api/search")
async def search_image(
    file: UploadFile = File(...),
    threshold: float = Query(SIM_THR, ge=0.0, le=1.0),
    top_k:     int   = Query(TOP_K,  gt=0,   le=1000)
):
    if not pipeline["ready"]:
        raise HTTPException(status_code=503, detail="Model initialising – retry.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    tmp = _save_upload(file)
    try:
        faces = pipeline["process"](str(tmp))
        if not faces:
            return {"message": "No faces detected.", "results": []}

        db = OracleVectorSearch()
        try:
            results = []
            for f in faces:
                db.insert_query_vector(f["embedding"])
                matches = db.search(top_k, threshold)
                results.append({
                    "query_face_id":   f["face_id"],
                    "query_bbox":      f["bbox"],
                    "query_confidence":round(f["conf"], 3),
                    "matches":         matches
                })
        finally:
            db.close()
        return {"results": results}
    except Exception as exc:
        log.error("Search error", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _cleanup(tmp)

@app.get("/health")
def health():
    return {"status": "up" if pipeline["ready"] else "starting",
            "ts": datetime.now(timezone.utc).isoformat()}

@app.get("/api/static", response_class=FileResponse)
def serve_static(path: str):
    """Serve images only from explicitly allowed directories."""
    allowed = [
        (ROOT_DIR / "cropped_faces").resolve(),
        (ROOT_DIR / "dataset").resolve(),
    ]

    extra = os.getenv("EXTRA_STATIC_ROOTS")
    if extra:
        for root in extra.split(os.pathsep):
            if root:
                allowed.append(Path(root).resolve())

    crop_env = os.getenv("CROPPED_FACES_DIR")
    if crop_env:
        allowed.append(Path(crop_env).resolve())
    else:
        meta_csv = ROOT_DIR / "face_metadata.csv"
        if meta_csv.exists():
            try:
                import csv
                with meta_csv.open(newline="", encoding="utf-8") as fh:
                    reader = csv.reader(fh)
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 2:
                            allowed.append(Path(row[1]).resolve().parent)
                        if len(row) >= 1:
                            allowed.append(Path(row[0]).resolve().parent)
                        break
            except Exception:
                pass

    p = Path(path).resolve()
    if not any(p.is_relative_to(root) for root in allowed):
        return JSONResponse(status_code=403, content={"error": "Forbidden"})
    if not p.exists():
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return p

# ═════════════════════ model start-up (GPU → CPU fallback) ═══════════
@app.on_event("startup")
def startup_models():
    if pipeline["ready"]:
        return

    # honour env DML_DEVICE_ID or default to 0 (discrete card on most laptops)
    preferred_id = os.getenv("DML_DEVICE_ID", "1")

    provider_sets = [
        (["DmlExecutionProvider", "CPUExecutionProvider"],
         [{"device_id": preferred_id}, {}]),
        (["CPUExecutionProvider"],
         [{}])   # provider_options must align with providers list
    ]

    last_err: Exception | None = None
    for providers, opts in provider_sets:
        try:
            tag = "GPU" if providers[0].startswith("Dml") else "CPU"
            log.info("Initialising models on %s (providers=%s)",
                     tag, providers)

            from insightface.app import FaceAnalysis
            det = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection"],
                providers=providers,
                provider_options=opts
            )
            det.prepare(ctx_id=0 if tag == "GPU" else -1,
                        det_size=(640, 640))

            from insightface.model_zoo import get_model
            emb = get_model("buffalo_l",
                            providers=providers,
                            provider_options=opts)
            emb.prepare(ctx_id=0 if tag == "GPU" else -1)

            # helper functions (kept small)
            import cv2
            from insightface.utils.face_align import norm_crop

            def _embed(bgr112: np.ndarray) -> np.ndarray:
                """Return flip-augmented embedding for a 112x112 BGR face."""
                face = cv2.resize(bgr112, (112, 112)) if bgr112.shape[:2] != (112, 112) else bgr112
                v1 = emb.get_feat(face).astype(np.float32).ravel()
                flip = cv2.flip(face, 1)
                v2 = emb.get_feat(flip).astype(np.float32).ravel()
                vec = v1 + v2
                return vec / (np.linalg.norm(vec) + 1e-7)

            def _process(img_path: str) -> List[Dict[str, Any]]:
                bgr = cv2.imread(img_path)
                if bgr is None:
                    return []
                faces = det.get(bgr, max_num=0)
                res: list[Dict[str, Any]] = []
                for i, f in enumerate(faces):
                    crop = norm_crop(bgr, f.kps, 112)
                    if crop is None or crop.size == 0:
                        continue
                    res.append({
                        "face_id":  i,
                        "bbox":     f.bbox.astype(int).tolist(),
                        "conf":     float(f.det_score),
                        "embedding":_embed(crop)
                    })
                return res

            pipeline["process"] = _process
            pipeline["ready"]   = True
            log.info("✅ RetinaFace + ArcFace ready on %s", tag)
            return

        except Exception as e:
            last_err = e
            log.warning("Model init failed on %s – %s", providers[0], e)

    log.error("🚨 Unable to initialise ANY backend (GPU or CPU)")
    raise last_err if last_err else RuntimeError("Unknown model init error")

# ═════════════════════ run standalone (optional) ══════════════════════
if __name__ == "__main__":          # pragma: no cover
    import uvicorn
    uvicorn.run("api_server_simple:app",
                host="0.0.0.0", port=8000, reload=False)
