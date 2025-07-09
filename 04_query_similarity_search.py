#!/usr/bin/env python3
"""
04_query_similarity_search.py  –  RetinaFace-aligned, ArcFace-embedded search
──────────────────────────────────────────────────────────────────────────────
• Detect faces with RetinaFace (InsightFace «buffalo_l») on GPU (DirectML) or CPU
• Align every face with its 5-point landmarks → 112 × 112 BGR
• Embed with ArcFace (512-D)  + horizontal-flip augmentation
• Push query vector(s) into QUERY_FACE (VECTOR(512,FLOAT32))  ❱  ONE ROW ONLY
• Run Oracle VECTOR_DISTANCE(COSINE) predicate (index built by 03_indexing.py)
• Display similarity = (1 – distance) × 100 %
"""

from __future__ import annotations
import argparse, array, json, logging, os, sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import oracledb
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop
from insightface.model_zoo import get_model

# ───────── configuration ─────────
ORACLE_DSN   = os.getenv("ORACLE_DSN",  "localhost:1521/FREEPDB1")
ORACLE_USER  = os.getenv("ORACLE_USER", "system")
ORACLE_PWD   = os.getenv("ORACLE_PWD",  "1123")

SIMILARITY_THRESHOLD = 0.35     # maximum cosine distance
TOP_K_RESULTS        = 100

ROOT     = Path(__file__).parent
LOG_DIR  = ROOT / "logs"; LOG_DIR.mkdir(exist_ok=True)

# ───────── logging ─────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"query_{datetime.now():%Y%m%d_%H%M%S}.log",
                            encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("query")

# ───────── Oracle client ─────────
try:
    oracledb.init_oracle_client(
        lib_dir=r"C:\Users\Agam\Downloads\intern\pathinfotech\oracle23ai\dbhomeFree\bin"
    )
    log.info("Oracle thick-client initialised")
except Exception as e:
    log.info("Using python-oracledb thin mode (%s)", e)

# ───────── DirectML check ─────────
USE_DML = True
try:
    import torch_directml  # noqa: F401
    DML_DEVICE = torch_directml.device()
    log.info("DirectML device: %s", DML_DEVICE)
except ImportError:
    USE_DML = False
    log.info("torch-directml not found – falling back to CPU")

# ───────── InsightFace models ─────────
log.info("Loading InsightFace «buffalo_l» (detection + landmarks) …")
providers = (["DmlExecutionProvider", "CPUExecutionProvider"]
             if USE_DML else ["CPUExecutionProvider"])
_det = FaceAnalysis(name="buffalo_l",
                    providers=providers)

# first try GPU @640×640; if DML reshape bug, fall back
if USE_DML:
    try:
        _det.prepare(ctx_id=0, det_size=(640, 640))
        log.info("RetinaFace on DirectML GPU (640×640)")
    except Exception as e:
        log.warning("DirectML failed (%s) – using CPU @1024×1024", e)
        _det.prepare(ctx_id=-1, det_size=(1024, 1024))
else:
    _det.prepare(ctx_id=-1, det_size=(1024, 1024))
    log.info("RetinaFace on CPU (1024×1024)")

log.info("Loading ArcFace head …")
os.environ["ORT_DML_ALLOW_LIST"] = "1"
_embedder = get_model("buffalo_l", providers=providers)
try:
    _embedder.prepare(ctx_id=0 if USE_DML else -1)
except Exception:
    _embedder.prepare(ctx_id=-1)  # ultimate fallback

# ───────── embedding helper ─────────
def arcface_embedding(bgr112: np.ndarray) -> np.ndarray:
    """512-D, L2-normalised, flip-augmented ArcFace vector (float32)"""
    rgb = bgr112[..., ::-1]
    v1  = _embedder.get_feat(rgb).astype(np.float32).flatten()
    v2  = _embedder.get_feat(rgb[:, ::-1, :]).astype(np.float32).flatten()
    vec = v1 + v2
    return vec / (np.linalg.norm(vec) + 1e-7)

# ───────── Oracle helper ─────────
class OracleVectorSearch:
    """Push ONE query vector into QUERY_FACE, then search FACES."""
    def __init__(self):
        self.con = oracledb.connect(user=ORACLE_USER,
                                    password=ORACLE_PWD,
                                    dsn=ORACLE_DSN)
        self.cur = self.con.cursor()
        self._ensure_query_table()

    # ------------------------------------------------------------------ #
    def _ensure_query_table(self):
        ddl = """
        BEGIN
          EXECUTE IMMEDIATE '
            CREATE TABLE query_face (
              id        NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
              embedding VECTOR(512, FLOAT32)
            )';
        EXCEPTION WHEN OTHERS THEN
          IF SQLCODE != -955 THEN RAISE; END IF;  -- -955: already exists
        END;"""
        self.cur.execute(ddl)
        self.con.commit()

    # ------------------------------------------------------------------ #
    def insert_query_vector(self, vec: np.ndarray):
        self.cur.execute("TRUNCATE TABLE query_face")
        arr = array.array("f", vec.tolist())          # true float32
        self.cur.execute("INSERT INTO query_face (embedding) VALUES (:v)",
                         {"v": arr})
        self.con.commit()

    # ------------------------------------------------------------------ #
    def search(self, k: int, thr: float) -> List[Dict[str,Any]]:
        sql = f"""
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
        FETCH  FIRST :k ROWS ONLY
        """
        self.cur.execute(sql, {"thr": thr, "k": k})
        cols = [d[0].lower() for d in self.cur.description]
        rows = [dict(zip(cols, r)) for r in self.cur]
        for r in rows:
            d = float(r["dist"])
            r["similarity"] = round(max(0.0, 1.0 - d) * 100, 2)
        return rows

    # ------------------------------------------------------------------ #
    def close(self):
        self.cur.close(); self.con.close()

# ───────── detect + align + embed ─────────
def process_query_image(path: str) -> List[Dict[str,Any]]:
    bgr = cv2.imread(path)
    if bgr is None:
        log.error("Cannot read %s", path); return []
    faces = _det.get(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), max_num=0)
    out   = []
    for i, f in enumerate(faces):
        crop = norm_crop(bgr, f.kps, 112)
        if crop is None or crop.size == 0:
            continue
        vec = arcface_embedding(crop)
        x1,y1,x2,y2 = f.bbox.astype(int).tolist()
        out.append(dict(face_id=i,
                        bbox=[x1,y1,x2,y2],
                        conf=float(f.det_score),
                        embedding=vec))
    log.info("Detected %d face(s) in query", len(out))
    return out

# ───────── pretty print ─────────
def print_results(payload: Dict[str,Any]):
    print("\n" + "="*70)
    print(f"QUERY  : {payload['query_image']}")
    print(f"FACES  : {payload['num_faces']}   "
          f"threshold={payload['threshold']}   "
          f"{payload['timestamp']}")
    print("-"*70)
    for res in payload["results"]:
        print(f"Face {res['query_face_id']}  bbox={res['query_bbox']}  "
              f"det={res['query_confidence']:.3f}")
        if not res["matches"]:
            print("   └ no matches ≤ threshold")
            continue
        for j,m in enumerate(res["matches"][:5], 1):
            print(f"   {j:2d}.  {m['similarity']:6.2f}%   "
                  f"dist={m['dist']:.4f}   {m['original_image']}")
    print("="*70)

# ───────── main ─────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Query image file")
    ap.add_argument("--threshold", type=float,
                    default=SIMILARITY_THRESHOLD,
                    help="max cosine distance (default 0.35)")
    ap.add_argument("--top-k", type=int, default=TOP_K_RESULTS)
    ap.add_argument("--output", help="save JSON result → file")
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"❌  image not found: {args.image}")

    faces = process_query_image(args.image)
    if not faces:
        sys.exit("❌  no faces detected")

    db = OracleVectorSearch()
    try:
        results = []
        for f in faces:
            db.insert_query_vector(f["embedding"])
            matches = db.search(args.top_k, args.threshold)
            results.append(dict(query_face_id=f["face_id"],
                                query_bbox=f["bbox"],
                                query_confidence=f["conf"],
                                matches=matches))
        payload = dict(query_image=args.image,
                       num_faces=len(faces),
                       threshold=args.threshold,
                       top_k=args.top_k,
                       timestamp=datetime.now().isoformat(),
                       results=results)
        print_results(payload)

        out = args.output or f"search_{datetime.now():%Y%m%d_%H%M%S}.json"
        Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("Results JSON → %s", out)
    finally:
        db.close()

if __name__ == "__main__":
    main()
