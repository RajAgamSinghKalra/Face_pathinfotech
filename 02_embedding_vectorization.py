#!/usr/bin/env python3
"""
02_embedding_vectorization.py
─────────────────────────────
* Reads face_metadata.csv (written by 01_face_detection_cropping.py)
* Keeps **only one** record per unique cropped_face file
* Re-aligns from stored 5-point landmarks when possible
* Flip-augmented ArcFace embeddings (GPU via DirectML, fallback CPU)
* Inserts into FACES (VECTOR(512,FLOAT32)) with true float32 arrays
"""

import argparse, csv, json, logging, os, sys, time, array
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Iterable

import numpy as np
from tqdm import tqdm
import oracledb
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization

# ─── optional OpenCV (Pillow fallback) ───────────────────────────────────
try:
    import cv2
    _USE_CV2 = True
except ImportError:                       # pragma: no-cover
    from PIL import Image
    _USE_CV2 = False
    logging.warning("⚠️  OpenCV unavailable – using Pillow resize stub")

# ─── Oracle client ───────────────────────────────────────────────────────
try:
    oracledb.init_oracle_client()
except Exception:
    pass                                   # thin mode is fine

# ─── ArcFace via DirectML ────────────────────────────────────────────────
print("Loading ArcFace (buffalo_l)…")
os.environ["ORT_DML_ALLOW_LIST"] = "1"
from insightface.model_zoo import get_model        # noqa: E402
from insightface.utils.face_align import norm_crop # for re-alignment

_ARCFACE = get_model("buffalo_l",
                     providers=["DmlExecutionProvider", "CPUExecutionProvider"])
try:
    _ARCFACE.prepare(ctx_id=0)
    print("✅ ArcFace running on GPU (DirectML)")
except Exception as e:
    print(f"⚠️  DirectML unavailable ({e}) – using CPU")
    _ARCFACE.prepare(ctx_id=-1)

print("Loading FaceNet (InceptionResnetV1)…")
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    _FACENET = InceptionResnetV1(pretrained="vggface2").to(_DEVICE).eval()
    FACENET_TRANSFORM = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        fixed_image_standardization,
    ])
    print("✅ FaceNet loaded")
except Exception as e:
    print(f"⚠️  FaceNet unavailable: {e}")
    _FACENET = None

# ─── paths & constants ───────────────────────────────────────────────────
ROOT     = Path(r"C:\Users\Agam\Downloads\intern\pathinfotech\face")
META_CSV = ROOT / "face_metadata.csv"

ORACLE_DSN  = os.getenv("ORACLE_DSN",  "localhost:1521/FREEPDB1")
ORACLE_USER = os.getenv("ORACLE_USER", "system")
ORACLE_PWD  = os.getenv("ORACLE_PWD",  "1123")

TABLESPACE  = "VECTOR_TS"
BATCH_SIZE  = 1_000

# ─── DDL (unchanged, includes landmark cols) ─────────────────────────────
CREATE_TABLE_SQL = f"""
CREATE TABLE faces (
  id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  original_image       VARCHAR2(500),
  cropped_face_path    VARCHAR2(500),
  face_id              NUMBER,
  bbox_x1 NUMBER, bbox_y1 NUMBER, bbox_x2 NUMBER, bbox_y2 NUMBER,
  confidence           NUMBER(5,4),
  bbox_width           NUMBER,
  bbox_height          NUMBER,
  embedding            VECTOR(512, FLOAT32),
  image_blob           BLOB,
  landmarks_5          CLOB,
  landmarks_106        CLOB,
  processing_timestamp TIMESTAMP,
  metadata             CLOB
) TABLESPACE {TABLESPACE}
"""

CREATE_VEC_INDEX_SQL = f"""
CREATE VECTOR INDEX faces_embedding_idx
  ON faces (embedding)
  ORGANIZATION NEIGHBOR PARTITIONS
  DISTANCE        COSINE
  TARGET ACCURACY 95
  TABLESPACE      {TABLESPACE}
"""

# ─── logging setup ───────────────────────────────────────────────────────
LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"embed_{datetime.now():%Y%m%d_%H%M%S}.log",
                            encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("embed")

# ─── Oracle helper (unchanged) ───────────────────────────────────────────
class OracleFaces:
    def __init__(self, reset: bool):
        self.conn = oracledb.connect(user=ORACLE_USER,
                                     password=ORACLE_PWD,
                                     dsn=ORACLE_DSN)
        self.cur  = self.conn.cursor()
        if reset:
            self.cur.execute("BEGIN "
                             "EXECUTE IMMEDIATE 'DROP INDEX faces_embedding_idx'; "
                             "EXECUTE IMMEDIATE 'DROP TABLE faces PURGE'; "
                             "EXCEPTION WHEN OTHERS THEN NULL; END;")
            self.conn.commit()
        self._ensure_schema()

    def _ensure_schema(self):
        self.cur.execute(
            "SELECT COUNT(*) FROM user_tables WHERE table_name='FACES'")
        if self.cur.fetchone()[0] == 0:
            log.info("Creating FACES table …")
            self.cur.execute(CREATE_TABLE_SQL); self.conn.commit()
        self.cur.execute(
            "SELECT COUNT(*) FROM user_indexes WHERE index_name='FACES_EMBEDDING_IDX'")
        if self.cur.fetchone()[0] == 0:
            log.info("Creating VECTOR index …")
            self.cur.execute(CREATE_VEC_INDEX_SQL); self.conn.commit()

    def insert(self, meta: Dict[str,Any],
               vec: np.ndarray, img_bytes: bytes):
        arr = array.array("f", np.asarray(vec, np.float32).ravel().tolist())
        self.cur.execute("""
            INSERT INTO faces (
              original_image, cropped_face_path, face_id,
              bbox_x1, bbox_y1, bbox_x2, bbox_y2,
              confidence, bbox_width, bbox_height,
              embedding, image_blob,
              landmarks_5, landmarks_106,
              processing_timestamp, metadata)
            VALUES (
              :orig,:crop,:fid,
              :x1,:y1,:x2,:y2,
              :conf,:bw,:bh,
              :vec,:blob,
              :lm5,:lm106,
              SYSTIMESTAMP,
              :meta)
        """, {
            "orig":  meta["original_image"],
            "crop":  meta["cropped_face"],
            "fid":   meta["face_id"],
            "x1":    meta["bbox_x1"],  "y1": meta["bbox_y1"],
            "x2":    meta["bbox_x2"],  "y2": meta["bbox_y2"],
            "conf":  meta["confidence"],
            "bw":    meta["bbox_width"], "bh": meta["bbox_height"],
            "vec":   arr,
            "blob":  img_bytes,
            "lm5":   meta.get("landmarks_5"),
            "lm106": meta.get("landmarks_106"),
            "meta":  json.dumps(meta, ensure_ascii=False)
        })

    def commit(self): self.conn.commit()
    def close (self): self.cur.close(); self.conn.close()

# ─── alignment + embedding ───────────────────────────────────────────────
def aligned_crop(meta: Dict[str,Any]) -> np.ndarray | None:
    try:
        lm5 = json.loads(meta.get("landmarks_5", "[]"))
        if len(lm5) == 5:
            img = cv2.imread(meta["original_image"])
            if img is not None:
                return norm_crop(img, np.asarray(lm5, np.float32), 112)
    except Exception:
        pass
    return cv2.imread(meta["cropped_face"])

def embed_face(img_bgr: np.ndarray) -> np.ndarray | None:
    """Return combined ArcFace + FaceNet embedding for a BGR crop."""
    if img_bgr is None or img_bgr.size == 0:
        return None
    # ArcFace 112x112 + horizontal flip
    face_arc = cv2.resize(img_bgr, (112, 112)) if img_bgr.shape[:2] != (112, 112) else img_bgr
    v1 = _ARCFACE.get_feat(face_arc).astype(np.float32).ravel()
    v2 = _ARCFACE.get_feat(cv2.flip(face_arc, 1)).astype(np.float32).ravel()
    arc_vec = v1 + v2
    arc_vec /= (np.linalg.norm(arc_vec) + 1e-7)

    # FaceNet 160x160
    if _FACENET is not None:
        face_fn = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (160, 160))
        tensor = FACENET_TRANSFORM(face_fn).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            fn_vec = _FACENET(tensor).cpu().numpy().ravel()
        fn_vec /= (np.linalg.norm(fn_vec) + 1e-7)
        vec = arc_vec + fn_vec
    else:
        vec = arc_vec

    vec /= (np.linalg.norm(vec) + 1e-7)
    return vec.astype(np.float32)

# ─── CSV helpers ─────────────────────────────────────────────────────────
INT_FIELDS = ("face_id","bbox_x1","bbox_y1","bbox_x2",
              "bbox_y2","bbox_width","bbox_height")
def read_meta(path: Path) -> Iterable[Dict[str,Any]]:
    with path.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            for k in INT_FIELDS:
                row[k] = int(float(row[k]))
            row["confidence"] = float(row["confidence"])
            yield row

def deduplicate(rows: Iterable[Dict[str,Any]]) -> list[Dict[str,Any]]:
    """Return one metadata row per unique cropped_face file."""
    seen: dict[str, Dict[str,Any]] = {}
    for r in rows:
        p = Path(r["cropped_face"]).resolve()
        if p.is_file() and p not in seen:   # keep the *first* occurrence
            seen[p] = r
    return list(seen.values())

# ─── main ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true",
                    help="Drop & recreate FACES table + index first")
    args = ap.parse_args()

    if not META_CSV.exists():
        sys.exit(f"❌  {META_CSV} not found")

    raw_rows   = list(read_meta(META_CSV))
    rows       = deduplicate(raw_rows)
    log.info("CSV rows = %d; unique existing crops = %d",
             len(raw_rows), len(rows))

    db  = OracleFaces(reset=args.reset)
    ok = bad = 0; t0 = time.time()

    for meta in tqdm(rows, desc="Embedding"):
        crop = aligned_crop(meta)
        vec  = embed_face(crop)
        if vec is None:
            bad += 1; continue
        with open(meta["cropped_face"], "rb") as fh:
            db.insert(meta, vec, fh.read())
        ok += 1
        if ok % BATCH_SIZE == 0:
            db.commit()

    db.commit(); db.close()
    dt = time.time() - t0
    log.info("✅ Finished – embedded=%d  failed=%d  (%.1fs, %.1f img/s)",
             ok, bad, dt, ok/max(dt,1))

if __name__ == "__main__":
    main()
