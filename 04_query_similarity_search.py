#!/usr/bin/env python3
"""
04_query_similarity_search.py
Improvements:
- Face alignment with 5 landmarks before embedding
- Use horizontal flip augmentation for embedding (average original+flipped)
- Remove unnecessary color channel conversion (use BGR as expected by model)
- Output similarity percentage instead of raw distance
"""
import argparse, json, logging, os, sys, array
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from insightface.utils.face_align import norm_crop
import oracledb
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization

# ─── Configuration ───
ORACLE_DSN      = "localhost:1521/FREEPDB1"
ORACLE_USER     = "system"
ORACLE_PASSWORD = "1123"
# Slightly stricter default threshold (cosine distance) for fewer false positives
SIMILARITY_THRESHOLD = 0.25  # 0.25 distance → 0.75 similarity
TOP_K_RESULTS        = 100

ROOT_DIR = Path(__file__).resolve().parent
LOG_DIR  = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


def setup_logging() -> logging.Logger:
    lf = LOG_DIR / f"query_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(lf, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("query")


log = setup_logging()

# ─── Load Oracle Client ───
try:
    oracledb.init_oracle_client(lib_dir=r"C:\Users\Agam\Downloads\intern\pathinfotech\oracle23ai\dbhomeFree\bin")
    log.info("Oracle thick client initialized")
except Exception as e:
    log.warning("Thick-client init failed (%s). Using thin mode.", e)

# ─── Load InsightFace (Detector + ArcFace Model) ───
log.info("Loading InsightFace (buffalo_l model)...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
_ARCFACE = get_model("buffalo_l")
_ARCFACE.prepare(ctx_id=-1)

log.info("Loading FaceNet (InceptionResnetV1)…")
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    _FACENET = InceptionResnetV1(pretrained="vggface2").to(_DEVICE).eval()
    FACENET_TRANSFORM = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        fixed_image_standardization,
    ])
    log.info("FaceNet loaded")
except Exception as e:
    log.warning("FaceNet unavailable: %s", e)
    _FACENET = None

# Pre-defined reference 5 landmarks for 112x112 ArcFace alignment
REF_LANDMARKS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041]    # right mouth corner
], dtype=np.float32)


def align_face(img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Align face using InsightFace's canonical norm_crop."""
    assert landmarks.shape == (5, 2), "landmarks must be 5x2"
    return norm_crop(img, landmarks, 112)


def embed_face(face_img: np.ndarray) -> np.ndarray:
    """Generate a normalized embedding using ArcFace and FaceNet ensemble."""
    face_arc = cv2.resize(face_img, (112, 112)) if face_img.shape[:2] != (112, 112) else face_img
    a1 = _ARCFACE.get_feat(face_arc).flatten()
    a2 = _ARCFACE.get_feat(cv2.flip(face_arc, 1)).flatten()
    arc_vec = a1 + a2
    arc_vec /= (np.linalg.norm(arc_vec) + 1e-7)

    if _FACENET is not None:
        face_fn = cv2.resize(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), (160, 160))
        tensor = FACENET_TRANSFORM(face_fn).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            fn_vec = _FACENET(tensor).cpu().numpy().ravel()
        fn_vec /= (np.linalg.norm(fn_vec) + 1e-7)
        # weighted ensemble – ArcFace contributes more to final embedding
        vec = 0.75 * arc_vec + 0.25 * fn_vec
    else:
        vec = arc_vec

    vec = vec.astype(np.float32)
    vec /= (np.linalg.norm(vec) + 1e-7)
    return vec


class OracleVectorSearch:
    def __init__(self) -> None:
        oracledb.defaults.fetch_lobs = False
        self.con = oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN)
        self.cur = self.con.cursor()
        log.info("Connected to Oracle %s", ORACLE_DSN)

    def close(self) -> None:
        self.cur.close()
        self.con.close()

    def insert_query_vector(self, vec: np.ndarray) -> int:
        """Insert query vector into query_face table and return its ID."""
        self.cur.execute("TRUNCATE TABLE query_face")
        out_id = self.cur.var(oracledb.NUMBER)
        vec_arr = array.array('f', vec.tolist())
        self.cur.execute(
            """
            INSERT INTO query_face (embedding)
            VALUES (:vec) RETURNING id INTO :out_id
            """,
            {"vec": vec_arr, "out_id": out_id}
        )
        self.con.commit()
        return int(out_id.getvalue()[0])

    def search(self, k: int, thr: float) -> List[Dict[str, Any]]:
        sql = f"""
        SELECT f.id,
               f.original_image,
               f.cropped_face_path,
               f.face_id,
               f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2,
               f.confidence,
               f.bbox_width, f.bbox_height,
               f.metadata,
               VECTOR_DISTANCE(f.embedding, q.embedding, COSINE) AS dist
        FROM faces f
        CROSS JOIN query_face q
        WHERE VECTOR_DISTANCE(f.embedding, q.embedding, COSINE) <= :thr
        ORDER BY dist
        FETCH FIRST :k ROWS ONLY
        """
        self.cur.execute(sql, {"thr": thr, "k": k})
        cols = [d[0].lower() for d in self.cur.description]
        return [dict(zip(cols, row)) for row in self.cur]

    def search_auto(self, k: int, thr: float,
                    step: float = 0.05, max_thr: float = 0.5
                    ) -> List[Dict[str, Any]]:
        """Search with progressively larger threshold until matches found."""
        results = self.search(k, thr)
        cur_thr = thr
        while not results and cur_thr < max_thr:
            cur_thr = round(cur_thr + step, 3)
            results = self.search(k, cur_thr)
        return results


def detect_and_embed(image_path: str) -> List[Dict[str, Any]]:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        log.error("Cannot read image %s", image_path)
        return []
    faces = app.get(img_bgr)
    results = []
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = f.bbox.astype(int)
        face_region = img_bgr[y1:y2, x1:x2]
        if face_region.size == 0:
            continue
        if f.kps is not None:
            landmarks = f.kps.astype(np.float32)
            aligned_face = align_face(img_bgr, landmarks)
        else:
            aligned_face = cv2.resize(face_region, (112, 112))
        emb = embed_face(aligned_face)
        results.append({
            "face_id": i,
            "bbox": [int(x) for x in f.bbox],
            "conf": float(f.det_score),
            "embedding": emb
        })
    log.info("Detected %d face(s) in %s", len(results), image_path)
    return results

# Expose helper for API modules
process_query_image = detect_and_embed


def pretty_print(payload: Dict[str, Any]) -> None:
    print("\n" + "="*60)
    print("SIMILARITY SEARCH SUMMARY")
    print("="*60)
    print(f"Query image   : {payload['query_image']}")
    print(f"Faces detected: {payload['num_faces']}")
    print(f"Timestamp     : {payload['timestamp']}\n")
    for fr in payload['results']:
        qid = fr['query_face_id']
        qb = fr['query_bbox']; qc = fr['query_confidence']
        print(f"Face {qid} @ {qb} (det_conf={qc:.3f})")
        if not fr['matches']:
            print("   └ no match ≤ threshold")
            continue
        for j, m in enumerate(fr['matches'][:5], 1):
            sim_pct = (1 - m['dist']) * 100.0
            print(f"   {j:2}. similarity={sim_pct:.2f}%  img={m['original_image']}")
    print("="*60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Path to query image")
    ap.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD, help="Cosine distance threshold")
    ap.add_argument("--top-k", type=int, default=TOP_K_RESULTS, help="Max results to return")
    ap.add_argument("--output", help="Path to save output JSON")
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        log.error("Image not found: %s", args.image)
        sys.exit(1)

    faces = detect_and_embed(args.image)
    if not faces:
        log.error("No faces found – aborting")
        sys.exit(1)

    db = OracleVectorSearch()
    try:
        results = []
        for f in faces:
            _ = db.insert_query_vector(f["embedding"])
            matches = db.search_auto(args.top_k, args.threshold)
            results.append({
                "query_face_id": f["face_id"],
                "query_bbox": f["bbox"],
                "query_confidence": f["conf"],
                "matches": matches
            })

        payload = {
            "query_image": args.image,
            "num_faces": len(faces),
            "threshold": args.threshold,
            "top_k": args.top_k,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }

        pretty_print(payload)

        out_path = args.output or f"search_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        log.info("Result JSON saved to %s", out_path)
    finally:
        db.close()


if __name__ == "__main__":
    main()
