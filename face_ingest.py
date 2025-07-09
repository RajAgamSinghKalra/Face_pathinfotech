#!/usr/bin/env python
"""
Batch-detect faces (MTCNN) ▶ embed (ArcFace) ▶ store 512-D vectors in Oracle 23ai.

• Source images: LFW dataset under
      C:/Users/Agam/Downloads/intern/pathinfotech/face/dataset/lfw-deepfunneled/lfw-deepfunneled
• Oracle 23ai instance: container (or native install) listening on localhost:1521
• Schema user: face_app / face_pass
"""

import os, sys, time, logging, glob, cv2, numpy as np, torch, oracledb
from datetime import datetime
from tqdm import tqdm
from facenet_pytorch import MTCNN
from insightface.model_zoo import get_model

# ── CONFIG ────────────────────────────────────────────────────────────
IMG_DIR  = r"C:\Users\Agam\Downloads\intern\pathinfotech\face\dataset\lfw-deepfunneled\lfw-deepfunneled"
DB_DSN   = "localhost:1521/FREEPDB1"       # service of Oracle 23ai
DB_USER  = "face_app"
DB_PWD   = "face_pass"

# ── LOGGING ───────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logf = f"logs/face23ai_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(logf), logging.StreamHandler(sys.stdout)]
)
logging.info("Session started.")

# ── DEVICE ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Running on device: {device}")

# ── MODELS ────────────────────────────────────────────────────────────
mtcnn   = MTCNN(keep_all=True, device=device)
arcface = get_model("glint360k_r100_fp16_0.1")     # 512-D ArcFace
arcface.eval().to(device)

def embed_faces(img_path: str):
    """Return list of (bbox, embedding) pairs for one image."""
    bgr = cv2.imread(img_path)
    if bgr is None:
        logging.warning(f"Unreadable image: {img_path}")
        return []
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb)
    if boxes is None:
        return []
    out = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (112, 112))
        tensor = torch.from_numpy(crop).permute(2, 0, 1).float().to(device)
        tensor = (tensor - 127.5) / 128.0         # ArcFace normalisation
        with torch.no_grad():
            emb = arcface(tensor.unsqueeze(0)).cpu().numpy()[0]
        emb /= np.linalg.norm(emb)               # L2-normalise for cosine
        out.append((box, emb))
    return out

# ── ORACLE DB CONNECTION ──────────────────────────────────────────────
oracledb.defaults.fetch_lobs = False
conn   = oracledb.connect(user=DB_USER, password=DB_PWD,
                          dsn=DB_DSN, thick=False)
cursor = conn.cursor()
insert_sql = "INSERT INTO faces(img_path, embedding) VALUES(:1, :2)"

# ── INGEST LOOP ───────────────────────────────────────────────────────
patterns = ["*.jpg", "*.jpeg", "*.png"]
all_imgs = []
for pat in patterns:
    all_imgs += glob.glob(os.path.join(IMG_DIR, "**", pat), recursive=True)

tot_faces, errors = 0, 0
t0 = time.time()

for img_path in tqdm(all_imgs, desc="Processing images"):
    try:
        for _, emb in embed_faces(img_path):
            cursor.execute(insert_sql, (img_path, emb.tolist()))
            tot_faces += 1
        if tot_faces % 500 == 0:
            conn.commit()
    except Exception as e:
        errors += 1
        logging.exception(f"Failed on {img_path}: {e}")

conn.commit()
elapsed = time.time() - t0
logging.info(f"Scanned {len(all_imgs)} files.")
logging.info(f"Faces stored : {tot_faces}")
logging.info(f"Errors       : {errors}")
logging.info(f"Elapsed time : {elapsed/60:.1f} min")

cursor.close()
conn.close()
