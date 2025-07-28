#!/usr/bin/env python3
"""
01_face_detection_cropping.py  –  ArcFace-ready aligned crops with optional 106‑point landmarks
──────────────────────────────────────────────────────────────────────────
* RetinaFace detection + 5‑point (always) & 106‑point (when available) landmark extraction
* Align via 5‑point norm_crop → 112×112 BGR
* Save crops + CSV metadata (includes both landmark sets)
* Uses DirectML on AMD GPU @640×640, fallback to CPU @1024×1024
"""

import glob, hashlib, json, logging, os, sys, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# optional perceptual hash
try:
    import imagehash
    from PIL import Image
    HAS_HASH = True
except ImportError:
    HAS_HASH = False

# DirectML (or CPU)
USE_DML = True
try:
    import torch_directml  # noqa: F401
    DEVICE = torch_directml.device()
    print(f"[INFO] DirectML on {DEVICE}")
except ImportError:
    DEVICE = None
    USE_DML = False
    print("[INFO] torch-directml not found – using CPU")

# InsightFace (load everything so we get both kps & kps_106 if present)
try:
    from insightface.app import FaceAnalysis
    from insightface.utils.face_align import norm_crop
except ImportError as e:
    print("‼️  InsightFace not installed:", e)
    sys.exit(1)

# ───────── USER CONFIG ─────────
# Point to the “dataset” folder that lives next to this script
SCRIPT_DIR    = Path(__file__).resolve().parent
DATASET_PATH  = str((SCRIPT_DIR / "dataset").resolve())
OUTPUT_PATH   = r"C:\Users\Agam\Downloads\intern\pathinfotech\face\cropped_faces"
METADATA_FILE = "face_metadata.csv"
CONF_THR      = 0.50   # lower threshold for harder/angled faces
MIN_SIZE      = 60     # px on shorter side
IMAGE_EXTS    = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

# ASCII‐safe logging
class AsciiFilter(logging.Filter):
    def filter(self, record):
        record.msg, record.args = record.getMessage().encode("ascii","ignore").decode(), ()
        return True

def setup_logger() -> logging.Logger:
    logdir = Path("logs"); logdir.mkdir(exist_ok=True)
    fn     = logdir / f"face_crop_{datetime.now():%Y%m%d_%H%M%S}.log"
    fmt    = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    log    = logging.getLogger("Cropper"); log.setLevel(logging.INFO)
    fh     = logging.FileHandler(fn, encoding="utf-8"); fh.setFormatter(fmt)
    ch     = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); ch.addFilter(AsciiFilter())
    log.addHandler(fh); log.addHandler(ch)
    return log

LOG = setup_logger()

def list_images(root: str) -> List[str]:
    seen, out = set(), []
    for ext in IMAGE_EXTS:
        for p in glob.glob(f"{root}/**/*{ext}", recursive=True):
            k = os.path.normcase(p)
            if k not in seen:
                seen.add(k); out.append(p)
    return sorted(out)

class Aligner:
    def __init__(self):
        providers = (["DmlExecutionProvider","CPUExecutionProvider"] if USE_DML
                     else ["CPUExecutionProvider"])
        self.det = FaceAnalysis(providers=providers)
        if USE_DML:
            try:
                self.det.prepare(ctx_id=0, det_size=(640,640))
                LOG.info("Using DirectML GPU (640×640)")
                return
            except Exception:
                LOG.warning("DML reshape bug → CPU fallback")
        self.det.prepare(ctx_id=-1, det_size=(1024,1024))
        LOG.info("Using CPU (1024×1024)")

    def detect(self, img_bgr: np.ndarray):
        """Detect faces in a BGR image using RetinaFace."""
        return self.det.get(img_bgr, max_num=0)

    @staticmethod
    def crop(img_bgr: np.ndarray, kps5: np.ndarray) -> np.ndarray:
        return norm_crop(img_bgr, kps5, image_size=112)

def main():
    out_dir = Path(OUTPUT_PATH); out_dir.mkdir(exist_ok=True)
    imgs    = list_images(DATASET_PATH)
    LOG.info(f"Found {len(imgs)} images")

    worker = Aligner()
    records: List[Dict] = []
    t0 = time.time()

    for img_path in tqdm(imgs, desc="Cropping"):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            LOG.warning(f"Cannot read {img_path}")
            continue

        faces = worker.detect(img_bgr)
        for idx, face in enumerate(faces):
            x1,y1,x2,y2 = face.bbox.astype(int)
            fw, fh     = x2-x1, y2-y1
            if face.det_score < CONF_THR or min(fw,fh) < MIN_SIZE:
                continue

            crop = worker.crop(img_bgr, face.kps)
            if crop is None or crop.size == 0:
                continue

            # perceptual fingerprint
            if HAS_HASH:
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                fp  = str(imagehash.phash(pil))
            else:
                fp  = hashlib.sha256(crop.tobytes()).hexdigest()

            # safely handle missing 106‐point
            kps5   = face.kps.tolist() if face.kps is not None else []
            kps106 = (face.kps_106.tolist()
                      if getattr(face, "kps_106", None) is not None else [])

            name = f"{Path(img_path).stem}_face{idx}.jpg"
            outp = out_dir/name
            cv2.imwrite(str(outp), crop)

            LOG.info(f"Saved {name}  conf={face.det_score:.3f}")
            records.append({
                "original_image":   img_path,
                "cropped_face":     str(outp),
                "face_id":          idx,
                "confidence":       float(face.det_score),
                "bbox_x1":          x1, "bbox_y1": y1,
                "bbox_x2":          x2, "bbox_y2": y2,
                "bbox_width":       fw, "bbox_height": fh,
                "landmarks_5":      json.dumps(kps5),
                "landmarks_106":    json.dumps(kps106),
                "face_fingerprint": fp,
                "processing_time":  datetime.now().isoformat()
            })

    if records:
        pd.DataFrame(records).to_csv(METADATA_FILE, index=False)
        LOG.info(f"Wrote metadata → {METADATA_FILE}")

    LOG.info(f"Done: {len(records)} faces in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
