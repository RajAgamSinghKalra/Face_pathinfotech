#!/usr/bin/env python3
"""
03_indexing.py ─ Oracle 23ai VECTOR index + optional FAISS dump
────────────────────────────────────────────────────────────────
usage:
  python 03_indexing.py               # rebuild Oracle vector index
  python 03_indexing.py --faiss idx   # + write idx.faiss / idx.json
  python 03_indexing.py --oracle      # Oracle only (same as default)
  python 03_indexing.py --tablespace MY_TS   # use different TS
"""

from __future__ import annotations
import argparse, json, logging, os, site, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm
import oracledb

# ───────── logging ─────────
LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"index_{datetime.now():%Y%m%d_%H%M%S}.log",
                            encoding="utf-8", errors="replace"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("index")

# ───────── Oracle connection details ─────────
try:
    oracledb.init_oracle_client()                  # thick if libs are found
    log.info("Oracle thick-mode client ✓")
except Exception:
    log.info("python-oracledb thin mode in use")

ORACLE_DSN  = os.getenv("ORACLE_DSN",  "localhost:1521/FREEPDB1")
ORACLE_USER = os.getenv("ORACLE_USER", "system")
ORACLE_PWD  = os.getenv("ORACLE_PWD",  "1123")

FACES_TABLE        = "FACES"
VECTOR_COL         = "EMBEDDING"
BLOB_COL           = "EMBEDDING_BLOB"
VECTOR_IDX         = "FACES_EMBEDDING_IDX"
HELPER_TABLE       = "FACES_VEC_IDX"
HELPER_IDX         = "FVI_HELPER_IDX"
HELPER_TS          = "VECTOR_TS"      # keep with ASSM

ROOT       = Path(r"C:\Users\Agam\Downloads\intern\pathinfotech\face")
INDEX_DIR  = ROOT / "indexes"; INDEX_DIR.mkdir(exist_ok=True)

# ═════════════════════════════ helper functions ══════════════════════════════
def db_open():
    conn = oracledb.connect(user=ORACLE_USER,
                            password=ORACLE_PWD,
                            dsn=ORACLE_DSN)
    cur  = conn.cursor()
    cur.execute(f"""
        SELECT column_name, data_type
        FROM   user_tab_columns
        WHERE  table_name = :tbl
          AND  column_name IN ('{VECTOR_COL}','{BLOB_COL}')
    """, tbl=FACES_TABLE)
    cols = {n: t for n, t in cur.fetchall()}
    has_vector = cols.get(VECTOR_COL) == "VECTOR"
    return conn, cur, has_vector

def sql_drop_index(cur, name: str):
    cur.execute("""
        BEGIN
            EXECUTE IMMEDIATE 'DROP INDEX '||:n;
        EXCEPTION WHEN OTHERS THEN
            IF SQLCODE != -1418 THEN RAISE; END IF;  -- -1418 = index not found
        END;""", n=name)

def sql_drop_table(cur, name: str):
    cur.execute(f"""
        BEGIN
            EXECUTE IMMEDIATE 'DROP TABLE {name} PURGE';
        EXCEPTION WHEN OTHERS THEN
            IF SQLCODE != -942 THEN RAISE; END IF;   -- -942 = table not found
        END;""")

def vector_index_ddl(idx: str, tbl: str, col: str, ts: str | None) -> str:
    ts_clause = f"TABLESPACE {ts}" if ts else ""
    return f"""CREATE VECTOR INDEX {idx}
               ON {tbl} ({col})
               ORGANIZATION NEIGHBOR PARTITIONS
               DISTANCE        COSINE
               TARGET ACCURACY 99
               {ts_clause}"""

def create_vector_index(cur, tbl: str, col: str,
                        idx: str, ts: str | None):
    sql_drop_index(cur, idx)
    log.info("Creating VECTOR index %s …", idx)
    cur.execute(vector_index_ddl(idx, tbl, col, ts))
    cur.connection.commit()
    log.info("VECTOR index %s ready", idx)

# ═════════════════════ FAISS helpers (optional) ═══════════════════════════════
def ensure_faiss():
    try:
        import faiss           # type: ignore
        return faiss
    except ImportError:
        user_site = site.getusersitepackages()
        if user_site not in sys.path and os.path.isdir(user_site):
            sys.path.append(user_site)
            try:
                import faiss   # type: ignore
                return faiss
            except ImportError:
                pass
        sys.exit("Install FAISS:  pip install faiss-cpu")

def build_faiss(ids: np.ndarray, vecs: np.ndarray, out: Path):
    faiss = ensure_faiss()
    d = vecs.shape[1]
    hnsw  = faiss.IndexHNSWFlat(d, 32)
    hnsw.hnsw.efSearch = 64
    index = faiss.IndexIDMap2(hnsw)
    index.add_with_ids(vecs.astype(np.float32), ids.astype(np.int64))
    faiss.write_index(index, str(out))
    log.info("FAISS index → %s (%.1f MB)", out, out.stat().st_size / 1_048_576)

    meta = dict(dim=d, alg="HNSW-Flat", vectors=int(vecs.shape[0]),
                created=datetime.now(timezone.utc).isoformat())
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))

# ═════════════════════ vector extraction ═════════════════════════════════════
def fetch_embeddings(cur, via_blob: bool):
    col = BLOB_COL if via_blob else VECTOR_COL
    cur.execute(f"SELECT id,{col} FROM {FACES_TABLE} ORDER BY id")
    ids, vecs = [], []
    for rid, val in tqdm(cur.fetchall(), desc="Fetch"):
        ids.append(rid)
        vec = (np.frombuffer(val.read(), "<f4") if via_blob
               else np.asarray(val, np.float32))
        vecs.append(vec)
    return np.asarray(ids, np.int64), np.vstack(vecs)

# ═════════════════════════════ main ═══════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true",
                    help="(re)build Oracle VECTOR index (default)")
    ap.add_argument("--faiss", metavar="FILE",
                    help="write FAISS index (HNSW-Flat) to FILE.faiss")
    ap.add_argument("--tablespace", metavar="TS",
                    help="override tablespace for VECTOR index")
    args = ap.parse_args()

    want_oracle = args.oracle or args.faiss is None
    want_faiss  = bool(args.faiss)

    conn, cur, has_vector = db_open()

    # Oracle VECTOR index
    if want_oracle:
        if has_vector:
            ts = args.tablespace or "VECTOR_TS"
            create_vector_index(cur, FACES_TABLE, VECTOR_COL,
                                VECTOR_IDX, ts)
        else:
            # legacy path – build helper VECTOR table from BLOBs
            log.warning("No VECTOR column; creating helper table/index …")
            ids, vecs = fetch_embeddings(cur, via_blob=True)
            sql_drop_table(cur, HELPER_TABLE)
            cur.execute(f"""
                CREATE TABLE {HELPER_TABLE} (
                    id        NUMBER PRIMARY KEY,
                    embedding VECTOR(512,FLOAT32)
                ) TABLESPACE {HELPER_TS}""")
            cur.executemany(
                f"INSERT /*+ APPEND */ INTO {HELPER_TABLE}(id,embedding) "
                f"VALUES (:1,:2)",
                [(int(i), v.tolist()) for i, v in zip(ids, vecs)])
            conn.commit()
            create_vector_index(cur, HELPER_TABLE, VECTOR_COL,
                                HELPER_IDX, HELPER_TS)

    # FAISS build (optional)
    if want_faiss:
        if "ids" not in locals():
            ids, vecs = fetch_embeddings(cur, via_blob=not has_vector)
        out = INDEX_DIR / args.faiss
        build_faiss(ids, vecs, out)

    cur.close(); conn.close()
    log.info("DONE")

if __name__ == "__main__":
    main()
