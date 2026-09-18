"""Build signed sparse connectome. DuckDB filter-first, bounded RAM.

    python process.py

Needs data/malecns/raw/*.feather from downloader.py.
Writes unsigned connectome.npz + neurons.parquet (sign per source row).
(plus weights.parquet cache reused across runs).

RAM design: feather->parquet streams once; DuckDB (capped 4 GB + disk
spill) decides kept bodies and writes filtered signed edges; CSR builds
from the small edge file only. Refuses to run with <6 GB free.
"""

import gc
import resource
import shutil
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
from pyarrow import ipc

OUT = Path("data/malecns")
RAW = OUT / "raw"
TMP = OUT / "tmp_coo"
DTMP = OUT / "tmp_duck"
AFILE = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NFILE = "body-neurotransmitters-male-cns-v1.0.feather"
WFILE = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
WPAR = OUT / "weights.parquet"
EPAR = OUT / "edges_kept.parquet"

MIN_FREE_GB = 6.0  # ponytail: refuse instead of letting the kernel pick victims

# Mirrors the SQL CASE below; used only for the neurons table sign column.
# ACh excites, GABA inhibits, glutamate mostly inhibitory (GluCl-alpha),
# monoamines modulatory at low gain.
NT_SIGN = {
    "acetylcholine": +1.0, "gaba": -1.0, "glutamate": -1.0,
    "dopamine": +0.2, "serotonin": +0.2, "octopamine": +0.2,
    "unknown": 0.0, "unclear": 0.0, "": 0.0,
}

VISUAL_MOTOR_DROP = (
    "'ol_intrinsic','ol_sensory','visual_projection','visual_centrifugal',"
    "'vnc_motor','cb_motor','vnc_efferent'"
)  # ponytail: keep descending_neuron as action readout


def find_col(df: pd.DataFrame, *candidates: str) -> str:
    """Janelia renames columns between releases -- fail loudly, not silently."""
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    sys.exit(f"None of {candidates} found. Columns are: {list(df.columns)}")


def free_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1e6
    return 0.0


def peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def feather_to_parquet() -> None:
    """Stream 1 GB feather -> parquet once. Tiny RAM, ~65k rows at a time."""
    rdr = ipc.RecordBatchFileReader(str(RAW / WFILE))
    names = rdr.schema.names
    low = {c.lower(): c for c in names}

    def _pick(*cands: str) -> int:
        for c in cands:
            if c.lower() in low:
                return rdr.schema.get_field_index(low[c.lower()])
        sys.exit(f"None of {cands} found. Columns are: {names}")

    pi, qi, wi = (_pick("body_pre", "bodyId_pre", "pre"),
                  _pick("body_post", "bodyId_post", "post"),
                  _pick("weight", "synapse_count", "count"))
    n = rdr.num_record_batches
    schema = pa.schema([("pre", pa.int64()), ("post", pa.int64()),
                        ("weight", pa.int64())])
    print(f"  converting {n:,} batches -> {WPAR} ...")
    with pq.ParquetWriter(WPAR, schema) as w:
        buf = []
        for i in range(n):
            b = rdr.get_batch(i)
            buf.append(pa.record_batch(
                [b.column(pi), b.column(qi), b.column(wi)],
                names=["pre", "post", "weight"]))
            del b
            if len(buf) >= 20 or i == n - 1:
                w.write_table(pa.Table.from_batches(buf))
                buf = []
            if i % 500 == 0 or i == n - 1:
                print(f"  batch {i + 1:,}/{n:,}", end="\r")
    print()


def main() -> None:
    free = free_gb()
    if free < MIN_FREE_GB:
        sys.exit(f"only {free:.1f} GB free, need {MIN_FREE_GB:.0f}+ — "
                 "close browser/VS Code/Discord first")
    for f in (AFILE, NFILE, WFILE):
        if not (RAW / f).exists():
            sys.exit(f"missing {RAW / f} — run python downloader.py first")

    print("Loading annotations (small)...")
    ann = pd.read_feather(RAW / AFILE)
    nt = pd.read_feather(RAW / NFILE)
    a_id = find_col(ann, "bodyId", "body_id", "body")
    n_id = find_col(nt, "bodyId", "body_id", "body")
    n_nt = find_col(nt, "consensus_nt", "predictedNt", "nt", "top_nt")
    ann_t = ann[[a_id, "superclass"]].rename(columns={a_id: "body"})
    nt_t = nt[[n_id, n_nt]].rename(columns={n_id: "body", n_nt: "nt"})
    del ann, nt
    gc.collect()

    if not WPAR.exists():
        feather_to_parquet()
    else:
        print(f"  {WPAR} exists, skipping conversion")

    DTMP.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA temp_directory='{DTMP}'")
    con.execute("PRAGMA memory_limit='4GB'")  # ponytail: spill, never OOM
    con.register("ann", ann_t)
    con.register("nt", nt_t)

    print("\nSQL pass 1: kept bodies (distinct, out-of-core)...")
    con.execute(f"""
        CREATE OR REPLACE TABLE kept AS
        -- ponytail: weights list 88M raw segments; restrict to proofread bodies in ann
        SELECT pre AS body FROM '{WPAR}' WHERE pre IN (SELECT body FROM ann)
        UNION
        SELECT post AS body FROM '{WPAR}' WHERE post IN (SELECT body FROM ann);
        CREATE OR REPLACE TABLE kept_ok AS
        SELECT k.body FROM kept k LEFT JOIN ann a ON a.body = k.body
        WHERE COALESCE(a.superclass, '') NOT IN ({VISUAL_MOTOR_DROP});
    """)
    n_kept = con.execute("SELECT COUNT(*) FROM kept").fetchone()[0]
    n_ok = con.execute("SELECT COUNT(*) FROM kept_ok").fetchone()[0]
    print(f"  bodies {n_kept:,} -> kept {n_ok:,} (dropped {n_kept - n_ok:,})")

    print("\nSQL pass 2: unsigned edges for kept bodies (sign in neurons.sign)...")
    con.execute(f"""
        COPY (
          SELECT w.pre AS pre, w.post AS post, w.weight AS weight
          FROM '{WPAR}' w
          JOIN kept_ok k1 ON k1.body = w.pre
          JOIN kept_ok k2 ON k2.body = w.post
        ) TO '{EPAR}' (FORMAT PARQUET);
    """)
    n_edges = con.execute(f"SELECT COUNT(*) FROM '{EPAR}'").fetchone()[0]
    print(f"  kept edges {n_edges:,}")
    kept_ids = np.array(con.execute(
        "SELECT body FROM kept_ok ORDER BY body").fetchall(),
        dtype=np.int64).ravel()
    con.close()
    shutil.rmtree(DTMP, ignore_errors=True)
    N = len(kept_ids)
    print(f"  peak so far {peak_gb():.1f} GB")

    print("\nNeuron table (small, pandas)...")
    neurons = (pd.DataFrame({"bodyId": kept_ids})
                 .merge(ann_t, left_on="bodyId", right_on="body", how="left")
                 .merge(nt_t, left_on="bodyId", right_on="body", how="left"))
    neurons = neurons.drop(columns=["body_x", "body_y"])
    neurons["nt"] = (neurons["nt"].fillna("unknown").astype(str).str.lower())
    neurons["sign"] = neurons["nt"].map(NT_SIGN).fillna(0.0)
    print(neurons["nt"].value_counts().head(8).to_string())

    print("\nCSR build: mapping kept edges in 1M-row batches...")
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(EPAR)
    R, C, V, flushed, n_files, seen = [], [], [], 0, 0, 0
    for bi, batch in enumerate(pf.iter_batches(batch_size=1_000_000,
                                               columns=["pre", "post", "weight"])):
        d = batch.to_pydict()
        P, Q, W = (np.asarray(d["pre"]), np.asarray(d["post"]), np.asarray(d["weight"]))
        del d, batch
        # kept_ids sorted+complete: searchsorted result IS the row index.
        r = np.searchsorted(kept_ids, P)
        c = np.searchsorted(kept_ids, Q)
        ok = ((r < N) & (kept_ids[np.clip(r, 0, N - 1)] == P)
              & (c < N) & (kept_ids[np.clip(c, 0, N - 1)] == Q) & (W > 0))
        R.append(r[ok].astype(np.int32))
        C.append(c[ok].astype(np.int32))
        V.append(np.asarray(W, dtype=np.float32)[ok])
        del P, Q, W, r, c
        seen += int(ok.size)
        if (bi + 1) % 10 == 0:
            np.savez(TMP / f"coo_{n_files:04d}.npz",
                     rows=np.concatenate(R), cols=np.concatenate(C),
                     vals=np.concatenate(V))
            n_files += 1
            flushed += sum(map(len, V))
            R.clear()
            C.clear()
            V.clear()
            gc.collect()
        if (bi + 1) % 5 == 0:
            print(f"  seen {seen:>,} peak {peak_gb():.1f} GB", end="\r")
    print()
    if R:
        np.savez(TMP / f"coo_{n_files:04d}.npz",
                 rows=np.concatenate(R), cols=np.concatenate(C),
                 vals=np.concatenate(V))
        flushed += sum(map(len, V))
        del R, C, V
        gc.collect()

    print("\nAssembling CSR...")
    sizes = []
    for f in sorted(TMP.glob("coo_*.npz")):
        with np.load(f) as z:
            sizes.append(len(z["vals"]))
    total = sum(sizes)
    rows = np.empty(total, dtype=np.int32)
    cols = np.empty(total, dtype=np.int32)
    vals = np.empty(total, dtype=np.float32)
    off = 0
    for f in sorted(TMP.glob("coo_*.npz")):
        with np.load(f) as z:
            n = len(z["vals"])
            rows[off:off + n] = z["rows"]
            cols[off:off + n] = z["cols"]
            vals[off:off + n] = z["vals"]
            off += n
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"  assembled {total:,} edges, peak {peak_gb():.1f} GB")
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    del rows, cols, vals
    gc.collect()
    A.sum_duplicates()
    A.sort_indices()

    sp.save_npz(OUT / "connectome.npz", A)
    neurons.to_parquet(OUT / "neurons.parquet")

    density = A.nnz / N ** 2
    print(f"  matrix {N:,} x {N:,}, {A.nnz:,} edges, density {density:.2e}")
    print(f"  wrote {OUT / 'connectome.npz'} and {OUT / 'neurons.parquet'}")
    print(f"  peak RSS {peak_gb():.1f} GB")


if __name__ == "__main__":
    main()
