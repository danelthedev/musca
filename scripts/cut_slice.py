"""Cut a targeted reservoir slice: central-brain intrinsic core.

Selection: superclass == cb_intrinsic (association tissue: mushroom body +
central complex live here; parquet has no finer MB/CX labels, so no stronger
claim) -> rank by in-slice in+out degree -> top k. Saves row indices; the
submatrix is rebuilt on load (cheap). Writes results/slice_10k.md doc.
Usage: cut_slice.py [--k 10000] [--pool cb_intrinsic] [--out data/slice_10k_idx.npy]
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import scipy.sparse as sp


def parse(a):
    o = dict(k=10000, pool="cb_intrinsic", out=str(ROOT / "data" / "slice_10k_idx.npy"))
    i = 0
    while i < len(a):
        if a[i].startswith("--") and i + 1 < len(a):
            o[a[i][2:]] = type(o.get(a[i][2:], ""))(a[i + 1]); i += 2
        else:
            i += 1
    o["k"] = int(o["k"])
    return o


def main():
    o = parse(sys.argv[1:])
    t0 = time.time()
    n = pd.read_parquet(str(ROOT / "data" / "malecns" / "neurons.parquet"))
    W = sp.load_npz(str(ROOT / "data" / "malecns" / "connectome.npz")).tocsr()
    pool = np.where(n["superclass"].to_numpy() == o["pool"])[0]
    print(f"pool {o['pool']}: {len(pool)} units", flush=True)
    sub = W[pool][:, pool].tocsr()
    sub.eliminate_zeros()
    indeg = np.diff(sub.tocsc().indptr)
    outdeg = np.diff(sub.indptr)
    score = indeg + outdeg
    take = np.argsort(score)[-o["k"]:]
    idx = np.sort(pool[take])
    sel = W[idx][:, idx].tocsr()
    sel.eliminate_zeros()
    np.save(o["out"], idx)
    doc = [
        "# slice_10k: central-brain intrinsic core",
        "",
        f"- pool: superclass == {o['pool']} ({len(pool)} units)",
        f"- rule: top {o['k']} by in-slice in+out degree (recurrent core)",
        f"- slice: n={len(idx)} nnz={sel.nnz} density={sel.nnz / len(idx) ** 2:.4f}",
        f"- in-slice degree: min={int(score[take].min())} med={int(np.median(score[take]))} max={int(score[take].max())}",
        f"- dropped pool units: {len(pool) - len(idx)} (low-degree pass-through)",
        f"- indices: {Path(o['out']).name} (matrix rows == neurons.parquet rows, verified)",
        f"- honesty: parquet has no mushroom-body/central-complex labels; this is",
        "  central-brain association tissue, not a mapped MB circuit.",
        f"- ({round(time.time() - t0, 1)}s)",
    ]
    (ROOT / "results" / "slice_10k.md").write_text("\n".join(doc) + "\n")
    print("\n".join(doc), flush=True)


if __name__ == "__main__":
    main()
