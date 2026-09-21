"""Top-down anatomical brain panel: real soma positions + live firing overlay.

somaLocation xyz (nm) comes from the already-downloaded raw annotations; matrix
row i == neurons.parquet row i (verified). No new downloads. Dots mark cell
bodies (NOT synapses -- arbors need the meshes); pathways = talking pairs.
Top-down axes auto-picked (lateral symmetry x long AP tail); --axes overrides.
Usage: brain_topo.py [--out docs/brain_topo.png] [--seed 7] [--axes 0,2]
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np

XYZ_CACHE = ROOT / "data" / "soma_xyz.npz"


def build_xyz():
    """bodyId -> xyz aligned to matrix order. Cached on disk."""
    if XYZ_CACHE.exists():
        z = np.load(XYZ_CACHE)
        if z["n"] == 191696:
            return z["xyz"], z["valid"]
    import pandas as pd
    raw = pd.read_feather(str(ROOT / "data" / "malecns" / "raw" /
                              "body-annotations-male-cns-v1.0-minconf-0.5.feather"),
                          columns=["bodyId", "somaLocation"])
    tab = {}
    for bid, loc in zip(raw["bodyId"].to_numpy(), raw["somaLocation"]):
        a = np.asarray(loc, dtype=float).ravel()
        if a.shape == (3,) and np.all(np.isfinite(a)):
            tab[int(bid)] = a
    order = pd.read_parquet(str(ROOT / "data" / "malecns" / "neurons.parquet"),
                            columns=["bodyId"])["bodyId"].to_numpy()
    xyz = np.full((len(order), 3), np.nan)
    valid = np.zeros(len(order), bool)
    for i, bid in enumerate(order):
        p = tab.get(int(bid))
        if p is not None:
            xyz[i] = p
            valid[i] = True
    np.savez(XYZ_CACHE, xyz=xyz.astype(np.float32), valid=valid, n=len(order))
    return xyz.astype(np.float32), valid


def pick_axes(xyz, valid):
    """AP = longest range (VNC cord tail); LR = best mirror symmetry; DV = rest."""
    p = xyz[valid]
    rng = p.max(0) - p.min(0)
    ap = int(np.argmax(rng))
    rest = [a for a in range(3) if a != ap]
    scores = {}
    for a in rest:
        h, _ = np.histogram(p[:, a], bins=60)
        h = h / h.sum()
        scores[a] = float(np.abs(h - h[::-1]).sum())
    lr = min(rest, key=lambda a: scores[a])
    dv = [a for a in rest if a != lr][0]
    print(f"axis ranges: {rng.astype(int).tolist()} AP={ap} LR={lr}(sym={scores[lr]:.3f}) "
          f"DV={dv}(sym={scores[dv]:.3f}) coverage={valid.mean() * 100:.1f}%", flush=True)
    return lr, ap


def main():
    a = sys.argv[1:]
    out = str(ROOT / "docs" / "brain_topo.png")
    seed = 7
    axes = None
    i = 0
    while i < len(a):
        if a[i] == "--out":
            out = a[i + 1]; i += 2
        elif a[i] == "--seed":
            seed = int(a[i + 1]); i += 2
        elif a[i] == "--axes":
            axes = tuple(int(v) for v in a[i + 1].split(",")); i += 2
        else:
            i += 1
    t0 = time.time()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xyz, valid = build_xyz()
    lr, ap = axes or pick_axes(xyz, valid)
    from brain_viz import sample_position
    head, W, g, m, phi = sample_position(seed)
    n = head.n
    x = phi[:n]
    fire = np.abs(x)
    pos = xyz[:n]
    ok = valid[:n]
    print(f"slice units with positions: {ok.sum()}/{n}", flush=True)

    X, Y = xyz[valid][:, lr], xyz[valid][:, ap]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.set_facecolor("black")
    h, xe, ye = np.histogram2d(X, Y, bins=220)
    ax1.imshow(np.log1p(h.T), origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
               cmap="Greys_r", aspect="equal", alpha=0.95)
    # used pathways among positioned slice units
    import scipy.sparse as sp
    A = abs(W[:n, :n]).astype(float).tocsr()
    top_units = [u for u in np.argsort(fire)[-40:] if ok[u]]
    edges = []
    for i in top_units:
        row = A.getrow(i)
        for k in range(row.nnz):
            j = row.indices[k]
            if ok[j]:
                edges.append((abs(row.data[k] * x[j]), i, j))
    edges.sort(reverse=True)
    edges = edges[:300]
    for _, i, j in edges:
        ax1.plot([pos[i, lr], pos[j, lr]], [pos[i, ap], pos[j, ap]],
                 "r-", lw=0.5, alpha=0.5, zorder=3)
    sc = ax1.scatter(pos[ok, lr], pos[ok, ap], c=fire[ok], s=16, cmap="hot",
                     vmin=0, vmax=np.percentile(fire[ok], 99), zorder=4,
                     edgecolors="none")
    plt.colorbar(sc, ax=ax1, label="|KC firing|")
    ax1.set_title(f"top-down soma map: {n} reservoir cells at true positions")
    ax1.set_xticks([]); ax1.set_yticks([])
    w = head.w
    top5 = [u for u in np.argsort(fire)[-5:][::-1] if ok[u]]
    ax2.axis("off")
    _v = float(w @ phi)
    ax2.text(0.5, 0.65, f"V = {_v:.3f}", ha="center", va="center", fontsize=30, color="teal")
    ax2.text(0.5, 0.45, f"move {m['from']}->{m['to']}  die {m['die']}", ha="center", va="center", fontsize=14)
    ax2.text(0.5, 0.25, "top KCs: " + ", ".join(f"{u}({fire[u]:.2f})" for u in top5), ha="center", va="center", fontsize=10)
    fig.suptitle("fly brain top-down — real soma positions, live firing")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    print(f"wrote {out} in {round(time.time() - t0, 1)}s", flush=True)
    print("top KCs: " + ", ".join(f"{u}({fire[u]:.2f})" for u in top5), flush=True)


if __name__ == "__main__":
    main()
