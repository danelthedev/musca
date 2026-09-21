"""Brain activity visualization: light up used pathways. Pure observer.

Layout: spectral embedding of the frozen reservoir (|W| symmetrized) computed
once and cached — schematic positions from connectivity, not anatomy (MaleCNS
annotations carry no soma coordinates). Overlay: Kenyon-cell firing |x| for a
real game position + top-contribution edges + MBON readout weights.

Usage: brain_viz.py [--out docs/brain_activity.png] [--seed 7]
Writes PNG + prints top-firing units and chosen move.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from engine import sim
from brain import value as V, flybrain as F

LAYOUT = ROOT / "data" / "brain_layout.npz"


def layout(W):
    """Spectral embedding from symmetrized |W|. Cached on disk."""
    if LAYOUT.exists():
        z = np.load(LAYOUT)
        if z["n"] == W.shape[0]:
            return z["xy"]
    A = abs(W).astype(float)
    A = A + A.T
    d = np.asarray(A.sum(1)).ravel()
    d[d == 0] = 1.0
    Dm = 1.0 / np.sqrt(d)
    Ls = np.eye(W.shape[0]) - (A.toarray() * Dm[:, None] * Dm[None, :])
    vals, vecs = np.linalg.eigh(Ls)
    xy = vecs[:, 1:3]
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0) + 1e-12)
    np.savez(LAYOUT, xy=xy, n=W.shape[0])
    return xy


def sample_position(seed=7):
    import random
    g = sim.Game()
    r = random.Random(seed)
    head_rng = random.Random(seed + 1)
    W = F.load_connectome()[:512, :512].tocsr()
    head = V.Head.load(str(ROOT / "brain" / "weights_league2" / "stageC_n512.npz"), W)
    for _ in range(40):
        g.roll(r)
        if not g.has_any_legal():
            g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn
            continue
        ms = g.legal_moves()
        if not ms:
            break
        g.apply(head.choose(g, ms, head_rng))
        if g.check_win()[0] or not g.moves_left or not g.has_any_legal():
            g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn
    g.roll(r)
    ms = g.legal_moves()
    m = head.choose(g, ms, head_rng)
    c = g.clone()
    c.apply(m)
    phi = head.phi(c.board, c.bar, c.off, c.turn)
    return head, W, g, m, phi


def main():
    a = sys.argv[1:]
    out = str(ROOT / "docs" / "brain_activity.png")
    seed = 7
    i = 0
    while i < len(a):
        if a[i] == "--out":
            out = a[i + 1]; i += 2
        elif a[i] == "--seed":
            seed = int(a[i + 1]); i += 2
        else:
            i += 1
    t0 = time.time()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    head, W, g, m, phi = sample_position(seed)
    n = head.n
    x = phi[:n]
    fire = np.abs(x)
    xy = layout(W)
    # top-contribution edges: |W_ij * x_j| into most active units
    top_units = np.argsort(fire)[-40:]
    A = abs(W).astype(float)
    edges = []
    for i in top_units:
        row = A.getrow(i)
        for k in range(row.nnz):
            j = row.indices[k]
            edges.append((abs(row.data[k] * x[j]), i, j))
    edges.sort(reverse=True)
    edges = edges[:300]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.scatter(xy[:, 0], xy[:, 1], c="0.85", s=8, zorder=1)
    for _, i, j in edges:
        ax1.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], "r-", lw=0.4, alpha=0.5, zorder=2)
    sc = ax1.scatter(xy[:, 0], xy[:, 1], c=fire, s=14, cmap="hot", vmin=0,
                     vmax=np.percentile(fire, 99), zorder=3)
    plt.colorbar(sc, ax=ax1, label="|KC firing|")
    ax1.set_title(f"reservoir: {n} Kenyon cells, red = used pathways")
    ax1.set_xticks([]); ax1.set_yticks([])
    w = head.w
    ax2.barh(["MBON w·phi"], [float(w @ phi)], color="teal")
    ax2.set_title(f"V={float(w @ phi):.3f} move {m['from']}->{m['to']} die {m['die']}")
    top5 = np.argsort(fire)[-5:][::-1]
    ax2.text(0.02, 0.5, "top KCs: " + ", ".join(f"{u}({fire[u]:.2f})" for u in top5),
             transform=ax2.transAxes, fontsize=9, va="center")
    fig.suptitle(f"fly brain activity — league2 1-ply, turn {'white' if g.turn == 0 else 'black'}")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    print(f"wrote {out} in {round(time.time() - t0, 1)}s")
    print("top KCs: " + ", ".join(f"{u}({fire[u]:.2f})" for u in top5))


if __name__ == "__main__":
    main()
