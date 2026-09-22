"""Export live brain-panel assets for fair_backgammon/web/public/.

- brain_coords.json: top-down soma (x, y in 0..1) for the arena slice rows,
  plus extent. Schematic positions from connectivity are NOT used here --
  these are real MaleCNS soma coordinates (data/soma_xyz.npz).
- brain_bg.png: dark 2D-histogram silhouette of all positioned somas.
Usage: export_brain_assets.py [--n 512] [--out dir] [--axes 0,2]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np


def parse(a):
    o = dict(n=512, out="/var/home/danel/Projects/fair_backgammon/web/public",
             axes="0,2", slice=None)
    i = 0
    while i < len(a):
        if a[i].startswith("--") and i + 1 < len(a):
            k = a[i][2:]
            o[k] = a[i + 1] if k in ("out", "axes", "slice") else int(a[i + 1])
            i += 2
        else:
            i += 1
    return o


def main():
    import json
    o = parse(sys.argv[1:])
    lr, ap = (int(v) for v in o["axes"].split(","))
    z = np.load(str(ROOT / "data" / "soma_xyz.npz"))
    xyz, valid = z["xyz"], z["valid"]
    n = o["n"]
    rows = np.load(o["slice"]) if o.get("slice") else np.arange(n)
    pos = xyz[rows]
    ok = valid[rows]
    X, Y = xyz[valid][:, lr], xyz[valid][:, ap]
    lox, hix = float(X.min()), float(X.max())
    loy, hiy = float(Y.min()), float(Y.max())
    nx = lambda v: (v - lox) / (hix - lox)
    ny = lambda v: (v - loy) / (hiy - loy)
    coords = [[round(float(nx(p[lr])), 4) if o else None,
               round(float(ny(p[ap])), 4) if o else None] for p, o in zip(pos, ok)]
    out = Path(o["out"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "brain_coords.json").write_text(json.dumps({
        "n": n, "coords": coords,
        "extent": [lox, hix, loy, hiy], "axes": [lr, ap]}))
    print(f"coords: {n} units, positioned {int(ok.sum())}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4, 4), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    h, xe, ye = np.histogram2d(X, Y, bins=200)
    ax.imshow(np.log1p(h.T), origin="lower",
              extent=[nx(xe[0]), nx(xe[-1]), ny(ye[0]), ny(ye[-1])],
              cmap="Greys_r", aspect="equal")
    ax.set_axis_off()
    fig.savefig(str(out / "brain_bg.png"), facecolor="black")
    print(f"bg: {out / 'brain_bg.png'}", flush=True)


if __name__ == "__main__":
    main()
