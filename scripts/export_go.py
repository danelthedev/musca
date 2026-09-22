"""Export reservoir + weight map + test vectors for fair_backgammon/fly/.

Usage: export_go.py [--n 512] [--w brain/weights/best.npz] [--out dir] [--vectors 300] [--em-vectors 100]
fair_backgammon embeds fly.json (shared W/Win + per-variant w) and replays
vectors.json in its Go test. Re-run after retraining; models are ~12KB each.
"""
import json
import random
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import sim
from brain import search as S
from brain import value as V
from brain import flybrain as F


def parse(a):
    o = dict(n=512, w=str(ROOT / "brain" / "weights" / "best.npz"), expert=None,
             mlp_w=str(ROOT / "brain" / "weights_mlp" / "distilled.npz"),
             out="/var/home/danel/Projects/fair_backgammon/fly", vectors=300, em_vectors=100,
             name="lobotomized", mlp_vectors=100)
    i = 0
    while i < len(a):
        if a[i] in ("--n", "--vectors", "--em-vectors") and i + 1 < len(a):
            o[a[i][2:].replace("-", "_")] = int(a[i + 1]); i += 2
        elif a[i] in ("--w", "--out", "--expert", "--mlp-w", "--name", "--slice") and i + 1 < len(a):
            o[a[i][2:].replace("-", "_")] = a[i + 1]; i += 2
        elif a[i] == "--untrained":
            o["untrained"] = True; i += 1
        else:
            raise SystemExit(f"usage: export_go.py [--n N] [--w path] [--out dir] [--vectors K], got {a[i]}")
    return o


def main():
    o = parse(sys.argv[1:])
    W = F.load_connectome()
    if o.get("slice"):
        idx = np.load(o["slice"])
        W = W[idx][:, idx].tocsr()
    n = W.shape[0] if o.get("slice") else o["n"]
    W = W[:n, :n].tocsr() if W.shape[0] >= n else W
    prior = V.Head(W, seed=7)
    trained = prior if o.get("untrained") else V.Head.load(o["w"], W)
    assert len(trained.w) == W.shape[0] + prior.dim + 1, "weights dim mismatch"
    expert = V.Head.load(o["expert"], W) if o["expert"] else trained
    assert len(expert.w) == W.shape[0] + prior.dim + 1, "expert dim mismatch"
    out = Path(o["out"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "testdata").mkdir(exist_ok=True)
    # variants map: each variant owns its full reservoir (n may differ).
    variants = {}
    if (out / "fly.json").exists():
        old = json.loads((out / "fly.json").read_text())
        if "variants" in old:
            variants = old["variants"]
        elif old.get("n") is not None and old.get("weights"):
            # legacy flat format: one reservoir shared by every weight key
            base = {k: v for k, v in old.items() if k != "weights"}
            for name in old["weights"]:
                variants[name] = {**base, "weights": {name: old["weights"][name]}}
    blob = {"n": W.shape[0], "dim": prior.dim, "steps": prior.steps,
            "indptr": W.indptr.tolist(), "indices": W.indices.tolist(),
            "data": [float(v) for v in W.data],
            "win": prior.Win.tolist(),
            "weights": {o["name"]: [float(v) for v in trained.w]}}
    variants[o["name"]] = blob
    (out / "fly.json").write_text(json.dumps({"variants": variants}))
    print(f"fly.json: {len(json.dumps(variants)) // 1024}KB variants={sorted(variants)} nnz={W.nnz}")

    # vectors: sampled positions, trained pick, near-ties excluded
    vecs, s, rng = [], 0, random.Random(7)
    while len(vecs) < o["vectors"] and s < o["vectors"] * 50:
        s += 1
        g = sim.Game()
        r = random.Random(1000 + s)
        for _ in range(rng.randint(0, 120)):  # random prefix
            g.roll(r)
            if not g.has_any_legal():
                g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn
                continue
            ms = g.legal_moves()
            if not ms:
                break
            g.apply(r.choice(ms))
            if g.check_win()[0] or not g.moves_left or not g.has_any_legal():
                g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn
        g.roll(r)
        if not g.has_any_legal():
            continue
        ms = g.legal_moves()
        if len(ms) < 2:
            continue
        vals = []
        for m in ms:
            c = g.clone()
            c.apply(m)
            vals.append(float(trained.w @ trained.phi(c.board, c.bar, c.off, c.turn)))
        order = sorted(range(len(ms)), key=lambda i: vals[i], reverse=g.turn == 0)
        if abs(vals[order[0]] - vals[order[1]]) <= 1e-9:
            continue  # tie: Go coin-flip may differ, skip
        pick = ms[order[0]]
        assert pick == trained.choose(g, ms, random.Random(s)), "choose mismatch"
        vecs.append({"board": g.board, "bar": g.bar, "off": g.off, "turn": g.turn,
                     "moves": [[m["from"], m["to"], m["die"]] for m in ms],
                     "pick": [pick["from"], pick["to"], pick["die"]]})
    if o["vectors"]:
        (out / "testdata" / "vectors.json").write_text(json.dumps(vecs))
    print(f"vectors: {len(vecs)} (scanned {s})")

    # (pure export: no search/MLP vectors; arena decides 1-ply linear only)

    # mlp.json + mlp vectors (1-ply picks, near-ties excluded)
    # (pure export: no mlp.json; arena is linear readout only)


if __name__ == "__main__":
    main()
