"""Export reservoir + weight map + test vectors for fair_backgammon/fly/.

Usage: export_go.py [--n 512] [--w brain/weights/best.npz] [--out dir] [--vectors 300] [--em-vectors 100]
fair_backgammon embeds fly.json (shared W/Win + per-variant w) and replays
vectors.json in its Go test. Re-run after retraining; models are ~12KB each.
"""
import json
import random
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
             out="/var/home/danel/Projects/fair_backgammon/fly", vectors=300, em_vectors=100)
    i = 0
    while i < len(a):
        if a[i] in ("--n", "--vectors", "--em-vectors") and i + 1 < len(a):
            o[a[i][2:].replace("-", "_")] = int(a[i + 1]); i += 2
        elif a[i] in ("--w", "--out", "--expert") and i + 1 < len(a):
            o[a[i][2:]] = a[i + 1]; i += 2
        else:
            raise SystemExit(f"usage: export_go.py [--n N] [--w path] [--out dir] [--vectors K], got {a[i]}")
    return o


def main():
    o = parse(sys.argv[1:])
    n = o["n"]
    W = F.load_connectome()
    W = W[:n, :n].tocsr() if W.shape[0] >= n else W
    prior = V.Head(W, seed=7)
    trained = V.Head.load(o["w"], W)
    assert len(trained.w) == W.shape[0] + prior.dim + 1, "weights dim mismatch"
    expert = V.Head.load(o["expert"], W) if o["expert"] else trained
    assert len(expert.w) == W.shape[0] + prior.dim + 1, "expert dim mismatch"
    out = Path(o["out"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "testdata").mkdir(exist_ok=True)
    blob = {"n": W.shape[0], "dim": prior.dim, "steps": prior.steps,
            "indptr": W.indptr.tolist(), "indices": W.indices.tolist(),
            "data": [float(v) for v in W.data],
            "win": prior.Win.tolist(),
            "weights": {"untrained": [float(v) for v in prior.w],
                        "trained": [float(v) for v in trained.w],
                        "expert": [float(v) for v in expert.w]}}
    (out / "fly.json").write_text(json.dumps(blob))
    print(f"fly.json: {len(json.dumps(blob)) // 1024}KB (nnz={W.nnz})")

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
    (out / "testdata" / "vectors.json").write_text(json.dumps(vecs))
    print(f"vectors: {len(vecs)} (scanned {s})")

    # em vectors: 2-ply picks, near-ties excluded (Go must agree exactly)
    em, s2 = [], 0
    while len(em) < o["em_vectors"] and s2 < o["em_vectors"] * 60:
        s2 += 1
        g = sim.Game()
        r = random.Random(777000 + s2)
        for _ in range(rng.randint(0, 120)):
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
            vals.append(S.exp_value(expert, expert.w, c.board, c.bar, c.off, g.turn))
        order = sorted(range(len(ms)), key=lambda i: vals[i], reverse=g.turn == 0)
        if abs(vals[order[0]] - vals[order[1]]) <= 1e-9:
            continue
        pick = ms[order[0]]
        assert pick == S.choose(expert, expert.w, g, ms, random.Random(s2)), "em mismatch"
        em.append({"board": g.board, "bar": g.bar, "off": g.off, "turn": g.turn,
                     "moves": [[m["from"], m["to"], m["die"]] for m in ms],
                     "pick": [pick["from"], pick["to"], pick["die"]]})
    (out / "testdata" / "em_vectors.json").write_text(json.dumps(em))
    print(f"em vectors: {len(em)} (scanned {s2})")


if __name__ == "__main__":
    main()
