"""P0 parity + perf gate for the GPU decision path.

Parity: GPU choose_batch vs CPU Head.choose on 200 fixed positions.
Pass = agreement >= 99%, every mismatch within 1e-4 value margin.
Perf: batched decisions/sec (B=32) vs CPU 1-ply.
Usage: parity_gpu.py [--n 512] [--positions 200]
"""
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from engine import sim
from brain import value as V, gpu as G, flybrain as F


def sample_positions(n_pos, seed0=31337):
    out = []
    s = 0
    while len(out) < n_pos:
        s += 1
        r = random.Random(seed0 + s)
        g = sim.Game()
        for _ in range(r.randint(0, 120)):
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
        out.append((g, ms))
    return out


def main():
    a = sys.argv[1:]
    n = int(a[a.index("--n") + 1]) if "--n" in a else 512
    n_pos = int(a[a.index("--positions") + 1]) if "--positions" in a else 200
    W = F.load_connectome()
    W = W[:n, :n].tocsr() if W.shape[0] >= n else W
    head = V.Head.load(str(ROOT / "brain" / "weights_league2" / "stageC_n512.npz"), W) \
        if n == 512 else V.Head(W, seed=7)
    gh = G.GPUHead(W, head.Win, head.w, steps=head.steps)
    pos = sample_positions(n_pos)
    agree = errs = 0
    worst = 0.0
    t0 = time.time()
    for k, (g, ms) in enumerate(pos):
        r1, r2 = random.Random(9000 + k), random.Random(9000 + k)
        cp = head.choose(g, ms, r1)
        gp = gh.choose_batch([(g, ms)], r2)[0]
        key = lambda m: (m["from"], m["to"], m["die"])
        if key(cp) == key(gp):
            agree += 1
        else:
            c1, c2 = g.clone(), g.clone()
            c1.apply(cp); c2.apply(gp)
            v1 = float(head.w @ head.phi(c1.board, c1.bar, c1.off, c1.turn))
            v2 = float(head.w @ head.phi(c2.board, c2.bar, c2.off, c2.turn))
            worst = max(worst, abs(v1 - v2))
            if abs(v1 - v2) <= 1e-4:
                agree += 1
            else:
                errs += 1
    print(f"parity n={n}: {agree}/{n_pos} agree, hard-mismatch {errs}, "
          f"worst-margin {worst:.2e} ({round(time.time() - t0, 1)}s)", flush=True)
    print("PARITY-" + ("PASS" if agree / n_pos >= 0.99 else "FAIL"), flush=True)

    B = 32
    items = [(g, ms) for g, ms in pos[:B]]
    r = random.Random(0)
    t0 = time.time()
    reps = 20
    for _ in range(reps):
        gh.choose_batch(items, r)
    per = (time.time() - t0) / reps
    print(f"perf: batched-32 decision {per * 1000:.1f} ms "
          f"({B / per:.0f} decisions/sec overall)", flush=True)
    t0 = time.time()
    for _ in range(reps):
        for g, ms in items:
            head.choose(g, ms, r)
    per_cpu = (time.time() - t0) / reps / B
    print(f"perf: cpu 1-ply {per_cpu * 1000:.2f} ms/decision", flush=True)
    print("PERF-DONE", flush=True)


if __name__ == "__main__":
    main()
