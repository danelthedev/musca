"""Step-1 throughput benchmark: sizes the whole-brain program.

Measures: (A) full-brain reservoir step, scipy-CPU vs torch-GPU;
(B) play_game profile: rules-only vs 1-ply (sim vs brain split);
(C) phi cost scaling across reservoir sizes.
Usage: bench_throughput.py [--reps 20] [--games 12] [--phis 200]
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from brain import flybrain as F, value as V, features as FT
from engine import sim


def parse(a):
    o = dict(reps=20, games=12, phis=200)
    i = 0
    while i < len(a):
        if a[i].startswith("--") and i + 1 < len(a):
            o[a[i][2:]] = int(a[i + 1]); i += 2
        else:
            i += 1
    return o


def main():
    o = parse(sys.argv[1:])
    import random
    print("== load connectome ==", flush=True)
    t0 = time.time()
    W = F.load_connectome().tocsr()
    n_full = W.shape[0]
    print(f"full W: {W.shape} nnz={W.nnz} ({round(time.time() - t0, 1)}s)", flush=True)
    dim = len(FT.encode_state([0] * 24, [0, 0], [0, 0], 0))
    rng = np.random.default_rng(0)
    u = rng.standard_normal(dim)
    xf = rng.standard_normal(n_full) * 0.1

    print("== A: reservoir step, scipy CPU ==", flush=True)
    Win_full = V.win_mat(n_full, dim, 7)
    t0 = time.time()
    for _ in range(o["reps"]):
        for _ in range(2):  # steps=2 like Head
            xf = np.tanh(W @ xf + Win_full @ u)
    t_cpu = (time.time() - t0) / o["reps"]
    print(f"scipy full-brain step x2: {t_cpu * 1000:.1f} ms/game-ply", flush=True)

    print("== A: reservoir step, torch ==", flush=True)
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"torch {torch.__version__} device={dev}", flush=True)
        if dev == "cuda":
            print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)
        crow = torch.from_numpy(W.indptr).to(dev)
        col = torch.from_numpy(W.indices).to(dev)
        val = torch.from_numpy(W.data).float().to(dev)
        Wsp = torch.sparse_csr_tensor(crow, col, val, size=W.shape, device=dev)
        Win_t = torch.from_numpy(np.ascontiguousarray(Win_full)).float().to(dev)
        for bs in (1, 32):
            xb = torch.from_numpy(np.ascontiguousarray(
                rng.standard_normal((n_full, bs)) * 0.1)).float().to(dev)
            ub = torch.from_numpy(np.ascontiguousarray(
                np.tile(u[:, None], (1, bs)))).float().to(dev)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(o["reps"]):
                for _ in range(2):
                    xb = torch.tanh(torch.matmul(Wsp, xb) + torch.matmul(Win_t, ub))
            if dev == "cuda":
                torch.cuda.synchronize()
            print(f"torch-{dev} batch={bs}: {(time.time() - t0) / o['reps'] * 1000:.1f} ms/step-x2", flush=True)
    except ImportError:
        print("torch NOT installed", flush=True)

    print("== B: play_game profile ==", flush=True)
    W512 = W[:512, :512].tocsr()
    h512 = V.Head(W512, seed=7)
    fly = h512.policy()
    t0 = time.time()
    for s in range(o["games"]):
        sim.play_game(sim.random_policy, sim.random_policy, random.Random(s))
    t_rules = (time.time() - t0) / o["games"]
    print(f"random-vs-random (rules only): {t_rules:.2f} s/game", flush=True)
    t0 = time.time()
    for s in range(o["games"]):
        sim.play_game(fly, fly, random.Random(100 + s))
    t_full = (time.time() - t0) / o["games"]
    print(f"1-ply-512 self-play: {t_full:.2f} s/game (brain ~{t_full - t_rules:.2f}s)", flush=True)
    print(f"==> ~{3600 * 24 / t_full:.0f} games/day/core at n=512", flush=True)

    print("== C: phi scaling ==", flush=True)
    for n in (512, 2048, 10000):
        Wn = W[:n, :n].tocsr()
        hn = V.Head(Wn, seed=7)
        b, bar, off = [0] * 24, [0, 0], [0, 0]
        t0 = time.time()
        for _ in range(o["phis"]):
            hn.phi(b, bar, off, 0)
        ms = (time.time() - t0) / o["phis"] * 1000
        print(f"phi n={n}: {ms:.2f} ms/call (nnz={Wn.nnz})", flush=True)
    print("BENCH-DONE", flush=True)


if __name__ == "__main__":
    main()
