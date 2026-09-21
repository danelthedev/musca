"""Distill 2-ply targets into the MLP (supervised + Adam), then validate vs teacher.

Usage: distill_mlp.py [positions=8000] [jobs=4] [--teacher weights] [--w0 mlp.npz]
                      [--out mlp_dist.npz] [--epochs 20] [--lr 1e-3] [--seed 1]
Gate: distilled MLP beats linear teacher head-to-head.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from engine import sim
from brain import value as V, search as S, mlp as M
from brain import flybrain as F


def collect_q(head, w, n_positions, seed0, jobs=4):
    import concurrent.futures as cf
    import random

    def worker(k):
        r = random.Random(seed0 + k * 7919)
        Ps, Ts = [], []
        games = 0
        while len(Ps) < n_positions // jobs + 1:
            games += 1
            if games > 4000:
                break
            g = sim.Game()
            while len(Ps) < n_positions // jobs + 1:
                mover = g.turn
                g.roll(r)
                if not g.has_any_legal():
                    g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn
                    continue
                inner = 0
                while True:
                    ms = g.legal_moves()
                    if not ms:
                        break
                    m = head.choose(g, ms, r)
                    g.apply(m)
                    inner += 1
                    if inner > 2000:
                        break
                    Ps.append(head.phi(g.board, g.bar, g.off, g.turn))
                    win, wnr = g.check_win()
                    if win:
                        Ts.append(1.0 if wnr == 0 else 0.0)
                        break
                    Ts.append(S.exp_value(head, w, g.board, g.bar, g.off, mover))
                    if not g.moves_left or not g.has_any_legal():
                        break
                if win if 'win' in dir() else False:
                    pass
                if g.check_win()[0] or (g.check_technical_win(0) or g.check_technical_win(1)):
                    break
                g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn
        return np.array(Ps), np.array(Ts)

    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        parts = list(ex.map(worker, range(jobs)))
    P = np.concatenate([p[0] for p in parts])[:n_positions]
    T = np.concatenate([p[1] for p in parts])[:n_positions]
    return P, T


def train_adam(net, P, T, epochs, lr, seed=1):
    rng = np.random.default_rng(seed)
    mW1 = np.zeros_like(net.W1)
    vW1 = np.zeros_like(net.W1)
    mb1 = np.zeros_like(net.b1)
    vb1 = np.zeros_like(net.b1)
    mW2 = np.zeros_like(net.W2)
    vW2 = np.zeros_like(net.W2)
    mb2 = 0.0
    vb2 = 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    n = len(P)
    bs = 256
    for ep in range(epochs):
        idx = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, bs):
            batch = idx[s:s + bs]
            gW1 = np.zeros_like(net.W1)
            gb1 = np.zeros_like(net.b1)
            gW2 = np.zeros_like(net.W2)
            gb2 = 0.0
            for i in batch:
                v, cache = net.forward(P[i])
                err = v - T[i]
                tot += err * err
                dW1, dB1, dW2, dB2 = net.grad(cache)
                gW1 += err * dW1
                gb1 += err * dB1
                gW2 += err * dW2
                gb2 += err * dB2
            k = len(batch)
            gW1 /= k
            gb1 /= k
            gW2 /= k
            gb2 /= k
            step += 1
            for g, m, v_, p in ((gW1, mW1, vW1, "W1"), (gb1, mb1, vb1, "b1"),
                                (gW2, mW2, vW2, "W2")):
                m[:] = b1 * m + (1 - b1) * g
                v_[:] = b2 * v_ + (1 - b2) * (g * g)
                upd = m / (1 - b1 ** step) / (np.sqrt(v_ / (1 - b2 ** step)) + eps)
                if p == "W1":
                    net.W1 -= lr * upd
                elif p == "b1":
                    net.b1 -= lr * upd
                else:
                    net.W2 -= lr * upd
            mb2 = b1 * mb2 + (1 - b1) * gb2
            vb2 = b2 * vb2 + (1 - b2) * (gb2 * gb2)
            net.b2 -= lr * (mb2 / (1 - b1 ** step)) / (np.sqrt(vb2 / (1 - b2 ** step)) + eps)
        print(f"  epoch {ep + 1}/{epochs} mse={tot / n:.5f}", flush=True)
    return net


def main():
    a = sys.argv[1:]
    npos = int(a[0]) if len(a) > 0 else 8000
    jobs = int(a[1]) if len(a) > 1 else 4
    teacher_p = str(ROOT / "brain" / "weights" / "expert.npz")
    w0 = str(ROOT / "brain" / "weights_mlp" / "stageC_n512.npz")
    out = str(ROOT / "brain" / "weights_mlp" / "distilled.npz")
    epochs, lr, seed = 20, 1e-3, 1
    i = 2
    while i < len(a):
        if a[i] == "--teacher":
            teacher_p = a[i + 1]; i += 2
        elif a[i] == "--w0":
            w0 = a[i + 1]; i += 2
        elif a[i] == "--out":
            out = a[i + 1]; i += 2
        elif a[i] == "--epochs":
            epochs = int(a[i + 1]); i += 2
        elif a[i] == "--lr":
            lr = float(a[i + 1]); i += 2
        elif a[i] == "--seed":
            seed = int(a[i + 1]); i += 2
        else:
            i += 1
    W = F.load_connectome()[:512, :512].tocsr()
    head = V.Head(W, seed=7)
    teacher = V.Head.load(teacher_p, W)
    t0 = time.time()
    print("collecting positions...", flush=True)
    P, T = collect_q(teacher, teacher.w, npos, seed, jobs)
    print(f"collected {len(P)} in {round(time.time() - t0)}s", flush=True)
    net = M.Net.load(w0, head)
    train_adam(net, P, T, epochs, lr, seed)
    net.save(out)
    print(f"saved -> {out} ({round(time.time() - t0)}s total)", flush=True)
    # validate vs teacher
    import concurrent.futures as cf
    import random
    sp, tp = net.policy(), teacher.policy()

    def run(s):
        r = random.Random(55000 + s)
        if s % 2 == 0:
            wnr, _, _, _ = sim.play_game(sp, tp, r, max_plies=20000)
            return (wnr == 0)
        wnr, _, _, _ = sim.play_game(tp, sp, r, max_plies=20000)
        return (wnr == 1)

    N = 200
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        res = list(ex.map(run, range(N)))
    print(f"DISTILLED-MLP vs TEACHER 1-ply: {sum(res)}/{N} = {100 * sum(res) / N:.1f}%", flush=True)


if __name__ == "__main__":
    main()
