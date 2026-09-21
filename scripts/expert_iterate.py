"""Expert-iteration pilot: distill 2-ply search into linear weights.

Collect turn-boundary positions from teacher self-play; target = expected
2-ply position value (avg over own 21 dice of best-reply V); ridge refit;
validate student vs teacher head-to-head + test positions.

Usage: expert_iterate.py [games=300] [jobs=4] [--w teacher.npz] [--out student.npz] [--lam 1.0]
Success: student beats teacher 1-ply (>55% over 200 games) or diagnose.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from engine import sim
from brain import value as V, search as S
from brain import flybrain as F


def collect(head, w, games, seed0):
    """Post-move samples: (phi(after), 2-ply value) per applied move.
    Matches exactly how S.choose scores candidates (usage-matched)."""
    import random
    Phis, Ts = [], []
    for gi in range(games):
        r = random.Random(seed0 + gi)
        g = sim.Game()
        while True:
            mover = g.turn
            g.roll(r)
            if not g.has_any_legal():
                g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn
                continue
            done, made = False, 0
            while True:
                ms = g.legal_moves()
                if not ms:
                    break
                m = head.choose(g, ms, r)
                g.apply(m)
                made += 1
                Phis.append(head.phi(g.board, g.bar, g.off, g.turn))
                win, wnr = g.check_win()
                if win:
                    Ts.append(1.0 if wnr == 0 else 0.0)
                    done = True
                    break
                Ts.append(S.exp_value(head, w, g.board, g.bar, g.off, mover))
                if not g.moves_left or not g.has_any_legal():
                    break
            if done:
                break
            if made and g.check_technical_win(mover):
                Ts[-1] = 1.0 if mover == 0 else 0.0
                break
            g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn
    return np.array(Phis), np.array(Ts)


def _pos_value(head, w, board, bar, off, turn):
    """Expected V at turn start: avg over own 21 dice of best-reply V."""
    white = turn == 0
    tot = 0.0
    for _, _, prob, ml in S.DICE:
        g = S._clone(board, bar, off, turn, ml)
        replies = g.legal_moves()
        if not replies:
            tot += prob * float(w @ head.phi(board, bar, off, turn))
            continue
        best, first = 0.0, True
        for m in replies:
            c = S._clone(board, bar, off, turn, ml)
            c.apply(m)
            v = float(w @ head.phi(c.board, c.bar, c.off, c.turn))
            if first or (white and v > best + 1e-9) or (not white and v < best - 1e-9):
                best, first = v, False
        tot += prob * best
    return tot


def main():
    import concurrent.futures as cf
    import random
    a = sys.argv[1:]
    games = int(a[0]) if len(a) > 0 else 300
    jobs = int(a[1]) if len(a) > 1 else 4
    wpath = str(ROOT / "brain" / "weights" / "expert.npz")
    out = str(ROOT / "brain" / "weights" / "student.npz")
    lam = 1.0
    i = 2
    while i < len(a):
        if a[i] == "--w":
            wpath = a[i + 1]; i += 2
        elif a[i] == "--out":
            out = a[i + 1]; i += 2
        elif a[i] == "--lam":
            lam = float(a[i + 1]); i += 2
        else:
            i += 1
    W = F.load_connectome()[:512, :512].tocsr()
    teacher = V.Head.load(wpath, W)
    t0 = time.time()
    # parallel collection in shards
    per = max(1, games // jobs)
    Phis, Ts = [], []
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(collect, teacher, teacher.w, per, 7000 + k * per) for k in range(jobs)]
        for f in futs:
            p, t = f.result()
            Phis.append(p)
            Ts.append(t)
    P, T = np.concatenate(Phis), np.concatenate(Ts)
    print(f"collected {len(P)} positions in {round(time.time() - t0)}s", flush=True)
    # ridge toward teacher: (P'P + lam I) w = P' T + lam w_teacher
    # (plain ridge collapses ||w|| and destroys move ordering)
    A = P.T @ P + lam * np.eye(P.shape[1])
    b = P.T @ T + lam * teacher.w
    w = np.linalg.solve(A, b)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(P))
    cut = int(0.8 * len(P))
    Ptr, Ttr, Pte, Tte = P[idx[:cut]], T[idx[:cut]], P[idx[cut:]], T[idx[cut:]]
    A = Ptr.T @ Ptr + lam * np.eye(P.shape[1])
    b = Ptr.T @ Ttr + lam * teacher.w
    w = np.linalg.solve(A, b)
    pred = Pte @ w
    print(f"heldout RMSE {float(np.sqrt(((pred - Tte) ** 2).mean())):.4f} target std {float(Tte.std()):.4f} n={len(P)}", flush=True)
    student = V.Head(W, seed=7, w=w)
    student.save(out)
    print(f"saved -> {out}", flush=True)
    # validate: student vs teacher, 1-ply, 200 games
    sp, tp = student.policy(), teacher.policy()
    wins = 0
    N = 200
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = []
        for s in range(N):
            if s % 2 == 0:
                futs.append(ex.submit(sim.play_game, sp, tp, random.Random(9000 + s)))
            else:
                futs.append(ex.submit(sim.play_game, tp, sp, random.Random(9000 + s)))
        for k, f in enumerate(futs):
            wnr, _, _, _ = f.result()
            wins += ((wnr == 0) == (k % 2 == 0))
    print(f"STUDENT vs TEACHER 1-ply: {wins}/{N} = {100 * wins / N:.1f}%", flush=True)


if __name__ == "__main__":
    main()
