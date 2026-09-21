"""B1 distillation: 2-ply teacher targets -> big linear student (any n/slice).

Teacher (small, e.g. league2-512) self-play positions; target = teacher 2-ply
exp value; streaming float32 ridge (P'P and P'T accumulated per game, full P
never formed); prior = sane Head init (teacher w is the wrong dim).
Validates student vs teacher 1-ply head-to-head (gate >= 53%).
Usage: distill_student.py [--teacher w] [--teacher-n 512]
                [--slice data/slice_10k_idx.npy | --n 512]
                [--games 2500] [--jobs 4] [--lam 1.0] [--seed 7000]
                [--out brain/weights_10k/student10k.npz] [--eval-games 200]
"""
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from engine import sim
from brain import value as V, search as S
from brain import flybrain as F


def parse(a):
    o = dict(teacher=str(ROOT / "brain" / "weights_league2" / "stageC_n512.npz"),
             teacher_n=512, slice=str(ROOT / "data" / "slice_10k_idx.npy"), n=512,
             games=2500, jobs=4, lam=1.0, seed=7000,
             out=str(ROOT / "brain" / "weights_10k" / "student10k.npz"),
             eval_games=200)
    i = 0
    while i < len(a):
        k = a[i]
        if k.startswith("--"):
            k = k[2:].replace("-", "_")
            if k not in o:
                raise SystemExit(f"unknown flag {a[i]}")
            v = a[i + 1]
            o[k] = v if isinstance(o[k], str) or o[k] is None else type(o[k])(v)
            i += 2
        else:
            raise SystemExit(f"unexpected arg {a[i]}")
    return o


def collectShard(teacher, games, seed0):
    """Teacher self-play; per post-move state: (state-tuple, 2-ply target)."""
    out = []
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
                m = teacher.choose(g, ms, r)
                g.apply(m)
                made += 1
                st = (list(g.board), list(g.bar), list(g.off), g.turn)
                win, wnr = g.check_win()
                if win:
                    out.append((st, 1.0 if wnr == 0 else 0.0))
                    done = True
                    break
                out.append((st, S.exp_value(teacher, teacher.w, g.board,
                                           g.bar, g.off, mover)))
                if not g.moves_left or not g.has_any_legal():
                    break
            if done:
                break
            if made and g.check_technical_win(mover):
                s, t = out[-1]
                out[-1] = (s, 1.0 if mover == 0 else 0.0)
                break
            g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn
    return out


def main():
    o = parse(sys.argv[1:])
    t0 = time.time()
    Wt = F.load_connectome()
    Wt = Wt[:o["teacher_n"], :o["teacher_n"]].tocsr()
    teacher = V.Head.load(o["teacher"], Wt)
    W = F.load_connectome()
    if o["slice"] and Path(o["slice"]).exists():
        idx = np.load(o["slice"])
        W = W[idx][:, idx].tocsr()
    else:
        W = W[:o["n"], :o["n"]].tocsr()
    student = V.Head(W, seed=7)
    d = len(student.w)
    print(f"distill: teacher n={teacher.n} student n={student.n} dim={d} "
          f"games={o['games']} jobs={o['jobs']} lam={o['lam']}", flush=True)
    A = np.zeros((d, d), dtype=np.float32)
    b = np.zeros(d, dtype=np.float32)
    per = max(1, o["games"] // o["jobs"])
    npos = 0
    with ThreadPoolExecutor(max_workers=o["jobs"]) as ex:
        futs = [ex.submit(collectShard, teacher, per, o["seed"] + k * per)
                for k in range(o["jobs"])]
        for f in futs:
            rows = f.result()
            P = np.empty((len(rows), d), dtype=np.float32)
            T = np.empty(len(rows), dtype=np.float32)
            for j, ((bo, ba, of, t), tv) in enumerate(rows):
                P[j] = student.phi(bo, ba, of, t)
                T[j] = tv
            A += P.T @ P
            b += P.T @ T
            npos += len(rows)
            print(f"  accumulated {npos} positions ({round(time.time() - t0)}s)", flush=True)
    print(f"collected {npos} positions in {round(time.time() - t0)}s", flush=True)
    M = A.astype(np.float64)
    M.flat[::d + 1] += o["lam"]
    rhs = b.astype(np.float64) + o["lam"] * student.w
    w = np.linalg.solve(M, rhs)
    pred_resid = float(np.mean((b - A @ w.astype(np.float32)) ** 2))
    print(f"solve done, mean train resid^2={pred_resid:.5f}", flush=True)
    student.w = w
    out = Path(o["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    student.save(out)
    print(f"saved -> {out}", flush=True)
    # validate: student vs teacher 1-ply
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except ImportError:
        use_gpu = False
    if use_gpu:
        from brain import gpu as Gmod
        gh = Gmod.GPUHead(W, student.Win, student.w, steps=student.steps)
        sp = lambda g_, m_, r_: gh.choose_batch([(g_, m_)], r_)[0]
    else:
        sp = student.policy()
    tp = teacher.policy()
    wins = 0
    N = o["eval_games"]
    with ThreadPoolExecutor(max_workers=o["jobs"]) as ex:
        futs = []
        for s in range(N):
            if s % 2 == 0:
                futs.append(ex.submit(sim.play_game, sp, tp, random.Random(9000 + s)))
            else:
                futs.append(ex.submit(sim.play_game, tp, sp, random.Random(9000 + s)))
        for k, f in enumerate(futs):
            wnr, _, _, _ = f.result()
            wins += ((wnr == 0) == (k % 2 == 0))
    print(f"STUDENT-10k vs TEACHER 1-ply: {wins}/{N} = {100 * wins / N:.1f}%", flush=True)
    print("DISTILL-DONE", flush=True)


if __name__ == "__main__":
    main()
