"""Expert-iteration lap: batched-GPU 2-ply teacher for the 10k head.

Probe: sample turn-boundary positions from teacher self-play, count how often
the 1-ply pick differs from the 2-ply pick -> disagreement rate. Predicts lap
value before paying for it.

Lap: same sampler, 2-ply targets for the APPLIED moves, ridge toward teacher,
student saved for h2h verify + TD resume.

Usage:
  probe: expert_lap.py --w W.npz --slice data/slice_10k_idx.npy --probe 400
  lap:   expert_lap.py --w W.npz --slice data/slice_10k_idx.npy \
             --sample 400 --out student.npz --lam 1.0
"""
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np

from engine import sim
from brain import flybrain as F
from brain import features as FT
from brain import value as V
from brain import search as S


def load_head(wpath, rows):
    W = F.load_connectome()[rows][:, rows].tocsr()
    prior = V.Head(W, seed=7)
    head = V.Head.load(wpath, W)
    assert len(head.w) == W.shape[0] + prior.dim + 1, "weights dim mismatch"
    return head, W


def sample_positions(head, games, seed0):
    """Usage-matched for the LAP: post-move states of actual teacher play
    + mover id. Decisions are re-derived by applying the played move."""
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
            done = False
            while True:
                ms = g.legal_moves()
                if not ms:
                    break
                m = head.choose(g, ms, r)
                g.apply(m)
                out.append((list(g.board), list(g.bar), list(g.off), mover))
                win, wnr = g.check_win()
                if win:
                    done = True
                    break
                if not g.moves_left or not g.has_any_legal():
                    break
            if done:
                break
            if g.check_technical_win(mover):
                break
            g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn
    return out

def sample_decisions(head, games, seed0):
    """Decision points for the probe: (board, bar, off, mover, moves_left)
    exactly at a roll, so legal moves are re-derivable."""
    out = []
    for gi in range(games):
        r = random.Random(seed0 + gi)
        g = sim.Game()
        while True:
            g.roll(r)
            if not g.has_any_legal():
                g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn
                continue
            while True:
                ms = g.legal_moves()
                if not ms:
                    break
                out.append((list(g.board), list(g.bar), list(g.off), g.turn,
                            list(g.moves_left)))
                g.apply(head.choose(g, ms, r))
                win, wnr = g.check_win()
                if win:
                    break
                if not g.moves_left or not g.has_any_legal():
                    break
            if g.check_win()[0]:
                break
            if g.check_technical_win(g.turn):
                break
            g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn
    return out


def two_ply_values(states, gh, chunk=20):
    """Expected V over opp's 21 dice of opp's best 1-ply reply, per state.
    Mirrors S.exp_value. One batched GPU pass per chunk."""
    out = np.zeros(len(states))
    empty = {}
    for st in range(0, len(states), chunk):
        feats, owner = [], []
        for si, (board, bar, off, mover) in enumerate(states[st:st + chunk]):
            opp = 1 - mover
            white_opp = opp == 0
            for di, (_d1, _d2, prob, ml) in enumerate(S.DICE):
                reps = S._clone(board, bar, off, opp, ml).legal_moves()
                if not reps:
                    # no reply: value the pre-reply state itself (opp to move)
                    feat = FT.encode_state(board, bar, off, opp)
                    feats.append(feat)
                    owner.append((si, di, prob, white_opp, "leaf"))
                    continue
                for r in reps:
                    c = S._clone(board, bar, off, opp, ml)
                    c.apply(r)
                    feats.append(FT.encode_state(c.board, c.bar, c.off, c.turn))
                    owner.append((si, di, prob, white_opp, "rep"))
        vals = gh.values(np.array(feats)) if feats else np.zeros(0)
        if st // chunk % 40 == 0:
            import torch as _t
            _t.cuda.empty_cache()
        best = {}  # (si, di) -> best reply value (max white opp, min black opp)
        for (si, di, prob, white_opp, kind), v in zip(owner, vals):
            key = (si, di)
            if key not in best:
                best[key] = [v, 1.0 / 36.0 if False else prob]
            else:
                if (white_opp and v > best[key][0] + 1e-9) or (
                        not white_opp and v < best[key][0] - 1e-9):
                    best[key][0] = v
        acc = {}
        for (si, di), (b, prob) in best.items():
            acc[si] = acc.get(si, 0.0) + prob * b
        for si, v in acc.items():
            out[st + si] = v
    return out


def one_ply_pick(state, moves, gh, rng):
    """1-ply pick: best V over legal moves (white max, black min, 1e-9 tie)."""
    board, bar, off, mover = state
    feats = []
    for m in moves:
        c = sim.Game.__new__(sim.Game)
        c.board, c.bar, c.off = list(board), list(bar), list(off)
        c.turn = mover
        c.dice = [0, 0]
        c.moves_left = list(m.moves_left if hasattr(m, "moves_left") else [m["die"]])
        c.has_rolled = True
        c.apply(m)
        feats.append(FT.encode_state(c.board, c.bar, c.off, c.turn))
    vals = gh.values(np.array(feats))
    white = mover == 0
    best, bestv = [], None
    for m, v in zip(moves, vals):
        if bestv is None or (white and v > bestv + 1e-9) or (not white and v < bestv - 1e-9):
            best, bestv = [m], v
        elif abs(v - bestv) <= 1e-9:
            best.append(m)
    return rng.choice(best)


def two_ply_pick(state, moves, gh, rng):
    """2-ply pick: best expected value over opp replies."""
    vals = two_ply_values([(state[0], state[1], state[2], state[3])
                           for _m in moves], gh, chunk=1) if False else \
        two_ply_values([state] * len(moves), gh, chunk=1)
    white = state[3] == 0
    best, bestv = [], None
    for m, v in zip(moves, two_ply_per_move(state, moves, gh)):
        if bestv is None or (white and v > bestv + 1e-9) or (not white and v < bestv - 1e-9):
            best, bestv = [m], v
        elif abs(v - bestv) <= 1e-9:
            best.append(m)
    return rng.choice(best)


def two_ply_per_move(state, moves, gh):
    """2-ply value per candidate move: state after m, mover=state mover."""
    return two_ply_values([(c.board, c.bar, c.off, state[3]) for c in _clones(state, moves)], gh, chunk=1)


def _clones(state, moves):
    board, bar, off, mover = state[:4]
    out = []
    for m in moves:
        c = sim.Game.__new__(sim.Game)
        c.board, c.bar, c.off = list(board), list(bar), list(off)
        c.turn = mover
        c.dice = [0, 0]
        c.moves_left = list(m.moves_left if hasattr(m, "moves_left") else [m["die"]])
        c.has_rolled = True
        c.apply(m)
        out.append(c)
    return out


def probe(head, W, n):
    import concurrent.futures as cf
    from brain import gpu as G
    gh = G.GPUHead(W, head.Win, head.w, steps=head.steps, device="cuda")
    jobs = 8
    per = max(1, (max(40, n // 20)) // jobs)
    t0 = time.time()
    pos = []
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {k: ex.submit(sample_decisions, head, per, 70001 + k * 997) for k in range(jobs)}
        done = 0
        for k, f in futs.items():
            pos.extend(f.result())
            done += 1
            print(f"  shard {done}/{jobs}: +{len([1])} games done ({round(time.time() - t0)}s)", flush=True)
    print(f"sampled {len(pos)} decisions in {round(time.time() - t0)}s", flush=True)
    pos = pos[:n]
    # per-position candidate moves (ml-aware)
    groups = []
    for st in pos:
        ms = _legal(st)
        if len(ms) < 2:
            continue
        groups.append((st, ms))
    print(f"{len(groups)} usable decisions", flush=True)
    if not groups:
        print("DISAGREEMENT: 0/0 = 0.0%"); return
    # 1-ply: one shot over every (pos, move) leaf
    f1, own = [], []
    for pi, (st, ms) in enumerate(groups):
        for m in ms:
            c = _clone_after(st, m)
            f1.append(FT.encode_state(c.board, c.bar, c.off, c.turn))
            own.append(pi)
    p1v = gh.values(np.array(f1))
    # 2-ply: one shot over every post-move state (chunked inside)
    post = []
    for (st, ms) in groups:
        for m in ms:
            c = _clone_after(st, m)
            post.append((c.board, c.bar, c.off, st[3]))
    t1 = time.time()
    p2v = two_ply_values(post, gh, chunk=60)
    print(f"2-ply evals {len(post)} in {round(time.time() - t1)}s", flush=True)
    # group slices
    disagr, total = 0, 0
    idx = 0
    for pi, (st, ms) in enumerate(groups):
        k = len(ms)
        white = st[3] == 0
        a = _pick_index(ms, p1v[idx:idx + k], white)
        b = _pick_index(ms, p2v[idx:idx + k], white)
        idx += k
        if a != b:
            disagr += 1
        total += 1
    print(f"DISAGREEMENT: {disagr}/{total} = {100 * disagr / max(1, total):.1f}% "
          f"(total {round(time.time() - t0)}s)")


def main():
    a = sys.argv[1:]
    wpath = str(ROOT / "brain" / "weights_10k_td" / "ckpt_150000.npz")
    sl = str(ROOT / "data" / "slice_10k_idx.npy")
    probe_n, sample, out, lam = 0, 0, None, 1.0
    from_cache = False
    i = 0
    while i < len(a):
        if a[i] == "--w":
            wpath = a[i + 1]; i += 2
        elif a[i] == "--slice":
            sl = a[i + 1]; i += 2
        elif a[i] == "--probe":
            probe_n = int(a[i + 1]); i += 2
        elif a[i] == "--sample":
            sample = int(a[i + 1]); i += 2
        elif a[i] == "--out":
            out = a[i + 1]; i += 2
        elif a[i] == "--lam":
            lam = float(a[i + 1]); i += 2
        elif a[i] == "--from-cache":
            from_cache = True; i += 1
        else:
            i += 1
    rows = np.load(sl)
    head, W = load_head(wpath, rows)
    print(f"head n={W.shape[0]} w={len(head.w)}", flush=True)
    if probe_n:
        probe(head, W, probe_n)
        return
    if sample and out:
        lap(head, W, sample, out, lam, from_cache=from_cache)


def lap(head, W, games, out, lam, from_cache=False):
    """All-moves lap: 2-ply targets for EVERY candidate move of each decision,
    so the ridge aligns the student's argmax with search's argmax (policy-style
    distillation), not just its average value on played moves."""
    from brain import gpu as G
    gh = G.GPUHead(W, head.Win, head.w, steps=head.steps, device="cuda")
    t0 = time.time()
    if from_cache:
        z = np.load(out + ".cache.npz")
        feats = z["feats"]
        boards, bars, offs, movers = z["boards"], z["bars"], z["offs"], z["movers"]
        rows = [(list(b), [int(bar[0]), int(bar[1])], [int(o[0]), int(o[1])], int(m))
                for b, bar, o, m in zip(boards, bars, offs, movers)]
        print(f"loaded cache: {len(rows)} targets in {round(time.time() - t0)}s", flush=True)
        return _lap_gpu(head, gh, rows, feats, out, lam, t0)
    rows, feats = [], []
    for gi in range(games):
        for st in sample_decisions(head, 1, 70001 + gi):
            ms = _legal(st)
            if len(ms) < 2:
                continue
            for m in ms:
                c = _clone_after(st, m)
                rows.append((c.board, c.bar, c.off, st[3]))
                feats.append(FT.encode_state(c.board, c.bar, c.off, c.turn))
        if (gi + 1) % 50 == 0:
            print(f"  sampled {len(rows)} targets ({gi + 1}/{games} games, {round(time.time() - t0)}s)", flush=True)
    print(f"sample done: {len(rows)} targets in {round(time.time() - t0)}s", flush=True)
    feats = np.array(feats, dtype=np.float32)
    return _lap_gpu(head, gh, rows, feats, out, lam, t0)


def _lap_gpu(head, gh, rows, feats, out, lam, t0):
    boards = np.array([r[0] for r in rows], dtype=np.int16)
    bars = np.array([r[1] for r in rows], dtype=np.int16)
    offs = np.array([r[2] for r in rows], dtype=np.int16)
    movers = np.array([r[3] for r in rows], dtype=np.int8)
    np.savez(out + ".cache.npz", feats=feats, boards=boards, bars=bars, offs=offs, movers=movers)
    print(f"cache saved ({len(rows)} targets) in {round(time.time() - t0)}s", flush=True)
    T = two_ply_values(rows, gh, chunk=20)
    print(f"targets done in {round(time.time() - t0)}s", flush=True)
    # streamed ridge: materializing P (7.7GB) OOMs the box; accumulate instead.
    d = P_dim = head.w.shape[0]
    A = np.zeros((d, d), dtype=np.float64)
    b = np.zeros(d, dtype=np.float64)
    CH = 3000
    for q in range(0, len(feats), CH):
        import torch as _t
        if q // CH % 8 == 0:
            _t.cuda.empty_cache()
        Pi = gh.phi_batch(feats[q:q + CH]).astype(np.float64)
        A += Pi.T @ Pi
        b += Pi.T @ T[q:q + CH]
        if (q // CH) % 8 == 0:
            print(f"  ridge chunk {q // CH + 1}/{(len(feats) - 1) // CH + 1} ({round(time.time() - t0)}s)", flush=True)
    A += lam * np.eye(d)
    b += lam * head.w
    w = np.linalg.solve(A, b)
    from brain import value as Vv
    stu = Vv.Head(head.W, seed=7)
    stu.w = w
    stu.save(out)
    print(f"saved {out} | ridge ok | ||w||={np.linalg.norm(w):.2f}", flush=True)





def _legal(state):
    board, bar, off, mover = state[:4]
    ml = state[4] if len(state) > 4 else []
    g = sim.Game.__new__(sim.Game)
    g.board, g.bar, g.off = list(board), list(bar), list(off)
    g.turn = mover
    g.dice = [0, 0]
    g.moves_left = list(ml)
    g.has_rolled = True
    return g.legal_moves()


def _clone_after(state, m):
    c = sim.Game.__new__(sim.Game)
    c.board, c.bar, c.off = list(state[0]), list(state[1]), list(state[2])
    c.turn = state[3]
    c.dice = [0, 0]
    c.moves_left = list(m.moves_left if hasattr(m, "moves_left") else [m["die"]])
    c.has_rolled = True
    c.apply(m)
    return c


def _pick_index(ms, vals, white):
    best, bestv = [], None
    for i, v in enumerate(vals):
        if bestv is None or (white and v > bestv + 1e-9) or (not white and v < bestv - 1e-9):
            best, bestv = [i], v
        elif abs(v - bestv) <= 1e-9:
            best.append(i)
    return best[0]


def one_ply_values(state, moves, gh):
    feats = []
    for m in moves:
        c = _clone_after(state, m)
        feats.append(FT.encode_state(c.board, c.bar, c.off, c.turn))
    return gh.values(np.array(feats))


if __name__ == "__main__":
    main()
