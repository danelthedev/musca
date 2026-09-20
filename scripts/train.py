"""TD(lambda) training for the value head. Frozen reservoir, linear w only.

Usage: train.py [--n 512] [--gamesA 300] [--gamesB 200] [--gamesC 300] [--alpha 3e-4]
                [--lambda 0.7] [--eps0 0.15] [--eps1 0.02] [--seed 1] [--eval-every 100]
                [--w0 weights.npz] [--outdir brain/weights]

V(s) = white-win prob, shared head. Every ply: delta = V(s') - V(s),
terminal V = reward (1 white wins else 0). e = lam*e + phi(s). w += a*delta*e.
Alpha auto-scales with reservoir size: a_eff = alpha * (2000/n).
Curriculum: A vs random, B vs greedy, C self-play vs snapshot (refresh 50 games).
Checkpoints each stage + best.npz on eval-vs-random improvement (40 games).
"""
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import sim
from opponents import greedy as G
from brain import value as V
from brain import flybrain as F


def parse(a):
    o = dict(n=512, gamesA=300, gamesB=200, gamesC=300, alpha=1e-4, lam=0.7,
             eps0=0.15, eps1=0.02, seed=1, eval_every=100, w0=None, equity=0,
             outdir=str(ROOT / "brain" / "weights"))
    i = 0
    while i < len(a):
        k = a[i]
        if k.startswith("--"):
            k = k[2:].replace("-", "_")
            if k == "lambda":
                k = "lam"
            if k in o and not isinstance(o[k], str) or k in ("w0", "outdir"):
                v = a[i + 1]
                o[k] = v if k in ("w0", "outdir") else type(o[k])(v)
                i += 2
            else:
                raise SystemExit(f"unknown flag {a[i]}")
        else:
            raise SystemExit(f"unexpected arg {a[i]}")
    return o


def eval_vs_random(head, n_games=40, seed=99999):
    fly = head.policy()
    w = 0
    for s in range(n_games):
        r = random.Random(seed + s)
        if s % 2 == 0:
            win, _, _, _ = sim.play_game(fly, sim.random_policy, r)
            w += (win == 0)
        else:
            win, _, _, _ = sim.play_game(sim.random_policy, fly, r)
            w += (win == 1)
    return w / n_games


def train_stage(head, games, opp_fn, o, prog, tag, snap_every=0):
    import numpy as np
    a_eff = o["alpha"] * (2000.0 / head.n)
    lam = o["lam"]
    e = np.zeros_like(head.w)
    total = prog["total"]
    span = o["gamesA"] + o["gamesB"] + o["gamesC"]
    for gi in range(games):
        rng = random.Random(o["seed"] * 1000003 + total)
        frac = total / max(1, span)
        eps = o["eps0"] + (o["eps1"] - o["eps0"]) * frac
        trainee_white = (total % 2 == 0)
        g = sim.Game()
        phis, vs = [], []
        # play full game, record positions after each ply
        while True:
            mover = g.turn
            g.roll(rng)
            if not g.has_any_legal():
                g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn
                continue
            done = False
            while True:
                moves = g.legal_moves()
                if not moves:
                    break
                is_trainee = (mover == 0) == trainee_white
                if is_trainee and rng.random() < eps:
                    m = rng.choice(moves)
                elif is_trainee:
                    m = head.choose(g, moves, rng)
                else:
                    m = opp_fn(g, moves, rng)
                g.apply(m)
                phis.append(head.phi(g.board, g.bar, g.off, g.turn))
                vs.append(float(head.w @ phis[-1]))
                win, wnr = g.check_win()
                if win:
                    mult, _ = g.win_multiplier(wnr)
                    reward = (mult / 3.0 if o["equity"] else 1.0) if wnr == 0 else 0.0
                    done = "win"
                    break
                if not g.moves_left or not g.has_any_legal():
                    break
            if done:
                break
            if g.check_technical_win(mover):
                reward = (2.0 / 3.0 if o["equity"] else 1.0) if mover == 0 else 0.0
                break
            g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn
        # backward TD(lambda) over ply sequence
        e[:] = 0.0
        for t in range(len(phis)):
            v_next = reward if t == len(phis) - 1 else vs[t + 1]
            delta = max(-1.0, min(1.0, v_next - vs[t]))
            e = lam * e + phis[t]
            scale = 1.0 + float(np.dot(e, e)) / len(e)  # trace damping
            head.w += a_eff * delta * e / scale
        total += 1
        prog["total"] = total
        if snap_every and total % snap_every == 0 and opp_fn is prog.get("snap_pol"):
            prog["snap_w"] = head.w.copy()
        if total % 25 == 0:
            print(f"  [{tag}] game {gi + 1}/{games} (total {total})", flush=True)
        if o["eval_every"] and total % o["eval_every"] == 0:
            sc = eval_vs_random(head)
            print(f"  eval@{total}: {100 * sc:.1f}% vs random", flush=True)
            if sc > prog["best"]:
                prog["best"] = sc
                head.save(Path(o["outdir"]) / "best.npz")
                print(f"  new best -> best.npz", flush=True)
    return head


def main():
    o = parse(sys.argv[1:])
    t0 = time.time()
    W = F.load_connectome()
    n = o["n"]
    W = W[:n, :n].tocsr() if W.shape[0] >= n else W
    head = V.Head(W, seed=7)
    if o["w0"]:
        head = V.Head.load(o["w0"], W)
    print(f"train: n={head.n} dim={head.dim} alpha_eff={o['alpha'] * 2000.0 / head.n:.2e} "
          f"lam={o['lam']} eps {o['eps0']}->{o['eps1']}")
    out = Path(o["outdir"])
    prog = {"total": 0, "best": eval_vs_random(head)}
    print(f"start eval: {100 * prog['best']:.1f}% vs random")
    head.save(out / "stage0_start.npz")
    train_stage(head, o["gamesA"], sim.random_policy, o, prog, "A-vs-random")
    head.save(out / f"stageA_n{head.n}.npz")
    train_stage(head, o["gamesB"], G.policy, o, prog, "B-vs-greedy")
    head.save(out / f"stageB_n{head.n}.npz")

    # stage C: self-play vs snapshot pool
    import numpy as np
    prog["snap_w"] = head.w.copy()
    snap_head = V.Head(head.W, seed=head.seed, steps=head.steps, w=head.w.copy())

    def snap_pol(g, moves, rng):
        if rng.random() < 0.05:
            return rng.choice(moves)
        return snap_head.choose(g, moves, rng)

    prog["snap_pol"] = snap_pol
    train_stage(head, o["gamesC"], snap_pol, o, prog, "C-selfplay", snap_every=50)
    head.save(out / f"stageC_n{head.n}.npz")
    sc = eval_vs_random(head, n_games=100)
    print(f"FINAL: {100 * sc:.1f}% vs random (100 games), best {100 * prog['best']:.1f}%, "
          f"{round(time.time() - t0, 1)}s -> {out}")


if __name__ == "__main__":
    main()
