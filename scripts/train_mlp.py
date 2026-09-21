"""TD(lambda) training for the MLP head. Usage like train.py plus --hidden 64.

Usage: train_mlp.py [--n 512] [--hidden 64] [--gamesA 300] [--gamesB 200] [--gamesC 300]
                    [--alpha 1e-3] [--lambda 0.7] [--eps0 0.15] [--eps1 0.02]
                    [--seed 1] [--eval-every 100] [--equity 1] [--w0 mlp.npz] [--outdir brain/weights_mlp]
"""
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from engine import sim
from opponents import greedy as G
from brain import value as V
from brain import mlp as M
from brain import flybrain as F


def parse(a):
    o = dict(n=512, hidden=64, gamesA=300, gamesB=200, gamesC=300, alpha=1e-3,
             lam=0.7, eps0=0.15, eps1=0.02, seed=1, eval_every=100, equity=1,
             w0=None, outdir=str(ROOT / "brain" / "weights_mlp"))
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


def eval_vs(net, opp_fn, n_games=40, seed=99999):
    fly = net.policy()
    w = 0
    for s in range(n_games):
        r = random.Random(seed + s)
        if s % 2 == 0:
            win, _, _, _ = sim.play_game(fly, opp_fn, r)
            w += (win == 0)
        else:
            win, _, _, _ = sim.play_game(opp_fn, fly, r)
            w += (win == 1)
    return w / n_games


def train_stage(net, games, opp_fn, o, prog, tag, snap_every=0):
    lam = o["lam"]
    alpha = o["alpha"]
    total = prog["total"]
    span = o["gamesA"] + o["gamesB"] + o["gamesC"]
    for gi in range(games):
        rng = random.Random(o["seed"] * 1000003 + total)
        frac = total / max(1, span)
        eps = o["eps0"] + (o["eps1"] - o["eps0"]) * frac
        trainee_white = (total % 2 == 0)
        g = sim.Game()
        phis = []
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
                    m = net.choose(g, moves, rng)
                else:
                    m = opp_fn(g, moves, rng)
                g.apply(m)
                phis.append(net.head.phi(g.board, g.bar, g.off, g.turn))
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
        net._zero_traces()
        for t in range(len(phis)):
            v_next = reward if t == len(phis) - 1 else net.forward(phis[t + 1])[0]
            net.td_step(phis[t], v_next, alpha, lam)
        total += 1
        prog["total"] = total
        if snap_every and total % snap_every == 0 and opp_fn is prog.get("snap_pol"):
            prog["snap_w"] = (net.W1.copy(), net.b1.copy(), net.W2.copy(), net.b2)
        if total % 25 == 0:
            print(f"  [{tag}] game {gi + 1}/{games} (total {total})", flush=True)
        if o["eval_every"] and total % o["eval_every"] == 0:
            sc = eval_vs(net, sim.random_policy)
            print(f"  eval@{total}: {100 * sc:.1f}% vs random", flush=True)
            if sc > prog["best"]:
                prog["best"] = sc
                net.save(Path(o["outdir"]) / "best.npz")
                print("  new best -> best.npz", flush=True)
    return net


def main():
    o = parse(sys.argv[1:])
    t0 = time.time()
    W = F.load_connectome()
    n = o["n"]
    W = W[:n, :n].tocsr() if W.shape[0] >= n else W
    head = V.Head(W, seed=7)
    net = M.Net(head, hidden=o["hidden"])
    if o["w0"]:
        net = M.Net.load(o["w0"], head)
    print(f"train_mlp: n={head.n} hidden={o['hidden']} alpha={o['alpha']:.0e} lam={o['lam']}")
    out = Path(o["outdir"])
    prog = {"total": 0, "best": eval_vs(net, sim.random_policy)}
    print(f"start eval: {100 * prog['best']:.1f}% vs random")
    net.save(out / "stage0_start.npz")
    train_stage(net, o["gamesA"], sim.random_policy, o, prog, "A-vs-random")
    net.save(out / f"stageA_n{head.n}.npz")
    train_stage(net, o["gamesB"], G.policy, o, prog, "B-vs-greedy")
    net.save(out / f"stageB_n{head.n}.npz")

    prog["snap"] = (net.W1.copy(), net.b1.copy(), net.W2.copy(), net.b2.copy())
    snap_net = M.Net(head, hidden=o["hidden"],
                     params={"W1": prog["snap"][0], "b1": prog["snap"][1],
                             "W2": prog["snap"][2], "b2": prog["snap"][3]})

    def snap_pol(g, moves, rng):
        if rng.random() < 0.05:
            return rng.choice(moves)
        return snap_net.choose(g, moves, rng)

    prog["snap_pol"] = snap_pol
    train_stage(net, o["gamesC"], snap_pol, o, prog, "C-selfplay", snap_every=50)
    net.save(out / f"stageC_n{head.n}.npz")
    sc = eval_vs(net, sim.random_policy, n_games=100)
    print(f"FINAL: {100 * sc:.1f}% vs random (100 games), best {100 * prog['best']:.1f}%, "
          f"{round(time.time() - t0, 1)}s -> {out}")


if __name__ == "__main__":
    main()
