"""TD(lambda) for the 5 MBON compartments. Frozen reservoir, linear heads only.

Usage: train_multihead.py [--n 512] [--gamesA 300] [--gamesB 200] [--gamesC 300]
                [--alpha 1e-4] [--lambda 0.7] [--eps0 0.15] [--eps1 0.02]
                [--seed 1] [--eval-every 100] [--w0-win weights.npz]
                [--outdir brain/weights_multi]

Terminal targets (white perspective): win=1 if white wins; wg/wb=1 if white
wins gammon (mult>=2) / backgammon (mult==3); lg/lb mirror for black wins.
Each head gets its own TD(lambda) trace. 1-ply on equity. numpy only.
"""
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import sim
from opponents import greedy as G
from brain import multihead as M
from brain import flybrain as F


def parse(a):
    o = dict(n=512, gamesA=300, gamesB=200, gamesC=300, alpha=1e-4, lam=0.7,
             eps0=0.15, eps1=0.02, seed=1, eval_every=100, w0_win=None,
             outdir=str(ROOT / "brain" / "weights_multi"))
    i = 0
    while i < len(a):
        k = a[i]
        if k.startswith("--"):
            k = k[2:].replace("-", "_")
            if k == "lambda":
                k = "lam"
            if k in o and not isinstance(o[k], str) or k in ("w0_win", "outdir"):
                v = a[i + 1]
                o[k] = v if k in ("w0_win", "outdir") else type(o[k])(v)
                i += 2
            else:
                raise SystemExit(f"unknown flag {a[i]}")
        else:
            raise SystemExit(f"unexpected arg {a[i]}")
    return o


def terminal_targets(wnr, mult):
    t = {h: 0.0 for h in M.HEADS}
    if wnr == 0:
        t["win"] = 1.0
        if mult >= 2:
            t["wg"] = 1.0
        if mult >= 3:
            t["wb"] = 1.0
    else:
        if mult >= 2:
            t["lg"] = 1.0
        if mult >= 3:
            t["lb"] = 1.0
    return t


def eval_policy(mh, opp_fn, n_games=60, seed=99999):
    fly = mh.policy()
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


def train_stage(mh, games, opp_fn, o, prog, tag, snap_every=0):
    import numpy as np
    a_eff = o["alpha"] * (2000.0 / mh.n)
    lam = o["lam"]
    traces = {h: np.zeros(mh.n + mh.dim + 1) for h in M.HEADS}
    total = prog["total"]
    span = o["gamesA"] + o["gamesB"] + o["gamesC"]
    for gi in range(games):
        rng = random.Random(o["seed"] * 1000003 + total)
        frac = total / max(1, span)
        eps = o["eps0"] + (o["eps1"] - o["eps0"]) * frac
        trainee_white = (total % 2 == 0)
        g = sim.Game()
        phis, vss = [], []
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
                    m = mh.choose(g, moves, rng)
                else:
                    m = opp_fn(g, moves, rng)
                g.apply(m)
                phi = mh.phi(g.board, g.bar, g.off, g.turn)
                phis.append(phi)
                vss.append({h: float(mh.ws[h] @ phi) for h in M.HEADS})
                win, wnr = g.check_win()
                if win:
                    mult, _ = g.win_multiplier(wnr)
                    targets = terminal_targets(wnr, mult)
                    done = "win"
                    break
                if not g.moves_left or not g.has_any_legal():
                    break
            if done:
                break
            if g.check_technical_win(mover):
                targets = terminal_targets(mover, 1)
                break
            g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn
        # backward TD(lambda) per head over ply sequence
        for h in M.HEADS:
            traces[h][:] = 0.0
        for t in range(len(phis)):
            for h in M.HEADS:
                v_next = targets[h] if t == len(phis) - 1 else vss[t + 1][h]
                delta = max(-1.0, min(1.0, v_next - vss[t][h]))
                traces[h] = lam * traces[h] + phis[t]
                scale = 1.0 + float(np.dot(traces[h], traces[h])) / len(traces[h])
                mh.ws[h] += a_eff * delta * traces[h] / scale
        total += 1
        prog["total"] = total
        if snap_every and total % snap_every == 0 and opp_fn is prog.get("snap_pol"):
            prog["snap_ws"] = {h: mh.ws[h].copy() for h in M.HEADS}
        if total % 25 == 0:
            print(f"  [{tag}] game {gi + 1}/{games} (total {total})", flush=True)
        if o["eval_every"] and total % o["eval_every"] == 0:
            sc = eval_policy(mh, sim.random_policy)
            print(f"  eval@{total}: {100 * sc:.1f}% vs random", flush=True)
            if sc > prog["best"]:
                prog["best"] = sc
                mh.save(Path(o["outdir"]) / "best_multi.npz")
                print("  new best -> best_multi.npz", flush=True)
    return mh


def main():
    o = parse(sys.argv[1:])
    t0 = time.time()
    W = F.load_connectome()
    n = o["n"]
    W = W[:n, :n].tocsr() if W.shape[0] >= n else W
    import numpy as np
    ws = None
    if o["w0_win"]:
        z = np.load(o["w0_win"])
        ws = {"win": z["w"]}
    mh = M.MultiHead(W, seed=7, ws=ws)
    print(f"train_multi: n={mh.n} dim={mh.dim} alpha_eff={o['alpha'] * 2000.0 / mh.n:.2e} "
          f"lam={o['lam']} eps {o['eps0']}->{o['eps1']} w0_win={o['w0_win']}")
    out = Path(o["outdir"])
    prog = {"total": 0, "best": eval_policy(mh, sim.random_policy)}
    print(f"start eval: {100 * prog['best']:.1f}% vs random")
    mh.save(out / "stage0_start.npz")
    train_stage(mh, o["gamesA"], sim.random_policy, o, prog, "A-vs-random")
    mh.save(out / f"stageA_n{mh.n}.npz")
    train_stage(mh, o["gamesB"], G.policy, o, prog, "B-vs-greedy")
    mh.save(out / f"stageB_n{mh.n}.npz")

    prog["snap_ws"] = {h: mh.ws[h].copy() for h in M.HEADS}
    snap = M.MultiHead(mh.W, seed=7, ws={h: mh.ws[h].copy() for h in M.HEADS})

    def snap_pol(g, moves, rng):
        if rng.random() < 0.05:
            return rng.choice(moves)
        if prog.get("snap_ws") is not None:
            snap.ws = {h: prog["snap_ws"][h] for h in M.HEADS}
        return snap.choose(g, moves, rng)

    prog["snap_pol"] = snap_pol
    train_stage(mh, o["gamesC"], snap_pol, o, prog, "C-selfplay", snap_every=50)
    mh.save(out / f"stageC_n{mh.n}.npz")
    sc = eval_policy(mh, sim.random_policy, n_games=100)
    scg = eval_policy(mh, G.policy, n_games=100)
    print(f"FINAL: {100 * sc:.1f}% vs random, {100 * scg:.1f}% vs greedy (100 games), "
          f"best {100 * prog['best']:.1f}%, {round(time.time() - t0, 1)}s -> {out}")


if __name__ == "__main__":
    main()
