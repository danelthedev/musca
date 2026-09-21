"""Frozen evaluation protocol for all future comparisons.

Usage: eval_standard.py --w weights.npz [--n 512] [--games 200] [--tag NAME]
                [--opp random|greedy|solid|all] [--h2h other.npz] [--h2h-games 300]
                [--seed 0] [--outdir results]

Ladder: N games each vs random/greedy/solid, alternating colors, fixed seeds.
h2h: 300 games vs another weights file. Appends to results/eval_<tag>.log.
"""
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import sim
from opponents import greedy as G
from opponents import solid as S
from brain import value as V
from brain import flybrain as F

OPPS = {"random": sim.random_policy, "greedy": G.policy, "solid": S.policy}


def series(fly_pol, opp_pol, N, seed0):
    wins = mults = gammons = 0
    for s in range(N):
        r = random.Random(seed0 + s)
        if s % 2 == 0:
            w, mult, _, _ = sim.play_game(fly_pol, opp_pol, r)
            won = (w == 0)
        else:
            w, mult, _, _ = sim.play_game(opp_pol, fly_pol, r)
            won = (w == 1)
        wins += won
        mults += mult if won else 0
        gammons += 1 if won and mult > 1 else 0
    return wins, (mults / max(1, wins)), 100 * gammons / max(1, wins)


def parse(a):
    o = dict(w=None, n=512, games=200, tag="eval", opp="all", h2h=None,
             h2h_games=300, seed=0, outdir=str(ROOT / "results"))
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
    if not o["w"]:
        raise SystemExit("--w required")
    return o


def main():
    o = parse(sys.argv[1:])
    t0 = time.time()
    W = F.load_connectome()
    W = W[:o["n"], :o["n"]].tocsr() if W.shape[0] >= o["n"] else W
    head = V.Head.load(o["w"], W)
    fly = head.policy()
    opps = [o["opp"]] if o["opp"] != "all" else ["random", "greedy", "solid"]
    lines = [f"# {o['tag']} w={o['w']} n={o['n']} games={o['games']}"]
    for opp in opps:
        wins, avm, gam = series(fly, OPPS[opp], o["games"], o["seed"] + 7919 * ["random", "greedy", "solid"].index(opp))
        lines.append(f"ladder-vs-{opp}: {wins}/{o['games']} = {100 * wins / o['games']:.1f}% avm={avm:.2f} gammon={gam:.1f}%")
    if o["h2h"]:
        other = V.Head.load(o["h2h"], W)
        wins, avm, gam = series(fly, other.policy(), o["h2h_games"], 55000)
        lines.append(f"h2h-vs-{Path(o['h2h']).stem}: {wins}/{o['h2h_games']} = {100 * wins / o['h2h_games']:.1f}%")
    lines.append(f"({round(time.time() - t0, 1)}s)")
    out = Path(o["outdir"]) / f"eval_{o['tag']}.log"
    with open(out, "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
