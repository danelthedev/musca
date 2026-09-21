"""Frozen evaluation protocol for all future comparisons.

Usage: eval_standard.py --w weights.npz [--n 512] [--games 200] [--tag NAME]
                [--opp random|greedy|solid|all] [--h2h other.npz] [--h2h-games 300]
                [--seed 0] [--outdir results] [--device cpu|cuda] [--jobs 1]

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


def series(fly_pol, opp_pol, N, seed0, jobs=1):
    def one(s):
        r = random.Random(seed0 + s)
        if s % 2 == 0:
            w, mult, _, _ = sim.play_game(fly_pol, opp_pol, r)
            won = (w == 0)
        else:
            w, mult, _, _ = sim.play_game(opp_pol, fly_pol, r)
            won = (w == 1)
        return won, (mult if won else 0), (1 if won and mult > 1 else 0)
    wins = mults = gammons = 0
    if jobs > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            it = list(ex.map(one, range(N)))
    else:
        it = [one(s) for s in range(N)]
    for won, mult, gam in it:
        wins += won
        mults += mult
        gammons += gam
    return wins, (mults / max(1, wins)), 100 * gammons / max(1, wins)


def parse(a):
    o = dict(w=None, n=512, games=200, tag="eval", opp="all", h2h=None,
             h2h_games=300, seed=0, outdir=str(ROOT / "results"), slice=None,
             device="cpu", jobs=1)
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
    if o["slice"]:
        import numpy as _np
        _idx = _np.load(o["slice"])
        W = W[_idx][:, _idx].tocsr()
    else:
        W = W[:o["n"], :o["n"]].tocsr() if W.shape[0] >= o["n"] else W
    head = V.Head.load(o["w"], W)
    gh = None
    if o["device"] == "cuda":
        from brain import gpu as Gmod
        gh = Gmod.GPUHead(W, head.Win, head.w, steps=head.steps)
        fly = lambda g_, m_, r_: gh.choose_batch([(g_, m_)], r_)[0]
    else:
        fly = head.policy()
    opps = [o["opp"]] if o["opp"] != "all" else ["random", "greedy", "solid"]
    lines = [f"# {o['tag']} w={o['w']} n={o['n']} games={o['games']} dev={o['device']} jobs={o['jobs']}"]
    for opp in opps:
        wins, avm, gam = series(fly, OPPS[opp], o["games"], o["seed"] + 7919 * ["random", "greedy", "solid"].index(opp), o["jobs"])
        lines.append(f"ladder-vs-{opp}: {wins}/{o['games']} = {100 * wins / o['games']:.1f}% avm={avm:.2f} gammon={gam:.1f}%")
    if o["h2h"]:
        import numpy as _np
        _z = _np.load(o["h2h"])
        _n = int(_z["n"])
        Wh = W if _n == W.shape[0] else F.load_connectome()[:_n, :_n].tocsr()
        other = V.Head.load(o["h2h"], Wh)
        other_pol = other.policy()
        if o["device"] == "cuda":
            from brain import gpu as Gmod
            _gh2 = Gmod.GPUHead(Wh, other.Win, other.w, steps=other.steps)
            other_pol = lambda g_, m_, r_: _gh2.choose_batch([(g_, m_)], r_)[0]
        wins, avm, gam = series(fly, other_pol, o["h2h_games"], 55000, o["jobs"])
        lines.append(f"h2h-vs-{Path(o['h2h']).stem}: {wins}/{o['h2h_games']} = {100 * wins / o['h2h_games']:.1f}%")
    lines.append(f"({round(time.time() - t0, 1)}s)")
    out = Path(o["outdir"]) / f"eval_{o['tag']}.log"
    with open(out, "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
