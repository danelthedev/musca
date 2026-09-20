"""Fast server-free eval. Usage: eval_local.py [N=200] [opp=random|greedy|self] [jobs=4] [--n 2000] [--w weights.npz] [--seed 0]

Fly (value head, 1-ply) plays half games white / half black. Prints win% + gammon rate.
"""
import concurrent.futures as _cf
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import sim
from opponents import greedy as G
from brain import value as V
from brain import flybrain as F


def load_W(n):
    if n <= 0:
        return None
    W = F.load_connectome()
    if W.shape[0] >= n:
        return W[:n, :n].tocsr()
    return W


def main():
    a = sys.argv[1:]
    N = int(a[0]) if len(a) > 0 else 200
    opp = a[1] if len(a) > 1 else "random"
    jobs = int(a[2]) if len(a) > 2 and a[2].isdigit() else 4
    n = 2000
    wpath = None
    seed0 = 0
    i = 1
    while i < len(a):
        if a[i] == "--n":
            n = int(a[i + 1]); i += 2
        elif a[i] == "--w":
            wpath = a[i + 1]; i += 2
        elif a[i] == "--seed":
            seed0 = int(a[i + 1]); i += 2
        else:
            i += 1
    W = load_W(n)
    if W is None:
        W = F.load_connectome(n=64, seed=7)
    head = V.Head(W, seed=7)
    if wpath:
        head = V.Head.load(wpath, W)
    fly = head.policy()

    def opp_pol(g, moves, rng):
        if opp == "greedy":
            return G.policy(g, moves, rng)
        if opp == "self":
            return head.policy()(g, moves, rng)
        return rng.choice(moves)

    def one(s):
        if s % 2 == 0:
            w, mult, reason, plies = sim.play_game(fly, opp_pol, random.Random(seed0 + s))
            fly_white = True
        else:
            w, mult, reason, plies = sim.play_game(opp_pol, fly, random.Random(seed0 + s))
            fly_white = False
        return (w == 0) == fly_white, mult, reason, plies

    wins = mults = gammons = 0
    plies_all = []
    with _cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        for k, (won, mult, reason, plies) in enumerate(ex.map(one, range(N))):
            wins += won
            mults += mult if won else 0
            gammons += 1 if won and mult > 1 else 0
            plies_all.append(plies)
            if (k + 1) % max(1, N // 10) == 0:
                print(f"  {k + 1}/{N} fly={wins}", flush=True)
    print(f"fly {wins}/{N} = {100 * wins / N:.1f}% vs {opp} (n={W.shape[0]}, w={'prior' if not wpath else wpath})")
    print(f"gammon rate {100 * gammons / max(1, wins):.1f}% of wins, med plies {int(statistics.median(plies_all))}")


if __name__ == "__main__":
    main()
