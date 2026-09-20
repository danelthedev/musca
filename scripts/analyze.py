"""Diagnostics scoreboard. Usage: analyze.py [N=200] [jobs=4] [--n 512] [--w brain/weights/best.npz]

Plays fly vs greedy/random/prior locally, reports win%, gammons, hits for/against,
blots left in danger, primes formed, home stacking. JSON to results/analysis.json.
"""
import concurrent.futures as _cf
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import sim
from opponents import greedy as G
from brain import value as V
from brain import flybrain as F


def struct(g, p):
    s = 1 if p == 0 else -1
    blots = made = prime = run = 0
    home = range(6) if p == 0 else range(18, 24)
    stack = 0
    for i in list(home):
        c = g.board[i] * s
        if c > 0:
            stack = max(stack, c)
    for v in g.board:
        if v == s:
            blots += 1
        elif v * s >= 2:
            made += 1
            run += 1
            prime = max(prime, run)
        else:
            run = 0
    return blots, made, prime, stack


def play(pol_w, pol_b, seed):
    rng = random.Random(seed)
    g = sim.Game()
    pols = [pol_w, pol_b]
    hits = [0, 0]
    samp = []
    plies = 0
    while True:
        mover = g.turn
        g.roll(rng)
        if not g.has_any_legal():
            g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn
            continue
        while True:
            ms = g.legal_moves()
            if not ms:
                break
            bar_before = list(g.bar)
            g.apply(pols[mover](g, ms, rng))
            plies += 1
            if g.bar[1 - mover] > bar_before[1 - mover]:
                hits[mover] += 1
            if plies % 10 == 0:
                samp.append((struct(g, 0), struct(g, 1)))
            win, w = g.check_win()
            if win:
                mult, reason = g.win_multiplier(w)
                return {"winner": w, "mult": mult, "reason": reason, "plies": plies,
                        "hits": hits, "samp": samp}
            if not g.moves_left or not g.has_any_legal():
                break
        if g.check_technical_win(mover):
            return {"winner": mover, "mult": 2, "reason": "tehnic", "plies": plies,
                    "hits": hits, "samp": samp}
        g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn


def main():
    a = sys.argv[1:]
    N = int(a[0]) if len(a) > 0 else 200
    jobs = int(a[1]) if len(a) > 1 else 4
    n, wpath = 512, str(ROOT / "brain" / "weights" / "best.npz")
    i = 2
    while i < len(a):
        if a[i] == "--n":
            n = int(a[i + 1]); i += 2
        elif a[i] == "--w":
            wpath = a[i + 1]; i += 2
        else:
            i += 1
    W = F.load_connectome()
    W = W[:n, :n].tocsr() if W.shape[0] >= n else W
    head = V.Head.load(wpath, W)
    prior = V.Head(W, seed=7)
    fly, pri = head.policy(), prior.policy()

    def rnd(g, moves, rng):
        return rng.choice(moves)

    matchups = {"greedy": G.policy, "random": rnd, "prior": pri}
    out = {}
    for name, opp in matchups.items():
        recs = []
        with _cf.ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = []
            for s in range(N):
                if s % 2 == 0:
                    futs.append(ex.submit(play, fly, opp, 5000 + s))
                else:
                    futs.append(ex.submit(play, opp, fly, 5000 + s))
            for k, f in enumerate(futs):
                r = f.result()
                fly_white = (k % 2 == 0)
                r["fly_won"] = (r["winner"] == 0) == fly_white
                recs.append(r)
        wins = [r for r in recs if r["fly_won"]]
        losses = [r for r in recs if not r["fly_won"]]
        d = {"n": N, "fly_win%": round(100 * len(wins) / N, 1),
             "gammon%_of_wins": round(100 * sum(1 for r in wins if r["mult"] > 1) / max(1, len(wins)), 1),
             "gammon%_of_losses": round(100 * sum(1 for r in losses if r["mult"] > 1) / max(1, len(losses)), 1),
             "med_plies": int(statistics.median([r["plies"] for r in recs]))}
        # structural means over samples, split by who won (fly-centric: samples labeled by fly side)
        agg = {"blots": [], "made": [], "prime": [], "stack": [], "hits_for": [], "hits_vs": []}
        for k, r in enumerate(recs):
            fly_side = 0 if k % 2 == 0 else 1
            for s0, s1 in r["samp"]:
                s = s0 if fly_side == 0 else s1
                agg["blots"].append(s[0])
                agg["made"].append(s[1])
                agg["prime"].append(s[2])
                agg["stack"].append(s[3])
            agg["hits_for"].append(r["hits"][fly_side])
            agg["hits_vs"].append(r["hits"][1 - fly_side])
        for k in ("blots", "made", "prime", "stack", "hits_for", "hits_vs"):
            d["fly_" + k] = round(statistics.mean(agg[k]), 2) if agg[k] else 0.0
        out[name] = d
        print(f"vs {name}: win {d['fly_win%']}%  gammons W/L {d['gammon%_of_wins']}/{d['gammon%_of_losses']}  "
              f"blots {d['fly_blots']} made {d['fly_made']} prime {d['fly_prime']} stack {d['fly_stack']}  "
              f"hits {d['fly_hits_for']}/{d['fly_hits_vs']} medplies {d['med_plies']}", flush=True)
    p = ROOT / "results" / "analysis.json"
    p.write_text(json.dumps(out, indent=1))
    print("->", p)


if __name__ == "__main__":
    main()
