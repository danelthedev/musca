"""Curated decision test set. Usage: test_positions.py generate [--n 20] | eval [--w weights] [--ply 1|2]

OPENERS: standard opening replies (Woolsey/Kaplan-era theory).
Generated: positions where 1-ply/2-ply disagree, verdict = 2-ply expected value with margin.
JSON: results/test_positions.json. Eval prints agreement %.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import sim
from brain import value as V, search as S
from brain import flybrain as F

OPENERS = [  # (dice, expected) ENGINE coords (std point - 1), white replies
    ((3, 1), [(7, 4, 3), (5, 4, 1)]),
    ((4, 2), [(7, 3, 4), (5, 3, 2)]),
    ((5, 1), [(12, 7, 5), (5, 4, 1)]),
    ((6, 1), [(12, 6, 6), (7, 6, 1)]),
    ((2, 1), [(12, 10, 2), (5, 4, 1)]),
    ((3, 2), [(12, 10, 2), (12, 9, 3)]),
    ((5, 4), [(12, 8, 4), (12, 7, 5)]),
    ((6, 2), [(12, 6, 6), (12, 10, 2)]),
]

OUT = ROOT / "results" / "test_positions.json"


def opener_states():
    out = []
    for dice, exp in OPENERS:
        g = sim.Game()
        g.turn = 0
        g.dice = list(dice)
        g.moves_left = list(dice)
        g.has_rolled = True
        out.append({"board": g.board, "bar": g.bar, "off": g.off, "turn": 0,
                    "moves": [[m["from"], m["to"], m["die"]] for m in g.legal_moves()],
                    "expect": [list(e) for e in exp], "kind": "opening"})
    return out


def rollout_winner(head, w, board, bar, off, turn, first_move, seed, n=200):
    """Win rate of committing to first_move then 1-ply both sides."""
    import concurrent.futures as cf

    def one(s):
        r = random.Random(seed + s)
        g = sim.Game.__new__(sim.Game)
        g.board, g.bar, g.off = list(board), list(bar), list(off)
        g.turn, g.dice, g.moves_left, g.has_rolled = turn, [0, 0], [], True
        return _play_rest(head, w, g, first_move, r)

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(one, range(n)))
    return sum(res) / n


def _play_rest(head, w, g, first_move, rng):
    pol = head.policy()
    g.moves_left = []
    g.has_rolled = False
    # apply forced first move consumes its die; simplest: full move object
    fm = {"from": first_move[0], "to": first_move[1], "die": first_move[2]}
    # reconstruct dice context: roll fresh then force legality is complex;
    # approximate: play out with 1-ply from the AFTER position, turn flipped
    c = sim.Game.__new__(sim.Game)
    c.board, c.bar, c.off = list(g.board), list(g.bar), list(g.off)
    c.turn, c.dice, c.moves_left, c.has_rolled = g.turn, [0, 0], [], True
    # apply first move manually via temp game with matching die
    t = sim.Game.__new__(sim.Game)
    t.board, t.bar, t.off = list(g.board), list(g.bar), list(g.off)
    t.turn, t.dice, t.moves_left, t.has_rolled = g.turn, [0, 0], [fm["die"]], True
    try:
        t.apply(fm)
    except Exception:
        return 0.5
    wnr, _, _, _ = _finish(head, w, t, rng)
    return 1.0 if wnr == 0 else 0.0


def _finish(head, w, g, rng):
    pol = head.policy()

    def rnd(g_, moves, rng_):
        return rng_.choice(moves)

    # continue as 1-ply white vs 1-ply black from g; ply cap breaks hit-loops
    plies = 0
    while True:
        mover = g.turn
        g.roll(rng)
        if not g.has_any_legal():
            g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn
            continue
        done = False
        while True:
            ms = g.legal_moves()
            if not ms:
                break
            m = pol(g, ms, rng)
            g.apply(m)
            plies += 1
            if plies > 3000:  # hit-loop: leader by pip takes it
                from brain.features import pip as _pip
                return (0 if _pip(g.board, g.bar, 0) <= _pip(g.board, g.bar, 1) else 1), 0, "", 0
            win, wnr = g.check_win()
            if win:
                return wnr, 0, "", 0
            if not g.moves_left or not g.has_any_legal():
                break
        if g.check_technical_win(mover):
            return mover, 0, "", 0
        g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn


def generate(n, head, w):
    rng = random.Random(99)
    one = head.policy()
    out = opener_states()
    s = 0
    while len([x for x in out if x["kind"] == "hard"]) < n and s < n * 40:
        s += 1
        g = sim.Game()
        r = random.Random(5000 + s)
        for _ in range(rng.randint(10, 120)):
            g.roll(r)
            if not g.has_any_legal():
                g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn
                continue
            ms = g.legal_moves()
            if not ms:
                break
            g.apply(r.choice(ms))
            if g.check_win()[0] or not g.moves_left or not g.has_any_legal():
                g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn
        g.roll(r)
        if not g.has_any_legal():
            continue
        ms = g.legal_moves()
        if len(ms) < 2:
            continue
        p1 = one(g, ms, random.Random(s))
        p2 = S.choose(head, w, g, ms, random.Random(s))
        if p1 == p2:
            continue  # only genuinely hard positions
        # verdict: 2-ply pick wins if its expected value clears by margin
        # (2-ply validated stronger: 28/40 + 76/100; rollouts too slow)
        vals = []
        for m in ms:
            cc = g.clone()
            cc.apply(m)
            vals.append(S.exp_value(head, w, cc.board, cc.bar, cc.off, g.turn))
        order = sorted(range(len(ms)), key=lambda i: vals[i], reverse=g.turn == 0)
        if abs(vals[order[0]] - vals[order[1]]) < 0.005:
            continue  # too close to call
        win = ms[order[0]]
        win = [win["from"], win["to"], win["die"]]
        out.append({"board": g.board, "bar": g.bar, "off": g.off, "turn": g.turn,
                    "moves": [[m["from"], m["to"], m["die"]] for m in ms],
                    "expect": [win], "kind": "hard",
                    "note": f"2ply margin {abs(vals[order[0]] - vals[order[1]]):.3f}"})
        print(f"hard {len([x for x in out if x['kind'] == 'hard'])}/{n} (scanned {s})", flush=True)
    OUT.write_text(json.dumps(out, indent=0))
    print(f"wrote {len(out)} positions -> {OUT}")


def evaluate(wpath, ply):
    data = json.loads(OUT.read_text())
    W = F.load_connectome()[:512, :512].tocsr()
    try:
        head = V.Head.load(wpath, W)
        one, is_mlp = head.policy(), False
    except KeyError:
        from brain import mlp as M
        _net = M.Net.load(wpath, V.Head(W, seed=7))
        head, one, is_mlp = _net.head, _net.policy(), True
        print("MLP weights: ply2 search unavailable, scoring ply1", flush=True)
    hit = tot = 0
    hard_hit = hard_tot = 0
    for v in data:
        g = sim.Game.__new__(sim.Game)
        g.board, g.bar, g.off = list(v["board"]), list(v["bar"]), list(v["off"])
        g.turn, g.dice, g.has_rolled = v["turn"], [0, 0], True
        ms = [{"from": a, "to": b, "die": d} for a, b, d in v["moves"]]
        g.moves_left = sorted({m["die"] for m in ms})  # legality needs dice present
        if v["kind"] == "opening":
            # opening: full-turn sequence; check first move matches any expected first
            got = one(g, ms, random.Random(0)) if (ply == 1 or is_mlp) else S.choose(head, head.w, g, ms, random.Random(0))
            ok = [got["from"], got["to"], got["die"]] in v["expect"] or \
                any([got["from"], got["to"]] == e[:2] for e in v["expect"])
        else:
            got = one(g, ms, random.Random(0)) if (ply == 1 or is_mlp) else S.choose(head, head.w, g, ms, random.Random(0))
            ok = [got["from"], got["to"], got["die"]] in v["expect"]
        tot += 1
        hit += ok
        if v["kind"] == "hard":
            hard_tot += 1
            hard_hit += ok
    print(f"{wpath} ply{ply}: {hit}/{tot} = {100 * hit / tot:.1f}% (hard {hard_hit}/{hard_tot})")


def main():
    a = sys.argv[1:]
    if not a or a[0] == "generate":
        n = 20
        for i, x in enumerate(a):
            if x == "--n":
                n = int(a[i + 1])
        W = F.load_connectome()[:512, :512].tocsr()
        head = V.Head.load(str(ROOT / "brain" / "weights" / "expert.npz"), W)
        generate(n, head, head.w)
    elif a[0] == "eval":
        wpath, ply = str(ROOT / "brain" / "weights" / "expert.npz"), 1
        for i, x in enumerate(a):
            if x == "--w":
                wpath = a[i + 1]
            if x == "--ply":
                ply = int(a[i + 1])
        evaluate(wpath, ply)
    else:
        raise SystemExit("usage: generate [--n 20] | eval [--w path] [--ply 1|2]")


if __name__ == "__main__":
    main()
