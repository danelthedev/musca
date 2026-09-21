"""Solid sparring partner: hand-eval leaf + 2-ply expectimax.
Beats greedy via lookahead (replies included). Target: beats greedy ~70%.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brain.features import move_features, pip, is_race, shots
from engine.sim import BAR, OFF


# tunable knobs (set from tests for tuning)
EXPO_BEHIND, EXPO_AHEAD = 0.8, 0.3
ANCHOR, HOME_PT, FIVE_PT, RACE_RUN = 0.5, 0.3, 0.4, 0.5

def score(g, m, c, mover):
    f = move_features(g, m, c, mover)
    progress, opp_pip, hit, made, neg_blot, off, bar = f
    val = (1.0 * progress + 0.5 * opp_pip + 1.5 * hit + 1.0 * made + 1.0 * neg_blot
           + 1.5 * off + 0.3 * bar)
    # blot exposure after the move, scaled by race lead
    expo = shots(c.board, c.bar, mover) / 6.0
    pw, pb = pip(c.board, c.bar, 0), pip(c.board, c.bar, 1)
    lead = (pb - pw) / 100.0 if mover == 0 else (pw - pb) / 100.0
    val += -EXPO_BEHIND * expo if lead < 0.2 else -EXPO_AHEAD * expo
    # anchors + key points
    if m["to"] != OFF:
        v = c.board[m["to"]]
        home = range(6) if mover == 0 else range(18, 24)
        if abs(v) >= 2 and m["to"] in home:
            val += HOME_PT
        anchor = range(18, 24) if mover == 0 else range(6)
        if abs(v) >= 2 and m["to"] in anchor:
            val += ANCHOR
        five = 4 if mover == 0 else 19
        if m["to"] == five and abs(v) >= 2:
            val += FIVE_PT
    # race: run, don't dance
    if is_race(c.board, c.bar):
        val += RACE_RUN * progress + 1.0 * off
    return val


DICE = []
for _d1 in range(1, 7):
    for _d2 in range(_d1, 7):
        if _d1 == _d2:
            DICE.append(([_d1] * 4, 1.0 / 36.0))
        else:
            DICE.append(([_d1, _d2], 2.0 / 36.0))


def leaf(board, bar, off, mover):
    """Static eval of a position from mover's perspective (higher = better for mover)."""
    pw, pb = pip(board, bar, 0), pip(board, bar, 1)
    mine = pw if mover == 0 else pb
    theirs = pb if mover == 0 else pw
    return (theirs - mine) / 50.0 - shots(board, bar, mover) / 12.0


def exp_value(board, bar, off, mover):
    """Expected leaf after opp's best reply (replies scored by leaf too)."""
    from engine import sim as _sim
    opp = 1 - mover
    tot = 0.0
    for ml, prob in DICE:
        g = _sim.Game.__new__(_sim.Game)
        g.board, g.bar, g.off = list(board), list(bar), list(off)
        g.turn, g.dice, g.moves_left, g.has_rolled = opp, [0, 0], list(ml), True
        replies = g.legal_moves()
        if not replies:
            tot += prob * leaf(board, bar, off, mover)
            continue
        best, first = 0.0, True
        for r in replies:
            c = _sim.Game.__new__(_sim.Game)
            c.board, c.bar, c.off = list(board), list(bar), list(off)
            c.turn, c.dice, c.moves_left, c.has_rolled = opp, [0, 0], list(ml), True
            c.apply(r)
            v = leaf(c.board, c.bar, c.off, mover)
            if first or v > best:
                best, first = v, False
        tot += prob * best
    return tot


def policy(g, moves, rng):
    """2-ply: each candidate scored by expected leaf after opp reply."""
    best, bestv = [], None
    for m in moves:
        c = g.clone()
        c.apply(m)
        v = exp_value(c.board, c.bar, c.off, g.turn) + rng.random() * 1e-9
        if bestv is None or v > bestv:
            best, bestv = [m], v
    return rng.choice(best)
