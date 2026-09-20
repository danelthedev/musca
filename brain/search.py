"""2-ply expectimax over the value head. No deps beyond sim/features.

For each own candidate: average V over all 21 opp dice, opp replying 1-ply.
Turn is treated as passing after the candidate (standard 2-ply approx).
"""
from engine import sim
from engine.sim import BAR, OFF

DICE = []
for _d1 in range(1, 7):
    for _d2 in range(_d1, 7):
        if _d1 == _d2:
            DICE.append((_d1, _d2, 1.0 / 36.0, [_d1] * 4))
        else:
            DICE.append((_d1, _d2, 2.0 / 36.0, [_d1, _d2]))


def _clone(board, bar, off, turn, moves_left):
    g = sim.Game.__new__(sim.Game)
    g.board, g.bar, g.off = list(board), list(bar), list(off)
    g.turn = turn
    g.dice = [0, 0]
    g.moves_left = list(moves_left)
    g.has_rolled = True
    return g


def _v(head, w, g):
    return float(w @ head.phi(g.board, g.bar, g.off, g.turn))


def exp_value(head, w, board, bar, off, mover):
    """Expected V after opp's best reply to the post-move position."""
    opp = 1 - mover
    white_opp = opp == 0
    tot = 0.0
    for _, _, prob, ml in DICE:
        g = _clone(board, bar, off, opp, ml)
        replies = g.legal_moves()
        if not replies:
            tot += prob * float(w @ head.phi(board, bar, off, opp))
            continue
        best, first = 0.0, True
        for r in replies:
            c = _clone(board, bar, off, opp, ml)
            c.apply(r)
            v = _v(head, w, c)
            if first or (white_opp and v > best + 1e-9) or (not white_opp and v < best - 1e-9):
                best, first = v, False
        tot += prob * best
    return tot


def choose(head, w, g, moves, rng):
    """2-ply pick. White max, black min, ties within 1e-9 random."""
    white = g.turn == 0
    best, bestv, first = [], 0.0, True
    for m in moves:
        c = g.clone()
        c.apply(m)
        v = exp_value(head, w, c.board, c.bar, c.off, g.turn)
        if first or (white and v > bestv + 1e-9) or (not white and v < bestv - 1e-9):
            best, bestv, first = [m], v, False
        elif abs(v - bestv) <= 1e-9:
            best.append(m)
    return rng.choice(best)


def policy(head, w=None):
    w = head.w if w is None else w

    def pol(g, moves, rng):
        return choose(head, w, g, moves, rng)

    return pol
