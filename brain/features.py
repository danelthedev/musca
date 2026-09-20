"""State + move features. No deps. Side-to-move-agnostic, absolute encoding.

V(s) reads white-win probability, so features stay absolute (turn included).
"""
from engine.sim import BAR, OFF

PIP_NORM = 100.0


def pip(board, bar, p):
    tot = bar[p] * 25
    if p == 0:
        for i, v in enumerate(board):
            if v > 0:
                tot += v * (i + 1)
    else:
        for i, v in enumerate(board):
            if v < 0:
                tot += -v * (24 - i)
    return tot


def encode_state(board, bar, off, turn):
    """~37-dim vector: signed board + bar/off/turn + structural extras."""
    b = [v / 5.0 for v in board]
    f = list(b) + [bar[0] / 2.0, bar[1] / 2.0, off[0] / 15.0, off[1] / 15.0,
                   1.0 if turn == 0 else -1.0]
    pw, pb = pip(board, bar, 0), pip(board, bar, 1)
    f += [(pb - pw) / PIP_NORM, pw / 200.0, pb / 200.0]
    for p, s in ((0, 1), (1, -1)):
        blots = made = home = 0
        rng = range(6) if p == 0 else range(18, 24)
        for i in rng:
            if board[i] * s > 0:
                home += board[i] * s
        for v in board:
            if v == s:
                blots += 1
            elif v * s >= 2:
                made += 1
        f += [blots / 8.0, made / 6.0, home / 15.0]
    return f


def move_features(g_before, m, g_after, mover):
    """7-dim, from mover perspective. Negative pip delta = progress."""
    opp = 1 - mover
    dp_own = pip(g_after.board, g_after.bar, mover) - pip(g_before.board, g_before.bar, mover)
    dp_opp = pip(g_after.board, g_after.bar, opp) - pip(g_before.board, g_before.bar, opp)
    to = m["to"]
    hit = 1.0 if (to != OFF and
                  ((mover == 0 and g_before.board[to] == -1) or
                   (mover == 1 and g_before.board[to] == 1))) else 0.0
    made = 0.0
    left_blot = 0.0
    if to != OFF:
        v = g_after.board[to]
        if (mover == 0 and v >= 2) or (mover == 1 and v <= -2):
            made = 1.0
        if abs(v) == 1:
            left_blot = 1.0
    return [-dp_own / 25.0, dp_opp / 25.0, hit, made, -left_blot,
            1.0 if to == OFF else 0.0, 1.0 if m["from"] == BAR else 0.0]
