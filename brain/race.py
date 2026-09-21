"""Race equity + expert move choice. No deps beyond sim/features/search.

Race positions (no contact) are scored by rollout equity instead of V:
white win -> mult/3, black win -> 0 (same units as equity-trained V).
Close calls (top-2 2-ply margin < TIE_MARGIN) go to short V-policy rollouts.
"""
from engine import sim
from engine.sim import BAR, OFF
from brain.features import pip, is_race
from brain import search as S

TIE_MARGIN = 0.01
ROLL_N = 6
ROLL_CAP = 600


def greedy_race_pick(g, moves):
    best, bestv = None, None
    for m in moves:
        c = g.clone()
        c.apply(m)
        v = pip(c.board, c.bar, g.turn)
        if bestv is None or v < bestv:
            best, bestv = m, v
    return best


def race_equity(board, bar, off, turn, n=24, rng=None, seed=0):
    """White-normalized equity of a race position via greedy rollouts."""
    import random as _r
    rng = rng or _r.Random(seed)
    tot = 0.0
    for _ in range(n):
        g = sim.Game.__new__(sim.Game)
        g.board, g.bar, g.off = list(board), list(bar), list(off)
        g.turn, g.dice, g.moves_left, g.has_rolled = turn, [0, 0], [], True
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
                g.apply(greedy_race_pick(g, ms))
                plies += 1
                if plies > ROLL_CAP:
                    break
                win, wnr = g.check_win()
                if win:
                    mult, _ = g.win_multiplier(wnr)
                    tot += (mult / 3.0) if wnr == 0 else 0.0
                    break
                if not g.moves_left or not g.has_any_legal():
                    break
            else:
                continue
            win, wnr = g.check_win()
            if win:
                break
            if plies > ROLL_CAP:
                tot += 0.5
                break
            if g.check_technical_win(mover):
                tot += (2.0 / 3.0) if mover == 0 else 0.0
                break
            g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn
    return tot / n


def rollout_equity(head, w, board, bar, off, turn, n=ROLL_N, seed=0):
    """White-normalized equity via 1-ply V-policy rollouts (tiebreaks)."""
    import random as _r
    pol = head.policy() if hasattr(head, "policy") else head
    tot = 0.0
    for i in range(n):
        rng = _r.Random(seed + i)
        g = sim.Game.__new__(sim.Game)
        g.board, g.bar, g.off = list(board), list(bar), list(off)
        g.turn, g.dice, g.moves_left, g.has_rolled = turn, [0, 0], [], True
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
                g.apply(pol(g, ms, rng))
                plies += 1
                if plies > ROLL_CAP:
                    break
                win, wnr = g.check_win()
                if win:
                    mult, _ = g.win_multiplier(wnr)
                    tot += (mult / 3.0) if wnr == 0 else 0.0
                    break
                if not g.moves_left or not g.has_any_legal():
                    break
            else:
                continue
            win, wnr = g.check_win()
            if win:
                break
            if plies > ROLL_CAP:
                tot += 0.5
                break
            if g.check_technical_win(mover):
                tot += (2.0 / 3.0) if mover == 0 else 0.0
                break
            g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn
    return tot / n


def expert_choose(head, w, g, moves, rng, seed=0):
    """2-ply per candidate, race positions scored by rollout equity,
    top-2 within TIE_MARGIN decided by short V-policy rollouts."""
    white = g.turn == 0
    scored = []
    for m in moves:
        c = g.clone()
        c.apply(m)
        if is_race(c.board, c.bar):
            v = race_equity(c.board, c.bar, c.off, c.turn, rng=rng)
        else:
            v = S.exp_value(head, w, c.board, c.bar, c.off, g.turn)
        scored.append((v, m))
    scored.sort(key=lambda t: t[0], reverse=white)
    if len(scored) >= 2 and abs(scored[0][0] - scored[1][0]) < TIE_MARGIN:
        top = scored[:2]
        rs = []
        for v, m in top:
            c = g.clone()
            c.apply(m)
            rs.append((rollout_equity(head, w, c.board, c.bar, c.off, c.turn, seed=seed), m))
        rs.sort(key=lambda t: t[0], reverse=white)
        return rs[0][1]
    return scored[0][1]
