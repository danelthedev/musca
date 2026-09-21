"""Legal move gen. Mirrors game/state.go + web/rules.ts. No deps."""


def all_in_home(board, bar, p):
    if bar[p] > 0:
        return False
    if p == 0:
        return all(v <= 0 for v in board[6:])
    return all(v >= 0 for v in board[:18])


def is_blocked(board, to, p):
    if to < 0 or to >= 24:
        return False
    v = board[to]
    return v <= -2 if p == 0 else v >= 2


def is_legal(board, bar, frm, to, die, moves_left, has_rolled, turn):
    if not has_rolled or die not in moves_left:
        return False
    return _is_legal_inner(board, bar, frm, to, die, turn)

def _is_legal_inner(board, bar, frm, to, die, turn):
    if bar[turn] > 0 and frm != -1:
        return False
    if bar[turn] == 0 and frm == -1:
        return False
    if frm != -1:
        if frm < 0 or frm >= 24:
            return False
        v = board[frm]
        if turn == 0 and v <= 0:
            return False
        if turn == 1 and v >= 0:
            return False
    if to == -2:
        if not all_in_home(board, bar, turn):
            return False
        if frm == -1:
            return False
        dist = frm + 1 if turn == 0 else 24 - frm
        if die < dist:
            return False
        if die > dist:
            if turn == 0:
                for i in range(frm + 1, 6):
                    if board[i] > 0:
                        return False
            else:
                for i in range(18, frm):
                    if board[i] < 0:
                        return False
        return True
    if to < 0 or to >= 24:
        return False
    if frm == -1:
        entry = 24 - die if turn == 0 else die - 1
        if to != entry:
            return False
    else:
        exp = frm - to if turn == 0 else to - frm
        if exp != die:
            return False
    if is_blocked(board, to, turn):
        return False
    return True


def legal_moves(board, bar, moves_left, has_rolled, turn):
    if not has_rolled or not moves_left:
        return []
    # ponytail: same nested order as brute force (d, frm, to), but only
    # arithmetically-possible (frm, to) pairs reach _is_legal_inner.
    # Every skipped pair provably returns False there: entry must equal
    # 24-d/d-1, normal moves must span exactly d, empty points hold nothing.
    # Emission order identical -> choose()/policies behave bit-identically.
    out = []
    for d in set(moves_left):
        for frm in [-1] + list(range(24)):
            if frm == -1:
                if bar[turn] == 0:
                    continue
                entry = 24 - d if turn == 0 else d - 1
                if _is_legal_inner(board, bar, -1, entry, d, turn):
                    out.append({"from": -1, "to": entry, "die": d})
                continue
            v = board[frm]
            if turn == 0 and v <= 0:
                continue
            if turn == 1 and v >= 0:
                continue
            if _is_legal_inner(board, bar, frm, -2, d, turn):
                out.append({"from": frm, "to": -2, "die": d})
            to = frm - d if turn == 0 else frm + d
            if 0 <= to < 24 and _is_legal_inner(board, bar, frm, to, d, turn):
                out.append({"from": frm, "to": to, "die": d})
    return out
