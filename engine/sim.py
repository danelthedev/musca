"""Local backgammon simulator. Mirrors fair_backgammon/game/state.go + lobby/turn.go.

No deps, no server. Training loop plays here; WS eval stays for final check.
Board: [24]int, +white(P0, moves down) -black(P1, moves up). Bar/off per player.
"""
import random as _r
from . import bg_rules as R

BAR, OFF = -1, -2


class Game:
    def __init__(self):
        self.board = [0] * 24
        self.board[23] = 2
        self.board[12] = 5
        self.board[7] = 3
        self.board[5] = 5
        self.board[0] = -2
        self.board[11] = -5
        self.board[16] = -3
        self.board[18] = -5
        self.bar = [0, 0]
        self.off = [0, 0]
        self.turn = 0
        self.dice = [0, 0]
        self.moves_left = []
        self.has_rolled = False

    def clone(self):
        g = Game.__new__(Game)
        g.board = list(self.board)
        g.bar = list(self.bar)
        g.off = list(self.off)
        g.turn = self.turn
        g.dice = list(self.dice)
        g.moves_left = list(self.moves_left)
        g.has_rolled = self.has_rolled
        return g

    def roll(self, rng):
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        self.dice = [d1, d2]
        self.moves_left = [d1] * 4 if d1 == d2 else [d1, d2]
        self.has_rolled = True

    def legal_moves(self):
        return R.legal_moves(self.board, self.bar, self.moves_left, self.has_rolled, self.turn)

    def has_any_legal(self):
        return len(self.legal_moves()) > 0

    def apply(self, m):
        """Port of Game.Apply. Raises ValueError if illegal."""
        ok = R.is_legal(self.board, self.bar, m["from"], m["to"],
                        m["die"], self.moves_left, self.has_rolled, self.turn)
        if not ok:
            raise ValueError(f"illegal {m} left={self.moves_left} turn={self.turn}")
        p = self.turn
        self.moves_left.remove(m["die"])
        if m["from"] == BAR:
            self.bar[p] -= 1
        elif p == 0:
            self.board[m["from"]] -= 1
        else:
            self.board[m["from"]] += 1
        if m["to"] == OFF:
            self.off[p] += 1
        else:
            v = self.board[m["to"]]
            if p == 0 and v == -1:  # hit black blot
                self.board[m["to"]] = 0
                self.bar[1] += 1
            elif p == 1 and v == 1:  # hit white blot
                self.board[m["to"]] = 0
                self.bar[0] += 1
            if p == 0:
                self.board[m["to"]] += 1
            else:
                self.board[m["to"]] -= 1

    def check_win(self):
        if self.off[0] >= 15:
            return True, 0
        if self.off[1] >= 15:
            return True, 1
        return False, -1

    def win_multiplier(self, w):
        l = 1 - w
        if self.off[l] > 0:
            return 1, ""
        if self.bar[l] > 0:
            return 3, "backgammon"
        if w == 0:
            for i in range(6):
                if self.board[i] < 0:
                    return 3, "backgammon"
        else:
            for i in range(18, 24):
                if self.board[i] > 0:
                    return 3, "backgammon"
        return 2, "gammon"

    def check_technical_win(self, p):
        """Port of CheckTechnicalWin. Valid at turn end only."""
        if self.bar[p] > 0:
            return False
        if p == 0:
            home = 0
            for i in range(24):
                if self.board[i] > 0:
                    if i >= 6:
                        return False
                    home += self.board[i]
            if self.off[0] + home != 15 or home not in (12, 6):
                return False
            return all(self.board[i] == home // 6 for i in range(6))
        home = 0
        for i in range(24):
            if self.board[i] < 0:
                if i < 18:
                    return False
                home -= self.board[i]
        if self.off[1] + home != 15 or home not in (12, 6):
            return False
        return all(self.board[i] == -(home // 6) for i in range(18, 24))

    def invariant_ok(self):
        tot = sum(abs(v) for v in self.board) + sum(self.bar) + sum(self.off)
        return tot == 30 and all(v >= 0 for v in self.bar + self.off)


def play_game(pol0, pol1, rng=None, max_plies=20000):
    """Full game. pol(game, moves, rng) -> move. Returns (winner, mult, reason, plies)."""
    rng = rng or _r.Random()
    g = Game()
    pols = [pol0, pol1]
    plies = 0
    while plies < max_plies:
        mover = g.turn
        g.roll(rng)
        if not g.has_any_legal():
            g.moves_left = []
            g.has_rolled = False
            g.turn = 1 - g.turn
            continue
        while True:
            moves = g.legal_moves()
            if not moves:
                break
            m = pols[mover](g, moves, rng)
            g.apply(m)
            plies += 1
            win, w = g.check_win()
            if win:
                mult, reason = g.win_multiplier(w)
                return w, mult, reason, plies
            if not g.moves_left or not g.has_any_legal():
                break
        if g.check_technical_win(mover):
            return mover, 2, "tehnic", plies
        g.moves_left = []
        g.has_rolled = False
        g.turn = 1 - g.turn
    raise RuntimeError("game did not finish")


def random_policy(g, moves, rng):
    return rng.choice(moves)
