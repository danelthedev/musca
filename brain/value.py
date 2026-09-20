"""Linear value head over frozen reservoir features. V(s) = white-win prob.

phi(s) = [x, u]: x = tanh reservoir state (W frozen), u = encode_state.
Policy: 1-ply. White argmax V(s'), black argmin. numpy only.
"""
import numpy as np
from pathlib import Path
from . import features as FT

WTS = Path(__file__).resolve().parent / "weights"

_cache = {}  # (n, dim, seed) -> Win; ponytail: build once


def win_mat(n, dim, seed):
    k = (n, dim, seed)
    m = _cache.get(k)
    if m is None:
        m = np.random.default_rng(seed).standard_normal((n, dim)) * 0.5
        _cache[k] = m
    return m


class Head:
    def __init__(self, W, seed=7, steps=2, w=None):
        self.W = W
        self.seed = seed
        self.steps = steps
        self.n = W.shape[0]
        self.dim = len(FT.encode_state([0] * 24, [0, 0], [0, 0], 0))
        self.Win = win_mat(self.n, self.dim, seed)
        if w is None:
            w = np.zeros(self.n + self.dim + 1)  # +1 bias
            # ponytail: sane start, V in [0,1]. diff idx n+29, bias last
            w[self.n + 29] = 0.4
            w[-1] = 0.5
        self.w = np.asarray(w, float)

    def phi(self, board, bar, off, turn):
        u = np.asarray(FT.encode_state(board, bar, off, turn))
        x = np.zeros(self.n)
        for _ in range(self.steps):
            x = np.tanh(self.W @ x + self.Win @ u)
        return np.concatenate([x, u, [1.0]])  # trailing bias

    def choose(self, g, moves, rng=None):
        """1-ply over clones. White max, black min. Ties broken randomly (symmetric)."""
        import random as _r
        rng = rng or _r
        best, bestv = [], None
        for m in moves:
            c = g.clone()
            c.apply(m)
            val = self.w @ self.phi(c.board, c.bar, c.off, c.turn)
            if bestv is None or (g.turn == 0 and val > bestv + 1e-9) or (g.turn == 1 and val < bestv - 1e-9):
                best, bestv = [m], val
            elif abs(val - bestv) <= 1e-9:
                best.append(m)
        return rng.choice(best)

    def policy(self, eps=0.0):
        def pol(g, moves, rng):
            if eps > 0 and rng.random() < eps:
                return rng.choice(moves)
            return self.choose(g, moves, rng)
        return pol

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, w=self.w, seed=self.seed, steps=self.steps, n=self.n, dim=self.dim)

    @classmethod
    def load(cls, path, W):
        z = np.load(path)
        return cls(W, seed=int(z["seed"]), steps=int(z["steps"]), w=z["w"])
