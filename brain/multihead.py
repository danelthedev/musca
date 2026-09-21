"""Multi-head MBON compartments over the frozen reservoir. Pure fly anatomy:
5 linear readouts (win, win-gammon, win-bg, lose-gammon, lose-bg), white
perspective, sharing one frozen reservoir (Kenyon cells). Dopamine-style TD
tunes each compartment. Equity E = 2Pw-1 + (Pwg-Plg) + (Pwb-Plb); 1-ply
argmax on E. numpy only.
"""
import numpy as np
from pathlib import Path
from . import value as V

HEADS = ("win", "wg", "wb", "lg", "lb")
# sane starting biases: base win rate 0.5, gammon ~0.2, backgammon ~0.05
BIAS0 = {"win": 0.5, "wg": 0.2, "wb": 0.05, "lg": 0.2, "lb": 0.05}


def _clamp(v):
    return max(0.0, min(1.0, v))


class MultiHead:
    def __init__(self, W, seed=7, steps=2, ws=None):
        self.base = V.Head(W, seed=seed, steps=steps)
        self.W = W
        self.seed = seed
        self.steps = steps
        self.n = self.base.n
        self.dim = self.base.dim
        d = self.n + self.dim + 1
        self.ws = {}
        for h in HEADS:
            if ws and h in ws:
                self.ws[h] = np.asarray(ws[h], float)
            else:
                w = np.zeros(d)
                w[-1] = BIAS0[h]
                self.ws[h] = w

    def phi(self, board, bar, off, turn):
        return self.base.phi(board, bar, off, turn)

    def probs(self, phi):
        return {h: _clamp(float(self.ws[h] @ phi)) for h in HEADS}

    def equity(self, phi):
        p = self.probs(phi)
        return 2 * p["win"] - 1 + (p["wg"] - p["lg"]) + (p["wb"] - p["lb"])

    def choose(self, g, moves, rng=None):
        """1-ply on equity. White max, black min. Ties broken randomly."""
        import random as _r
        rng = rng or _r
        best, bestv = [], None
        for m in moves:
            c = g.clone()
            c.apply(m)
            val = self.equity(self.phi(c.board, c.bar, c.off, c.turn))
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
        np.savez(p, seed=self.seed, steps=self.steps, n=self.n, dim=self.dim,
                 **{f"w_{h}": self.ws[h] for h in HEADS})

    @classmethod
    def load(cls, path, W):
        z = np.load(path)
        return cls(W, seed=int(z["seed"]), steps=int(z["steps"]),
                   ws={h: z[f"w_{h}"] for h in HEADS})
