"""MLP value head over frozen reservoir features. V(s) = sigmoid(W2.tanh(W1.phi+b1)+b2).

numpy only. TD(lambda) with eligibility traces (accumulating). Save/load npz.
Go port needs only two matvecs + tanh/sigmoid.
"""
import numpy as np


class Net:
    def __init__(self, head, hidden=64, seed=7, params=None):
        self.head = head  # frozen reservoir features (brain.value.Head)
        self.dim = head.n + head.dim + 1
        self.hidden = hidden
        if params is None:
            rng = np.random.default_rng(seed)
            self.W1 = rng.standard_normal((hidden, self.dim)) * np.sqrt(1.0 / self.dim)
            self.b1 = np.zeros(hidden)
            self.W2 = rng.standard_normal(hidden) * np.sqrt(1.0 / hidden)
            self.b2 = 0.0
        else:
            self.W1, self.b1, self.W2, self.b2 = (np.asarray(params[k], float) for k in ("W1", "b1", "W2", "b2"))
        self._zero_traces()

    def _zero_traces(self):
        self.eW1 = np.zeros_like(self.W1)
        self.eB1 = np.zeros_like(self.b1)
        self.eW2 = np.zeros_like(self.W2)
        self.eB2 = 0.0

    def forward(self, phi):
        z = self.W1 @ phi + self.b1
        h = np.tanh(z)
        a = self.W2 @ h + self.b2
        v = 1.0 / (1.0 + np.exp(-a))
        cache = (phi, z, h, a, v)
        return v, cache

    def v(self, board, bar, off, turn):
        return self.forward(self.head.phi(board, bar, off, turn))[0]

    def grad(self, cache):
        """dV/dparams at cached forward pass."""
        phi, z, h, a, v = cache
        dsig = v * (1.0 - v)
        dtanh = 1.0 - h * h
        dW2 = dsig * h
        dB2 = dsig
        back = dsig * self.W2 * dtanh
        dW1 = np.outer(back, phi)
        dB1 = back
        return dW1, dB1, dW2, dB2

    def td_step(self, phi, v_next, alpha, lam):
        """One TD update from stored features phi with bootstrap v_next. Returns delta."""
        v, cache = self.forward(phi)
        delta = max(-1.0, min(1.0, v_next - v))
        dW1, dB1, dW2, dB2 = self.grad(cache)
        self.eW1 = lam * self.eW1 + dW1
        self.eB1 = lam * self.eB1 + dB1
        self.eW2 = lam * self.eW2 + dW2
        self.eB2 = lam * self.eB2 + dB2
        scale = 1.0 + (float((self.eW1 ** 2).sum()) + float((self.eB1 ** 2).sum())
                       + float((self.eW2 ** 2).sum()) + self.eB2 ** 2) / self.dim
        self.W1 += alpha * delta * self.eW1 / scale
        self.b1 += alpha * delta * self.eB1 / scale
        self.W2 += alpha * delta * self.eW2 / scale
        self.b2 += alpha * delta * self.eB2 / scale
        return delta

    def choose(self, g, moves, rng=None):
        import random as _r
        rng = rng or _r
        best, bestv = [], None
        for m in moves:
            c = g.clone()
            c.apply(m)
            val = self.v(c.board, c.bar, c.off, c.turn)
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
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 seed=self.head.seed, steps=self.head.steps, n=self.head.n, hidden=self.hidden)

    @classmethod
    def load(cls, path, head):
        z = np.load(path)
        return cls(head, hidden=int(z["hidden"]),
                   params={"W1": z["W1"], "b1": z["b1"], "W2": z["W2"], "b2": z["b2"]})
