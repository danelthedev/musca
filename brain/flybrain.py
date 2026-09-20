"""Fly connectome -> reservoir matrix. Tries real data, falls back synthetic."""
from pathlib import Path
import numpy as np
import scipy.sparse as sp

DATA = Path(__file__).resolve().parent.parent / "data" / "malecns"
N = 1000  # reservoir size, tiny RAM

def _load_real():
    for p in [DATA / "connectome.npz", DATA / "edges_kept.parquet", DATA / "weights.parquet"]:
        if p.exists():
            return p
    return None

def load_connectome(n=None, seed=7):
    """Return signed sparse CSR. Real full matrix if present else seeded synthetic."""
    real = _load_real()
    if real and real.suffix == ".npz":
        try:
            m = sp.load_npz(real)
            if n is None:
                return m.tocsr()
            if m.shape[0] >= n:
                return m[:n, :n].tocsr()
            return m.tocsr()
        except Exception:
            pass
    if n is None:
        n = N
    rng = np.random.default_rng(seed)
    # 5% density, signs: 70% excite / 30% inhibit (fly-like E/I)
    rows = rng.integers(0, n, size=n * 50)
    cols = rng.integers(0, n, size=n * 50)
    vals = np.where(rng.random(n * 50) < 0.7, 1.0, -1.0) * rng.uniform(0.1, 1.0, n * 50)
    w = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    # spectral scaling ~0.9 for stable reservoir
    w = (w * (0.9 / max(1e-6, abs(w).sum(axis=1).mean()))).tocsr()
    return w

def encode_board(board, bar, off, turn):
    """28-dim input vector from game state."""
    b = np.asarray(board, dtype=float) / 5.0
    v = np.concatenate([b, np.asarray(bar, float) / 2.0, np.asarray(off, float) / 15.0,
                        np.array([1.0 if turn == 0 else -1.0])])
    return v[:28] if len(v) >= 28 else np.pad(v, (0, 28 - len(v)))

_CACHE = {}  # (kind, n, dim, seed) -> matrix; ponytail: build once


def _win(n, dim, seed):
    k = ("win", n, dim, seed)
    m = _CACHE.get(k)
    if m is None:
        m = np.random.default_rng(seed).standard_normal((n, dim)) * 0.5
        _CACHE[k] = m
    return m


def _readout(n, seed):
    k = ("readout", n, seed)
    m = _CACHE.get(k)
    if m is None:
        m = np.random.default_rng(seed + 99).standard_normal((n, 4)) * 0.3
        _CACHE[k] = m
    return m


def reservoir_state(W, u, seed=1, steps=2):
    """Run x=tanh(Wx+Win*u). Deterministic."""
    Win = _win(W.shape[0], len(u), seed)
    x = np.zeros(W.shape[0])
    for _ in range(steps):
        x = np.tanh(W @ x + Win @ u)
    return x

def score_moves(W, board, bar, off, turn, moves, seed=1, steps=2):
    """Rank legal moves via reservoir readout. Returns moves sorted best-first."""
    if not moves:
        return []
    u = encode_board(board, bar, off, turn)
    x = reservoir_state(W, u, seed=seed, steps=steps)
    R = _readout(W.shape[0], seed)  # readout: [from,to,die,pip]
    scored = []
    for m in moves:
        f = np.array([m["from"] / 24.0, (m["to"] + 2) / 26.0, m["die"] / 6.0,
                      _pip_delta(board, bar, m, turn) / 50.0])
        scored.append((float(x @ (R @ f)), m))
    scored.sort(key=lambda t: -t[0])
    return [m for _, m in scored]

def _pip_delta(board, bar, m, turn):
    if m["to"] == -2:
        return -float(m["die"])  # bearing off always good, mover-agnostic
    return 0.0
