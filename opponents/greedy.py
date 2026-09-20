"""Greedy heuristic opponent: linear weights over move_features. Intermediate gate.

Stronger than random (hits, makes points, bears off), weaker than trained value.
No deps. Works as local-sim policy pol(g, moves, rng).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brain.features import move_features

# weights over [progress, opp_pip, hit, made, -blot, off, bar_entry]; higher better
W = [1.0, 0.5, 1.5, 1.0, 1.0, 1.5, 0.3]


def policy(g, moves, rng):
    best, bestv = [], None
    for m in moves:
        c = g.clone()
        c.apply(m)
        f = move_features(g, m, c, g.turn)
        val = sum(wi * fi for wi, fi in zip(W, f)) + rng.random() * 1e-6
        if bestv is None or val > bestv:
            best, bestv = [m], val
    return rng.choice(best)
