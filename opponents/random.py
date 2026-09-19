"""Opponent policies: fn(moves, rng) -> move or None. Add new file per policy."""
import random as _r


def choose(moves, rng=None):
    """Random legal move. Baseline opponent for fly."""
    if not moves:
        return None
    return (rng or _r).choice(moves)
