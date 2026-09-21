"""Parallel game generation for train_resume.py --workers N.

Lives in its own importable module because worker entry points must pickle by
reference (forkserver start method): scripts run as __main__ are not importable
in children. Parent sets PYTHONPATH so children find this package; W ships once
per worker via initializer (pickled, ~35MB at 10k — one-time cost).
"""
from engine import sim
from opponents import greedy as G
from brain import features as FT

_WG = {}  # worker globals: W (once), Win, gh, snap, snapver


def winit(W):
    _WG["W"] = W
    _WG["Win"] = None
    _WG["gh"] = None
    _WG["snap"] = None
    _WG["snapver"] = -1


def play_game_data(trainee_fn, opp_fn, trainee_white, eps, rng):
    """Play one full game, record post-move states. Shared by sequential and
    parallel paths: same call order, same RNG consumption, same result.
    trainee_fn/opp_fn take (g, moves, rng). Returns (states, reward) with
    states = list of (board, bar, off, turn) tuples; reward white-perspective."""
    g = sim.Game()
    states = []
    while True:
        mover = g.turn
        g.roll(rng)
        if not g.has_any_legal():
            g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn
            continue
        while True:
            moves = g.legal_moves()
            if not moves:
                break
            is_trainee = (mover == 0) == trainee_white
            if is_trainee and rng.random() < eps:
                m = rng.choice(moves)
            elif is_trainee:
                m = trainee_fn(g, moves, rng)
            else:
                m = opp_fn(g, moves, rng)
            g.apply(m)
            states.append((list(g.board), list(g.bar), list(g.off), g.turn))
            win, wnr = g.check_win()
            if win:
                mult, _ = g.win_multiplier(wnr)
                return states, ((mult / 3.0) if wnr == 0 else 0.0)
            if not g.moves_left or not g.has_any_legal():
                break
        if g.check_technical_win(mover):
            return states, ((2.0 / 3.0) if mover == 0 else 0.0)
        g.moves_left, g.has_rolled, g.turn = [], False, 1 - g.turn


def _gen(task):
    """Generate one game in a worker. task = (total, kind, w, snap_w, snap_ver,
    eps, seed, use_cuda, steps). Returns (total, states, reward). Per-game
    seeded RNG -> reproducible; path differs from --workers 1 (documented).
    Decisions use wave-start weights (stale-sync, chunk-bounded)."""
    import random as _r2
    import numpy as _np
    from brain import value as _V
    total, kind, w, snap_w, snap_ver, eps, seed, use_cuda, steps = task
    W = _WG["W"]
    rng = _r2.Random(seed * 1000003 + total)
    if use_cuda:
        if _WG.get("gh") is None:
            from brain import gpu as _G
            _WG["Win"] = _V.Head(W, seed=7).Win
            _WG["gh"] = _G.GPUHead(W, _WG["Win"], _np.array(w, float), steps=steps)
            _WG["snap"] = _G.GPUHead(W, _WG["Win"], _np.array(w, float), steps=steps)
        import torch as _t
        _dev = _WG["gh"].dev
        _WG["gh"].w_t.copy_(_t.from_numpy(_np.array(w, float)).to(_dev))
        if _WG["snapver"] != snap_ver:
            _WG["snap"].w_t.copy_(_t.from_numpy(_np.array(snap_w, float)).to(_dev))
            _WG["snapver"] = snap_ver
        trainee = lambda g_, m_, r_: _WG["gh"].choose_batch([(g_, m_)], r_)[0]
        snap_pol = lambda g_, m_, r_: _WG["snap"].choose_batch([(g_, m_)], r_)[0]
    else:
        head = _V.Head(W, seed=7)
        head.w = _np.array(w, float)
        snap_head = _V.Head(W, seed=7, w=_np.array(snap_w, float))
        trainee = lambda g_, m_, r_: head.choose(g_, m_, r_)
        snap_pol = lambda g_, m_, r_: (r_.choice(m_) if r_.random() < 0.05
                                       else snap_head.choose(g_, m_, r_))
    opp = {"random": sim.random_policy, "greedy": G.policy, "snap": snap_pol}[kind]
    states, reward = play_game_data(trainee, opp, (total % 2 == 0), eps, rng)
    return (total, states, reward)
