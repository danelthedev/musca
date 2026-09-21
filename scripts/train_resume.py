"""Resumable TD(lambda) trainer: arbitrary reservoir (top-left n or --slice),
CPU or GPU (CUDA) rollout decisions, exact stop/continue.

Usage: train_resume.py [--n 512 | --slice data/slice_10k_idx.npy]
                [--gamesA 300] [--gamesB 200] [--gamesC 300]
                [--workers 4] (parallel game generation; default 1 = exact sequential)
                [--alpha 1e-4] [--lambda 0.7] [--eps0 0.15] [--eps1 0.02]
                [--seed 1] [--eval-every 100] [--eval-games 20]
                [--ckpt-every 50] [--w0 weights.npz] [--device cpu|cuda]
                [--outdir brain/weights_10k] [--resume outdir]

Resume: --resume loads state.npz (weights, game counter, curriculum position,
RNG state, snapshot pool, best) and continues the exact schedule. Stop anytime;
checkpoints land every --ckpt-every games plus best_*.npz on improvement.
GPU path: rollout decisions via brain/gpu.py (parity-gated); traces stay numpy.
"""
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import sim
from opponents import greedy as G
from brain import value as V
from brain import flybrain as F
from brain import features as FT


def parse(a):
    o = dict(n=512, slice=None, gamesA=300, gamesB=200, gamesC=300, alpha=1e-4,
             lam=0.7, eps0=0.15, eps1=0.02, seed=1, eval_every=100,
             eval_games=20, ckpt_every=50, w0=None, device="cpu",
             outdir=str(ROOT / "brain" / "weights_10k"), resume=None, workers=1)
    i = 0
    while i < len(a):
        k = a[i]
        if k.startswith("--"):
            k = k[2:].replace("-", "_")
            if k == "lambda":
                k = "lam"
            if k in o or k in ("w0", "outdir", "resume", "slice"):
                v = a[i + 1]
                o[k] = v if k in ("w0", "outdir", "resume", "slice") or isinstance(o.get(k), str) \
                    else type(o[k])(v)
                i += 2
            else:
                raise SystemExit(f"unknown flag {a[i]}")
        else:
            raise SystemExit(f"unexpected arg {a[i]}")
    return o


def load_W(o):
    import numpy as np
    W = F.load_connectome()
    if o["slice"]:
        idx = np.load(o["slice"])
        return W[idx][:, idx].tocsr()
    n = o["n"]
    return W[:n, :n].tocsr() if W.shape[0] >= n else W


def eval_vs(head, opp_fn, n_games, seed, chooser=None):
    fly = chooser or head.policy()
    w = 0
    for s in range(n_games):
        r = random.Random(seed + s)
        if s % 2 == 0:
            win, _, _, _ = sim.play_game(fly, opp_fn, r)
            w += (win == 0)
        else:
            win, _, _, _ = sim.play_game(opp_fn, fly, r)
            w += (win == 1)
    return w / n_games


def save_state(out, head, prog):
    import numpy as np
    np.savez(out / "state.npz", w=head.w, total=np.array(prog["total"]),
             stage=np.array(prog["stage"]), done=np.array(prog["done"]),
             sdone=np.array(prog["sdone"]),
             seed=np.array(prog["seed"]), best=np.array(prog["best"]),
             snap_w=prog["snap_w"],
             rngstate=np.array(prog["rng"].getstate(), dtype=object),
             allow_pickle=True)


def load_state(out):
    import numpy as np
    z = np.load(out / "state.npz", allow_pickle=True)
    prog = {"total": int(z["total"]), "stage": int(z["stage"]),
            "done": int(z["done"]), "seed": int(z["seed"]),
            "best": float(z["best"]), "snap_w": z["snap_w"],
            "sdone": list(z["sdone"]) if "sdone" in z else [0, 0, 0]}
    r = random.Random()
    r.setstate(tuple(z["rngstate"]))
    prog["rng"] = r
    return z["w"], prog


from brain import pargen as _pargen
from brain.pargen import play_game_data

def train_loop_par(head, stages, o, prog, out, pool, gh):
    """Parallel game generation, ordered updates (stale-sync). Games generate in
    worker processes with wave-start weights; TD updates apply on main in
    exact game order. Snapshot waves align to absolute multiples of 50 ==
    sequential refresh points. Only difference vs sequential: decisions use
    weights up to one chunk stale (small alpha -> validate curves)."""
    import numpy as np
    a_eff = o["alpha"] * (2000.0 / head.n)
    lam = o["lam"]
    e = np.zeros_like(head.w)
    span = sum(g for _, g, _ in stages)
    use_cuda = o["device"] == "cuda"
    workers = pool._max_workers
    CH = max(8, 4 * workers)
    si = 0
    req = [g for _, g, _ in stages]
    while si < len(stages) and prog["sdone"][si] >= req[si]:
        si += 1
    prog["stage"] = si
    while si < len(stages):
        tag, games, kind = stages[si]
        while prog["sdone"][si] < games:
            if kind == "snap":
                boundary = ((prog["total"] // 50) + 1) * 50
                take = max(1, min(games - prog["sdone"][si], boundary - prog["total"]))
                snap_w = np.array(prog["snap_w"], float)
                snap_ver = prog["total"]
            else:
                take = min(games - prog["sdone"][si], CH)
                snap_w, snap_ver = prog["snap_w"], -1
            base = prog["total"]
            w0 = head.w.copy()
            tasks = []
            for k in range(take):
                total = base + k
                frac = total / max(1, span)
                eps = o["eps0"] + (o["eps1"] - o["eps0"]) * frac
                tasks.append((total, kind, w0, snap_w, snap_ver, eps,
                              o["seed"], use_cuda, head.steps))
            for total, states, reward in pool.map(_pargen._gen, tasks):
                if gh is not None:
                    phis = [p for p in gh.phi_batch(np.array(
                        [FT.encode_state(b, bar, off, t) for (b, bar, off, t) in states]))]
                else:
                    phis = [head.phi(b, bar, off, t) for (b, bar, off, t) in states]
                vs = [float(head.w @ p) for p in phis]
                e[:] = 0.0
                for t in range(len(phis)):
                    v_next = reward if t == len(phis) - 1 else vs[t + 1]
                    delta = max(-1.0, min(1.0, v_next - vs[t]))
                    e = lam * e + phis[t]
                    scale = 1.0 + float(np.dot(e, e)) / len(e)
                    head.w += a_eff * delta * e / scale
                prog["total"] += 1
                prog["sdone"][si] += 1
                if kind == "snap" and prog["total"] % 50 == 0:
                    prog["snap_w"] = head.w.copy()
                if prog["total"] % 25 == 0:
                    print(f"  [{tag}] total {prog['total']}", flush=True)
                if o["ckpt_every"] and prog["total"] % o["ckpt_every"] == 0:
                    head.save(out / f"ckpt_{prog['total']}.npz")
                    save_state(out, head, prog)
                if o["eval_every"] and prog["total"] % o["eval_every"] == 0:
                    sc = eval_vs(head, sim.random_policy, o["eval_games"], 99999,
                                 chooser=(lambda g_, m_, r_: gh.choose_batch([(g_, m_)], r_)[0]) if gh else None)
                    print(f"  eval@{prog['total']}: {100 * sc:.1f}% vs random", flush=True)
                    if sc > prog["best"]:
                        prog["best"] = sc
                        head.save(out / "best.npz")
                        print("  new best -> best.npz", flush=True)
        prog["stage"] += 1
        head.save(out / f"stage_{si}.npz")
        save_state(out, head, prog)
        si += 1
    return head

def train_loop(head, stages, o, prog, out, gh=None, gh_snap=None, snap_head=None):
    import numpy as np
    a_eff = o["alpha"] * (2000.0 / head.n)
    lam = o["lam"]
    e = np.zeros_like(head.w)
    span = sum(g for _, g, _ in stages)
    rng = prog["rng"]
    snap_pol = prog.get("snap_pol")

    def trainee_choose(g, moves):
        if gh is not None:
            gh.w_t.copy_(__import__("torch").from_numpy(head.w).to(gh.dev))
            return gh.choose_batch([(g, moves)], rng)[0]
        return head.choose(g, moves, rng)

    si = 0  # recompute from sdone: allows extending the schedule on resume
    req = [g for _, g, _ in stages]
    while si < len(stages) and prog["sdone"][si] >= req[si]:
        si += 1
    prog["stage"] = si
    while si < len(stages):
        tag, games, kind = stages[si]
        opp_fn = {"random": sim.random_policy, "greedy": G.policy,
                  "snap": snap_pol}[kind]
        gi0 = prog["sdone"][si]
        for gi in range(gi0, games):
            frac = prog["total"] / max(1, span)
            eps = o["eps0"] + (o["eps1"] - o["eps0"]) * frac
            trainee_white = (prog["total"] % 2 == 0)
            states, reward = play_game_data(
                lambda g_, m_, r_: trainee_choose(g_, m_),
                opp_fn, trainee_white, eps, rng)
            if gh is not None:
                phis = [p for p in gh.phi_batch(np.array(
                    [FT.encode_state(b, bar, off, t) for (b, bar, off, t) in states]))]
            else:
                phis = [head.phi(b, bar, off, t) for (b, bar, off, t) in states]
            vs = [float(head.w @ p) for p in phis]
            e[:] = 0.0
            for t in range(len(phis)):
                v_next = reward if t == len(phis) - 1 else vs[t + 1]
                delta = max(-1.0, min(1.0, v_next - vs[t]))
                e = lam * e + phis[t]
                scale = 1.0 + float(np.dot(e, e)) / len(e)
                head.w += a_eff * delta * e / scale
            prog["total"] += 1
            prog["sdone"][si] += 1
            if kind == "snap" and prog["total"] % 50 == 0:
                prog["snap_w"] = head.w.copy()
                if snap_head is not None:
                    snap_head.w = head.w.copy()
                if gh_snap is not None:
                    import torch
                    gh_snap.w_t.copy_(torch.from_numpy(head.w).to(gh_snap.dev))
            if prog["total"] % 25 == 0:
                print(f"  [{tag}] game {gi + 1}/{games} (total {prog['total']})", flush=True)
            if o["ckpt_every"] and prog["total"] % o["ckpt_every"] == 0:
                head.save(out / f"ckpt_{prog['total']}.npz")
                save_state(out, head, prog)
            if o["eval_every"] and prog["total"] % o["eval_every"] == 0:
                sc = eval_vs(head, sim.random_policy, o["eval_games"], 99999,
                             chooser=(lambda g_, m_, r_: gh.choose_batch([(g_, m_)], r_)[0]) if gh else None)
                print(f"  eval@{prog['total']}: {100 * sc:.1f}% vs random", flush=True)
                if sc > prog["best"]:
                    prog["best"] = sc
                    head.save(out / "best.npz")
                    print("  new best -> best.npz", flush=True)
        prog["stage"] += 1
        head.save(out / f"stage_{si}.npz")
        save_state(out, head, prog)
        si += 1
    return head


def main():
    o = parse(sys.argv[1:])
    t0 = time.time()
    out = Path(o["outdir"])
    out.mkdir(parents=True, exist_ok=True)
    W = load_W(o)
    stages = [("A-vs-random", o["gamesA"], "random"),
              ("B-vs-greedy", o["gamesB"], "greedy"),
              ("C-selfplay", o["gamesC"], "snap")]
    pool = None
    if o["workers"] > 1:
        import os
        from concurrent.futures import ProcessPoolExecutor
        os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
        pool = ProcessPoolExecutor(max_workers=o["workers"],
                                   initializer=_pargen.winit, initargs=(W,))
    gh = gh_snap = snap_head = None
    if o["resume"] and (Path(o["resume"]) / "state.npz").exists():
        w0, prog = load_state(Path(o["resume"]))
        head = V.Head(W, seed=7, w=w0)
        print(f"resumed: total={prog['total']} stage={prog['stage']} done={prog['done']} best={100 * prog['best']:.1f}%")
    else:
        head = V.Head(W, seed=7)
        if o["w0"]:
            head = V.Head.load(o["w0"], W)
        prog = {"total": 0, "stage": 0, "done": 0, "seed": o["seed"],
                "best": 0.0, "snap_w": head.w.copy(), "rng": random.Random(o["seed"]),
                "sdone": [0, 0, 0]}
        prog["best"] = -1.0
        head.save(out / "stage_init.npz")
    if o["device"] == "cuda":
        from brain import gpu as Gmod
        gh = Gmod.GPUHead(W, head.Win, head.w, steps=head.steps)
        gh_snap = Gmod.GPUHead(W, head.Win, prog["snap_w"], steps=head.steps)
        prog["snap_pol"] = lambda g_, m_, r_: gh_snap.choose_batch([(g_, m_)], r_)[0]
        ev_chooser = lambda g_, m_, r_: gh.choose_batch([(g_, m_)], r_)[0]
    else:
        snap_head = V.Head(W, seed=7, w=prog["snap_w"].copy())
        prog["snap_pol"] = lambda g_, m_, r_: (r_.choice(m_) if r_.random() < 0.05
                                               else snap_head.choose(g_, m_, r_))
        ev_chooser = None
    if prog["best"] < 0 and prog["total"] == 0:
        prog["best"] = eval_vs(head, sim.random_policy, min(20, o["eval_games"]), 99999, chooser=ev_chooser)
        print(f"start eval: {100 * prog['best']:.1f}% vs random")
    print(f"train_resume: n={head.n} dim={head.dim} device={o['device']} "
          f"alpha_eff={o['alpha'] * 2000.0 / head.n:.2e} lam={o['lam']}")
    try:
        if pool is not None:
            train_loop_par(head, stages, o, prog, out, pool, gh)
        else:
            train_loop(head, stages, o, prog, out, gh, gh_snap, snap_head)
    finally:
        if pool is not None:
            pool.shutdown()
    sc = eval_vs(head, sim.random_policy, 100, 99999, chooser=ev_chooser)
    print(f"FINAL: {100 * sc:.1f}% vs random (100), best {100 * prog['best']:.1f}%, "
          f"{round(time.time() - t0, 1)}s -> {out}")


if __name__ == "__main__":
    main()
