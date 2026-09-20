"""Play N games, record results to JSONL. Usage: eval.py [N] [opp] [jobs] [steps] [out] [--w path|prior] [--n N]

Fly side defaults to untrained reservoir (policy fly). --w switches it to the
1-ply value head: a weights .npz (trained) or 'prior' (pip-prior, untrained). --n
slices the connectome (auto-sized to weights when --w names a file)."""
import ast
import sys
import threading
import time
from pathlib import Path

import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bot as B
import brain.flybrain as F
import engine.client as C


def one_game(W, gid, opp, timeout=240, steps=2, fly_policy="fly", head=None):
    t0 = time.time()
    rec = {"game": gid, "opp": opp, "winner": None, "winnerName": None,
           "mult": None, "reason": None, "moves": 0, "errors": 0, "secs": 0.0, "code": None, "fly": fly_policy}
    try:
        ck0 = C.session("fly0")
        code = C.create_lobby(ck0)
        opp_name = "rnd1" if opp == "random" else "fly1"
        ck1 = C.session(opp_name)
        C.join_lobby(code, ck1)
        rec["code"] = code
        ws0 = C.WS(code, ck0)
        ws1 = C.WS(code, ck1)
        log, stop = [], [False]
        opp_policy = "random" if opp == "random" else "fly"
        tA = threading.Thread(target=B.play_one, args=(ws0, W, 0, 1000 + gid, log, stop, fly_policy, steps, head), daemon=True)
        tB = threading.Thread(target=B.play_one, args=(ws1, W, 1, 2000 + gid, log, stop, opp_policy, steps), daemon=True)
        tA.start(); tB.start()
        tA.join(timeout=timeout); tB.join(timeout=10)
        stop[0] = True
        try:
            ws0.close(); ws1.close()
        except Exception:
            pass
        rec["moves"] = len([l for l in log if " move " in l])
        rec["errors"] = len([l for l in log if "ERROR" in l])
        for l in log:
            if l.startswith("WIN"):
                try:
                    w = ast.literal_eval(l[4:].strip())
                    rec.update(winner=w.get("winner"), winnerName=w.get("winnerName"),
                               mult=w.get("mult"), reason=w.get("reason"))
                except Exception:
                    rec["winnerName"] = l
    except Exception as e:
        rec["errors"] = 1
        rec["reason"] = f"harness: {e}"
    rec["secs"] = round(time.time() - t0, 1)
    rec["wall"] = round(time.time(), 1)  # end epoch; start = wall - secs
    return rec


def main():
    import concurrent.futures as _cf
    a, pos, weights, wn = sys.argv[1:], [], None, None
    i = 0
    while i < len(a):  # strip flags; rest stays positional
        if a[i] == "--w" and i + 1 < len(a):
            weights, i = a[i + 1], i + 2
        elif a[i] == "--n" and i + 1 < len(a):
            wn, i = int(a[i + 1]), i + 2
        else:
            pos.append(a[i]); i += 1
    n = int(pos[0]) if len(pos) > 0 else 100
    opp = pos[1] if len(pos) > 1 else "random"
    jobs = int(pos[2]) if len(pos) > 2 else 4
    steps, out_arg = 2, None
    if len(pos) > 3:  # steps or out name
        if pos[3].isdigit():
            steps = int(pos[3])
        else:
            out_arg = pos[3]
    if len(pos) > 4:
        out_arg = pos[4]
    wtag = "" if not weights else ("_prior" if weights == "prior" else "_" + Path(weights).stem)
    if out_arg:  # explicit output name
        out = Path(out_arg)
        out = out if out.parent != Path(".") else Path(__file__).resolve().parent.parent / "results" / out.name
    else:
        out = Path(__file__).resolve().parent.parent / "results" / f"eval_{opp}_{n}_s{steps}{wtag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    have, fly_wins_resumed = 0, 0
    if out.exists():
        for line in open(out):
            try:
                r0 = json.loads(line)
                have += 1
                fly_wins_resumed += (r0.get("winner") == 0)
            except Exception:
                pass
        print(f"resuming: {have} games already in {out.name}")
    from brain import value as Vh
    W = F.load_connectome()
    head, fly_policy = None, "fly"
    if weights:
        if weights != "prior":  # slice W to trained size
            import numpy as _np
            wn = int(_np.load(weights)["n"])
        if wn is not None and W.shape[0] > wn:
            W = W[:wn, :wn].tocsr()
        head = Vh.Head(W, seed=7) if weights == "prior" else Vh.Head.load(weights, W)
        fly_policy = "value"  # wtag already set during flag parsing
    elif wn is not None and W.shape[0] > wn:
        W = W[:wn, :wn].tocsr()
    if fly_policy == "value":
        head.phi([0] * 24, [0, 0], [0, 0], 0)  # warm cache
    else:
        F.score_moves(W, [0]*24, [0,0], [0,0], 0, [{"from": 5, "to": 3, "die": 2}])  # warm cache
    print(f"eval: {n} games fly({fly_policy}{wtag}) vs {opp} x{jobs}, W={W.shape}")
    lock, done = threading.Lock(), [0]
    fly_wins = [fly_wins_resumed]
    def run(gid):
        rec = one_game(W, gid, opp, steps=steps, fly_policy=fly_policy, head=head)
        with lock:
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            done[0] += 1
            fly_wins[0] += (rec["winner"] == 0)
            print(f"game {gid + 1}/{n}: winner={rec['winnerName']} moves={rec['moves']} "
                  f"errors={rec['errors']} {rec['secs']}s [{done[0]}/{n - have}] fly={fly_wins[0]}", flush=True)
        return rec
    with _cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        list(ex.map(run, range(have, n)))
    print(f"\nFINAL fly {fly_wins[0]}/{n} vs {opp} -> {out}")


if __name__ == "__main__":
    main()
