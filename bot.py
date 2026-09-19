"""Fly bot: connectome reservoir picks moves. Demo: two bots play full game."""
import sys, threading, time
import engine.client as C
import engine.bg_rules as R
import brain.flybrain as F
from opponents.random import choose as random_choose

def play_one(ws, W, idx, seed, log, stop, policy="fly", steps=2):
    import random as _rand
    rng = _rand.Random(seed)
    last_state = None
    while not stop[0]:
        try:
            m = ws.recv(timeout=30)
        except Exception as e:
            log.append(f"p{idx} recv end: {e}")
            return
        t = m.get("t")
        if t == "win":
            log.append(f"WIN {m}")
            stop[0] = True
            return
        if t != "state":
            if t == "error":
                msg = str(m.get("msg", ""))
                if "already rolled" in msg or "not your turn" in msg:
                    continue
                log.append(f"p{idx} ERROR {m}")
                stop[0] = True
                return
            continue
        if m.get("turn") != idx:
            continue
        if m.get("doubleOffer"):
            continue
        sk = (tuple(m["board"]), tuple(m["bar"]), tuple(m.get("movesLeft") or []), m.get("hasRolled"), m.get("turn"))
        if sk == last_state:
            time.sleep(0.1)
            continue
        last_state = sk
        if not m.get("hasRolled"):
            ws.send({"t": "roll"})
            continue
        if "legalMoves" in m:  # ponytail: server rules authoritative, fallback local
            moves = [{"from": int(x["from"]), "to": int(x["to"]), "die": int(x["die"])} for x in m["legalMoves"] or []]
        else:
            moves = R.legal_moves(m["board"], m["bar"], m.get("movesLeft") or [], True, idx)
        if not moves:
            ws.send({"t": "pass"})
            time.sleep(0.2)
            continue
        best = random_choose(moves, rng) if policy == "random" else F.score_moves(W, m["board"], m["bar"], m["off"], idx, moves, seed=seed, steps=steps)[0]
        tag = "rnd" if policy == "random" else "fly"
        log.append(f"p{idx}({tag}) move {best} dice={m.get('dice')} left={m.get('movesLeft')}")
        ws.send({"t": "move", "from": best["from"], "to": best["to"], "die": best["die"]})

def demo(opp="fly", steps=2):
    W = F.load_connectome() if opp != "rndonly" else None
    ck0 = C.session("fly0")
    opp_policy = "random" if opp == "random" else "fly"
    opp_name = "rnd1" if opp == "random" else "fly1"
    code = C.create_lobby(ck0)
    ck1 = C.session(opp_name)
    C.join_lobby(code, ck1)
    ws0 = C.WS(code, ck0)
    ws1 = C.WS(code, ck1)
    log, stop = [], [False]
    t0 = threading.Thread(target=play_one, args=(ws0, W, 0, 7, log, stop, "fly", steps), daemon=True)
    t1 = threading.Thread(target=play_one, args=(ws1, W, 1, 21, log, stop, opp_policy, steps), daemon=True)
    t0.start(); t1.start()
    t0.join(timeout=180); t1.join(timeout=10)
    stop[0] = True
    ws0.close(); ws1.close()
    errs = [l for l in log if "ERROR" in l]
    print(f"code={code} moves={len([l for l in log if 'move' in l])} errors={len(errs)}")
    for l in log[-30:]:
        print(" ", l)
    if errs:
        sys.exit(1)
    if not any("WIN" in l for l in log) and len(log) < 5:
        print("WARN: no win yet, game incomplete")
        sys.exit(2)

if __name__ == "__main__":
    demo(sys.argv[1] if len(sys.argv) > 1 else "fly", int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 2)
