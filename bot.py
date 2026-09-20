"""Fly bot: connectome reservoir picks moves. Demo: two bots play full game."""
import sys, threading, time
import engine.client as C
import engine.bg_rules as R
import brain.flybrain as F
from engine.sim import Game as SimGame
from brain import value as V
from opponents.random import choose as random_choose

def play_one(ws, W, idx, seed, log, stop, policy="fly", steps=2, head=None):
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
        if policy == "random":
            best, tag = random_choose(moves, rng), "rnd"
        elif policy == "value":  # trained head; server state -> sim.Game
            g = SimGame()
            g.board, g.bar, g.off = list(m["board"]), list(m["bar"]), list(m["off"])
            g.turn, g.moves_left, g.has_rolled = idx, list(m.get("movesLeft") or []), True
            best, tag = head.choose(g, moves, rng), "val"
        else:
            best = F.score_moves(W, m["board"], m["bar"], m["off"], idx, moves, seed=seed, steps=steps)[0]
            tag = "fly"
        log.append(f"p{idx}({tag}) move {best} dice={m.get('dice')} left={m.get('movesLeft')}")
        ws.send({"t": "move", "from": best["from"], "to": best["to"], "die": best["die"]})

def load_side(weights, n):
    """W (+head, policy) for one seat. Untrained fly loads full W; weights force slice."""
    W = F.load_connectome()
    if weights and weights != "prior":
        import numpy as _np
        n = int(_np.load(weights)["n"])  # weights dim must match W exactly
    if n is not None and W.shape[0] > n:
        W = W[:n, :n].tocsr()
    if not weights:
        return W, None, "fly"
    head = V.Head(W, seed=7) if weights == "prior" else V.Head.load(weights, W)
    return W, head, "value"


def play_seat(base, code, name, W, head, policy="fly", steps=2):
    """Single vs-fly seat. Color from server players (rematch swaps); survives wins."""
    import random as _rand
    import socket as _sock
    C.BASE = base
    rng = _rand.Random(hash((code, name)) & 0xFFFF)
    for _ in range(30):  # reconnect budget
        try:
            ck = C.session(name)
            C.join_lobby(code, ck)  # idempotent for our seat
            ws = C.WS(code, ck)
        except Exception as e:
            print(f"fly {name}@{code}: connect fail {e}", flush=True)
            time.sleep(2)
            continue
        try:
            last_state, want_rematch, last_progress, was_rolled = None, False, time.time(), None
            while True:
                try:
                    m = ws.recv(timeout=60)
                except _sock.timeout:
                    if time.time() - last_progress > 90:  # stale seat? reconnect for fresh state
                        break
                    continue  # idle while human thinks
                except Exception:
                    break  # reconnect
                t = m.get("t")
                if t == "win":
                    if not want_rematch:  # accept rematch so humans can replay
                        ws.send({"t": "rematch"})
                        want_rematch = True
                    last_state = None  # new game starts fresh; keep seat
                    continue
                if t == "opponent_left":
                    return
                if t != "state":
                    continue
                players = m.get("players") or []
                if name not in players:
                    continue
                idx = players.index(name)
                if m.get("turn") != idx or m.get("doubleOffer"):
                    continue
                if last_state is None and list(m.get("off") or []) == [0, 0]:
                    want_rematch = False  # fresh game after rematch
                sk = (tuple(m["board"]), tuple(m["bar"]), tuple(m.get("movesLeft") or []), m.get("hasRolled"), m.get("turn"))
                if sk == last_state:
                    time.sleep(0.2)
                    continue
                first_move = was_rolled is False and m.get("hasRolled")
                last_state, last_progress, was_rolled = sk, time.time(), m.get("hasRolled")
                if not m.get("hasRolled"):
                    ws.send({"t": "roll"})
                    continue
                if "legalMoves" in m:  # ponytail: server rules authoritative
                    moves = [{"from": int(x["from"]), "to": int(x["to"]), "die": int(x["die"])} for x in m["legalMoves"] or []]
                else:
                    moves = R.legal_moves(m["board"], m["bar"], m.get("movesLeft") or [], True, idx)
                if not moves:
                    ws.send({"t": "pass"})
                    time.sleep(0.2)
                    continue
                if policy == "value":
                    g = SimGame()
                    g.board, g.bar, g.off = list(m["board"]), list(m["bar"]), list(m["off"])
                    g.turn, g.moves_left, g.has_rolled = idx, list(m.get("movesLeft") or []), True
                    best = head.choose(g, moves, rng)
                else:
                    best = F.score_moves(W, m["board"], m["bar"], m["off"], idx, moves, seed=hash(name) & 0xFFFF, steps=steps)[0]
                time.sleep(0.9 if first_move else 0.25)  # dice animation finishes first
                ws.send({"t": "move", "from": best["from"], "to": best["to"], "die": best["die"]})
        finally:
            try:
                ws.close()
            except Exception:
                pass
        time.sleep(2)
    print(f"fly {name}@{code}: giving up", flush=True)

def demo(opp="fly", steps=2, weights=None, n=None):
    W = F.load_connectome() if opp != "rndonly" else None
    head, fly_policy = None, "fly"
    if W is not None and (weights or n is not None):
        W, head, fly_policy = load_side(weights, n)
    ck0 = C.session("fly0")
    opp_policy = "random" if opp == "random" else "fly"
    opp_name = "rnd1" if opp == "random" else "fly1"
    code = C.create_lobby(ck0)
    ck1 = C.session(opp_name)
    C.join_lobby(code, ck1)
    ws0 = C.WS(code, ck0)
    ws1 = C.WS(code, ck1)
    log, stop = [], [False]
    t0 = threading.Thread(target=play_one, args=(ws0, W, 0, 7, log, stop, fly_policy, steps, head), daemon=True)
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

if __name__ == "__main__":  # bot.py [opp] [steps] [--w path|prior] [--n N] [--join CODE --name N --url U]
    a, pos, weights, n, join, name, url = sys.argv[1:], [], None, None, None, "Fly", None
    i = 0
    while i < len(a):
        if a[i] == "--w" and i + 1 < len(a):
            weights, i = a[i + 1], i + 2
        elif a[i] == "--n" and i + 1 < len(a):
            n, i = int(a[i + 1]), i + 2
        elif a[i] == "--join" and i + 1 < len(a):
            join, i = a[i + 1], i + 2
        elif a[i] == "--name" and i + 1 < len(a):
            name, i = a[i + 1], i + 2
        elif a[i] == "--url" and i + 1 < len(a):
            url, i = a[i + 1], i + 2
        else:
            pos.append(a[i]); i += 1
    if join:  # vs-fly seat: join room, play one color, survive rematches
        W, head, policy = load_side(weights, n)
        play_seat(url or C.BASE, join, name, W, head, policy)
    else:
        demo(pos[0] if pos else "fly", int(pos[1]) if len(pos) > 1 and pos[1].isdigit() else 2, weights, n)
