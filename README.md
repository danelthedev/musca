# musca fly-backgammon bot

Fly MaleCNS connectome reservoir plays backgammon via fair_backgammon API.

## Layout

- `brain/` — `downloader.py` (fetch MaleCNS raw), `process.py` (max-brain signed CSR), `flybrain.py` (load + reservoir scoring)
- `engine/` — `bg_rules.py` (legal moves, mirrors `game/state.go`), `client.py` (session/lobby/ws, stdlib only)
- `opponents/` — one file per policy: `random.py` (more later)
- `bot.py` — demo runner
- `scripts/` — eval helpers (later)

## Run

```sh
# terminal 1: API
cd ../fair_backgammon && go run .
# terminal 2: fly vs fly
cd musca && .venv/bin/python bot.py
# fly vs random baseline
cd musca && .venv/bin/python bot.py random
```

`BG_URL` overrides API base (default `http://localhost:8080`).

## Brain data

- `brain/downloader.py` fetches ~1.1GB raw into `data/malecns/raw/`
- `brain/process.py` builds max-brain signed `connectome.npz` (191,696 bodies, ~25M edges, no superclass drops)
- `data/` stays gitignored; without it bot falls back to seeded synthetic reservoir
