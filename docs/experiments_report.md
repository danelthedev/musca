# Experiments report: fly strength program

All numbers measured locally unless noted. Weights in musca/brain/weights*.

## Scoreboard baselines (200 games, 1-ply, alternating colors)
| model | vs greedy | vs random | prime | stack |
|---|---|---|---|---|
| 38-dim head (pre-phase-2) | 63.0% | 97.5% | ~1.4 | ~6.0 |
| trained 52-dim (500 games) | 74.0% | 95.5% | 1.53 | 6.23 |
| expert 52-dim (10k equity) | 89.5% | 99.5% | 1.94 | 5.15 |
| league 52-dim (100k equity) | 86.7% | 99.0% | 2.27 | 4.54 |

## Head-to-head (200 games unless noted)
- 2-ply vs 1-ply, same expert weights: 28/40 (70%).
- expert-2ply vs trained-1ply: 76/100.
- student (linear distill) vs teacher: 172/300 = 57.3%.
- distilled-MLP vs teacher: 111/200 = 55.5%.
- MLP-TD (7k games) vs expert: 71/200 = 35.5%.
- league-100k vs expert 1-ply: 96/200 = 48.0% (volume alone did not pass expert).

## Test positions (results/test_positions.json: 8 openings + 20 hard)
- expert 1-ply: 0/28 (hard 0/20 by construction; openings non-standard).
- distilled-MLP 1-ply: 9/28 (hard 9/20).
- expert 2-ply: 20/28 (hard 20/20).

## Search (Go, results in fly/go_test_results.txt)
- TestChooseMatchesPython, TestExpectimaxMatchesPython 100/100,
  TestExpectimaxLegal, race properties: PASS.
- ExpertChoose (race-aware + rollout tiebreak): ~66ms/decision.
- Go mlp.go mirrors brain/mlp.py (TestMlpMatchesPython PASS).

## Arena variants (fly → Play vs Fly)
- untrained: prior weights, 1-ply. Fly.
- trained: best.npz, 1-ply. Fly+.
- expert: expert.npz + ExpertChoose (2-ply, race equity, rollout tiebreaks). Fly*.
- All three verified with live server games (fly/variant_verification.log).

## Conclusions
- 2-ply search is the largest single strength contributor.
- Distillation (linear and MLP) absorbs search knowledge into weights.
- 100k-game volume improved structure (prime/stack) but not head-to-head vs expert.
- Next loop rounds: distill league/expert-2ply targets into MLP; league with
  snapshot pool; selective 3-ply. Stop after two consecutive no-gain rounds.
