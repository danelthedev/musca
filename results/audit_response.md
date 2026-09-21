# Auditor objection -> evidence (round 2)

1. "fly.json still only untrained/trained/expert" — FALSE now.
   fly.json weights keys: expert, league2, student_league, trained, untrained.
   Export log: extra variants: ['league2', 'student_league'].
   Live server lists variants=[untrained trained expert league2 student_league].

2. "Impure decision code remains compiled in" — REMOVED.
   Deleted: fly/search.go, search_test.go, race.go, race_test.go,
   mlp.go, mlp_test.go, mlp.json, testdata/em_vectors.json, mlp_vectors.json.
   grep for ExpertChoose/MlpChoose/RaceEquity/RolloutEquity/Expectimax/ExpValue
   across fly/ api/ lobby/ game/ tests/ main.go: CLEAN (only stale
   go_test_results.txt mentioned them; file regenerated, 3 PASS).
   export_go.py em/mlp sections removed.

3. "P3 no linear-distillation script/log" — script: scripts/expert_iterate.py
   (linear ridge-toward-teacher, usage-matched). Log:
   results/phase3_distill_train.log (329935 positions, RMSE 0.0290,
   STUDENT vs TEACHER 115/200=57.5%). Eval: phase3_studleague_eval.log
   (85.0% greedy, 57.0% vs league, 54.5% vs expert).

4. "P4 league2 missing logs" — results/phase4_league2_train.log (FINAL 98.0%
   vs random, best 100%) and results/phase4_league2_eval.log
   (91.0% greedy, 57.0% vs studleague, 59.5% vs league).

5. "Verdict missing" — staged, needs human play (by design: blind vs user).
   Protocol docs/match_protocol.md; blind variants fly-a/fly-b via
   FLY_A_VARIANT/FLY_B_VARIANT env override (api/handlers.go); seal
   results/verdict_seal.log; sheet results/blind_match.md; staging verified
   live (room 9JHU, BOT-MOVED x3). Awaiting user to play first-to-5.

6. "Go full suite FAIL TestDoubleAcceptScoresStake" — pre-existing, unrelated:
   results/note_pristine_fail.log; tests/ lobby/ game/ untouched
   (git diff --name-only: api/handlers.go, bin, fly/* only).
