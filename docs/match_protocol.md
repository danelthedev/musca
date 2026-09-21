# Blind match protocol: human verdict

Goal: decide whether the fly "reasonably beats a human player".

## Setup
- Mode: Play vs Fly (doubling disabled, both sides).
- Anonymization: the two candidate variants run under neutral botnames
  (`Fly-A`, `Fly-B`), assigned randomly per match. Handler maps variant by
  env override, not by the select list, so the player cannot tell which is which.
- Match: first to 5 game wins. Alternate colors each game (server rematch
  already swaps colors).

## Rules
- The player plays normally, no engine help.
- Either side may resign a clearly lost game (counts as a win).
- Disconnects: replay the game, same colors.

## Scoring
- Winner = first to 5. Record per-game: winner, mult/reason, plies, variant key
  (sealed until the match ends).
- Bar: challenger needs 5-2 or better to claim "beats me" (a 5-4 squeak
  triggers one more match round, swapped anonymization).

## After
- Unseal variant keys, publish the sheet to results/blind_match.md.
- A decisive loss sends the fly back to the improvement loop; a decisive win
  closes the verdict phase.
