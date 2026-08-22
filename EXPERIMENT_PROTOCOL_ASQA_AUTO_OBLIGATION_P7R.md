# ASQA automatic obligation parser replay P7r (frozen protocol)

Date frozen: 2026-08-22 (Asia/Shanghai), after P7x job 749083 and its frozen
case-level diagnostic, before any P7r coverage score or append was generated.

## Motivation and permitted repair

P7x marked 18/73 induction outputs invalid. Complete inspection showed that
none were missing node sets: 17 outputs used four lines, each a valid
one-element JSON array, while one four-element array used the non-JSON `\'`
escape inside English possessives. The frozen P7x parser accepted one valid
four-element JSON array or a numbered list, so it replaced all 18 semantically
complete sets with the original question. P7r repairs only these two exhausted
serialization mismatches.

## Frozen replay

- Reuse every raw P7x induction output. Do not regenerate induction text.
- Keep the original P7x parser first. For a P7x-invalid output only, first
  normalize `\'` to a literal apostrophe and accept it only if this yields one
  valid four-element array; otherwise accept exactly four nonempty lines when
  every line is a JSON array containing one string. In both cases the four
  cleaned strings must be unique and at most 35 words.
- Reuse the 220 frozen candidate coverage scores and appends from the 55
  originally valid cases exactly.
- For the recovered 18 cases only, score the 72 recovered obligations with the
  unchanged P6x A/B readout and generate one unchanged 96-token append each.
- Recompute automatic-logit, hash-random, and post-hoc action-Oracle selections
  on all 73 cases. Replay the frozen P7x/P6x gold Oracle, gold logit, and generic
  rows.
- Retain every P7x scientific gate and threshold unchanged.

## Outcomes

- `APPARATUS_FAIL`: not all 73 outputs recover to exact four-node sets, frozen
  rows do not align, scores are non-finite, or P6x baselines do not replay;
- `AUTO_OBLIGATION_LEDGER_PASS`: every unchanged P7x ledger gate passes;
- `INDUCED_ACTION_ONLY`: the unchanged induced action-Oracle gate passes but an
  automatic selection/action gate fails;
- `AUTO_OBLIGATION_FAIL`: the repaired induced-set Oracle still fails.

P7r is a parser repair, not evidence for a new method variant. No prompt,
obligation text, selector, generation instruction, metric, threshold, or
outcome rule may change after P7r output is observed.
