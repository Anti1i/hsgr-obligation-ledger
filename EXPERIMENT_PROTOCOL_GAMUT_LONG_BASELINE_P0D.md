# GAMUT long-baseline audit P0d

## Purpose

Several P0b answers stop mid-sentence because generation was capped at 256 tokens. P0d tests whether
the apparent missing-node and relation errors survive when the same 7B model receives a 768-token
answer budget. Repair methods remain out of scope until this confound is resolved.

## Frozen comparison

- Same 192 P0b case IDs, questions, fixed evidence, prompts, model and greedy decoding.
- Old arm: saved P0b answer with a 256-token generation cap.
- New arm: regenerate with a 768-token cap.
- Judge: P0c's answer-only `extract_then_check` procedure using Qwen2.5-14B-Instruct. This procedure
  achieved 100% positive-control accuracy, 100% reversed-control accuracy and 100% parse validity
  on the 192 P0c cases.
- Old extractions are reused exactly from P0c; only new answers are judged in P0d.

## Measurements

- Output token count and cap-hit proxy (`>= cap - 2`).
- Fraction with all mandatory process steps detected.
- Fraction with a detected order reversal.
- Relation-only cases: all steps detected but their extracted order is not canonical.
- Paired changes in component completeness and relation status.

## Frozen decision rule

- If the 768-token arm improves all-step coverage by at least 10 percentage points or reduces the
  cap-hit proxy by at least 20 points, P0b is considered materially length-confounded.
- A natural-process repair experiment proceeds only if manual review confirms at least four
  relation-only failures in the 768-token arm.
- Otherwise, GAMUT process order may be retained as a controlled stress test, but not as the natural
  main benchmark for the method.

## Limits

The extraction judge is same-family and uses oracle step descriptions. Cap hits are a proxy rather
than direct generation stop reasons. P0d measures problem incidence, not repair performance, hidden
state utility, automatic obligation induction, or novelty.

