# GAMUT relation-judge calibration P0c

## Purpose

P0b cannot be interpreted because its 14B A/B judge accepted only 33.3% of deliberately
reversed answers as wrong. P0c isolates two plausible causes before any more repair claims:
single-label preference and leakage from showing the gold evidence to the judge.

This is an evaluator calibration study, not a method-performance experiment.

## Frozen inputs

- The 192 fresh P0b cases and their saved 7B baseline answers.
- The same oracle process steps parsed before P0/P0b.
- The positive control lists steps in canonical order.
- The negative control lists the same steps in reverse order.
- The 14B judge remains independent from the 7B answer generator, but is from the same model family.

## Judge variants

1. `original_direct`: the already saved P0b single A/B score with gold evidence visible.
2. `counterbalanced_with_evidence`: average semantic margins from two prompts whose A/B meanings
   are swapped. Gold evidence remains visible.
3. `counterbalanced_answer_only`: the same counterbalancing, but the judge sees only the answer
   and requirement.
4. `extract_then_check`: without gold evidence, generate JSON containing the matched step IDs in
   answer order; code, rather than the model, checks whether the IDs are increasing.

For the extraction variant, missing steps follow the GAMUT rubric and do not by themselves fail
the relation. A malformed extraction is never counted as a correct control.

## Frozen gates

- A judge is usable only if positive-control accuracy is at least 95% and negative-control
  accuracy is at least 95% over all 192 cases.
- `extract_then_check` additionally requires at least 95% parse validity.
- Natural relation-only failures are screened only with usable judges.
- The GAMUT process slice supports a follow-up repair study only if at least four of 192 natural
  baseline answers contain all mandatory steps but put at least two in the wrong order.
- Disagreements between usable judges require human inspection before a repair run.

## Interpretation limits

Passing controls does not make an LLM judge equivalent to human annotation. Oracle steps are still
used, and the study neither establishes automatic graph induction nor hidden-state control. If fewer
than four relation-only failures remain, the correct conclusion is that this GAMUT slice does not
instantiate the proposed relation-repair problem often enough—not that the repair method failed.

