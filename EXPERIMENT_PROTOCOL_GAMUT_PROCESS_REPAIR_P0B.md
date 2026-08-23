# GAMUT fresh independent-judge process repair P0b (frozen protocol)

Date frozen: 2026-08-23 (Asia/Shanghai), after P0 job 751023 was diagnosed and
before any P0b model output was generated.

## Why P0b exists

P0 found 241 parseable GAMUT process cases, so the benchmark contains a real
ordered-relation apparatus. However, its same-7B A/B judge failed both frozen
control gates (79.17% correct-order and 89.58% reverse-order accuracy), and only
one of 48 baselines was repair eligible. Manual inspection of that one case
showed that the judge accepted a flat rewrite that still placed grounding
before skidding/jacking, while a typed rewrite actually restored the required
order. Exact-string JSON patches were also invalid because the generator copied
text from evidence rather than an exact span of the saved answer.

P0b repairs those measurement/action interfaces without reusing the observed
case for evaluation.

## Frozen apparatus

- Reconstruct the same SHA-256 ordering as P0. Exclude the first 48 P0 cases
  and use the next 192 parseable process cases. This is a fresh, disjoint set.
- Generator: frozen `Qwen/Qwen2.5-7B-Instruct`, greedy, 256-token answer and
  repair budgets.
- Judge: frozen `Qwen/Qwen2.5-14B-Instruct`, greedy A/B logit readout. A means
  fully meets; B combines partial/missing/contradictory. The judge never
  generates answers or repairs. The zero logit-difference threshold is frozen.
- The baseline prompt, evidence cap, target parsing, component checks,
  relation-only definition, and previously-met Answer-Critical preservation
  set are unchanged from P0.
- Correct-order and reverse-order controls are retained for every case. Both
  must reach at least 95% accuracy. Otherwise P0b is `APPARATUS_FAIL` and no
  method comparison is interpreted.

## Repair arms

The full-rewrite arms are unchanged:

1. `flat_full_rewrite` sees the ordinary-text target.
2. `typed_full_rewrite` additionally sees ordered nodes and directed edges.

The minimal arms use sentence addresses instead of exact copied strings:

3. `flat_sentence_patch` returns JSON with one-based `start_sentence`,
   `end_sentence`, and `replacement` using the ordinary target.
4. `typed_sentence_patch` uses the same action schema plus the typed graph.

The indices must select one to four consecutive sentences in range, and the
replacement must be nonempty and at most 180 words. An invalid action leaves
the baseline unchanged and remains in the denominator. There is no repair
retry or fallback rewrite.

## Metrics and frozen gates

Metrics are target recovery, preservation of every previously met
Answer-Critical requirement, safe success (both), edit ratio, patch validity,
candidate diversity, and the post-hoc four-candidate safe-success Oracle.
Relation-only cases are always reported separately.

Apparatus gates require 192 fresh cases, zero overlap with P0, both controls at
least 95%, and a complete candidate denominator. The problem gate requires at
least four natural relation-only failures. The action-space gate requires at
least 30% safe-success Oracle.

Typed sentence repair is provisionally supported only if it reaches at least
25% safe success and 85% no-regression, exceeds flat sentence repair by at
least five points, and has no more than half the median edit ratio of flat full
rewrite (unless flat full rewrite has zero median edit ratio).

Hidden-state selection is allowed only if the Oracle exceeds the best fixed
arm by at least ten points and at least three cases disagree across arms.

Outcomes are `APPARATUS_FAIL`, `PROBLEM_NOT_ESTABLISHED`,
`ACTION_SPACE_FAIL`, `REPAIR_WORKS_STRUCTURE_NOT_ADDED`, and
`STRUCTURED_REPAIR_P0B_PASS`.

P0b remains an oracle-rubric, same-family-model case study. It does not claim
automatic obligation induction, independent human validation, benchmark-level
performance, causal hidden-state control, or novelty.
