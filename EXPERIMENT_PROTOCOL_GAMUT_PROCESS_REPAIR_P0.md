# GAMUT typed process-repair P0 (frozen protocol)

Date frozen: 2026-08-23 (Asia/Shanghai), before any P0 model output was
generated.

## Question under test

The earlier ASQA experiments tested flat answer-facet sets. They did not show
that a model needs, or can use, relations between obligations. P0 therefore
uses naturally occurring ordered-process requirements in the public GAMUT
text-only split.

The first question is diagnostic: does a frozen model sometimes mention the
individual process steps while failing the required order? If this does not
occur often enough, the proposed process-state problem is not established on
this apparatus and repair results are not interpreted as support.

The second question is interventional: on those natural process failures, does
an explicit typed graph of ordered nodes and directed edges enable a smaller,
safer repair than the same requirement written as ordinary text?

## Frozen apparatus

- Dataset: `facebook/GAMUT`, configuration `default`, split
  `test_text_only`; no images are used. Evidence is restricted to the supplied
  snippets. The released rubric is used under its CC-BY-NC benchmark license
  and is never committed to this repository.
- Structural cases: an Answer-Critical element must explicitly ask for a
  sequence/order and expose a numbered master list from which at least two
  ordered steps can be mechanically parsed. One target is retained per
  example. Cases are SHA-256 ordered with the salt
  `20260823-gamut-process-repair-p0`, and the first 48 are used.
- Model: `Qwen/Qwen2.5-7B-Instruct`, bfloat16, SDPA, greedy decoding. The same
  frozen model supplies a deliberately low-cost A/B requirement check. This is
  a case-study screen, not the final paper metric; any promoted result must be
  replicated with an independent judge and human audit.
- Baseline: question plus fixed evidence only, with no rubric or Guide shown;
  maximum 256 new tokens.
- Binary check: A means fully meets the requirement; B includes partial,
  missing, or contradictory. The readout is `logit(B)-logit(A)` at the answer
  position, with zero as the frozen threshold. A correctly ordered answer and
  an answer containing the same steps in reverse order are scored for every
  target as calibration controls. A content-free answer is not a valid
  negative because GAMUT's sequence item deliberately does not penalize
  omitted steps.
- A relation-only failure means the composite process requirement fails while
  every mechanically extracted component step passes. A broader repair case
  may miss at most one component step, but must mention at least two steps and
  already meet at least one other Answer-Critical requirement. Results are
  always reported separately for relation-only cases.

## 2 x 2 repair arms

All arms see the same question, evidence, saved answer, failed target, and list
of previously met Answer-Critical requirements.

1. `flat_full_rewrite`: ordinary-text target; return a complete rewritten
   answer.
2. `typed_full_rewrite`: the same target plus ordered nodes and directed edges;
   return a complete rewritten answer.
3. `flat_span_patch`: ordinary-text target; return one exact JSON
   `old_text`/`new_text` replacement.
4. `typed_span_patch`: the same typed graph; return one exact JSON replacement.

A span patch is invalid if `old_text` is empty, does not occur exactly once in
the saved answer, or the replacement is empty/over 180 words. Invalid patches
leave the baseline unchanged and remain in the denominator. There is no
retry, fallback rewrite, or post-output prompt change.

## Metrics

- target recovery: the failed process requirement changes to A;
- preservation: fraction of baseline Answer-Critical A requirements that stay
  A;
- no regression: every previously met Answer-Critical requirement stays A;
- safe success: target recovery and no regression both hold;
- edit ratio: one minus token-level `SequenceMatcher` similarity to baseline;
- patch validity and number of distinct candidate texts;
- action Oracle: for each case, whether any of the four frozen candidates is a
  safe success. The Oracle is diagnostic only and never reported as an
  automatic method.

## Frozen decision rules

Apparatus passes only if at least 24 structural targets are found and both
positive- and negative-control accuracy are at least 90%.

The problem is provisionally established only if at least four relation-only
natural failures are found. The repair action space is viable only if the
four-candidate safe-success Oracle is at least 30% on all repair cases.

Typed minimal repair is provisionally supported only if, on all repair cases:

- `typed_span_patch` safe success is at least 25%;
- its no-regression rate is at least 85%;
- it exceeds `flat_span_patch` safe success by at least five points; and
- its median edit ratio is no more than half that of `flat_full_rewrite`, or
  `flat_full_rewrite` has zero median edit ratio because it copied the answer.

Hidden-state selection is considered only if the action Oracle exceeds the
best fixed arm by at least ten points and at least three cases have different
safe-success outcomes across arms. Otherwise P0 explicitly blocks a hidden
selector experiment.

Possible outcomes are `APPARATUS_FAIL`, `PROBLEM_NOT_ESTABLISHED`,
`ACTION_SPACE_FAIL`, `REPAIR_WORKS_STRUCTURE_NOT_ADDED`, and
`STRUCTURED_REPAIR_P0_PASS`.

P0 does not establish benchmark-level performance, causal hidden-state
control, hierarchy beyond one typed process relation, or novelty over prior
checklist/refinement work.
