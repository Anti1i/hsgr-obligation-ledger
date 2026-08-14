# ASQA set-Guide answer-boundary patch P4x (frozen protocol)

Date frozen: 2026-08-14 (Asia/Shanghai), after P1x job 733287, P2x job
733309, and P3x job 733401, but before any P4x model output was generated.

## 1. Claim under test

P1x showed that a complete gold facet checklist improves answer coverage. P2x
did not find useful node-level readout in answer prefixes. P3x then showed that
a single facet intervention focuses its named facet but crowds out other
facets, so choosing one node is the wrong action space.

P4x tests a narrower set-level mediation claim:

> The semantic difference between the complete correct facet set and a
> matched wrong facet set is represented at the answer-boundary hidden state,
> and adding that per-problem difference to the checklist-free prompt can
> causally recover a material part of the correct checklist's coverage gain.

This is an oracle latent-mediation screen. Gold facet questions are used to
construct the Guide in separate source forward passes. The generation prompt
itself contains no checklist. Passing P4x would not make activation addition,
steering vectors, hidden routing, or gold facet outlines novel, and would not
yet define a deployable HSGR method.

## 2. Frozen examples and split

Reconstruct the exact 192 clean P1x cases selected by ascending SHA-256 of
`20260815-clean-p1x|released_id`, together with the exact P1x same-facet-count,
nearest-word-count decoy mapping. Require the P1x generation file to contain
exactly one row for every case and each of `closedbook`, `fixed_direct`,
`true_facets`, and `decoy_facets`.

Order the 192 cases by ascending SHA-256 of
`20260815-asqa-set-guide-p4x|released_id`. The first 64 cases form the
calibration set and the remaining 128 cases form the untouched heldout set.
No P4x layer or strength decision may use heldout outputs.

## 3. Model, Guide, and intervention

- Model: `Qwen/Qwen2.5-7B-Instruct`, bfloat16, SDPA, greedy decoding.
- Maximum 192 new tokens, no sampling, retrieval, reranking, judge model,
  regeneration, or answer-dependent modification.
- Candidate transformer blocks are zero-based layers 13, 20, and 27.
- Candidate strengths are alpha in {0.5, 1.0, 2.0}.
- Fixed documents, chat template, prompt text, tokenizer, and scorer are
  identical to P1x.

For case `i` and block `l`, separately run the complete true-checklist and
matched decoy-checklist prompts and capture their last-prompt-token states
immediately after block `l`:

`g(i,l) = h_true(i,l) - h_decoy(i,l)`.

Generate from the `fixed_direct` prompt. On the first model call only (the
prefill), add `alpha * g(i,l)` to the final prompt token immediately after
block `l`. Do not patch any document token or any autoregressively generated
token. Gold aliases never enter a prompt or hidden feature.

## 4. Calibration and heldout controls

Run all nine layer/alpha combinations on the 64 calibration cases. Select one
combination by, in order: highest STR-HIT, highest STR-EM, smaller alpha, then
shallower layer. This rule is fixed even if every combination hurts.

Run only the selected combination on the 128 heldout cases under three arms:

1. `correct_guide`: the case's own `g(i,l)`;
2. `wrong_guide`: another heldout case's Guide, matched to the same facet
   count by the deterministic ordering
   `20260815-asqa-set-guide-p4x-wrong|source_id|candidate_id`, excluding self,
   and rescaled to the norm of the source case's correct Guide;
3. `random_guide`: a deterministic isotropic Gaussian direction seeded by
   `20260815-asqa-set-guide-p4x-random|case_id|layer`, rescaled to the same
   norm as the correct Guide.

The existing P1x `fixed_direct`, `true_facets`, and `decoy_facets` generations
are the paired no-patch baseline and textual ceilings. Controls are never used
to choose the layer or alpha.

## 5. Frozen metrics and gates

Use the frozen ALCE-normalized alias-substring scorer:

- `STR-EM`: mean facet coverage per problem;
- `STR-HIT`: fraction of problems covering every facet.

Report all calibration cells, all heldout arms, exact paired two-sided
McNemar tests, word lengths, Guide norms, two fixed P1x ID-hash halves, and
facet-count strata. Report results even after any failed gate.

### Apparatus gates

1. exactly 427 eligible cases, exactly the 192 P1x-selected cases, a 64/128
   disjoint calibration/heldout split, and exactly 768 aligned P1x rows;
2. all selected cases have 2--6 unique facets, five fixed documents, a
   same-count non-self decoy, and every expected hidden state is finite;
3. on the full 192 P1x rows, `true_facets - fixed_direct` is at least +5
   points STR-HIT and +3 points STR-EM;
4. the target generation prompts are byte-for-byte `fixed_direct` prompts and
   the hook is applied exactly once per generated sequence, during prefill.

### Heldout causal-mediation gates

5. `correct_guide - fixed_direct` is at least +5 points STR-HIT;
6. `correct_guide - fixed_direct` is at least +3 points STR-EM;
7. the correct Guide recovers at least 50% of the positive heldout
   `true_facets - fixed_direct` STR-HIT gain;
8. `correct_guide - wrong_guide` is at least +5 points STR-HIT;
9. `correct_guide - random_guide` is at least +5 points STR-HIT;
10. the exact paired two-sided McNemar p-value for correct Guide versus direct
    STR-HIT is below 0.05, with more correct-only successes;
11. `correct_guide - fixed_direct` STR-HIT is strictly positive in both fixed
    P1x ID-hash halves;
12. correct-Guide median answer length is 30--160 words and differs from the
    direct median by at most 40 words.

## 6. Outcomes and stop rule

- `SET_LATENT_MEDIATION_PASS`: all 12 gates pass. This supports a causal
  set-level latent mediator and licenses a separately frozen experiment that
  replaces gold facet-derived Guides with a test-time-available set encoder or
  structural state estimator. It does not establish novelty or a final method.
- `APPARATUS_FAIL`: any gate 1--4 fails. Repair only the predeclared apparatus;
  do not interpret intervention scores.
- `SET_LATENT_MEDIATION_FAIL`: gates 1--4 pass but any gate 5--12 fails. Do not
  continue with per-example answer-boundary activation addition on this
  apparatus. Diagnose whether the failure is lack of latent mediation,
  insufficient single-position causal leverage, or non-specific perturbation,
  using only the frozen wrong/random controls and reported calibration cells.

The experiment may not be rescued by changing the split, layers, alpha grid,
prompt, hidden position, controls, metric, or thresholds after heldout results
are observed. Any materially different intervention requires a new protocol
and new heldout cases.
