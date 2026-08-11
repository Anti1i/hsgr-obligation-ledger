# Route-Counterfactual Hidden Guide: frozen development protocol v1

Frozen on 2026-08-11 before extracting any structure-de-oracled route-counterfactual
features.  The remaining 377 MuSiQue problems stay sealed.  This document may
be superseded only by a new, versioned protocol written before the affected
result is observed; failed gates must not be relaxed retroactively.

## 1. Claim and non-claims

### Testable claim

Given a fixed pool of answer candidates, fixed open-book evidence, and a
hierarchy predicted without gold structure annotations, the frozen language
model's hidden response to a controlled
predecessor-state intervention supplies candidate-selection information beyond
vote mass, explicit/text features, an ordinary hidden correctness verifier,
and a non-structural activation-delta verifier.

For candidate `a_i`, let the two verifier prompts be identical in question,
full evidence, candidate, format, and token budget.  They differ only in the
predicted predecessor assignment:

```
z_i = concat_l([
    h_l(a_i, predicted matched predecessor state)
      - h_l(a_i, predicted counterfactual predecessor state),
    h_l(a_i, predicted matched predecessor state)
      * h_l(a_i, predicted counterfactual predecessor state)
])

score_i = vote_i + explicit_i + lambda * Guide(z_i, predicted hierarchy)
```

The Guide changes trajectory/candidate commitment only.  It does not modify
generation-time residuals, attention, or token probabilities.

### Explicit non-claims

The method is not claimed to introduce any of the following in isolation:

- best-of-N verification or verifier-weighted voting;
- probing hidden states for answer correctness;
- start-to-end or temporal activation-delta reranking;
- process reward modelling from correct/incorrect trajectories;
- hierarchical or topology-aware reward propagation;
- counterfactual data augmentation in general;
- generation-time hidden-state steering;
- hierarchy prediction, decomposition, retrieval, or credit assignment.

The candidate contribution is narrower: a verifier reads the *within-candidate
hidden response to an intervention on a predicted dependency predecessor
state*, and this response is tested with route-specific causal controls.

## 2. Overlap boundary with current methods

| Existing method family | Already covered by prior work | What must be additional here |
|---|---|---|
| Outcome verifier / Best-of-N | train a verifier and select one generated completion | improvement beyond an equal-data ordinary verifier |
| Hidden correctness probe | predict correctness from an unperturbed hidden state | improvement from the predecessor intervention contrast, not absolute state |
| Hidden activation delta / centroid reranking | rank by a trajectory's internal start-to-end change | improvement beyond an equal-dimension non-structural delta baseline |
| Process reward model | label or synthesize correct/error reasoning steps | no labelled textual error is injected; the same candidate is observed under a predecessor-state intervention |
| Hierarchical/topological reward | propagate reward over steps or a graph | no reward propagation; hierarchy defines the intervention variable only |
| Counterfactual supervision | construct paired correct/error trajectories | both arms preserve candidate text and evidence; the primary arm must be fully predicted and must pass route swap/mismatch controls |

If a full related-work audit finds an earlier method with the same intervention,
representation contrast, and selection role, novelty is not established and
the paper claim must be narrowed or abandoned regardless of accuracy.

Primary sources included in the pre-experiment overlap audit:

- Cobbe et al., *Training Verifiers to Solve Math Word Problems*:
  <https://arxiv.org/abs/2110.14168>
- Zhang et al., *Reasoning Models Know When They're Right*:
  <https://arxiv.org/abs/2504.05419>
- Liang et al., *CLUE: Non-parametric Verification from Experience via
  Hidden-State Clustering*: <https://arxiv.org/abs/2510.01591>
- Chi and Wang, *Verifiable Counterfactual Supervision for Process Reward
  Models*: <https://arxiv.org/abs/2605.02395>
- Wang et al., *Towards Hierarchical Multi-Step Reward Models*:
  <https://arxiv.org/abs/2503.13551>
- Feng et al., *RewardFlow: Topology-Aware Reward Propagation on State Graphs*:
  <https://arxiv.org/abs/2603.18859>

This is an initial boundary audit, not a claim that the literature search is
exhaustive.

## 3. Data boundary

- **Development only:** the previously consumed MuSiQue sets of 200 + 320 +
  320 problems (840 total).  They are pooled for problem-disjoint nested OOF;
  the previously observed "second 320" is no longer treated as held-out.
- **Sealed confirmatory set:** the remaining 377 problems.  No prompts,
  candidates, structures, features, labels, or aggregate metrics may be read
  before every development gate in section 7 passes.
- Candidate pools remain the frozen SC@8 outputs already generated for each
  consumed problem.  Candidate labels are used only for training/evaluation,
  never in hierarchy prediction, predecessor execution, prompt construction,
  or feature extraction.
- All split membership is by problem ID.  No candidate from one problem may
  appear in another fold.

## 4. No structure-oracle primary condition

The primary method must satisfy all of the following at inference time:

1. hierarchy/dependency state is predicted from the original question and the
   same fixed open-book evidence available to SC@8;
2. predecessor values are model predictions, not decomposition annotations;
3. both intervention arms receive the identical dataset-provided open-book
   evidence seen by SC@8; neither receives a paragraph-to-hop mapping or a
   separately routed support subset;
4. no gold hop count, gold subquestion, gold dependency edge, gold predecessor
   answer, answer alias, or final answer is used to construct either prompt;
5. the counterfactual is generated deterministically from the predicted state
   without consulting candidate correctness;
6. all candidates for a problem use the same predicted hierarchy and
   predecessor states; only the proposed final answer differs.

This is a selection experiment under an open-book evidence setting, not a
retrieval claim.  An oracle-decomposition or oracle-route arm may be reported
only as a labelled mechanism ceiling outside the primary result.

## 5. Frozen baselines and controls

All learned baselines use identical outer folds, inner validation policy,
candidate labels, layers/projection dimension, training epochs, and policy
weight search unless the baseline definition makes a component inapplicable.

1. `SC@8`: majority vote on the frozen candidates.
2. `explicit predicted-state`: vote plus deterministic text features derived
   only from the predicted state.
3. `ordinary hidden verifier`: the absolute hidden state from the same matched
   predicted-state prompt, with no predecessor contrast.
4. `activation-delta verifier`: an equal-dimension hidden delta between two
   fixed positions in that same matched prompt; predecessor assignment is not
   changed.
5. `non-hidden listwise`: count, normalized answer length, predicted-state text
   features, and the same listwise objective, but no LM hidden vectors.
6. `matched Guide`: the proposed route-counterfactual hidden response.
7. `route swap`: reverse matched and counterfactual arms at evaluation.
8. `route mismatch`: pair each candidate's matched state with another problem's
   counterfactual state within the same hop stratum.
9. `state-label permutation`: permute predicted hierarchy/depth labels across
   training problems while retaining all candidate labels.
10. `length/count`: single-feature controls.

The primary manuscript comparison is not allowed to omit baselines 3 and 4.

## 6. Compute and measurement validity

- Report candidate-generation cost and Guide-verification cost separately.
- Count prompt tokens, generated tokens, verifier forward tokens, and model
  calls per problem.  Re-sent evidence counts every time it is processed.
- Accuracy comparisons use the same frozen SC@8 candidate pool, so claimed
  gains are selection gains, not generation gains.
- Report an equal-total-token SC baseline when enough cached candidates exist;
  otherwise report the accuracy/compute Pareto curve and do not call SC@8
  compute-matched.
- Primary endpoint: normalized exact match per problem.  Token F1 is secondary
  and cannot reverse a failed primary decision.
- Report exact paired McNemar counts/tests, 95% paired bootstrap confidence
  intervals, and 2/3/4-hop strata.  The problem, not the candidate, is the
  statistical unit.
- The three primary superiority tests (vs SC@8, ordinary hidden verifier, and
  activation-delta verifier) use Holm correction at family-wise alpha 0.05.
- Hyperparameters and checkpoints are selected inside the training portion of
  each outer fold.  An outer-fold label may not influence its feature
  construction, early stopping, layer choice, policy weight, or threshold.

## 7. Frozen development gates

Every gate must pass on nested OOF over the consumed 840 problems:

1. **Selection value:** matched Guide is at least +2.0 percentage points over
   SC@8 and has Holm-adjusted `p < 0.05`.
2. **Beyond explicit/non-hidden:** at least +1.0 point over both the explicit
   predicted-state and equal-procedure non-hidden policies.
3. **Beyond ordinary hidden verification:** at least +1.0 point over the
   ordinary hidden verifier, with Holm-adjusted `p < 0.05`.
4. **Beyond generic activation delta:** at least +1.0 point over the
   activation-delta verifier, with Holm-adjusted `p < 0.05`.
5. **Route directionality:** at least +2.0 points over route swap.
6. **Route coupling:** at least +1.0 point over route mismatch.
7. **Hierarchy dependence:** at least +1.0 point over state-label permutation.
8. **Depth signature:** gain over SC@8 is non-negative for every hop stratum,
   and the 4-hop gain is no more than 1.0 point below the 2-hop gain.
9. **OOF stability:** matched Guide minus SC@8 is positive in at least four of
   five outer folds and in both pre-declared ID-hash halves.
10. **No structure-oracle and accounting audit:** section 4 has no violation and compute
    counters are complete.

Only after all ten gates pass may one fixed code commit/configuration be run
once on the 377-problem confirmatory set.  A paper-level general claim further
requires a second multi-hop QA dataset with the same structure-de-oracled
protocol.

## 8. Stop conditions

- Failure of gates 3 or 4 means the result is a generic hidden verifier, not a
  distinct HSGR Guide; stop the main-method claim.
- Failure of gates 5--7 means the signal is not causally tied to the predicted
  hierarchy; stop the hierarchy-specific claim.
- Failure of gate 8 repeats the established weak-depth pattern; do not repair
  it with post-hoc per-hop weights.
- Do not resume online residual/attention steering based on a reranking result.
