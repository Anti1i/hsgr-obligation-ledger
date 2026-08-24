# P0j: Can stale obligation verdicts be predicted after an answer revision?

## Status and scope

This protocol is frozen before the first P0j model run. P0j is a controlled
case study, not a claim about natural error prevalence and not a test of
whether witness overlap is a causal mechanism.

P0f/P0g established that repairing one part of an answer can invalidate a
previously satisfied requirement. P0h did not support sentence-level support
co-location as the explanation. P0j therefore asks the narrower engineering
question: after a revision, can we identify which **old satisfied,
non-target** verdicts need to be recomputed?

## Unit and label

Each saved answer has 12 atomic obligations and exactly one deliberately
failed target. A revision tries to repair that target. The target is always
rejudged and is excluded from the prediction dataset. Only parse-valid
revisions for which that mandatory check confirms target recovery enter the
stale-verdict prediction set; failed repair attempts are rejected rather than
treated as committed state transitions.

For every other obligation `i` that was satisfied before the revision:

`stale_i = 1` iff a full post-revision judgment says `i` is no longer met.

Thus one revision contributes 11 prediction units. The candidate label comes
from a full blind rejudgment. All candidate positives and a frozen sample of
candidate negatives must be manually audited before a scientific claim.

## Controlled matrix

- 8 held-out-by-scenario report settings.
- 3 target types: wrong number, wrong attribution, reversed order.
- 4 revision operators:
  1. forced target-sentence replacement;
  2. forced four-sentence section replacement;
  3. model-selected patch of at most four consecutive sentences;
  4. complete answer rewrite.
- 2 generator families: Qwen2.5-7B-Instruct and Qwen3-8B in non-thinking
  mode.

Total planned revisions: `8 * 3 * 4 * 2 = 192`, yielding at most 2,112
non-target transitions. Each clean answer has 12 short factual sentences, and
each old satisfied obligation has a frozen sentence witness. This construction
is intentionally controlled and more demanding than P0g's four-obligation
answers, but it must not be described as a natural benchmark.

The initial verifier is Qwen2.5-14B-Instruct so that existing control behavior
can be compared with P0g/P0h. Qwen3-14B is a later robustness check, not a
condition for this screen.

## Features available without a full rejudgment

Features may use the old answer, the revised answer, the frozen old witness,
the obligation text, the target type, the revision operator, and generator
identity. They may not use the new verifier verdict or its rationale.

- whether an actual sentence diff touches the old witness;
- distance from the witness to the changed span;
- whether the old witness text remains verbatim;
- maximum character similarity between the old witness and any new sentence;
- lexical overlap between the obligation and changed text;
- global character edit ratio;
- changes in numbers, citations, and negation terms;
- old witness position, changed-span size, operator, target type, and generator.

Witness overlap is treated only as a feature and heuristic baseline. P0h rules
out presenting it as an established causal explanation.

## Leakage prevention

- The repair target is excluded and always rejudged.
- Learned evaluation uses nested scenario-grouped cross-validation. No
  revision or obligation from a held-out scenario may enter its training or
  threshold-selection folds.
- Threshold selection occurs only inside each outer training fold and targets
  at least 90% stale-verdict recall.
- Labels are generated only after prompts, operators, features, baselines, and
  gates are frozen.

## Policies and metrics

Compare:

- full rejudgment;
- target-only rejudgment;
- diff/witness overlap;
- distance-to-change;
- witness-text similarity;
- a frozen union heuristic;
- random rejudgment with the learned policy's matched budget;
- grouped out-of-fold logistic regression.

Every policy pays for the mandatory target rejudgment. Primary metrics are:

- stale recall: fraction of stale non-target verdicts selected for rejudgment;
- verification saving: fraction of all 12 verdicts not recomputed;
- maximum saving attainable at 100% stale recall;
- learned-minus-random recall at matched budget.

Accuracy is not a primary metric because the negative class is expected to
dominate.

## Frozen gates

### Gate 1: verifier apparatus

On baseline and clean controls, JSON parse validity, positive accuracy, and
negative accuracy must each be at least 95%.

### Gate 2: phenomenon support

Before any learnability claim, manual audit must confirm at least 20 stale
non-target verdicts spanning at least 5 scenarios and at least 3 revision
operators. Both generator families are desirable and their coverage is
reported, but lack of one family does not convert a failed gate into success.

If Gate 2 fails, P0j is inconclusive because there are too few positives. A
classifier score cannot rescue it.

### Gate 3: useful predictability

After manual labels replace audited candidate labels, nested grouped
out-of-fold prediction must satisfy all of:

- stale recall at least 90%;
- verification saving at least 25%;
- at least 15 percentage points more recall than matched-budget random;
- at least 5 percentage points more saving than the best frozen single
  heuristic that also reaches 90% recall.

In addition, report the maximum saving at 100% recall. Failure of Gate 3 means
the stale state exists but this feature set does not support selective
invalidation.

## Manual audit

The review packet contains every candidate stale positive and 36 candidate
negative transitions chosen by a fixed SHA-256 ordering. Review checks only
whether the non-target obligation was satisfied before and is no longer
satisfied after. It also flags verifier leakage from the evidence and
generation/parse failures.

## Decision

- Gates 1-3 pass: selective invalidation is a viable mechanism prototype;
  proceed to natural long-form revisions and stronger verifier replication.
- Gate 1 fails: recalibrate the judge; do not interpret revision effects.
- Gate 2 fails: stop the learned invalidator line on this apparatus.
- Gate 2 passes but Gate 3 fails: retain full invalidation or a conservative
  deterministic policy; do not proceed to RL.

Reinforcement learning is explicitly out of scope until all three gates pass.
