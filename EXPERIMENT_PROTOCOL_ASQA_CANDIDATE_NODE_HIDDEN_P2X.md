# ASQA candidate-node hidden Guide P2x (frozen conditional protocol)

Date frozen: 2026-08-14 (Asia/Shanghai), after structural P0 but before any
ASQA P1x generation or P2x hidden-state result was observed.

## 1. Claim under test

On clean fixed-support ASQA questions, a partially generated long answer has a
small candidate-node domain consisting of its released disambiguated answer
facets.  P2x asks whether the frozen model's answer-prefix hidden state can
predict which currently uncovered facet nodes will still be missing when the
ordinary direct generation terminates.

This is a candidate-**node** experiment, not a candidate-value experiment.
P0 already showed that ASQA does not guarantee multiple meaningful values
inside each facet node.  P2x therefore does not sample wrong short answers or
claim to repair the old value-domain collapse.

## 2. Dependency on P1x

Use exactly the 192 examples and `fixed_direct` outputs from
`EXPERIMENT_PROTOCOL_ASQA_CLEAN_FIXED_SUPPORT_P1X.md`.  P2x hidden extraction
is licensed only if all three frozen P1x baseline-difficulty gates pass:

- fixed-direct STR-HIT is 40--75%;
- fixed evidence beats closed book by at least 10 points in STR-HIT or STR-EM;
- median fixed-direct output length is 30--160 words.

The P1x true-versus-decoy scaffold result is reported as independent evidence.
A failed scaffold gate does not get redefined after P2x, but it prevents a
later causal steering claim even if the P2x readout is predictive.

## 3. Candidate nodes, states, and labels

- One candidate node is one released disambiguated facet question.
- Candidate short-answer aliases are used only to construct training/evaluation
  labels.  They are never included in a model prompt or candidate feature.
- Every example has 2--6 unique candidate nodes and the same five fixed docs.
- From each deterministic `fixed_direct` answer, form answer-prefix states at
  25%, 50%, and 75% of its generated response tokens.
- At a state, retain only nodes whose aliases are not yet present in the
  prefix under official ALCE normalization.
- The binary future-omission target is 1 exactly when that retained node is
  also absent from the completed direct answer; it is 0 when ordinary
  continuation later covers the node.

Thus the target is not the trivial question "is the node covered now?".  It is
whether an uncovered node remains unresolved without a Guide.

## 4. Frozen features and split

Order the 192 IDs by SHA-256 of `20260815-asqa-node-p2x|released_id`.  The first
96 are calibration and the last 96 are untouched held-out cases.  All folds
and weights are problem-disjoint.

Frozen model: `Qwen/Qwen2.5-7B-Instruct`.  At blocks 13, 20, and 27 extract:

- last and mean response-token states for the answer prefix;
- mean facet-question-token state for each candidate node.

Each view is projected by a fixed signed random projection to 64 dimensions.
The full reader receives prefix-last, prefix-mean, node, elementwise
prefix/node product, absolute prefix/node difference, and the frozen surface
features below.  Ridge strengths are selected only by five-fold
problem-disjoint calibration OOF performance.

Surface-only controls receive:

- prefix token/word length and prefix fraction;
- candidate index, candidate count, and facet-question length;
- token overlap of facet question with the ambiguous question and prefix.

Additional controls are prefix-hidden-only, node-hidden-only, and a wrong-node
control that replaces the candidate representation with a deterministically
matched node from another problem while preserving candidate count and prefix.
No final answer, short-answer alias, coverage label, root outcome, or P1x
true/decoy output enters any feature.

## 5. Frozen endpoints

Primary endpoints on held-out candidate rows/states:

1. node-level AUROC for future omission;
2. Recall@1 on mixed states containing at least one future-missing and one
   future-covered candidate;
3. mean reciprocal rank of a future-missing node on those mixed states.

Report average precision, positive prevalence, all three prefix fractions,
facet-count strata, fixed ID-hash halves, and candidate/state/problem counts.
Each state has equal total evaluation weight.  Candidate rows from one problem
may never cross calibration, validation, or held-out boundaries.

## 6. Frozen apparatus and readout gates

The apparatus passes only if the held-out half contains:

1. all 96 expected problems and at least 250 candidate-state rows;
2. at least 30 problems whose final direct answer misses a facet;
3. at least 60 mixed candidate states;
4. future-omission prevalence between 10% and 60% among retained nodes.

The hidden candidate-node readout passes only if all conditions hold:

5. held-out AUROC is at least 0.70;
6. AUROC exceeds surface-only by at least 0.05;
7. mixed-state Recall@1 exceeds surface-only by at least 0.10;
8. the full reader has positive AUROC and Recall@1 gains over surface-only in
   both fixed ID-hash halves;
9. replacing the correct node representation with the wrong-node control
   reduces AUROC by at least 0.05 and Recall@1 by at least 0.05.

Thresholds may not change after P1x generations or P2x hidden states appear.

## 7. Interpretation and non-overlap boundary

A pass establishes only that hidden states contain candidate-specific advance
warning of a future omitted facet beyond the frozen observable controls.  It
does not establish that changing generation toward that node improves the
answer.

- Gold facet questions make P2x an oracle-node apparatus; a deployable method
  must obtain nodes without using answer aliases or test labels.
- P2x performs no retrieval, passage ranking, complete-path enumeration, tree
  search, regeneration, or prompt-action routing.
- Ordinary answer verification is insufficient: gains must depend on the
  correct candidate-node representation and survive the wrong-node control.
- A later causal test, if licensed, must be separately frozen and use hidden
  node conditioning with anti-node, random-node, text-only, and no-Guide
  controls.  Adding the facet question to the prompt is only a textual oracle
  comparator, not the proposed HSGR mechanism.
- ASQA is a one-root/multi-facet composition graph, not evidence for arbitrary
  deep hierarchical reasoning.  Claims must remain at this scope unless a
  deeper structured benchmark confirms them.

If any apparatus gate fails, stop.  If predictive gates fail, do not run causal
steering.  If P2x passes but P1x true facets do not beat decoys, report hidden
predictability as diagnostic only and stop before a Guide claim.
