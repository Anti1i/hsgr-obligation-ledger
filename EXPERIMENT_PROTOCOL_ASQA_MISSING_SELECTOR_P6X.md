# ASQA automatic missing-facet selector P6x (frozen protocol)

Date frozen: 2026-08-22 (Asia/Shanghai), after P5x aggregate results were
observed and before any P6x model output or hidden feature was generated.

## 1. Claims under test

P5x established only an Oracle action result: when the one missing facet is
given, a short append recovers 43.42% of eligible answers while preserving all
previously covered facets. P6x tests the unestablished bottleneck:

1. can a frozen model select the missing facet without gold aliases or gold
   status labels at test time;
2. does a trained hidden-state readout provide useful information beyond the
   frozen model's explicit A/B output logits; and
3. does automatic selection retain a material fraction of the P5x Oracle
   append ceiling end to end.

P6x does not test automatic facet induction: the ASQA facet-question set is
still supplied. It also does not contain hierarchical edges.

## 2. Disjoint train and evaluation sources

- Probe training/calibration source: the exact 192 fresh P3x cases selected by
  `20260815-asqa-single-node-p3x` and their frozen `fixed_direct` answers.
- Untouched P6x evaluation source: the exact 192 old P1x cases selected by
  `20260815-clean-p1x` and their frozen `fixed_direct` answers.
- Require zero case-ID overlap, exactly 1,108 P3x rows, and exactly 768 P1x
  rows.

For candidate-level selection evaluation, keep cases with at least one covered
and one missing facet under the frozen scorer. For end-to-end repair, keep the
evaluation cases with exactly one missing facet and at least one covered facet.
The latter set must contain at least 40 cases.

Gold aliases are used only to create training labels and final metrics. They
never enter selector or generation prompts.

## 3. Frozen selector prompt and scores

For every candidate facet, show only the ambiguous question, saved answer, and
candidate facet question. Ask whether the answer covers that candidate and
define two single-token labels:

- `A`: COVERED;
- `B`: MISSING.

From one frozen Qwen forward pass record:

1. `logit`: next-token `logit(B) - logit(A)`;
2. hidden states at zero-based transformer blocks 13, 20, and 27, at the last
   selector-prompt token;
3. a deterministic lexical missingness score based on candidate-question
   content-word recall in the saved answer;
4. a deterministic hash-random score.

Model: `Qwen/Qwen2.5-7B-Instruct`, bfloat16, SDPA. No documents are shown to
the selector because the target is answer coverage, not factual support.

## 4. Hidden probe without evaluation tuning

Fit L2 logistic probes on P3x candidate labels (`missing=1`) for each layer and
`C in {0.01, 0.1, 1.0}`. Use five deterministic case-grouped folds. Select one
cell by, in order:

1. highest out-of-fold exactly-one-missing target-selection accuracy;
2. highest candidate AUROC;
3. smaller C;
4. shallower layer.

Refit the selected cell on every mixed P3x training candidate. Apply it once to
P1x evaluation candidates. No P1x label, output, metric, or hidden state may
affect layer, C, prompt, or threshold selection.

Each selector chooses the maximum missingness score, with lower facet index as
a deterministic tie-break. Report candidate AUROC and exactly-one-missing
case-level target-selection accuracy.

## 5. End-to-end append arms

On every exactly-one-missing P1x evaluation case, generate one 96-token maximum
append under each arm:

1. `oracle_append`: actual missing facet;
2. `hidden_probe_append`: facet selected by the frozen hidden probe;
3. `logit_append`: facet selected by explicit A/B logits;
4. `lexical_append`: facet selected by lexical missingness;
5. `random_append`: facet selected by the deterministic random score;
6. `generic_append`: no facet target.

Every generated string is appended deterministically to the saved answer.
Generation uses the same five fixed documents, greedy decoding, and bounded
append instruction as P5x. Report strict recovery, target recovery, prior-facet
preservation, word length, and exact paired McNemar tests.

## 6. Frozen gates and outcomes

### Apparatus and Oracle replication

1. exact source/case/row counts and zero train-evaluation overlap;
2. at least 40 valid evaluation repair cases and mixed candidate labels in all
   five training folds;
3. exact saved-answer rescoring and finite selector features/scores;
4. A and B are distinct single tokens and all selector prompts exclude fixed
   documents and answer aliases as explicit fields;
5. Oracle append recovers at least 30%, preserves all prior facets in at least
   98%, and beats generic append by at least 20 points with `p<0.05`.

### Surface-selector gates

6. logit target-selection accuracy is at least 50%;
7. it beats random selection by at least 10 points with paired `p<0.05`;
8. logit append recovers at least 20%, at least half the Oracle recovery rate,
   and preserves prior facets in at least 98%;
9. logit append beats generic append by at least 10 points with `p<0.05`.

### Hidden-specific gates

10. hidden probe candidate AUROC is at least 0.70 and target-selection accuracy
    is at least 50%;
11. hidden probe selection beats logit selection by at least 5 points;
12. hidden-probe append beats logit append by at least 5 points STR-HIT;
13. hidden-probe append recovers at least 20%, at least half the Oracle rate,
    preserves prior facets in at least 98%, and beats generic by at least 10
    points with `p<0.05`.

Outcomes:

- `APPARATUS_FAIL`: any apparatus/Oracle gate fails;
- `HIDDEN_SELECTOR_PASS`: all surface and hidden-specific gates pass;
- `SURFACE_SELECTOR_ONLY_PASS`: all surface gates pass but any hidden gate
  fails;
- `SELECTOR_FAIL`: apparatus passes but surface gates fail.

A surface-only pass supports an explicit answer-coverage controller and is
evidence against needing hidden states on this apparatus. A hidden pass
supports only a prompted frozen-state selector; it does not establish automatic
facets, hierarchy, novelty, or a trained end-to-end HSGR system.

No prompt, split, layer, C grid, selector, arm, metric, threshold, or outcome
rule may change after P6x outputs or evaluation features are observed.
