# ASQA obligation-preserving local repair P5x (frozen protocol)

Date frozen: 2026-08-22 (Asia/Shanghai), after P3x aggregate results were
observed and before any P5x model output was generated.

## 1. Question under test

P3x showed that asking the model to emphasize one facet in a fresh answer
raises coverage of that facet but removes more coverage from the other facets.
P5x tests two narrower claims without training a hidden controller:

1. **Local-action feasibility:** when a direct answer has exactly one measured
   missing facet, a bounded append-only action can recover it while preserving
   the already covered facets.
2. **Obligation-state utility:** during full rewriting, explicitly marking the
   missing facet as `UNSATISFIED` and the others as `SATISFIED / PRESERVE`
   improves recovery and preservation relative to a status-swapped ledger and
   a target-only rewrite.

This is a gold-facet Oracle screen. It does not test automatic facet induction,
hidden-state reading, hierarchy, training, or method novelty.

## 2. Frozen examples

Reconstruct the exact 192 fresh P3x cases selected by
`20260815-asqa-single-node-p3x`. Load the immutable P3x generation artifact and
require exactly the expected 1,108 rows.

Select every case whose saved `fixed_direct` answer:

- fails strict full-facet coverage;
- has exactly one absent alias group under the frozen scorer; and
- has at least one present alias group.

No model output or newly generated hidden state is used for selection. The
single absent group is the Oracle repair target. A deterministic hash chooses
one already-present group as the status-swap target. At least 40 eligible cases
are required for an interpretable apparatus.

## 3. Model and common inputs

- Model: `Qwen/Qwen2.5-7B-Instruct`, bfloat16, SDPA, greedy decoding.
- The five fixed support documents, original question, and saved direct answer
  are present in every arm.
- No retrieval, sampling, reranking, judge model, or repeated attempts.
- Append arms receive at most 96 new tokens; rewrite arms receive at most 192.
- Gold answer aliases never enter a prompt. Facet questions do enter the Oracle
  target/ledger prompts.

## 4. Frozen arms

For each eligible case generate exactly one output under each arm:

1. `target_append`: output only a one- or two-sentence addition for the actual
   missing facet. The final answer is constructed deterministically by appending
   it to the saved answer.
2. `generic_append`: output only a one- or two-sentence factually supported
   missing interpretation, without being told which facet is missing. Append it
   identically.
3. `target_rewrite`: rewrite the complete answer, naming only the actual missing
   facet and generically asking to preserve existing material.
4. `correct_ledger_rewrite`: rewrite with the actual missing facet labelled
   `UNSATISFIED — ADD` and every scorer-present facet labelled
   `SATISFIED — PRESERVE`.
5. `swapped_ledger_rewrite`: show the identical complete facet set, but label a
   scorer-present facet `UNSATISFIED — ADD` and label the actual missing facet
   `SATISFIED — PRESERVE`.

The correct and swapped ledgers therefore differ in state labels, not in which
facet questions are visible.

## 5. Metrics

Use the unchanged ALCE-normalized alias-substring scorer. Report for every arm:

- strict recovery rate (`STR-HIT`);
- mean facet coverage (`STR-EM`);
- actual-target recovery rate;
- mean and all-or-none preservation of the facets present in the saved answer;
- answer length and number of newly lost facets.

Report exact paired two-sided McNemar tests for:

- `target_append` vs `generic_append`;
- `correct_ledger_rewrite` vs `swapped_ledger_rewrite`;
- `correct_ledger_rewrite` vs `target_rewrite`.

Also report the existing P3x `all_true` from-scratch result on the selected
cases as a contextual ceiling, not as a newly generated P5x arm.

## 6. Frozen gates and outcomes

### Apparatus gates

1. exactly 427 clean eligible source cases, 192 exact P3x cases, 235 in the
   pre-P3x fresh pool, zero overlap with P1x, and exactly 1,108 valid P3x rows;
2. at least 40 repair-eligible cases, each with exactly one missing and at least
   one present facet, two to six facets, and five fixed documents;
3. the saved answer rescoring exactly matches every saved P3x present vector;
4. correct and swapped ledger prompts expose identical facet-question sets,
   with only one present/missing status swap.

### Local-action gates

1. `target_append` strict recovery is at least 30%;
2. it exceeds `generic_append` strict recovery by at least 10 points;
3. the paired McNemar p-value is below 0.05 and target-only wins exceed
   generic-only wins;
4. all-present-facet preservation is at least 98%;
5. the median generated addition is between 5 and 60 words.

### Obligation-state gates

1. `correct_ledger_rewrite` strict recovery is at least 35%;
2. it exceeds `swapped_ledger_rewrite` by at least 5 points STR-HIT with paired
   McNemar p below 0.05 and more correct-only wins;
3. it exceeds `target_rewrite` by at least 5 points STR-HIT;
4. its all-present-facet preservation is at least 90%;
5. its all-present-facet preservation exceeds `target_rewrite` by at least 5
   points.

Outcome labels are `APPARATUS_FAIL`, `BOTH_PASS`, `LOCAL_ONLY_PASS`,
`OBLIGATION_STATE_ONLY_PASS`, or `BOTH_FAIL`. A local-only pass licenses work
on bounded repair actions but not a latent ledger. An obligation-state pass
licenses a separate automatic-state experiment, but still does not establish a
hidden-state or hierarchical contribution.

The split, eligibility rule, prompts, arms, scorer, thresholds, and outcome
logic may not be changed after P5x outputs are observed. Any modified protocol
requires a new experiment identifier and fresh outputs.
