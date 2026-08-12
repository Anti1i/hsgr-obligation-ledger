# FinQA atomic program-intervention P0: frozen protocol

Frozen on 2026-08-12 before downloading FinQA locally, generating a predicted
program, or observing an intervention outcome.  P0 is a falsification screen,
not a Hidden Guide experiment and not a novelty claim.

## 1. Question

> Does a fixed, predicted FinQA program admit a stable and non-trivial space of
> gold-free atomic interventions whose downstream execution utility depends
> jointly on program location and edit type?

The program is never expanded into alternative trajectories.  Every candidate
is one local edit of one fixed program followed by deterministic execution.

## 2. Stage A: data and structure audit

- Source: the corrected public `czyssrs/FinQA` repository.  The exact source
  commit and SHA-256 of every JSON split must be recorded.
- Splits: train, dev, and public test.  Private test is not used because its
  gold program is unavailable.
- Re-execute every gold program with an independent FinQA-compatible executor.
- Report parse/execution agreement, operation length, dependency depth,
  reference edges, two-reference joins, branching, and table operations.
- Frozen structural subsets:
  - `deep`: at least three operations and dependency depth at least three;
  - `join`: at least one operation consuming two distinct prior references;
  - `deep_join`: both conditions.

Stage A passes only if gold parse coverage and gold execution agreement are at
least 95% on every split, public test contains at least 150 `deep` examples,
and public test contains at least 100 `join` examples.  Failure prevents a
hierarchical HSGR interpretation; a chain-only diagnostic may be reported but
does not license Stage B.

## 3. Stage B: one fixed predicted program

Stage B is run on development data only.  Public test remains untouched until
all prompts, parsers, edit rules, thresholds, and statistical code are frozen.

- Model: `Qwen/Qwen2.5-7B-Instruct`, greedy decoding.
- Input: question plus the complete public table/text context.  `gold_inds`,
  gold program, gold answer, and gold execution result are forbidden inputs.
- Output: exactly one FinQA-format program, with no sampling, repair loop,
  reranking, hidden-state extraction, or self-consistency.
- Primary population: all parseable development predictions.
- Structural population: the predeclared `deep`, `join`, and `deep_join`
  subsets defined from gold structure only; selection never uses predictions
  or outcomes.
- Primary baseline gate: predicted execution accuracy must be in `[0.30,0.70]`
  on at least one structural population containing at least 150 examples.

## 4. Two candidate spaces

Every space includes `NOOP`.  Invalid candidates remain in the denominator and
score as incorrect.

### 4.1 Gold-diff ceiling (diagnostic only)

Deterministically align equal-length predicted and gold programs after syntax
normalization.  Construct complete edits using the gold operation/reference/
literal when one atomic replacement or one argument swap suffices.  Report
unalignable, length-mismatch, equivalent, one-edit, two-edit, and greater-than-
two-edit coverage separately.  This ceiling measures repairability only and
cannot pass the method-relevant gate by itself.

### 4.2 Deployable ceiling (main gate)

Candidates may use only the question, table/text context, and predicted
program.  Gold fields are forbidden.  The frozen atomic edit vocabulary is:

1. `replace_op`: replace a numeric operation with another numeric operation,
   or a table operation with another table operation;
2. `replace_ref`: replace one argument by a legal earlier `#k` reference;
3. `replace_literal`: replace one numeric argument by a deployable literal;
4. `replace_row`: replace the row argument of a table operation by a table row
   header;
5. `swap_args`: swap the two arguments of a numeric operation.

Deployable literals are deduplicated and capped at 32.  Ranking is fixed:
current-program literals, question literals, then literals from evidence units
ranked by lowercase alphanumeric token overlap with the question; ties use
source order.  Standard constants are appended only after observed literals.
Candidates are canonicalized, deduplicated, and capped at 512 per example by
the fixed edit order above, step index, argument index, and payload rank.

The selected action is therefore `(node, edit_type, payload)`.  Payload is not
silently supplied by gold in any deployable metric.

## 5. Frozen metrics and gates

Report candidate count, parse validity, executor validity, baseline-correct
corruption, baseline-error repair, and paired accuracy for every edit type and
node rank.  Nodes are indexed both from the start and from the final root.

- `B`: no edit.
- `O_gold`: per-example `NOOP` or best gold-diff one-edit outcome.
- `O_deploy`: per-example `NOOP` or best deployable one-edit outcome.
- `O_fixed_type`: best globally fixed edit type, with node and payload oracle
  within that type for each example.
- `O_fixed_rank`: best globally fixed node rank from the root, with edit type
  and payload oracle at that rank for each example.
- `O_joint`: identical to `O_deploy`; location, type, and payload may all vary.
- `O_two`: exhaustive deployable edits up to distance two, using the same
  frozen payload universe and a separately reported candidate count.

Stage B passes only if all conditions hold on a predeclared population:

1. at least 150 examples and at least 50 baseline errors;
2. predicted-program parse validity at least 95%;
3. `O_deploy - B >= 10pp`;
4. `O_joint - max(O_fixed_type, O_fixed_rank) >= 5pp`;
5. `(O_deploy-B)/(O_gold-B) >= 0.50` whenever `O_gold>B`;
6. `O_two - O_deploy <= 3pp`;
7. joint gain is positive in both SHA-256 ID halves and at least 3pp in each;
8. paired bootstrap 95% confidence intervals for conditions 3 and 4 exclude
   zero;
9. all denominators, invalid programs, length mismatches, and uncovered
   failures are retained in the report.

Passing development licenses one frozen public-test confirmation.  Only a
confirmation pass licenses a Hidden Guide experiment.  It does not establish
novelty.

## 6. Stop and claim boundaries

- Gold-diff passes but deployable fails: candidate construction is still the
  bottleneck; stop.
- One-edit fails but two-edit succeeds: the task is edit-sequence search; stop
  this single-intervention HSGR route.
- Joint does not beat either marginal: a hierarchy-conditioned typed router is
  unnecessary; stop.
- FinQA structure is predominantly short/linear: results may support atomic
  program repair but not hierarchical guidance.
- Program generation, execution, error localization, counterfactual repair,
  and financial reasoning correction are prior-work territory and are not
  claimed contributions.

