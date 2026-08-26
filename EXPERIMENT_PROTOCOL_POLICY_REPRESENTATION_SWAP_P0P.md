# P0p: Cross-policy task-representation swap

## Question

Does a task representation produced by a policy help that same policy execute a
task better than an equally complete representation produced by another policy?

This is a selected case study, not evidence that a model has a privileged
"native language" or that hidden states should be trained.  A diagonal pattern
in an uncontrolled matrix can be caused by checklist quality, length, wording,
or evaluator bias.  The controlled matrix below is the decisive test.

## Frozen data and models

- Dataset: `RefineBench/RefineBench` at revision
  `2777137e7c489f5049608f41d2432326429ea619`.
- Ten fresh long-answer tasks: two deterministic selections from each of five
  strata (math/statistics, STEM, law, humanities/social science, other).
- Exclude the eight P0o rescue cases.
- Require 5--12 canonical requirements, a reference answer of at least 400
  characters, and a query of at most 8,000 characters.
- Policies/checklist authors (similar 7--8B scale, different model families):
  `Qwen/Qwen3-8B`, `mistralai/Mistral-7B-Instruct-v0.3`, and
  `allenai/OLMo-2-1124-7B-Instruct`.
- Local evaluator: `Qwen/Qwen2.5-14B-Instruct`.  This is not the official
  RefineBench GPT-4.1 evaluator.

## Representations

For each task and each representation-author model, build two representations.

1. `native`: the author sees the task but not the canonical checklist and writes
   exactly the same number of atomic requirements as the canonical checklist.
   This matrix is exploratory because semantic coverage may differ.
2. `structural`: the author sees the task and the full, ID-labelled canonical
   requirements, but may output only an ordered grouping of those IDs into
   phases.  Every ID must occur exactly once.
   When rendered for the answer policy, canonical wording is copied verbatim;
   no author-generated wording or author identity is shown.  Therefore all
   cells have identical requirement content and differ only in grouping/order.

A third `canonical` arm presents the canonical requirements in their original
order with no model-specific grouping.

Each of the three target policies answers every task using every source's native
and structural representation, plus the canonical arm.  The target never sees
the representation author's identity.  Generation is greedy to remove sampling
variance from this small screen.

Total answer generations: `10 * 3 * (3 native + 3 structural + 1 canonical) =
210`.

## Evaluation

All answers, including reference answers, are evaluated against the canonical
RefineBench checklist, not against the representation used to generate them.
The evaluator receives the complete task context.  Native representations are
also evaluated for semantic coverage of each canonical requirement.

For each 3x3 matrix, define:

`diagonal advantage = mean(score(M_i, R_i)) - mean(score(M_i, R_j), i != j)`.

Report the overall advantage, each target-policy advantage, a task-level
bootstrap interval, and a task-wise representation-source permutation p-value.
The balanced statistic cancels a representation author's overall main effect;
the structural matrix additionally removes semantic-content and wording
differences by construction.

## Apparatus gates

1. All native checklists and structural plans parse after at most one
   pre-registered format-repair attempt.
2. Every structural plan contains every canonical ID exactly once.
3. All evaluator outputs parse.
4. Reference-answer canonical-criterion pass rate is at least 90%.
5. For interpretation of the native matrix only: each author's mean canonical
   coverage is at least 75%, and the largest source-to-source coverage gap is at
   most 10 percentage points.  This is only a coverage-matching screen, not a
   complete quality match for atomicity, redundancy, or wording.

## Mechanism gates

The policy-alignment hypothesis survives P0p only if the controlled structural
matrix satisfies all of:

1. overall diagonal advantage >= 3 percentage points;
2. diagonal advantage is positive for at least two of three target policies;
3. task-wise source-label permutation `p <= 0.10`;
4. each policy's own structural representation improves over the canonical arm
   by at least 3 points on average across policies.

The uncontrolled native matrix is supportive only if its coverage gate passes,
its diagonal advantage is >= 5 points, at least two target policies are
positive, and `p <= 0.10`.  Native-only success is treated as
`CONTENT_OR_WORDING_CONFOUNDED`, not as policy-native decomposition evidence.

## Decisions

- `PROVISIONAL_STRUCTURAL_PASS_REVIEW_REQUIRED`: all controlled gates pass.
  Manually audit task quality and evaluator decisions before any prospective
  extension.
- `CONTENT_OR_WORDING_CONFOUNDED`: only the native matrix passes.
- `STOP_NO_ALIGNMENT_SIGNAL`: controlled matrix does not pass.
- `APPARATUS_FAILURE`: any mandatory apparatus gate fails.

Even a pass would establish only a small text-level compatibility effect.  It
would not establish hidden-state alignment, architectural necessity, novelty,
or a scalable compiler.
