# P0l: causal dependency-recomputation Guide screen

## Frozen question

When a verifier is explicitly shown a cached old `SAT` verdict and the current
revised document, can a Guide that identifies the dependency to recompute
causally recover correct current-version judgments?

P0l is a controlled case study. It does not estimate natural prevalence, train
a hidden-state predictor, or evaluate reinforcement learning.

## Claim boundary inherited from P0k

P0k showed a controlled recomputation failure, but its flat verifier did not
receive an old verdict. Therefore `verdict persistence` was only a hypothesis.
P0l places the old `SAT` verdict in every arm, making persistence and recovery
directly testable.

The claim about surface rules is also restricted to the frozen P0k baselines
on the matched construction. P0l does not claim that all possible surface
methods fail.

## Cases and states

Reuse all 40 P0k-R1 cases: 8 domains crossed with comparison, attribution,
derived arithmetic, temporal ordering, and definition/threshold mechanisms.

Only the two revised states are used:

- dependency edit: the unchanged conclusion is now false;
- matched harmless edit: the unchanged conclusion remains true.

Both revisions change the same source sentence, at the same position, with the
same local entities. There are 80 current-version inputs per arm.

## Five frozen arms

All arms receive identical cached verdict, current document, audited
conclusion, criterion, recomputation instruction, and output schema. Only the
Guide block changes.

1. `flat`: no dependency map.
2. `source_only`: correct current source sentence IDs, without the operation.
3. `relation_only`: correct dependency operation, without sentence IDs.
4. `source_relation`: both correct source IDs and correct operation. This is
   the pre-specified primary Guide.
5. `shuffled_guide`: the same structured fields, but source IDs point to
   non-evidence sentences and the relation comes from a different mechanism.

The shuffled relation is selected deterministically from the wrong mechanism
whose surface-token length is closest to the correct relation. It controls for
Guide length and format, not merely for extra prompt tokens.

## No-answer-leakage constraints

- A Guide may name source sentence IDs and an operation such as strict
  comparison, subtraction, temporal order, source-claim match, or threshold
  application.
- A Guide never copies current numeric values, current claims, the expected
  Boolean, the edit arm, or the words stale/safe/correct/incorrect.
- The full current document contains the facts, as it does in every arm.
- Prompts are deterministically shuffled before generation. Arm names are not
  shown to the model.

## Models and decoding

- Qwen3-8B, non-thinking;
- Qwen2.5-14B-Instruct.

Generation is deterministic. The output is a one-field JSON Boolean. Parsing
accepts only an explicit `met: true/false`; invalid output counts as an error.

## Metrics

Primary:

- stale recall: fraction of dependency edits assigned current `FAIL`;
- harmless specificity: fraction of harmless edits retained as current `SAT`.

Secondary:

- balanced accuracy;
- paired wins/losses, exact McNemar p-value, and paired bootstrap confidence
  interval against `flat` and `shuffled_guide`;
- per-mechanism stale recall;
- factorial diagnostics for `source_only` and `relation_only`.

## Frozen gates

### G1: output validity

Every model-arm cell must have at least 98% parse validity.

### G2: primary causal rescue

For **each** model, `source_relation` must:

- reach at least 75% stale recall;
- improve stale recall by at least 20 points over `flat`;
- improve stale recall by at least 15 points over `shuffled_guide`;
- beat `flat` and `shuffled_guide` with paired exact McNemar `p <= 0.05`.

### G3: no destructive over-invalidation

For each model, `source_relation` must reach at least 95% harmless specificity
and lose no more than 5 points versus `flat`.

### G4: the relation contributes beyond location

For each model:

`max(relation_only recall, source_relation recall) - source_only recall >= 10pp`.

This gate prevents a result where sentence localization alone explains the
effect attributed to relation-aware recomputation.

### G5: mechanism breadth

For each model, `source_relation` must catch at least 6 of 8 dependency edits
in at least four of five mechanisms.

P0l passes only if G1-G5 pass for both models. A one-model effect is reported
as model-specific exploratory evidence, not a robust Guide result.

## Required manual audit

Inspect:

- every dependency edit missed by `source_relation`;
- every harmless case rejected by `source_relation` but accepted by `flat`;
- every case where `shuffled_guide` is correct and `source_relation` is wrong;
- all invalid generations.

## Decision rule

- Full pass: relation-aware Guide has controlled causal evidence; next test a
  natural revision sample, then ask whether hidden state can predict the Guide.
- Correct Guide beats flat but not shuffled: extra prompting/attention is a
  sufficient explanation; do not claim relation causality.
- Source-only matches the full Guide: localization, not relation structure, is
  the active ingredient.
- No reliable rescue: stop the current Guide formulation. Do not add hidden
  states, training, or RL to compensate.
