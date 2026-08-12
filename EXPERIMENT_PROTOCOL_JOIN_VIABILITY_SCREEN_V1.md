# Join Viability Screen V1: frozen protocol

Frozen on 2026-08-12 before inspecting any metadata-subset outcomes.

## Purpose

This is a benchmark/backbone viability diagnostic, not a method result.  It
asks whether `Qwen/Qwen2.5-7B-Instruct` on an outcome-independent easier stratum
of `gsm_join` has the regime needed by a future one-pass, counterfactually
trained hierarchical Guide.  No hidden states, edits, repair prompts, model
training, or method claims are allowed in this screen.

The prior typed-node experiment is not the baseline for this screen.  The
baseline is clean direct whole-graph solving, because the typed-node pipeline
contains possibly wrong parent assignments.  The gold-structure executor is
also not informative here: `gsm_join` answers were constructed by executing
the source GSM8K symbolic steps.  The unknown quantity is whether a model can
solve parents and compose gold parent values in a non-floor regime.

## Data and separation

- Calibration: all 400 rows of `data/gsm_join_train.jsonl` (GSM8K train,
  generation seed 29).
- Confirmation: `data/gsm_join_test.jsonl` (disjoint GSM8K test, seed 17).
- Phase A reuses the already generated calibration caches from Slurm job
  `728006`: 400 direct records with one greedy plus seven temperature-0.8
  samples, and 800 parent records with one greedy plus three samples.
- Calibration outcomes may select one metadata rule.  Confirmation outcomes
  may not alter the rule, thresholds, prompts, model, or decision.

## Frozen metadata rule family

Rules inspect only annotated difficulty metadata, never model outputs for an
individual item:

- `all`;
- `total_le_T`, `T in {8, 9, 10, 11, 12}`;
- `root_le_R`, `R in {2, 3, 4}`;
- `max_parent_le_P`, `P in {3, 4, 5}`;
- every conjunction `total_le_T__root_le_R` over the sets above;
- every conjunction `max_parent_le_P__root_le_R` over the sets above.

No rule based on correctness, candidate diversity, answer value, topic, or
manually inspected examples is eligible.

## Phase A: cached CPU prefilter

For each rule report:

- retained graph count;
- clean direct SC@1, SC@3, SC@5, SC@8 and oracle@8;
- mean greedy parent accuracy over both parent slots;
- parent oracle@4 and non-collapsed graph rate as diagnostics only.

A rule is pre-eligible only if all conditions hold:

1. at least 100 calibration graphs;
2. direct SC@1 is in `[0.30, 0.70]`;
3. direct SC@8 is in `[0.30, 0.70]`;
4. mean greedy parent accuracy is at least `0.70`.

If no rule is pre-eligible, V1 fails immediately.  Do not run gold-bound
roots, confirmation, hidden extraction, structural edits, or model training.
Move to a separately frozen screen for an easier generated regime or a larger
backbone; do not relax these thresholds after seeing the result.

## Phase B: calibration gold-bound root

Only if Phase A passes, run one deterministic root call per graph in the union
of pre-eligible rules, binding the two annotated gold parent values.  The
prompt and answer parser must be the same as the structural-hardness screen.

A rule remains eligible only if:

1. gold-bound root accuracy is at least `0.70`;
2. gold-bound root accuracy exceeds the stronger of direct SC@1 and SC@8 by
   at least `0.10`.

Choose one rule deterministically by: largest retained set, direct SC@1
closest to `0.50`, largest gold-root gap, then rule name.

## Phase C: untouched confirmation

Apply the chosen rule once to `gsm_join_test`.  Generate the same direct
SC@1/3/5/8 arm, one greedy call for each parent, and one gold-bound root call.
The confirmation gate requires all of:

1. at least 80 graphs;
2. direct SC@1 and SC@8 both in `[0.25, 0.75]`;
3. mean greedy parent accuracy at least `0.65`;
4. gold-bound root accuracy at least `0.65`;
5. gold-bound root exceeds the stronger direct baseline by at least `0.08`.

A pass establishes only that this benchmark/backbone regime can support a
separately frozen causal-latent Guide experiment.  It does not establish that
hierarchy helps, that counterfactual supervision beats flat SFT, or that the
proposed method is novel.

## Resource and accounting constraints

- Model: `Qwen/Qwen2.5-7B-Instruct`.
- Direct: one greedy plus seven temperature-0.8 samples, `max_new=512`.
- Parent greedy: one call, `max_new=192`.
- Gold-bound root: one deterministic call, `max_new=192`.
- GPU work must run under Slurm; Phase A is CPU-only and must not initialize
  Torch or load a model.
- Every generated stage must checkpoint JSONL rows and be resumable.

