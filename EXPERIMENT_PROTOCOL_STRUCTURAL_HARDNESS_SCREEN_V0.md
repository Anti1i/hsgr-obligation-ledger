# Structural-hardness screen V0: frozen protocol

## Purpose

This is a benchmark-selection diagnostic, not a method result.  It tests
whether a controlled two-parent arithmetic join creates the regime required by
HSGR: local nodes are solvable, global composition is difficult, and sampled
node domains still contain a recoverable correct assignment.

The target is not literal problem difficulty or a post-hoc set of model
mistakes.  A useful structural-hardness regime must satisfy all three:

1. a strong direct baseline is away from both ceiling and floor;
2. both parents and the root with gold bindings are usually solvable;
3. the candidate graph contains enough decisions where a non-modal candidate
   can recover a root error.

## Data separation

- Calibration: `data/gsm_join_train.jsonl`, deterministically constructed from
  the GSM8K train split with seed 29.
- Confirmation: `data/gsm_join_test.jsonl`, constructed from the disjoint
  GSM8K test split with seed 17.
- No confirmation outcome may select an item or alter a rule.

All rows contain two independently causal parent edges into one root.  Model
outcomes are never used as per-item inclusion criteria.

## Frozen generation

Model: `Qwen/Qwen2.5-7B-Instruct`.

- Direct baseline: one greedy and seven temperature-0.8 whole-graph samples.
  Report SC@1/3/5/8 and oracle@8; SC@8 is the primary direct baseline.
- Each parent: one greedy and three temperature-0.8 samples.  The modal
  normalized value is the ordinary node policy; any gold value is the
  candidate-availability oracle.
- Root: one greedy execution with modal parent bindings and one with gold
  parent bindings.

The screen does not extract hidden states and does not train a Guide.

## Frozen rule family

Only outcome-independent metadata rules are eligible:

- all rows;
- total annotated steps at least 8, 10, 12, 14, or 16;
- root steps at least 2, 3, or 4;
- minimum parent steps at least 2, 3, 4, or 5;
- every conjunction of one minimum-parent threshold and one root threshold;
- every conjunction of one total-step threshold and one root threshold.

Select a rule on calibration data only.  A rule is eligible when:

1. it retains at least 100 graphs;
2. direct SC@8 is between 40% and 60%, inclusive;
3. mean modal parent accuracy is at least 70%;
4. gold-bound root accuracy is at least 80%;
5. at least 15% of graphs have a non-collapsed parent domain;
6. candidate recoverability exceeds the stronger of direct SC@8 and the modal
   graph pipeline by at least 10 percentage points;
7. at least 20 graphs are actionable: the modal graph is wrong because at
   least one modal parent is wrong, but both parent domains contain gold and a
   gold-bound root execution is correct.

If several rules are eligible, choose deterministically by: largest retained
sample, SC@8 closest to 50%, largest recoverability gap, then rule name.

## Confirmation gate

Apply the selected metadata rule once to the confirmation set.  Continue to a
separate hidden-Guide protocol only if all conditions hold:

1. at least 100 graphs remain;
2. direct SC@8 is between 35% and 65%;
3. mean modal parent accuracy is at least 65%;
4. gold-bound root accuracy is at least 75%;
5. at least 15% have a non-collapsed parent domain;
6. recoverability exceeds the stronger ordinary baseline by at least 10 pp;
7. at least 20 graphs are actionable.

Passing establishes only that the benchmark has relevant action headroom.  It
does not establish that hidden states predict the action, that a Guide works,
or that the method is novel.  A later method comparison must match total
prompt-plus-generation tokens and report both the full set and the frozen
structural subset.
