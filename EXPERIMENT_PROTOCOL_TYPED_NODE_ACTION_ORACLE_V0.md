# Typed Node x Action Oracle V0: frozen falsification protocol

Frozen on 2026-08-12 before generating any completion from the intervention
prompts below.  This replaces the parent multi-view candidate-domain pilot as
the active route.  It does not train a Hidden Guide or an energy model.

## 1. Claim being tested

The old value-selection mechanism has only 0.7 percentage points of node
oracle headroom in the cached four-sample parent domains.  V0 asks a different
question:

> Under a one-intervention budget, do different fixed-hierarchy states require
> different **locations and types** of repair strongly enough to justify a
> learned hierarchy-conditioned router?

This is not claimed to prove novelty.  Error-source routing and targeted
repair already exist in this repository and in current work.  NCoTS selects
reasoning operators along a partial trajectory, while ATLAS selects latent
steering actions at thought boundaries.  The only candidate distinction left
for a later HSGR method is a fixed reasoning DAG whose single-node textual
intervention is supervised by the counterfactual final-root outcome.

## 2. Explicit non-overlap boundary

- No alternative full trajectories are retained, expanded, pruned, or
  backtracked.  Each arm performs exactly one intervention on one current DAG.
- The graph and all non-target node assignments are fixed.  Structure search,
  re-parenting, and latent steering are out of scope in V0.
- `node` is not an action type.  Earlier provenance experiments varied the
  repair source with essentially one recomputation behavior.  V0 must measure
  **within-node typed-action complementarity**.
- Hidden-state action selection, adaptive compute, repair prompting, and
  dependency localization are not contributions by themselves.
- Hindsight oracles use gold only during analysis and are not deployable.

## 3. Frozen data and initial state

- Development-only data: `data/gsm_join_train.jsonl`.
- Select exactly 128 of its 400 graphs by the ascending SHA-256 rank of
  `typed-node-action-v0|id`; selection is independent of every model outcome.
- Model: frozen `Qwen/Qwen2.5-7B-Instruct`.
- Initial parent assignments are the ordinary greedy candidates already
  cached in the 800-row structural-hardness parent cache.  No stochastic
  candidate domain is used.
- Generate one deterministic root from the two current parent assignments.
- `gsm_join_test.jsonl` remains untouched.  A V0 pass licenses a separately
  frozen confirmation experiment on that split, not hidden-state training.

## 4. Frozen intervention matrix

Target nodes are `parent_0`, `parent_1`, and `root`.  Four actions apply to
every node:

1. `equation`: express the target as minimal equations and recompute it;
2. `independent`: ignore the proposal and independently recompute it;
3. `backward`: test the proposal backward against quantities and constraints;
4. `redecompose`: use a different local decomposition into smaller subproblems.

One additional root-only action is:

5. `rebind`: audit the ordered binding of both fixed parent values and units,
   then recompute only the root.

This gives 13 `(node, action)` arms per graph.  Every graph receives every
arm, regardless of correctness.  Each arm is one deterministic call with a
512-token cap.  The prompt fixes non-target assignments and requests one
repaired target value and one propagated root value.  A separate equal-cap,
one-call `generic` repair may inspect all nodes.

## 5. Metrics

For every arm report parse validity, total accuracy, baseline-error repair
rate, baseline-correct corruption rate, accuracy delta, prompt/generated
tokens, and calls.  Also report:

- base and generic-repair accuracy;
- best fixed `(node, action)` accuracy;
- uniform one-intervention expected accuracy;
- all-13 majority-vote accuracy and its 13x call cost;
- **Action Oracle at a fixed node**:
  `max_v mean_i max(base_i, max_a outcome_i(v,a))`;
- **Node Oracle with a fixed common action**:
  `max_a mean_i max(base_i, max_v outcome_i(v,a))`;
- **Node x Action Oracle**:
  `mean_i max(base_i, max_{v,a} outcome_i(v,a))`;
- per-action and per-node exclusive repair counts on baseline errors;
- pairwise action overlap, Jaccard, and
  `P(action_i repairs, action_j fails | base error)`;
- the same main deltas in two fixed graph-ID hash halves.

The `max(base, ...)` terms give the oracle an `ACCEPT` option.  Raw arm
accuracies still measure the cost of corrupting initially correct states.

## 6. Frozen go/no-go gate

V0 passes only if all conditions hold:

1. exactly 128 base rows, 128 generic rows, and 128 rows for each of 13 arms;
2. root parse validity is at least 95% for generic and every arm, and target
   parse validity is at least 90% for every arm;
3. at least 40 initial roots are wrong;
4. Node x Action Oracle exceeds the stronger of generic-with-oracle-KEEP and
   best-fixed-arm-with-oracle-KEEP by at least 8.0 percentage points;
5. Node x Action Oracle exceeds both the Action Oracle and Node Oracle by at
   least 3.0 percentage points;
6. at least two action types each have at least three exclusive repairs, and
   at least two nodes each have at least three exclusive repairs;
7. joint-oracle gain over the strongest equal-call comparator is non-negative
   in both fixed hash halves;
8. every intervention and generic arm uses exactly one call and the identical
   512-token cap, with complete token accounting.

All thresholds are fixed before outcomes.  They may not be relaxed after the
run.

## 7. Stop rule

- Failure: do not train a hidden-state action router or graph energy on this
  action space.  Report whether the failure was lack of repair ability, lack
  of typed complementarity, lack of node interaction, or output validity.
- Pass: freeze a disjoint confirmation protocol first.  Only a confirmation
  pass can license hidden-state routing, with flat hidden, text, confidence,
  ordinary node-correctness, ATLAS-style state-only, NCoTS-style operator, and
  generic-repair controls.
