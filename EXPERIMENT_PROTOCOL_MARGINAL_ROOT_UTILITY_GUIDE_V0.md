# Marginal Root-Utility Guide V0: frozen action-headroom protocol

## Status and purpose

This is a zero-GPU falsification pretest for a possible training-time HSGR
route. It is frozen before inspecting the policy results.

The proposed signal is value-class marginal root utility: for node `v` and
candidate value class `k`, the Guide uses the hierarchy to measure how much
`k` participates in root-correct assignments. The final method, if pursued,
would train a hidden-state observer to predict that *future graph-global
utility* before the root answer is generated. V0 does **not** claim that the
oracle credit is deployable and does **not** evaluate a final method.

## Non-overlap boundary

- This is not an inference-time hidden direction, residual steering method,
  correctness verifier, route selector, or error-provenance router.
- The Guide is a training-time credit definition over a node's candidate
  domain and downstream root recoverability.
- The later hidden-state test, if V0 passes, must read the target node before
  seeing a proposed root answer and must predict node-level marginal utility.
- Whole-solution last-token hidden features are disallowed because they reduce
  the experiment to ordinary answer/trajectory verification.

## Frozen data and policies

Primary independent test sets:

- `outputs` (MATH pilot)
- `outputs_gsm_test` (GSM test)

Training splits are descriptive only:

- `outputs_math_train`
- `outputs_gsm_train`

For each problem, enumerate the existing aggregate assignments. For every
node/value class compute:

- `frequency`: candidate sampling frequency;
- `LOO`: number of root-correct assignments using the value class divided by
  the total number of assignments;
- `exact_cf`: `P(root correct | value=k) - P(root correct | value!=k)`.

Select one value per node with deterministic tie-breaking by frequency and
then the pre-existing domain order. Evaluate the aggregate row corresponding
to the selected tuple. Report:

- hard-commit and frequency baselines;
- LOO Guide;
- exact-CF Guide (diagnostic oracle);
- recoverability oracle (any correct assignment exists).

No hidden features are extracted in V0.

## Frozen validity checks

Report all problems, multi-class problems, and actionable problems where LOO
and frequency select different value tuples. Report missing selected tuples,
duplicate assignment tuples, candidate-domain collapse, and paired outcomes.

The LOO signal is considered structurally faithful only if its node-level
argmax agrees with exact CF on at least 95% of multi-class nodes in each
primary test set.

## Frozen go/no-go gate

V0 passes only if **all** conditions hold on both primary test sets:

1. at least 100 evaluable problems and at least 20 actionable problems;
2. node-level LOO/exact-CF argmax agreement is at least 95%;
3. LOO Guide improves over the more accurate of hard-commit and frequency by
   at least 3.0 percentage points over all evaluable problems;
4. the paired two-sided exact McNemar p-value for that improvement is below
   0.05;
5. LOO Guide improves over that same baseline on the actionable subset (no
   minimum magnitude beyond being positive).

If any condition fails, do not run a new GPU hidden-state extraction for this
route. A pass licenses only a separately frozen node-position hidden-state
pretest; it does not establish novelty, trainability, or end-to-end gain.

