# Join-Provenance HSGR Guide: result-free candidate protocol v2

Frozen on 2026-08-12 before any model is run on `gsm_join_test`.  This protocol
does not supersede the single-edge v1 protocol or change its gates.  V1 tests
whether source-specific repair actions have any value.  V2 is a stricter,
separate candidate designed to test whether that value is actually
hierarchy-specific.

## 1. Why a join graph is required

On a two-node chain, `UPSTREAM` versus `LOCAL` is almost identical to locating
the first wrong chronological step.  Generic mistake localization, hidden
arithmetic-error probes, targeted re-prompting, and uncertainty-triggered
redirection already cover that broad idea.  A two-node result therefore cannot
support an HSGR novelty claim even if its accuracy is positive.

V2 uses an explicit three-node join:

```text
parent_0 ---->
               root
parent_1 ---->
```

Both parent values have independently verified causal effects on the symbolic
root answer.  Parent problems and their assignment to root literals are
randomized, and their mean annotated depths must remain balanced.  A useful
Guide must distinguish an error propagated by edge `parent_0 -> root`, an
error propagated by edge `parent_1 -> root`, and an error introduced locally
at `root`.  This is causal-source routing on sibling incoming edges, not merely
early-versus-late step detection.

## 2. Data integrity

- Development data: `data/gsm_join_test.jsonl`, 400 unique graphs.
- Each row stores two exact parent answers, the original root answer, the
  answer after each single-edge intervention, and the answer after both
  interventions.
- Each single-edge answer must differ from the original root answer.
- Parent positions are randomized.  The absolute difference between the mean
  annotated step counts of `parent_0` and `parent_1` must be at most 0.25.
- The MuSiQue final377 set stays sealed and is unrelated to this experiment.

## 3. Frozen actions and source labels

A base run solves both parents independently and then solves the root using
their predicted values.  Before any repair, the exact development label is:

- `KEEP`: base root is correct;
- `P0`: base root is wrong, parent 0 is wrong, parent 1 is correct;
- `P1`: base root is wrong, parent 1 is wrong, parent 0 is correct;
- `LOCAL`: base root is wrong and both parents are correct;
- `BOTH`: base root is wrong and both parents are wrong.

Equal-cap one-call repair actions are `GENERIC`, `REPAIR_P0`, `REPAIR_P1`,
`REPAIR_LOCAL`, and `REPAIR_BOTH`.  A source oracle selects from the exact
label without examining repair outcomes.  Hindsight best-of-repairs is only an
unattainable diagnostic.

## 4. Oracle action gates

Do not extract hidden states unless every gate passes:

1. At least 30 base errors each in `P0`, `P1`, and `LOCAL`; report `BOTH`
   separately regardless of count.
2. Source-routed repair beats equal-call generic repair by at least +3.0pp on
   all eligible graphs with exact paired McNemar `p < 0.05`.
3. Source-routed repair beats the best fixed action by at least +3.0pp with
   exact paired McNemar `p < 0.05`.
4. On each of the `P0`, `P1`, and `LOCAL` strata, its matching repair beats
   every non-matching single-source repair by at least 5pp.
5. The routed advantage is non-negative in both fixed ID-hash halves and in
   low/high root-step-count strata.
6. Calls, prompt tokens, generated tokens, and caps are complete and balanced.

V1 passing is supporting feasibility evidence, not a substitute for these V2
gates.  V1 failing does not permit relaxed V2 thresholds.

## 5. Hidden-state Guide, only after the oracle gates

The end-to-end Guide must choose among `KEEP/P0/P1/LOCAL/BOTH` before seeing a
repair output.  Candidate node features are frozen-model hidden states from
the three base node executions.  The proposed structure-aware reader uses a
shared, permutation-equivariant edge scorer:

```text
edge_score_i = phi(h_parent_i, h_root, edge_features_i)
local_score  = psi(h_root)
action       = Guide({edge_score_i}, local_score, global_state)
```

No test gold, repair text, answer alias, or symbolic program may enter the
reader.  Labels are permitted only in training folds.

Required problem-disjoint nested-OOF baselines and controls:

1. KEEP and equal-call generic repair;
2. confidence-only action policy;
3. text-surface action classifier;
4. flat concatenated hidden-state classifier with no graph sharing;
5. per-node ordinary correctness probes followed by a deterministic router;
6. chronological step locator with the same hidden features and parameters;
7. proposed permutation-equivariant HSGR reader;
8. parent-edge swap, graph-edge deletion, and source-label permutation;
9. random repair and best fixed repair.

The HSGR reader must beat generic repair and the flat equal-data hidden
classifier by at least +2.0pp with Holm-adjusted paired `p < 0.05`, be positive
in four of five outer folds and both hash halves, and lose at least 2pp under
edge swap.  Otherwise the result is ordinary hidden error localization, not a
hierarchy-specific Guide.

## 6. Novelty and stop rule

The candidate contribution is only the combination of:

1. causal error-source attribution over sibling incoming dependency edges;
2. a permutation-equivariant hidden-state reader tied to that graph; and
3. source-conditioned local/subgraph repair actions.

Hidden error detection, mistake localization, repair prompting, graph
execution, and hidden-state probing are not claimed as new in isolation.  If
the flat or chronological control matches the proposed reader, abandon the
HSGR claim even if end-to-end accuracy improves.
