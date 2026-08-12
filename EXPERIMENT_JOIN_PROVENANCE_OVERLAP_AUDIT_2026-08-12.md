# Join-provenance route: post-protocol current-method overlap audit

Date: 2026-08-12.  This audit was completed after freezing
`EXPERIMENT_PROTOCOL_JOIN_PROVENANCE_GUIDE_V2.md` and before running its model
experiment.  It does not change any experimental gate.  It narrows what a
positive result would be allowed to claim.

## Verdict

The broad proposal -- dependency-aware failure attribution followed by
targeted repair -- **overlaps current methods and is not a defensible HSGR
novelty claim**.

In particular:

- FALAT explicitly traces dependencies to distinguish an error-introducing
  step from later steps that merely inherit or propagate it, and checks whether
  correcting a candidate step would recover the outcome:
  <https://arxiv.org/abs/2606.00765>.
- AgentTether builds a dependency-aware Critical Transition Graph, localizes a
  failure-critical subtrajectory, converts the cause into scoped guidance, and
  performs guarded runtime intervention:
  <https://arxiv.org/abs/2607.06273>.
- Trajectory Graph Copilot uses a probabilistic trajectory graph and a GNN for
  pre-execution error diagnosis and self-correction:
  <https://arxiv.org/abs/2607.27443>.
- AgentLocate attributes failure to a responsible component and earliest
  decisive step with a learned/adapted judge:
  <https://arxiv.org/abs/2607.07989>.
- Generic reasoning-error localization and hidden-state-guided arithmetic
  re-prompting also predate this route:
  <https://aclanthology.org/2024.findings-acl.826/> and
  <https://aclanthology.org/2025.emnlp-main.411/>.

## What remains testable

The v2 oracle experiment may still answer a useful mechanism question: on a
controlled single-model reasoning join, do source-specific actions have enough
counterfactual value to justify learning a reader?  It must not be described
as validating novelty.

If its oracle gates pass, the only potentially differentiating method claim is
much narrower:

> A frozen generator's per-node hidden states are read by a shared,
> permutation-equivariant incoming-edge scorer to route a repair action on a
> single-model reasoning DAG.

Even that claim is currently **unestablished**, not novel by default.  It must
beat the flat hidden classifier, chronological locator, per-node ordinary
correctness probes, edge-swap control, and graph deletion/permutation controls
already required by the v2 protocol.  A further full-paper audit of graph
debuggers and hidden-state verifiers is mandatory before promotion.

## Stop decision

- Do not use `dependency attribution`, `error propagation tracing`, `local
  repair`, or `graph-guided diagnosis` as claimed contributions.
- A positive oracle ceiling authorizes only a hidden-reader experiment, not a
  paper claim.
- If the flat/chronological hidden controls match the graph reader, close this
  route even if accuracy exceeds KEEP or generic repair.
- If the oracle gates fail, close the route before hidden-state extraction.
