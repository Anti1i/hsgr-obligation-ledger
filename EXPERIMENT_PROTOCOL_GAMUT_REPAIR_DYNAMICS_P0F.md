# GAMUT repair-dynamics audit P0f

## Question

P0f reinterprets the frozen P0e case study as an intervention audit.  It asks whether a targeted
repair of the failed process-order obligation changes obligations that were already satisfied.

This is the smallest test of the proposed repair-influence view:

- diagonal effect: did repairing `ORDER` make the required order correct?
- off-diagonal effect: did that intervention remove a previously present process component?
- operator dependence: do full rewrites and local sentence patches show different patterns?

## Frozen inputs

P0f consumes, without generating new answers:

- the four manually defensible P0d relation-only cases;
- all sixteen P0e outputs (four repair arms per case);
- P0c's calibrated extract-then-check results already stored in those outputs.

Every baseline must have all process components present and the process relation unmet.  The audit
fails closed if that precondition is not true.

## Outcomes

For each repair attempt, P0f records the state transition for `ORDER` and every process node.  The
complete target is recovered only when every component remains present and the canonical order is
restored.  A sorted subset is not counted as target recovery.

Automatically losing a previously present component is evidence that a repair *attempt* can have a
negative side effect, but it is not the headline "fix one, break another" result because the
composite process target was not fully fixed.  That stricter result requires a structurally safe
repair that manually breaks a separate, previously correct factual obligation.

Invalid extractions are reported as unknown, not counted as component regressions.

The generated review packet requires manual inspection of factual preservation for all sixteen
outputs.  Automatic structural results do not establish factual preservation.

## Interpretation limits

- One target obligation produces one row of a repair cross-effect matrix, not a full matrix.
- Four selected cases can show existence, not prevalence or statistical significance.
- Differences between repair operators are descriptive only at this sample size.
- P0f cannot establish asymmetry between two repair targets, learnability of the influence graph,
  superiority of a planner, or any hidden-state claim.

If a structurally safe repair has a manually confirmed factual regression, "fix one, break another"
is supported as an existence claim.  Component loss supports only the weaker claim that repair
attempts can damage the ledger.  If neither occurs, the broad hypothesis remains open, but this
GAMUT process-order slice does not support it.
