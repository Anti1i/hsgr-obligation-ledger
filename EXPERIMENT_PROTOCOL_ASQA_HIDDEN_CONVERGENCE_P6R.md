# ASQA hidden-probe convergence audit P6r (frozen protocol)

Date frozen: 2026-08-22 (Asia/Shanghai), after P6x job 749008 and before any
P6r hidden feature or model output was generated.

## Purpose

P6x passed every surface-selector gate but its dual liblinear hidden probes
emitted convergence warnings. P6r asks only whether the negative
hidden-specific result was an optimizer artifact. It is not a new prompt or
method search.

## Frozen reproduction

- Reconstruct the exact P3x 192-case training split and P1x 192-case evaluation
  split used by P6x, with zero overlap.
- Reuse the same saved direct answers, mixed-case filter, selector prompt,
  labels, layers 13/20/27, `C in {0.01, 0.1, 1.0}`, five case-grouped folds,
  cell-selection order, model, bfloat16 forward pass, and lower-index tie-break.
- Reuse `StandardScaler(with_mean=False)` and the same balanced L2 logistic
  objective. Change only liblinear from its non-converged dual optimizer to the
  mathematically equivalent primal optimizer, with `max_iter=5000` and
  `tol=1e-6`.
- Capture convergence warnings and iteration counts for every fold and final
  refit. Do not generate or rescore answers in P6r.
- Replay the frozen P6x candidate file and require its labels and explicit A/B
  logits to align with the P6r reconstruction.

## Gates and outcomes

Apparatus gates require exact counts, zero split overlap, exact saved-answer
rescoring, finite features/scores, exact P6x candidate alignment, and replayed
A/B logits within `1e-3` absolute error.

The convergence gate requires zero convergence warning in all 45 CV fits and
the final refit, with every recorded iteration count below 5000.

A converged hidden recovery additionally requires evaluation candidate AUROC
at least 0.70, exactly-one-missing target selection at least 50%, and hidden
selection to beat the unchanged explicit A/B logit selector by at least five
percentage points.

Outcomes:

- `APPARATUS_FAIL`: a reproduction/alignment gate fails;
- `SOLVER_STILL_FAIL`: the primal optimizer does not converge everywhere;
- `CONVERGED_HIDDEN_RECOVERY`: every convergence and hidden-recovery gate passes;
- `CONVERGED_HIDDEN_NO_ADVANTAGE`: optimization is valid but the stable hidden
  probe still does not beat the explicit output-head readout.

No split, prompt, layer, C value, preprocessing, metric, threshold, or outcome
rule may change after P6r features are observed.
