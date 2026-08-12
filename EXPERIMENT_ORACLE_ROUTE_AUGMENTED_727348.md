# Oracle-structure route-residual diagnostic (job 727348)

## Boundary

Commit `ac7ab78`; CPU Slurm job `727348` on `xcnz5`; 840 previously consumed
MuSiQue problems and 6,720 fixed SC@8 candidates; five-fold nested OOF.  The
three cached feature sets use oracle decomposition, verified predecessor
values, and gold support routing.  No new model prompts were generated and
final377 was not read.  The job completed in 1m30s with exit code zero and all
seven cluster-side tests passing.

This is a root-cause diagnostic, not an end-to-end result.  It asks whether the
route residual has value beyond an ordinary hidden verifier when structure
induction is made perfect by oracle information.

## Capacity-matched result

| Policy | Accuracy | vs SC@8 | vs route-augmented |
|---|---:|---:|---:|
| SC@8 | 48.4524% | -- | -8.9286pp |
| explicit oracle-route state | 53.3333% | +4.8810pp | -4.0476pp |
| ordinary-wide hidden verifier | 56.7857% | +8.3333pp | -0.5952pp |
| oracle route-augmented Guide | **57.3810%** | **+8.9286pp** | -- |

Trainable parameters were 40,513 for route-augmented and 42,193 for
ordinary-wide.  The ordinary control therefore had at least as much capacity.

Route-augmented vs SC@8 made 84 fixes and 9 breaks (`p=2.17e-16`,
Holm-adjusted `p=6.51e-16`, bootstrap 95% CI `[+6.79pp,+11.19pp]`).  It also
beat the explicit oracle-route policy by +4.0476pp (43/9,
Holm-adjusted `p=4.08e-6`).  These comparisons establish hidden-verification
selection value, not a distinct route effect.

The decisive comparison failed:

- vs parameter-matched ordinary hidden: **+0.5952pp**;
- 22 fixes / 17 breaks, exact paired `p=0.522`;
- Holm-adjusted `p=0.522`;
- paired-bootstrap 95% CI `[-0.83pp,+2.02pp]`.

The observed margin is below the frozen +1pp gate, non-significant, and
compatible with a small ordinary-verifier advantage.

## Structural controls and stability

The route perturbation was behaviorally coupled but not reliably useful:

- route swap: 49.4048%; primary gain +7.9762pp, `p=2.30e-11`;
- route mismatch: 54.7619%; primary gain +2.6190pp, `p=4.72e-4`;
- outer-fold deltas over ordinary-wide:
  `0.0000, -1.7857, +3.5714, +1.7857, -0.5952pp`;
- ID-hash-half deltas: `+0.4819pp` and `+0.7059pp`.

Only two of five outer folds were positive, so the frozen four-of-five
stability gate failed.  The hash halves alone do not override this failure.

Gold-hop deltas over ordinary-wide were:

| Gold hops | n | Route-augmented | Ordinary-wide | Delta |
|---|---:|---:|---:|---:|
| 2 | 453 | 58.7196% | 58.7196% | 0.0000pp |
| 3 | 240 | 55.8333% | 55.4167% | +0.4167pp |
| 4 | 147 | 55.7823% | 53.0612% | +2.7211pp |

The depth signature and route controls passed.  This supports the narrow
statement that oracle route contrast affects decisions, especially on 4-hop
examples.  It does not satisfy the primary requirement of predictive value
beyond ordinary hidden verification.

## Frozen-gate outcome

| Gate | Result |
|---|---|
| beyond parameter-matched ordinary hidden verifier | **fail** |
| selection value over SC@8 | pass |
| beyond explicit oracle-route state | pass |
| route swap and mismatch controls | pass |
| four-of-five OOF and hash-half stability | **fail** |
| depth signature vs ordinary | pass |
| feature and capacity validity | pass |

## Decision

`ORACLE ROUTE RESIDUAL FAILS.  DO NOT INVEST IN A STRUCTURE PREDICTOR FOR THIS
SELECTION MECHANISM.  FINAL377 REMAINS SEALED.`

Predicted-hierarchy collapse was a real defect in job `726645`, but it was not
the decisive cause: removing it with oracle structure still produced less
than one point of unstable residual gain over an ordinary hidden verifier.
Training a stronger parser would add another learned component without fixing
the primary identification problem.  The hidden Guide selection line is
therefore closed rather than retuned post hoc.

