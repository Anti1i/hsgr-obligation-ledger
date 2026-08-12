# Route-augmented hidden Guide development result (job 726645)

## Boundary

Commit `a9afdfa`; CPU Slurm job `726645`; 840 previously consumed MuSiQue
problems and 6,720 fixed SC@8 candidates; five-fold nested OOF.  The job reused
the frozen `1b074fc` matched/counterfactual feature payload and did not generate
new prompts or read any final377 example.  Job exit code was zero.

The model retained an absolute matched hidden-state encoder and added the
predeclared predecessor-role counterfactual residual.  Its controls used the
same candidate pool and nested-OOF procedure.  Trainable parameter counts were:

| Model | Parameters |
|---|---:|
| route-augmented | 40,513 |
| ordinary-wide hidden verifier | 42,193 |
| activation-wide delta verifier | 39,673 |

The ordinary control therefore had at least as much capacity as the proposed
model; the result cannot be attributed to a smaller ordinary baseline.

## Nested-OOF results

| Policy | Accuracy | vs SC@8 | vs route-augmented |
|---|---:|---:|---:|
| SC@8 | 48.4524% | -- | -5.9524pp |
| explicit predicted state | 50.9524% | +2.5000pp | -3.4524pp |
| activation-wide delta verifier | 54.1667% | +5.7143pp | -0.2381pp |
| ordinary-wide hidden verifier | **54.4048%** | **+5.9524pp** | **0.0000pp** |
| route-augmented hidden Guide | **54.4048%** | **+5.9524pp** | -- |

Route-augmented vs SC@8 had 63 fixes and 13 breaks, exact McNemar
`p=5.04e-9`, Holm-adjusted `p=1.51e-8`, with paired-bootstrap 95% CI
`[+4.05pp, +7.98pp]`.  This establishes selection value for hidden-state
verification, but it does not establish a hierarchy-specific Guide effect.

The decisive parameter-matched comparisons failed:

- vs ordinary-wide: `0.0000pp`, 23 fixes / 23 breaks, `p=1.0`,
  Holm-adjusted `p=1.0`, bootstrap CI `[-1.55pp, +1.55pp]`;
- vs activation-wide: `+0.2381pp`, 24 / 22, `p=0.883`,
  Holm-adjusted `p=1.0`, bootstrap CI `[-1.31pp, +1.90pp]`.

## Structural controls and stability

- route swap: 52.5000%; route-augmented gain `+1.9048pp`, below the frozen
  `+2pp` gate despite nominal paired `p=0.0440`;
- route mismatch: 52.1429%; gain `+2.2619pp`, paired `p=0.0110`;
- outer-fold deltas over ordinary-wide were
  `+1.7857, 0.0000, -1.1905, 0.0000, -0.5952pp`, so only one of five folds
  was positive;
- predeclared ID-hash-half deltas over ordinary-wide were
  `-0.2410pp` and `+0.2353pp`;
- predicted hierarchy depth remained collapsed: 2/3/4 = 16/791/33.

Gold-hop evaluation-only gains over SC@8 were `+6.4018pp`, `+7.9167pp`, and
`+1.3605pp` for 2/3/4-hop problems.  On 4-hop problems the route-augmented
model reached 47.6190%, below both ordinary-wide and activation-wide at
49.6599%.  The frozen depth-signature gate therefore failed.

## Frozen-gate outcome

| Gate | Result |
|---|---|
| beyond parameter-matched ordinary hidden verifier | fail |
| beyond parameter-matched activation delta | fail |
| selection value over SC@8 | pass |
| route controls | fail |
| depth signature | fail |
| OOF stability over ordinary hidden verifier | fail |
| feature validity / capacity accounting | pass |

## Decision

`STOP HIDDEN GUIDE SELECTION ROUTE.  DO NOT CONSUME FINAL377.`

The counterfactual route feature can perturb decisions, as shown by the swap
and mismatch controls, but it adds no reproducible predictive value beyond an
ordinary matched-state hidden verifier.  The exact tie in aggregate accuracy,
opposite-signed hash halves, and failure in four of five folds rule out treating
the SC@8 improvement as evidence for HSGR-style hierarchical guidance.

Per the protocol frozen before this job, further width, seed, or policy-weight
tuning on these 840 labels would be post-hoc optimization and is not a valid
rescue.  Any subsequent main-method attempt must change the source of
identifying evidence rather than repackage candidate-selection verification.
