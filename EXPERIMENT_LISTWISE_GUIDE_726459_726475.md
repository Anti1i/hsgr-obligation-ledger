# Hidden-state listwise Guide development results

## Claim boundary

All three runs train on the already consumed development 200 plus first fresh
320 problems and evaluate on the already observed second fresh 320 problems.
They use oracle decompositions, verified predecessor values, and gold support
routing.  These are development diagnostics, not end-to-end or fresh claims.
The final 377-problem split remains sealed.

## Shared architecture

For every sampled answer candidate and layer 14/21/28, the hidden reader uses
the directed route margin and interaction
`[h_correct - h_wrong, h_correct * h_wrong]`.  Candidate-outcome supervision
combines class-balanced BCE with within-problem pairwise ranking.  The policy
adds this Guide signal to the explicit predecessor guard.  Non-hidden, route
swap, route mismatch, and length controls use the same evaluation path.

## Results

| Run | Change | Accuracy | vs SC8 | vs explicit | Key controls | Frozen gates |
|---|---|---:|---:|---:|---|---|
| 726459 (`f2267ce`) | shared listwise Guide | **51.5625%** | **+7.5000pp** | **+2.8125pp**, 13 fixes / 4 breaks, p=0.0490 | non-hidden 48.7500%; swap 44.6875%; mismatch 50.0000%; length 49.6875% | 7/8; depth failed |
| 726471 (`1a4d68b`) | equal `(hop,class)` loss and equal-hop rank loss | **51.5625%** | **+7.5000pp** | **+2.8125pp**, 12 / 3, p=0.0352 | non-hidden 48.7500%; swap 40.3125%; mismatch 50.6250% | 6/8; coupling and depth failed |
| 726475 (`f9611f8`) | OOF-selected policy weights per hop | 50.9375% | +6.8750pp | +2.1875pp, 11 / 4, p=0.1185 | weights 2/3/4 = 0.5/0.3/0.2; mismatch 50.3125% | 6/8; coupling and depth failed |

SC8 is 44.0625% and the explicit predecessor guard is 48.7500% on the second
320-problem development set.

### Depth audit

The original shared model has gains over SC8 of +7.6503pp, +8.7500pp, and
+5.2632pp for 2/3/4-hop problems.  Its frozen depth gate requires the 4-hop
gain to be at least the 2-hop gain minus 1pp, or +6.6503pp.  It misses by
1.3871pp: one additional correct 4-hop problem would pass.  Importantly, the
original model's 4-hop changes relative to explicit are two fixes and zero
breaks, so this is not evidence of a harmful long-chain intervention.

Training has 270/160/90 problems at 2/3/4 hops.  Equal-hop training did not
change the 4-hop result and instead improved only 2-hop, rejecting simple
frequency imbalance as the cause.  Independent OOF calibration selected a
smaller 4-hop weight and removed both held-development 4-hop fixes, showing
that a separate estimate from only 90 training problems has excessive
variance and poor transfer.

## Decision

`DO NOT CONSUME FINAL HOLDOUT` remains binding.  The original shared model is
the strongest candidate: it demonstrates outcome value, hidden-state
contribution, route directionality, and predecessor coupling, but it does not
pass the frozen depth signature.

Do not repeat hop-balanced loss or independently tuned per-hop policy weights.
The next technically justified development is a shared hidden reader with a
small, shrinkage-regularized depth-conditioned modulation.  It should retain
pooled training and a single OOF policy weight while allowing the Guide to
interpret the same route-state feature differently by reasoning depth.
Required controls remain non-hidden, route swap, route mismatch, and a
depth-label permutation/ablation.  The eight frozen gates must remain
unchanged before the final split is considered.
