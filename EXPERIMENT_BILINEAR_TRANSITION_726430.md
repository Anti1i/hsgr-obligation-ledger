# Skew-bilinear transition diagnostic (job 726430)

## Status

**DO NOT CONSUME FINAL HOLDOUT.** This CPU-only diagnostic used the already observed second fresh feature set and left the remaining 377 problems untouched.

## Result

| Policy | Accuracy | Delta vs SC@8 |
|---|---:|---:|
| SC@8 | 44.0625% | - |
| Explicit predecessor guard | 48.7500% | +4.6875 pp |
| Length control | 49.6875% | +5.6250 pp |
| Mismatched predecessor control | 49.6875% | +5.6250 pp |
| Skew-bilinear transition | **50.6250%** | **+6.5625 pp** |
| Sign-swapped control | 46.5625% | +2.5000 pp |

The bilinear policy made 22 fixes and 1 break relative to SC@8 (`p=5.72e-6`). Relative to the explicit guard it made 8 fixes and 2 breaks, gaining 1.875 pp. The development-selected weight was 0.15.

The antisymmetric interaction produced clear directionality and a real mismatch effect, but both the beyond-length and route-coupling margins were 0.9375 pp, just below the frozen 1 pp gates. The held pooled structural AUROC was 0.6495, below 0.70, although strict within-problem AUROC was 0.8667 on 15 eligible problems. It also trailed the frozen TransitionGuard reference by 0.3125 pp.

The interaction fixes were not merely a subset of the linear paired reader: only 5 of 8 fixes overlapped; the bilinear arm had 3 unique fixes and 1 unique break. This motivates a development-selected two-channel consensus while keeping mismatch, direction, length, and paired-only controls mandatory.

