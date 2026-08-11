# Paired directed-transition diagnostic (job 726427)

## Status

**DO NOT CONSUME FINAL HOLDOUT.** This was a CPU-only development diagnostic on the already observed 320-problem second fresh set. It did not use the remaining 377 untouched problems.

## Method

A single structural reader was trained on the normalized hidden-state difference `h_correct - h_predecessor`. Training and weight selection used only problem-disjoint OOF predictions on the original 200-problem development set. Strict structural supervision used candidates mentioned by one Guide route but not the other.

The same trained reader was reused for two causal controls:

- sign swap: `h_predecessor - h_correct`;
- mismatch: rotate predecessor hidden states among the eight candidates of the same problem, preserving route marginals while breaking candidate-level coupling.

## Result

| Policy | Accuracy | Delta vs SC@8 |
|---|---:|---:|
| SC@8 | 44.0625% | - |
| Explicit predecessor guard | 48.7500% | +4.6875 pp |
| Length control | 49.6875% | +5.6250 pp |
| Mismatched predecessor control | 50.3125% | +6.2500 pp |
| Paired transition | **50.9375%** | **+6.8750 pp** |
| Sign-swapped control | 38.1250% | -5.9375 pp |

The paired transition made 26 fixes and 4 breaks relative to SC@8 (`p=5.95e-5`). Relative to the explicit guard it made 12 fixes and 5 breaks, gaining 2.1875 pp. The development-selected weight was 0.5.

The structural reader transferred (pooled AUROC 0.8152 dev / 0.7822 held; strict within-problem AUROC 1.0 on 11 / 15 eligible problems). Directionality, headroom, safety, length, and frozen-reference gates passed.

The route-coupling gate failed: correct candidate pairing beat the within-problem mismatch by only 0.625 pp, below the required 1 pp. A linear difference can decompose into independent route terms and remain dominated by the destination route. The next diagnostic therefore uses a skew-symmetric bilinear feature that has no single-route term.

