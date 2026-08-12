# Oracle relation-focus action ceiling (job 725772)

## Boundary

Commit `e1d3c6d`; Qwen2.5-7B-Instruct; 400 fresh-at-the-time MuSiQue units.
Every arm used the same programmatically compiled final-hop goal, verified
predecessor values, original question, and full gold-support evidence.  The
oracle selected the correct final-hop support block; a matched control selected
a predecessor block.  This is an oracle action ceiling, not a deployable
hidden-state method.

## Result

| Arm | Normalized EM | Official answer F1 |
|---|---:|---:|
| base execution | 48.50% | 0.6343 |
| neutral generic retry | **52.75%** | 0.6587 |
| oracle correct relation focus | 52.50% | 0.6740 |
| matched wrong relation focus | 52.50% | **0.6743** |

Correct focus was -0.25pp below neutral (5 fixes / 6 breaks, `p=1.0`) and
exactly tied wrong focus (3 / 3, `p=1.0`).  Correct-focus recovery among the
206 base errors was 7.77%.  Its +0.0153 F1 over neutral was not significant
under the paired sign-flip test (`p=0.0878`).

The correct-minus-neutral EM delta increased with gold depth
(-2.56/+0.75/+4.17pp for 2/3/4 hop), but both route-headroom and
route-specificity gates failed.  A depth trend cannot rescue zero aggregate
route specificity.

## Decision

`DO NOT TRAIN A HIDDEN ROUTE CONTROLLER FROM THIS ACTION.`

Knowing and marking the oracle relation did not improve normalized exact match
over a generic retry and did not outperform a wrong route.  The current
full-evidence MuSiQue execution setting therefore supplies no causal action
headroom for a relation-focus Guide.

