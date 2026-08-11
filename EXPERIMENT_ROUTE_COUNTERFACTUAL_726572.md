# Route-counterfactual hidden Guide development result (job 726572)

## Boundary

Commit `1b074fc`; Qwen2.5-7B-Instruct; 840 previously consumed MuSiQue
problems; five-fold nested OOF.  The experiment predicts the hierarchy and
predecessor values without gold structure annotations.  It retains the fixed
open-book evidence used by SC@8, so this is a selection result, not a retrieval
claim.  The final 377 problems were not read.

Matched and counterfactual prompts contain identical question, evidence,
predicted nodes, predicted values, candidate, and role-label multiset.  The
intervention swaps `PARENT` and `NONPARENT` roles between one predicted parent
and a predicted foil.  All 6,720 primary prompt pairs had equal token length;
no candidate prompt was truncated.

## Nested-OOF results

| Policy | Accuracy | vs SC@8 |
|---|---:|---:|
| SC@8 | 48.4524% | -- |
| explicit predicted state | 50.9524% | +2.5000pp |
| non-hidden listwise | 51.3095% | +2.8571pp |
| route-counterfactual Guide | **52.3810%** | **+3.9286pp** |
| same-prompt activation delta | 54.1667% | +5.7143pp |
| same-prompt ordinary hidden verifier | **55.0000%** | **+6.5476pp** |

Guide vs SC@8: 50 fixes / 17 breaks, exact McNemar
`p=6.74e-5`, Holm-adjusted `p=2.02e-4`, paired bootstrap 95% CI
`[+2.02pp, +5.83pp]`.

The decisive comparisons go the other way:

- Guide vs ordinary hidden: **-2.6190pp**, 20 fixes / 42 breaks,
  `p=0.00715`, Holm-adjusted `p=0.0143`;
- Guide vs activation delta: **-1.7857pp**, 18 / 33,
  `p=0.0489`, paired bootstrap CI `[-3.45pp, -0.12pp]`.

Controls:

- Guide vs route swap: +1.7857pp, below the frozen +2pp gate;
- Guide vs route mismatch: +1.5476pp, `p=0.0106`, pass;
- state/depth-label permutation: exactly 0 change, fail;
- predicted depth collapsed to 2/3/4 = 16/791/33 problems;
- gold-hop evaluation gains over SC@8 were +4.42/+4.58/+1.36pp for
  2/3/4-hop problems, so the depth signature failed.

## Decision

`DO NOT CONSUME FINAL377`.

The route intervention contains a real coupling signal, but using the
counterfactual response alone discards stronger absolute-state correctness
information.  It is not a viable main method and cannot be presented as
superior to current hidden-state verification or activation-delta reranking.

One final development diagnostic is justified without new feature extraction:
retain the absolute matched hidden representation and add a small structural
counterfactual residual.  It must beat parameter-matched wide ordinary-hidden
and activation-delta models, not merely SC@8.  If it fails, stop the hidden
Guide selection route.
