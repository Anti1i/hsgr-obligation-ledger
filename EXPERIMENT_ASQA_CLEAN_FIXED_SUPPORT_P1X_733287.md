# ASQA clean fixed-support candidate-node screen P1x — job 733287

## Verdict

`HEADROOM_AND_STRUCTURE_PASS`.  On the frozen clean/support-complete ASQA
subset, Qwen2.5-7B has meaningful strict all-facet headroom and the true facet
nodes improve coverage substantially beyond an equally sized irrelevant
checklist.  All eight frozen P1x gates pass.

This is an oracle textual-structure ceiling.  It licenses the separately
frozen candidate-node hidden-state P2x apparatus; it is not itself HSGR and
does not retroactively change the `BORDERLINE` P0 verdict.

## Reproducibility

- Protocol: `EXPERIMENT_PROTOCOL_ASQA_CLEAN_FIXED_SUPPORT_P1X.md`
- Code/protocol commit: `892a7b4`
- Slurm job: `733287`, `xgpi14`, H100 NVL MIG 3g.47gb, 8m53s, exit `0:0`
- Model: `Qwen/Qwen2.5-7B-Instruct`, bfloat16, greedy, 192 new-token cap
- Sample: 192 of 427 clean eligible examples
- Facet histogram: 125/33/21/9/4 examples with 2/3/4/5/6 nodes
- Result directory: `/mnt/scratch/z/zitong/dch-hsgr/results/asqa_clean_fixed_support_p1x_733287`
- Report SHA-256: `d584ed2aa8e0ff13312c02e8681dac0c9206cade10a862d0a05de9cbfbfd4912`
- Generations SHA-256: `4f39958c99b86987c49008b3abf22c0367274502950b45dabf66cf6ab50f1f28`
- Selected IDs SHA-256: `25f29940b9f4190fdc7b37ca2f6bc52ffd8bb1a6e7dfc3913f1179b847dce9d3`

## Absolute results

| Arm | STR-EM | STR-HIT | Median words |
|---|---:|---:|---:|
| closed book | 30.86% | 11.98% | 114.0 |
| fixed direct | 70.76% | 43.75% | 100.5 |
| true facet nodes | 78.96% | 57.81% | 109.5 |
| matched decoy nodes | 70.78% | 45.31% | 99.5 |

## Frozen comparisons

- Fixed direct minus closed book: +39.90 points STR-EM and +31.77 points
  STR-HIT.
- True nodes minus matched decoys: +8.18 points STR-EM and +12.50 points
  STR-HIT.
- True nodes minus fixed direct: +14.06 points STR-HIT.
- Decoys minus fixed direct: only +1.56 points STR-HIT.
- Paired true-versus-decoy strict successes: 34 true-only versus 10
  decoy-only; exact two-sided McNemar `p = 0.000388`.

All three difficulty gates and all five structure gates pass.  The gain is
therefore not explained by fixed evidence alone, output length, checklist
format, or checklist count.

## Corrected claim

ASQA supports a useful **candidate-node** domain on the frozen clean subset:
the identities of answer facets materially affect strict long-answer coverage.
It does not show that each node contains multiple useful candidate values, and
it does not yet show that hidden states can identify which node will be missed.
Those are separate questions.  P2x now tests the latter using answer-prefix
hidden states, surface controls, and a wrong-node representation control.
