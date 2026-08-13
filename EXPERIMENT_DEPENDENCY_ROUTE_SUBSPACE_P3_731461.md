# Value-orthogonal route-subspace P3 — job 731461

## Status

- Frozen protocol: `EXPERIMENT_PROTOCOL_DEPENDENCY_ROUTE_SUBSPACE_P3.md`
- Code commit: `6ee55c8`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Data: 96 calibration + 192 held-out cases, seed `20260818`
- Primary layer: 21; fixed diagnostic window: layers 19–21
- Host / accelerator: `xgpi20`, GPU 0, NVIDIA H100 NVL MIG 3g.47gb
- Environment: PyTorch `2.13.0+cu130`
- Slurm: `COMPLETED`, exit 0, elapsed 00:01:44
- Frozen verdict: **APPARATUS_FAIL**

P3 did not earn a route-subspace causal verdict because the model failed the
pre-registered clean task-accuracy requirement once the printed and decoy
branches were assigned distinct values.  Direction and intervention integrity
passed; causal results below are diagnostic only.

## Integrity and route direction

All **288/288** paired source prompts passed the tokenizer controls: identical
length, P checkpoint token index/ID, and exactly one earlier print-target token
difference.  P, X, and corrupted P were distinct digits, so clean, decoy, and
corrupted targets could not collapse to the same answer.

The calibration-only route direction was projected outside an 18-dimensional
digit-value basis.  Direction/value overlaps were `4.3e-8` to `6.5e-8`; sham
value overlaps were `7.7e-8` to `1.4e-7`; route/sham dot products were below
`1.5e-8`.  The direction integrity gate passes.

Route-on versus route-off held-out paired direction accuracy was:

| Layer | Calibration | Held-out | Fixed halves |
|---:|---:|---:|---:|
| 19 | 1.000 | 1.000 | 1.000 / 1.000 |
| 20 | 1.000 | 0.995 | 0.991 / 1.000 |
| 21 | 0.990 | 1.000 | 1.000 / 1.000 |

Thus a consistent route-dependent state displacement survives removal of the
linear digit-value subspace.  This is strong representation evidence, not by
itself evidence that the model uses the direction to decide.

## Why the apparatus failed

On the held-out distinct-value task:

- route-on clean accuracy: **0.760**;
- route-off clean accuracy against its decoy target: **0.714**;
- corrupted receiver accuracy against its own target: **0.776**;
- corrupted receiver accuracy against the clean target: **0.016**;
- clean-minus-corrupt clean-target logp: **+12.476 nats**,
  95% CI **[+11.634, +13.299]**.

The frozen apparatus required both route-on and route-off clean accuracy to be
at least 0.90.  Both failed.  Correct full-state patches strongly restored the
clean value, so hook efficacy was not the problem: their clean-target gain over
corrupt was +12.623 nats / +72.92 points at layer 21 and +12.815 nats / +76.04
points over layers 19–21.  The failure is specifically that the base model does
not reliably follow the print route when the two branches have different
values.

## Diagnostic intervention outcomes

All route-subspace edits were value-orthogonal and compared against the average
of equal-norm positive and negative sham edits.

| Mode | Sham − route, clean logp | 95% CI | Clean acc. change | Route − sham, decoy logp | Decoy acc. change |
|---|---:|---:|---:|---:|---:|
| Layer 21 | +0.0176 | [+0.0053,+0.0304] | +1.04 pp | +0.0594 | +0.00 pp |
| Layers 19–21 | +0.0758 | [+0.0494,+0.1052] | +2.60 pp | +0.2779 | +1.82 pp |

At layer 21, the targeted edit is practically equivalent to sham under the
frozen bounds.  Sustaining it through layers 19–21 produces the predicted
two-sided movement—lower clean-branch score and higher decoy score—and is
stable across fixed halves (+0.067/+0.087 clean-logp differences).  However it
misses the strict switch gate: clean logp is below +0.10, clean accuracy below
+3 points, and decoy accuracy below +3 points.  These outcomes cannot be
promoted after observing them, especially because the apparatus failed.

For comparison, the full route-off state over layers 19–21 reduced clean logp
by +0.195 nats, 95% CI [+0.130,+0.269], and clean accuracy by +3.65 points.
This suggests route information may be distributed and repeatedly
reconstructed, but remains diagnostic under the failed base task.

## Interpretation and next decision

P3 strengthens one conclusion and leaves another unresolved:

1. A value-orthogonal route representation is exceptionally stable and
   generalizes perfectly or nearly perfectly across held-out cases.
2. This run cannot establish its causal use because the model itself solves the
   distinct-value route task only about 71–76% of the time.

Do not tune thresholds or intervention scale on these held-out cases.  One
apparatus repair is justified: a new-seed, **apparatus-only prompt screen** over
pre-specified renderings, using only route-on/off clean accuracy, corrupt-own
accuracy, and corruption sensitivity.  It must extract no hidden states and
perform no patches.  If no rendering reaches both clean accuracies >=0.90, stop
the present causal mechanism line and move to external Guide utility.  If one
passes, rerun the unchanged route-subspace question on another disjoint seed.
