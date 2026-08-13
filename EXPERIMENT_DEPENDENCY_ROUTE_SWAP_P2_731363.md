# Same-position dependency route-swap P2 — job 731363

## Status

- Frozen protocol: `EXPERIMENT_PROTOCOL_DEPENDENCY_ROUTE_SWAP_P2.md`
- Code commit: `021fea5`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Data: 96 calibration + 192 held-out one-hop cases, seed `20260817`
- Locked residual layer: 21, inherited from P1 before P2 outcomes
- Host / accelerator: `xgpi20`, GPU 0, NVIDIA H100 NVL MIG 3g.47gb
- Environment: PyTorch `2.13.0+cu130`
- Slurm: `COMPLETED`, exit 0, elapsed 00:01:24
- Frozen verdict: **POSITION_CONFOUND_SUPPORTED**

P2 was a confirmatory control for P1's remaining donor-position confound.  The
result does not reproduce P1's large correct-role advantage when source token
position is held fixed.

## Control integrity

All **288/288** paired donors passed the runtime tokenizer invariants:

- route-on and route-off prompts had equal token length;
- their P checkpoint digit had the same token index and token ID;
- tokenized prompts differed at exactly one earlier print-target token;
- the donor token, digit, label, absolute position, checkpoint layout, and local
  checkpoint context were identical.

The corrupted receiver differed from route-on clean only at P's checkpoint
digit.  Thus P2 isolates the route declaration more strictly than P1, whose
correct and wrong-route donors came from different checkpoint positions.

## Readable route signal

The fixed P0/P1 linear probe at locked layer 21 strongly distinguished the
same-position P state according to whether the earlier program printed its
branch:

- calibration OOF paired accuracy: **0.979** (94 wins, 2 losses);
- held-out paired accuracy: **0.953** (183 wins, 9 losses);
- held-out pooled AUROC: **0.681**;
- one-sided sign-test p: **1.35e-43**;
- fixed hash halves: **0.978** and **0.932**;
- identical-metadata control: **0.500**, all 192 pairs tied.

The paired and pooled metrics answer different questions.  The high paired
accuracy shows a highly consistent within-case route-dependent displacement;
the lower pooled AUROC shows that absolute hidden states also vary substantially
across cases.  The frozen representation gate passes.

## Task and patch apparatus

The task and intervention remained valid on the held-out cases:

- route-on clean accuracy: **1.000**;
- route-off clean accuracy: **1.000**;
- corrupted receiver accuracy against its own answer: **0.750**;
- corrupted receiver accuracy against the route-on clean answer: **0.161**;
- clean-minus-corrupt clean-answer logp: **+10.398 nats**,
  95% CI **[+9.403, +11.381]**.

The route-on same-position patch restored accuracy to **1.000**, improved
clean-answer logp over corrupt by **+10.378 nats**, 95% CI
**[+9.360, +11.366]**, and recovered **99.81%** of the corruption gap.  The
apparatus gate passes.

## Same-position causal contrast

| Arm at locked layer 21 | Clean-answer accuracy | Mean clean-answer logp |
|---|---:|---:|
| Correct route, same P position | 1.000 | -0.0210 |
| Wrong route, same P position | 0.990 | -0.0303 |
| Cross-problem, value-matched P | 0.984 | -0.0786 |

The primary correct-minus-wrong contrast was:

- logp difference: **+0.00927 nats**, 95% CI
  **[+0.00218, +0.01949]**;
- accuracy difference: **+1.04 percentage points**, 95% CI
  **[0.00, +2.60]**;
- fixed hash-half logp differences: **+0.01125** and **+0.00755** nats.

The small effect is statistically direction-consistent, but it is far below
the frozen causal thresholds of +0.10 nats and +3 percentage points.  More
importantly, both confidence intervals lie wholly inside the preregistered
practical-equivalence regions of [-0.10,+0.10] nats and [-3,+3] percentage
points.  Therefore the frozen verdict is **POSITION_CONFOUND_SUPPORTED**, not
merely a non-significant or underpowered causal failure.

## Revised interpretation

P2 separates two facts:

1. The checkpoint hidden state contains a strong, held-out-readable signal
   about whether its branch is selected by the earlier print route.
2. Under a single-site layer-21 replacement, that route difference has only a
   practically negligible effect on the answer once token value and source
   position are matched.

Consequently, P1's +0.262-nat and +7.81-point correct-versus-wrong effect should
not be cited as clean dependency-role causality.  The large P1 contrast was
substantially driven by source-position/context compatibility.  P2 does not
prove that route information is never used: later computation can reconstruct
the route from the unchanged prompt, and a one-layer whole-state replacement
may not isolate the readable route component.  It does establish that the
specific P1 activation-patching evidence is insufficient for a causal HSGR or
hierarchy claim.

## Stop decision

Do not proceed directly to depth progression, because P2 did not earn the
protocol's `ROUTE_SWAP_CONFIRM` prerequisite.  The defensible result is now a
**representation–utilization separation** on a controlled one-hop route:
strong route readability with practical causal equivalence under the frozen
single-site intervention.

Any continuation should change the scientific question explicitly.  A
mechanistic follow-up could isolate the calibrated route direction and test a
pre-registered multi-layer or subspace intervention, with matched sham and
value-preservation controls.  A method-oriented HSGR follow-up could instead
test whether an external Guide can use the readable route signal to improve a
separately validated difficult task.  Neither continuation may treat P1 as
already proving endogenous hierarchical control.
