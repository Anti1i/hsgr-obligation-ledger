# One-hop dependency-role activation patching P1 — job 731174

## Status

- Frozen protocol: `EXPERIMENT_PROTOCOL_DEPENDENCY_PATCH_P1.md`
- Code commit: `67c04ac`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Data: 96 calibration + 192 held-out `chain1_copy` cases, seed `20260816`
- Candidate residual layers: 7, 14, 21
- Host / accelerator: `xgpi13`, GPU 0, NVIDIA H100 NVL (95,830 MiB)
- Environment: PyTorch `2.13.0+cu130`
- Slurm: `COMPLETED`, exit 0, elapsed 00:01:37
- Frozen verdict: **CAUSAL_PASS** at layer 21

**Subsequent-control note:** P2 job 731363 held the donor token position fixed
and found the route-on/off contrast practically equivalent.  P1's frozen
within-experiment verdict remains recorded, but its causal dependency-role
interpretation is superseded by
`EXPERIMENT_DEPENDENCY_ROUTE_SWAP_P2_731363.md`; the large P1 effect cannot be
used as clean hierarchy or HSGR-control evidence.

This is the first experiment in this line for which the base task, held-out
representation gate, patch apparatus, and dependency-specific causal gate all
pass their frozen criteria.  Its claim boundary remains deliberately narrow:
the result concerns a one-edge identity dependency, not a multi-level
hierarchy, an HSGR Guide, or downstream performance improvement.

## Representation result

Calibration selected layer 21 without observing held-out or causal results:

| Layer | Calibration paired accuracy | Calibration AUROC | One-sided sign p |
|---:|---:|---:|---:|
| 7 | 0.448 | 0.445 | 0.869 |
| 14 | 0.500 | 0.463 | 0.541 |
| 21 | **0.656** | **0.628** | **0.00144** |

At the selected layer, the disjoint held-out P-versus-X probe obtained:

- paired accuracy: **0.714** (137 wins, 55 losses);
- pooled AUROC: **0.697**;
- one-sided sign-test p: **1.46e-9**;
- fixed ID-hash halves: **0.693** and **0.736**;
- metadata-only paired accuracy: **0.510**;
- hidden-minus-metadata paired accuracy: **+0.203**.

All frozen representation requirements pass.  Randomized variable labels and
counterbalanced program/checkpoint order make a fixed label or simple position
explanation inadequate, although this probe alone does not identify the exact
encoded feature.

## Task and patch apparatus

The one-hop identity task remained valid on the new held-out cases:

- clean-answer accuracy: **1.000**;
- corrupted prompt accuracy against its own executable answer: **0.734**;
- corrupted prompt accuracy against the clean answer: **0.161**;
- clean-minus-corrupt clean-answer log probability: **+10.633 nats**,
  bootstrap 95% CI **[+9.649, +11.628]**.

At selected layer 21, the same-case clean-P patch restored clean-answer
accuracy to **0.990** and raised clean-answer log probability over the corrupt
run by **+10.616 nats**, 95% CI **[+9.610, +11.605]**.  The patch apparatus gate
therefore passes independently of the role-specific comparison.

## Dependency-specific causal result

All arms began from the same corrupted receiver.  The source states carried
the same clean digit; they differed in whether that source checkpoint was the
printed branch (`correct_role`), the unprinted branch in the same case
(`wrong_route`), or the printed branch in another value-matched case
(`cross_problem`).

| Layer | Correct acc. | Wrong-route acc. | Correct − wrong logp | 95% CI | Accuracy difference | Frozen causal gate |
|---:|---:|---:|---:|---:|---:|---|
| 7 | 1.000 | 1.000 | +0.0067 | [+0.0010, +0.0142] | +0.00 pp | FAIL |
| 14 | 0.995 | 1.000 | -0.0017 | [-0.0049, +0.0008] | -0.52 pp | FAIL |
| 21 | **0.990** | **0.911** | **+0.2619** | **[+0.1503, +0.3960]** | **+7.81 pp** | **PASS** |

At layer 21 the correct-minus-wrong log-probability differences in the two
fixed halves were **+0.3775** and **+0.1336** nats, and the correct patch
recovered **99.84%** of the clean-versus-corrupt log-probability gap.  Thus the
effect is layer-localized, exceeds both frozen effect-size thresholds, has a
strictly positive interval, and is directionally stable across the fixed
halves.  Layer 21 was selected by calibration representation performance, not
by these causal outcomes.

For context, the layer-21 cross-problem printed-branch arm reached 0.958
accuracy and mean correct-answer log probability -0.161, between the correct
same-case arm (-0.021) and wrong-route same-case arm (-0.283).  This comparison
was reported but was not a frozen causal gate.

## What the result does and does not establish

The frozen **CAUSAL_PASS** is evidence that, in this controlled one-hop task,
late residual states contain readable information associated with dependency
role and that replacing the corrupted checkpoint state with the same-case
printed-branch state is more effective than using an equal-valued unprinted-
branch state.  This is stronger than the earlier observation that hidden state
has generic correctness signal.

It is not yet clean evidence for hierarchy.  There is only one dependency
edge, while the apparatus screen showed that three-edge routing was unreliable.
There is also a remaining source-compatibility confound: the correct donor is
the clean version of the receiver's own token position, whereas the wrong-route
donor comes from another token position.  Randomization and the metadata
control weaken simple positional explanations but do not make the causal donor
positions identical.

The next confirmatory experiment should therefore be a **same-position
route-swap control**.  It must hold the donor token, value, absolute position,
labels, and surrounding checkpoint layout fixed, changing only an earlier
`print(P)` versus `print(H)` route declaration.  Correct- and wrong-role states
can then be taken from the same physical checkpoint position in matched
prompts.  Only if that contrast replicates should a separately frozen depth-1
versus depth-2/3 progression be used to test a genuinely hierarchical claim.
