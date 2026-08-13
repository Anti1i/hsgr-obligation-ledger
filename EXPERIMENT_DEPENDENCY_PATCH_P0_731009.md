# Dependency-role activation patching P0 — job 731009

## Status

- Frozen protocol: `EXPERIMENT_PROTOCOL_DEPENDENCY_PATCH_P0.md`
- Code commit: `cceab9f`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Data: 96 calibration + 192 held-out controlled programs
- Candidate layers: 7, 14, 21
- Host / accelerator: `xgpi20`, GPU 0, NVIDIA H100 NVL MIG 3g.47gb
- Slurm: `COMPLETED`, exit 0, elapsed 00:06:38
- Frozen verdict: **REPRESENTATION_FAIL**

The earlier job 729019 completed model computation but failed while serializing
a NumPy scalar.  Job 731009 is the unchanged experiment after a serialization-
only repair; no feature, intervention, split, layer, or gate was changed.

## Representation result

Calibration selected layer 21 by the frozen tie-breaking rule, despite all
three calibration results being below chance:

| Layer | Calibration paired accuracy | Calibration pooled AUROC |
|---:|---:|---:|
| 7 | 0.448 | 0.464 |
| 14 | 0.458 | 0.458 |
| 21 | 0.458 | 0.479 |

At selected layer 21, the held-out P-versus-X result was:

- paired accuracy: **0.552** (106 wins, 86 losses);
- pooled AUROC: **0.561**;
- one-sided sign-test p: **0.085**;
- fixed hash halves: **0.644** and **0.476**;
- metadata-only paired accuracy: **0.490**;
- hidden-minus-metadata: **+0.063**, below the frozen +0.10 requirement.

This does not establish a stable, held-out representation of whether a matched
checkpoint variable is an ancestor of the printed output.

## Task and intervention validity

The model did not solve the direct-answer executable-program task:

- clean digit accuracy: **0.099**;
- corrupted prompt accuracy against its own executable answer: **0.068**;
- clean-minus-corrupt clean-answer log probability: **+0.262 nats**,
  bootstrap 95% CI **[+0.089, +0.436]**.

Thus the corruption changed the model's score in the expected direction, but
the model's exact execution accuracy was approximately ten-class chance.  The
frozen apparatus gate required clean accuracy >=0.50 and corrupt-own accuracy
>=0.40, so it failed at every layer.

The root-positive patch strongly raised clean-answer log probability at every
layer (+2.27 to +2.41 nats over corrupt, all positive confidence intervals),
showing that the hook can causally change the output distribution.  It did not
raise digit accuracy, which is unsurprising given the invalid base task.

## Causal specificity (diagnostic only)

Because both the representation and apparatus gates failed, these values are
not evidence for or against endogenous structural use:

| Layer | correct − wrong logp (nats) | 95% CI | Accuracy difference |
|---:|---:|---:|---:|
| 7 | +0.061 | [+0.024, +0.098] | -1.56 pp |
| 14 | -0.006 | [-0.052, +0.037] | -1.04 pp |
| 21 (selected) | +0.026 | [-0.016, +0.067] | -1.56 pp |

Layer 7 has a small positive log-probability difference but misses the frozen
+0.10-nat threshold and has worse accuracy.  It cannot be reselected after
seeing causal outcomes.

## Interpretation and stop decision

P0 supports only the local conclusion that this model/prompt/probe did not
provide a stable dependency-role representation.  It does **not** establish a
representation-utilization gap, and it does not show that hierarchy is absent
from LLM hidden states generally, because the model could not perform the base
task.

Do not tune the probe, layer, or thresholds on these 192 held-out examples.
Before a new confirmatory patch experiment, run a disjoint apparatus-only
screen over pre-specified simpler but topology-matched programs.  Select a task
using only executable-answer accuracy and corruption sensitivity, without
observing P/X probe or correct-versus-wrong-route outcomes.  A new seed and new
examples are required for any subsequent representation/causal test.
