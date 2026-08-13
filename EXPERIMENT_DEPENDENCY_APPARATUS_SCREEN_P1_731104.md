# Dependency-role apparatus-only screen — job 731104

## Status

- Frozen protocol: `EXPERIMENT_PROTOCOL_DEPENDENCY_APPARATUS_SCREEN_P1.md`
- Code commit: `5050b69`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Seed: `20260815`; 96 examples per pre-specified family
- Host / accelerator: `xgpi13`, GPU 0, NVIDIA H100 NVL
- Slurm: `COMPLETED`, exit 0, elapsed 00:01:09
- Verdict: **APPARATUS_PASS**, selected family `chain1_copy`

This stage extracted no hidden states and performed no activation patches.  It
selected a task family only from executable-answer accuracy and corruption
sensitivity.

## Results

| Fixed order | Family | Clean accuracy | Corrupt-own accuracy | Clean − corrupt clean-answer logp | 95% CI | Gate |
|---:|---|---:|---:|---:|---:|---|
| 1 | `dag_add` | 0.063 | 0.104 | +0.431 | [+0.172, +0.706] | FAIL |
| 2 | `chain3_add` | 0.167 | 0.094 | +0.133 | [+0.017, +0.254] | FAIL |
| 3 | `chain3_copy` | 0.990 | 0.458 | +3.840 | [+3.001, +4.703] | FAIL |
| 4 | `chain1_copy` | 1.000 | 0.667 | +9.457 | [+7.920, +10.981] | PASS |

The frozen gate required clean accuracy >=0.60, corrupt-own accuracy >=0.50,
log-probability decrease >=0.20 nats, and a positive CI lower bound.  Selection
used the first passing family in the fixed hardest-to-easiest order.

## Interpretation

The arithmetic DAG and three-step addition chain are not valid direct-answer
tasks for this model.  More importantly, the model solves a three-step identity
chain almost perfectly only while the relevant P value equals the decoy X
value.  When P is corrupted away from X, it follows the true printed branch in
only 45.8% of cases.  Thus arithmetic difficulty is not the whole P0 failure:
dependency routing itself degrades across three variable-renaming edges.

Only the one-edge identity family provides a valid minimum apparatus.  It is
not sufficient evidence for hierarchical reasoning.  It is useful as a strict
basic-case falsification: a new-seed P1 may ask whether the hidden state can
distinguish and causally use the one-hop printed branch versus a value-matched
unprinted branch.  Failure there stops the dependency-role hidden-state route;
success would justify, but not replace, a later depth-progression experiment.

