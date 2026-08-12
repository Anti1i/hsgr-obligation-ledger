# Dependency-error provenance v1 result: job 727421

Run on 2026-08-12 from commit `dcdbb9f` on `xgpi13` with one H100-47.
Slurm state was `COMPLETED`, exit code `0:0`, elapsed time `00:07:29`.
This is a development result on 400 rows of `data/gsm_chain_test.jsonl`; it
does not use the sealed MuSiQue final377 set.

## Frozen v1 decision

The frozen v1 protocol **failed** and remains failed.  It must not be counted
as evidence that source-routed repair beats a fixed repair policy.

| Quantity | Result |
|---|---:|
| Base | 6.50% |
| Source-routed | 15.25% |
| Generic repair | 7.00% |
| Best fixed action (`LOCAL`) | 17.00% |
| Source-routed minus generic | +8.25 pp, exact McNemar p=2.10e-9 |
| Source-routed minus best fixed | -1.75 pp, exact McNemar p=0.015625 |
| UPSTREAM stratum: upstream minus local | -2.60 pp |
| LOCAL stratum: local minus upstream | +33.33 pp |

Source counts were 269 `UPSTREAM`, 105 `LOCAL`, and 26 `NONE`.  The sample
count, generic-comparison, and equal-call/cap gates passed.  The best-fixed
comparison and two-sided source-specific crossover gates failed, so the
protocol's stop rule applies: do not train a hidden-state provenance reader
from this v1 result and do not advance it to the join-graph experiment.

## Post-run validity diagnosis

An aggregate-only inspection of the already generated outputs found a severe
length-censoring confound:

| Arm | Parse rate | Outputs at 191+ of 192 tokens | Accuracy among parsed |
|---|---:|---:|---:|
| Generic | 3.00% | 98.50% | 58.33% |
| Upstream | 1.75% | 98.25% | 57.14% |
| Local | 29.25% | 71.75% | 53.85% |

Base Question 1 and Question 2 parse rates were only 35.50% and 11.75%,
respectively.  Thus the observed zero upstream-stratum success for the
upstream arm is not a clean estimate of action quality: the longer upstream
action was almost always cut off before its final boxed answer.  This does not
reverse the v1 decision.  It means v1 is an invalid mechanism test in addition
to being a gate failure.

## Corrective replication rule

A separately frozen v1.1 replication may change only the common generation
cap and add an output-validity gate.  It must retain the same data, model,
greedy decoding, one call per arm, action prompts, effect-size thresholds, and
paired tests.  V1.1 cannot retroactively convert v1 into a positive result.

