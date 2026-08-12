# Dependency-error provenance corrective v1.1: job 727774

Run on 2026-08-12 from commit `2975cb0` on `xgpi13` with one Slurm-assigned
H100 NVL (95,830 MiB). Slurm state was `COMPLETED`, exit code `0:0`, elapsed
time `00:35:42`. The run used PyTorch `2.13.0+cu130` and the frozen 400-row
development set `data/gsm_chain_test.jsonl`. It did not read the sealed
MuSiQue final377 set.

## Decision

Corrective v1.1 **failed**. Do not submit join-v2, do not train a hidden-state
provenance reader, and do not run a v1.2 that moves the cap or thresholds
again on the same development outcomes.

The precise supported conclusion is limited: source-specific repair actions
show conditional crossover on this synthetic chain, but even a gold source
router does not beat generic repair or the best fixed action. Therefore this
action space has no demonstrated end-to-end routing value on the tested
distribution.

## Frozen results

| Quantity | Result |
|---|---:|
| Base | 7.25% |
| Generic repair | 59.50% |
| Fixed upstream repair | 58.00% |
| Fixed local repair | 11.25% |
| Gold source-routed repair | 58.50% |
| Hindsight best of repairs (non-deployable) | 65.50% |
| Source-routed minus generic | -1.00 pp, McNemar p=0.6655 |
| Source-routed minus best fixed | +0.50 pp, McNemar p=0.5 |

The v1.1 base produced 356 `UPSTREAM`, 15 `LOCAL`, and 29 `NONE` examples.
The source-specific crossover was in the intended direction:

- on `UPSTREAM`, upstream repair was 57.02% versus local repair 3.93%
  (+53.09 pp);
- on `LOCAL`, local repair was 13.33% versus upstream repair 0.00%
  (+13.33 pp).

However, the `LOCAL` stratum had only 15 examples, below the frozen minimum of
30. More importantly, the gold routed policy still lost to generic repair and
improved over the best fixed action by only two examples out of 400.

## Output validity and cost

Raising the common cap to 512 substantially fixed v1's severe truncation, but
did not fully pass the frozen validity gate:

| Output | Parse rate | Near-cap rate where available |
|---|---:|---:|
| Base Question 1 | 98.25% | — |
| Base Question 2 | 99.50% | — |
| Generic repair | 95.25% | 4.75% |
| Upstream repair | 94.00% | 6.25% |
| Local repair | 98.75% | 1.25% |

All repair arms used one call and the same 512-token cap. Mean generated
tokens were 334.41 (generic), 339.99 (upstream), and 220.11 (local).

## Gate table

| Frozen gate | Result |
|---|---|
| At least 30 UPSTREAM and LOCAL errors | FAIL |
| Source router ≥3 pp over generic, p<0.05 | FAIL |
| Source router ≥3 pp over best fixed, p<0.05 | FAIL |
| Source-specific crossover ≥5 pp in both strata | PASS |
| One repair call and equal cap | PASS |
| Every output parse rate ≥95% | FAIL |

The negative decision does not claim that error provenance can never be
useful. It rejects this two-node source-routed repair route as the paper's main
mechanism because the required action value was not established after the
pre-declared truncation correction.
