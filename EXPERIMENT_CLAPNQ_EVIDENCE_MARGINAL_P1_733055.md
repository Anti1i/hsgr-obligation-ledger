# CLAPnQ fixed-support evidence-marginal P1 — job 733055

## Status

- Frozen protocol: `EXPERIMENT_PROTOCOL_CLAPNQ_EVIDENCE_MARGINAL_P1.md`
- Code commit: `9ca14b5`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Data: official CLAPnQ answerable dev; 96 calibration + 96 held-out cases
- Contextual layers: 13, 20, 27; calibration selected layer 20
- Host / accelerator: `xgpi14`, GPU 0, NVIDIA H100 NVL MIG 3g.47gb
- Environment: PyTorch `2.13.0+cu130`
- Slurm: `COMPLETED`, exit 0, elapsed 00:01:31
- Frozen outcome: **MODEL_RELEVANT_NOT_READABLE**

The outcome name means “not readable beyond the frozen surface-control gate.” The hidden readout is correlated with the target, but it does not add the required information over simple observable features.

## Candidate-domain result: strong pass

All seven model-relevance gates pass on 96 held-out questions / 273 evidence nodes.

| Metric | Held-out result | Frozen gate |
|---|---:|---:|
| Median `drop_all - full` gold-answer NLL | +2.242 nats/token | >= +0.020 |
| Positive total-evidence effect | 100.0% | >= 60% |
| Median `drop_all - support_only` NLL | +2.298 nats/token | >= +0.020 |
| Median leave-one-evidence-out marginal | +0.886 nats/token | diagnostic |
| Active evidence nodes (`u_i >= .005`) | 98.17% | >= 35% |
| Cases with at least two active nodes | 100.0% | >= 25% |
| Cases with distinct within-case marginals | 98.96% | >= 30% |

The node-utility p10 / p50 / p90 values are +0.320 / +0.886 / +1.705 nats/token. Different support nodes therefore have strongly different, model-relevant effects. This is not the numerical candidate collapse seen on GSM/MATH.

This result is deliberately limited. Gold answers often reuse wording from their supporting passage, so teacher-forced NLL can be dominated by lexical copyability. It establishes a usable intervention domain, not a hierarchy-specific reasoning gain.

## Hidden Guide result: conditional-signal gate fails

The calibration-only procedure selected block 20 and ridge alpha 100. On held-out nodes:

| Readout | RMSE | Pearson | Spearman |
|---|---:|---:|---:|
| Surface controls | 0.4963 | 0.5031 | 0.4967 |
| Hidden + surface | 0.4935 | 0.5065 | 0.5057 |

The hidden readout is stable across fixed halves (Spearman 0.470 and 0.532), so the raw correlation is real. However:

- Spearman gain over surface is only +0.009, below the frozen +0.08 gate;
- RMSE improves only 0.56%, below the frozen 5% gate.

Thus most predictable marginal utility is already explained by sentence length, position, question overlap, number of supports, and support-support overlap. P1 does not support the claim that hidden states contain a useful additional Guide signal for this target.

## Apparatus note

The first submission, job `733036`, passed data download, all nine unit tests, and model loading, then exited in 58 seconds with the known NUS cuDNN SDPA sublibrary mismatch on its first forward pass. Commit `9ca14b5` disables cuDNN SDPA inside the actual model process. No data, sample, threshold, layer, or analysis choice changed. Job `733055` is the clean rerun and is the only scientific result.

## Objective conclusion

The original suggestion is **partly correct**:

1. Length alone is not the solution.
2. Fixed support plus several human-annotated evidence units does solve the candidate-domain-collapse problem on CLAPnQ.
3. The current marginal-NLL Guide target is too easy to predict from surface form, so it does not yet justify a hidden-state HSGR mechanism.

CLAPnQ is therefore a valid benchmark/control for evidence composition, but not yet the strongest main benchmark for the desired paper claim.

## Recommended next route

Prioritize **ASQA with fixed/oracle support passages**. ASQA has multiple disambiguated QA pairs and short answers per question, giving explicit answer facets. The next target should be “which gold facet remains uncovered by the current answer prefix,” not “which source sentence is easiest to copy.”

The HSGR object would be:

```text
fixed support passages
  -> human answer-facet nodes
  -> covered / uncovered facet state
  -> one coherent long answer
```

The Guide reads the current answer-prefix hidden state and predicts marginal uncovered-facet utility. Retrieval remains fixed; nodes are facets rather than complete paths; there is no tree search or iterative prompt-action routing. A later causal test may guide generation toward an uncovered facet, with text-only, anti-direction, and surface-only controls.

Before any steering, freeze two gates on a new split:

1. the direct model must leave meaningful strict all-facet headroom (target 40–75%);
2. hidden state must predict held-out residual facet coverage beyond an exact lexical/surface coverage baseline.

FACTS Grounding should remain a later external grounding stress test because it lacks released node-level gold facets/support spans.

