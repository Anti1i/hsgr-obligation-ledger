# P0m result: persistence versus recomputation

## Bottom line

P0m reaches the frozen **C_STOP_CONTROLLED_LINE** branch.

- Showing an old `SAT` verdict did not reduce verification accuracy. There is
  no evidence for cache-induced verdict anchoring in this apparatus.
- Forced structured execution improved one model in one condition, but the
  effect was not robust across models or cache conditions and did not satisfy
  the frozen safety/recall gates.
- No strong final-verdict override was observed. When the operator, operands,
  external computation, and the model's reported computation were all correct,
  the final verdict was also correct.

The controlled stale-verdict line should therefore stop here as a proposed
HSGR main mechanism. P0m does not justify adding hidden-state intervention or
RL to this line.

## Frozen setup

- 40 executable semantic-dependency cases;
- one dependency-changing edit and one matched harmless edit per case;
- four arms: Fresh/Cached x Free/Structured;
- Qwen3-8B in non-thinking mode and Qwen2.5-14B-Instruct;
- 640 deterministic generations in total;
- model final verdict and independently executed checker verdict scored
  separately.

The harmless controls prevent a system from appearing better merely by
answering `FAIL` more often.

## Primary results

All values below are percentages. `Stale` is recall on revisions that make the
unchanged conclusion false. `Safe` is specificity on harmless revisions where
the conclusion remains true.

| Model | Arm | Model stale | Model safe | Checker stale | Checker safe |
|---|---|---:|---:|---:|---:|
| Qwen3-8B | Fresh-Free | 0.0 | 100.0 | - | - |
| Qwen3-8B | Cached-Free | 0.0 | 100.0 | - | - |
| Qwen3-8B | Fresh-Structured | 10.0 | 100.0 | 55.0 | 70.0 |
| Qwen3-8B | Cached-Structured | 5.0 | 100.0 | 55.0 | 67.5 |
| Qwen2.5-14B | Fresh-Free | 37.5 | 100.0 | - | - |
| Qwen2.5-14B | Cached-Free | 40.0 | 100.0 | - | - |
| Qwen2.5-14B | Fresh-Structured | 62.5 | 100.0 | 77.5 | 87.5 |
| Qwen2.5-14B | Cached-Structured | 60.0 | 100.0 | 75.0 | 85.0 |

### Cache effect

- Qwen3-8B: Fresh-Free minus Cached-Free = 0 points, exact paired
  `p = 1.0`.
- Qwen2.5-14B: Fresh-Free minus Cached-Free = -2.5 points, exact paired
  `p = 1.0`.

The old `SAT` line did not create the failure. Qwen3 failed equally without
it, while Qwen2.5 was numerically one case better with it.

### Structured-execution effect

- Qwen3-8B Fresh: +10 points over Free, `p = 0.125`.
- Qwen3-8B Cached: +5 points, `p = 0.5`.
- Qwen2.5-14B Fresh: +25 points, `p = 0.0213`.
- Qwen2.5-14B Cached: +20 points, `p = 0.0768`.

The Qwen2.5 Fresh effect is real within this sample, but it is not enough to
support the proposed mechanism: final stale recall remained 62.5%, the Cached
effect was not significant, and Qwen3 did not reproduce it. The external
checker also introduced false failures on 12.5%-32.5% of harmless controls.

## Where the failures occurred

The output parser was valid in every cell except two duplicated-source outputs
from Qwen2.5 Cached-Structured; that cell still had 97.5% validity and passed
the frozen apparatus gate.

### Qwen3-8B

- operator accuracy: 100% in both Structured arms;
- operand accuracy: 56.25% Fresh and 51.25% Cached;
- complete trace rate: 5% Fresh and 0% Cached;
- reported-computation/checker agreement: 45% Fresh and 40% Cached.

Qwen3 often named the right relation but copied labels or incomplete values
instead of executable operands. Even when the program could execute, Qwen3
frequently reported the old conclusion's truth value rather than the program's
result. This happened equally in Fresh and Cached conditions, so it is generic
recomputation failure rather than old-verdict anchoring.

### Qwen2.5-14B

- operator accuracy: 93.75% Fresh and 91.25% Cached;
- operand accuracy: 52.5% in both arms;
- complete trace rate: 6.25% Fresh and 1.25% Cached;
- reported-computation/checker agreement: 81.25% Fresh and 77.5% Cached.

Qwen2.5 benefited from the explicit operator/operand format, especially on
comparison, temporal, and threshold cases. The gain did not generalize to
attribution, and on all eight Fresh derived-arithmetic stale cases the program
obtained the false relation while the model final verdict remained true.
Inspection showed that the model also reported the computation as true, so
these were computation failures, not a later final-verdict override.

## Strong-override audit

The frozen strict count was zero for every model and cache condition. Because
models often added the audited conclusion as a source or omitted an alias-link
sentence, an additional audit ignored source-ID scoring entirely. The count
remained zero.

Thus the appealing case study -- correct source facts, correct operands,
correct operator, correct reported computation, but a final `SAT` that ignores
all of them -- did not occur. Apparent checker/final disagreements always also
contained an earlier extraction or reported-computation error.

## Scientific conclusion

P0k established that unchanged conclusions can become false after upstream
semantic edits. P0l showed that natural-language Guides do not reliably repair
them. P0m now shows that:

1. the failure is not caused by displaying a cached old verdict;
2. forcing a structured record can help a stronger model, but is model-specific
   and trades recall for checker false alarms;
3. the evidence does not isolate a ledger-specific propagation mechanism;
4. structured extraction plus deterministic checking is a crowded current
   method family and cannot carry the paper's novelty by itself.

This is a useful negative result, but not a sufficient main paper contribution.
The defensible action is to stop this controlled line rather than adding a
hidden-state predictor, more prompt wording, or RL on top of an unsupported
mechanism.

## Reproducibility

- Git commit: `e55ca84`
- Slurm job: `753636`
- Compute node: `xgpi13`
- GPU: one H100-47 allocation (`CUDA_VISIBLE_DEVICES=0`)
- Runtime: 4 minutes 4 seconds; exit code 0
- Result directory:
  `/mnt/scratch/z/zitong/hsgr-obligation-ledger/results/persistence_recomputation_p0m_753636`
- Report SHA-256:
  `5bb7abc0a7df9e3eb972ee7762cbb39fbe4acd6301a53b4ec70d60ad91cb525c`
- Rows SHA-256:
  `ef6331ddedbdd2406c850be0a89bcccc1af5630630d3f3713727ee95bd22363c`
- Review SHA-256:
  `57f14997af9c8e90fae1299573ed6eab9b6176cad0653d7fecad28ba5d4182f7`

The full raw outputs remain in the project-specific scratch result directory;
they were not copied through the SSH gateway or added to Git.
