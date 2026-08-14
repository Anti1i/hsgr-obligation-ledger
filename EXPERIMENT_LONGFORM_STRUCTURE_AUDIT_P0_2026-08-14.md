# Long-form grounded-QA structure audit P0

Date: 2026-08-14

## Verdict

**CLAPnQ GOLD passes the frozen structural apparatus gate.** It is the best current benchmark for testing whether HSGR can guide composition of several evidence units into a long answer without the numerical candidate-domain collapse seen on the math tasks.

This is an apparatus result, not an HSGR performance result. It establishes that annotated, non-collapsed structure exists; it does not yet establish model headroom, hidden-state signal, causal Guide utility, or an end-task gain.

## What was audited

- Official CLAPnQ answerable train and dev records from the [PrimeQA CLAPnQ repository](https://github.com/primeqa/clapnq).
- 2,254 examples total: one fixed gold passage, one long answer, and an explicit list of human-selected support sentences per record.
- No retrieval, language-model call, GPU, semantic judge, or generated candidate was used.
- Frozen protocol: `EXPERIMENT_PROTOCOL_LONGFORM_STRUCTURE_AUDIT_P0.md`.

## Main results

| Diagnostic | Result |
|---|---:|
| Valid / malformed examples | 2,254 / 0 |
| At least two unique support sentences | 98.98% |
| At least three unique support sentences | 48.18% |
| Multi-support and annotated non-consecutive | 98.89% |
| Frozen structural-headroom rate | 98.89% |
| Frozen pilot-eligible rate | 81.37% |
| Selected support missing from passage sentence list | 0.00% |
| Exact duplicate selected-support items | 0.17% |
| Annotated minimal-answer examples | 1.42% |

Support-sentence count has median 2, mean 2.90, p75 3, p90 4, and maximum 40. Long answers have median 48 words (p25 34, p75 65); fixed passages have median 169 words (p25 135, p75 213).

The median token-Jaccard overlap between pairs of selected support sentences is 0.107. Thus the multiple nodes are generally lexically distinct rather than duplicated copies of the same sentence. This is only a lexical redundancy check, not a semantic-independence proof.

All eight pre-registered apparatus gates pass.

## Why this is different from the failed math candidate domain

The math experiments repeatedly sampled a node answer and often normalized many samples to the same numeric value. The apparent candidate set then had little real variation, and adding deliberately wrong numbers did not create useful reasoning states.

CLAPnQ supplies a different object:

- atomic nodes are human-selected evidence sentences;
- partial states omit different genuinely relevant evidence units;
- the observable state space is the set of evidence subsets, not a set of fabricated answer values;
- gold support annotations let us measure which unit is absent and whether the final answer remains complete.

For a record with `k` support units there are `2^k` evidence-subset states in principle. We will not enumerate paths or turn the method into ToT/DPTS. The next experiment uses controlled leave-one-evidence-out interventions and asks whether a contextual hidden-state Guide can predict the marginal effect of each node.

## Benchmark comparison

### 1. CLAPnQ GOLD — proceed first

The released answerable data directly provides fixed passages, long answers, selected support sentences, and non-consecutive annotations. It therefore supports an objective evidence-composition experiment with no retrieval confound.

Risk: the passages are moderate rather than extremely long (median 169 words), and multiple support sentences can still express one compact fact. A model-level marginal-utility screen is required before claiming useful hierarchy.

### 2. ASQA fixed support — strong secondary benchmark

The [ASQA schema](https://huggingface.co/datasets/din0s/asqa) provides multiple disambiguated QA pairs with short answers, long answers, and annotation knowledge passages. The QA pairs are unusually useful because they provide automatically scorable answer facets. This may ultimately be better for strict completeness evaluation than CLAPnQ.

The official Google Storage file timed out from the present local network, so no empirical ASQA distribution is claimed in this report. Retry from the cluster or use the official fixed/oracle-passage release before a model run; do not silently substitute retrieved passages.

### 3. FACTS Grounding QA-like subset — do not use as the first HSGR benchmark

The [FACTS Grounding public set](https://www.kaggle.com/datasets/deepmind/FACTS-grounding-examples) has 856 examples with a fixed long context, system instruction, and user request. However, it mixes task types and supplies no human answer facets or support-span graph. Its official evaluation relies on model judges. Length is high, but node-level structural supervision is absent.

FACTS is useful later as an external grounding stress test after the HSGR mechanism is fixed. It is not suitable for deciding whether the Guide learned a hierarchy, unless an independent subset is manually annotated and frozen.

## Next experiment

Run a 192-example CLAPnQ dev evidence-marginal screen with Qwen2.5-7B-Instruct:

1. score the released gold answer under the full passage, all-support-deleted passage, support-only passage, and every leave-one-support-out passage;
2. measure whether removing different support nodes causes distinct held-out losses rather than another collapsed domain;
3. extract contextual hidden states at each support node in the full passage;
4. train a calibration-only Guide readout to predict each node's leave-one-out marginal loss;
5. compare against surface controls and evaluate on disjoint questions.

Only if evidence effects are material and the hidden Guide generalizes should we proceed to generated-answer steering. A readable hidden signal alone is not a causal result.

