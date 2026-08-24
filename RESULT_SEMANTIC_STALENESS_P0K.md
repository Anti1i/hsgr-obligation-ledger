# P0k-R1 semantic-staleness screen: result

## Decision

P0k-R1 does **not** pass its frozen hard-case gate. It establishes a clean
controlled failure mode and removes the obvious surface shortcut, but it does
not yet justify hidden-state learning or RL.

The most useful new observation is narrower: a flat LLM verifier often reuses
the visible old conclusion instead of recomputing whether it is still true
from the current upstream facts. This should be tested with an explicit,
relation-aware recomputation guide before any hidden-state work.

## Run record

- code commit: `7c0ce91`
- Slurm job: `753423` (`COMPLETED`, exit `0:0`, 5m54s)
- compute host: `xgpi20`
- allocated device: `CUDA_VISIBLE_DEVICES=0`, H100 NVL MIG 3g.47gb
- judges: Qwen3-8B non-thinking and Qwen2.5-14B-Instruct
- cases: 8 domains x 5 mechanisms = 40 base cases
- states per case: old, dependency edit, matched harmless edit = 120 per judge
- result directory:
  `/mnt/scratch/z/zitong/hsgr-obligation-ledger/results/semantic_staleness_p0k_753423`

All nine unit tests passed on the compute node, including the domain-grouped
surface predictor test.

## Apparatus repair before the model run

The first R0 preflight was rejected before using a model. Its dependency and
harmless edits occurred in different source sentences, allowing the frozen
surface union to reach 95% stale recall with 35% verification saving. R0 is
therefore leakage evidence, not research evidence.

R1 changed both arms to edit the same composite source sentence, at the same
position, involving the same local entities. For example, one arm changes the
first value so `A < B` becomes false; the paired harmless arm changes the
second value while preserving `A < B`. No gate or threshold was relaxed.

## Frozen surface results on R1

| Policy | Stale recall | Verification saving |
|---|---:|---:|
| recheck everything | 100% | 0% |
| direct witness overlap | 0% | 100% |
| token Jaccard | 100% | 0% |
| character similarity | 45% | 56.25% |
| entity overlap | 40% | 60% |
| citation overlap | 10% | 90% |
| frozen surface union | 100% | 0% |
| learned surface model, leave-one-domain-out | 90% | 10% |
| matched-budget random for the learned policy | 90% | 10% |

Thus G2 passes: no frozen surface rule reaches both 90% stale recall and 25%
saving. The learned surface model is indistinguishable from matched-budget
random at its operating point, so G4 fails.

## Judge-format defect and raw semantic audit

The original strict parser required integer sentence IDs. Qwen3 usually wrote
`"S9"`; Qwen2.5 often also wrote unquoted lists such as `[S2, S3]`. Therefore
the frozen report's parse-validity numbers (0% and 16.67%) mostly measure a
serialization mismatch and must not be interpreted as semantic ambiguity.

For diagnosis only, the Boolean `met` field was extracted directly from every
raw output without changing any prediction. The result was:

| Judge | Old SAT | Dependency-edit FAIL | Harmless-edit SAT |
|---|---:|---:|---:|
| Qwen3-8B | 40/40 | 0/40 | 40/40 |
| Qwen2.5-14B-Instruct | 40/40 | 14/40 | 40/40 |

Raw Boolean agreement was 106/120 = 88.33%, below the frozen 95% requirement.
All 14 disagreements were dependency edits; Qwen2.5 matched the executable
label and Qwen3 returned SAT.

Dependency-edit detection by mechanism:

| Mechanism | Qwen3-8B | Qwen2.5-14B |
|---|---:|---:|
| attribution/source binding | 0/8 | 0/8 |
| comparison | 0/8 | 2/8 |
| derived arithmetic | 0/8 | 0/8 |
| temporal ordering | 0/8 | 4/8 |
| definition/threshold | 0/8 | 8/8 |

The old and harmless controls being 100% while stale recall collapses rules
out random guessing. It is consistent with a strong SAT/status-quo bias: the
flat verifier sees the conclusion sentence and tends to accept it without
recomputing the dependency. It does not prove why that happens internally.

## Manual audit

The generated labels are executable comparisons, subtraction checks, date
orders, source-claim matches, or score-threshold checks. The conclusion is
byte-for-byte unchanged, both revisions modify the same non-conclusion source
sentence, and each harmless arm preserves the Boolean condition.

The 14 raw judge disagreements were manually checked. They contain two direct
comparison flips, four alias-mediated temporal flips, and eight threshold
flips. Each is unambiguous under its stated glossary and values. The strongest
surface union has no false negatives because it rechecks all 80 revisions, so
there are no additional surface-miss rows to audit.

## Frozen gates

| Gate | Result | Reason |
|---|---|---|
| G1 semantic apparatus | FAIL | strict format failure; raw stale recognition also far below 95% |
| G2 surface baselines insufficient | PASS | high recall requires rechecking nearly/all rows |
| G3 mechanism coverage | FAIL | neither judge recognizes four mechanisms reliably |
| G4 learned surface screen | FAIL | 90% recall saves only 10%, equal to matched random |
| P0k hard-case gate | **FAIL** | G1 and G3 fail |

## What is and is not supported

Supported in this controlled case study:

1. An unchanged conclusion can become false because a different sentence
   changed.
2. After matching edit position and local vocabulary, cheap surface cues do
   not tell which same-sentence edit invalidates the conclusion.
3. A generic flat verifier is not a reliable oracle for this task; larger
   Qwen2.5-14B is better than Qwen3-8B here, but still misses 65% of stales.

Not supported:

1. the prevalence of this failure in natural long-document revision;
2. that a relation-aware ledger fixes it;
3. that the required control signal is present in hidden states;
4. any benefit from learned routing or reinforcement learning.

## Next justified case study

Use the same 40 paired cases for a new, separately frozen recomputation-guide
screen. Compare a flat verifier against a relation-aware ledger that supplies
only the dependency structure (source sentence IDs and operation type), never
the current values or answer. Add a matched-length shuffled/incorrect guide as
a control. Require improved stale recall while retaining old and harmless
specificity, and test each mechanism separately.

This directly asks whether HSGR's Guide has a useful role: not predicting a
verdict from hidden state, but telling the model what upstream relation must be
recomputed after an edit. Hidden-state prediction is justified only if that
explicit Guide intervention first produces a reliable causal improvement.

## Result hashes

- report: `78583d39539e3b5ad63095f8889d73488b18a24db5e2a7eaddf513b1b9c0e2e8`
- cases: `1f6d48bc97c2a6eded5088630af8822854c9ac61cc0e894a6a6dea7f1847bff4`
- judgments: `3f9043ca989c3fc98d6cde3e915f131708b926c4abc271b96c3e32c74969381e`
- rows: `d9e4e248bfe7454f4c873d7ca484b9b35dbc9506cdadcf29209945a7244f0a3a`
- review: `9c686aa0fcf461c03f5418be2989a54e7768389cf39d558811dc65b5515c9c9f`
