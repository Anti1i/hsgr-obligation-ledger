# Witness-interference mechanism study P0g — result

## Verdict

**Repair regression is supported as a controlled existence phenomenon.  Witness overlap is useful
for conservative invalidation, but the proposed witness-preserving prompt does not prevent
regression.  The full four-gate mechanism claim therefore fails.**

This result keeps repair dynamics scientifically interesting, but it does not yet justify a main
method, planner, or hidden-state module.

## Runs and apparatus

- Frozen generation job: `751560`, Qwen2.5-7B-Instruct, 96 candidates, completed on `xgpi13`.
- Initial verifier in job `751560`: apparatus failure.  Its negative-control accuracy was 83.3%,
  below the frozen 95% gate.  All four errors came from silently correcting an answer's wrong
  citation using the fixed evidence.
- Frozen recalibration: `EXPERIMENT_PROTOCOL_WITNESS_INTERFERENCE_P0G_R1.md`.
- Rejudge job: `751577`, Qwen2.5-14B-Instruct, completed on `xgpi13`; it reused all 96 candidate
  texts from `751560` and did not regenerate or filter outputs.
- R1 code: `f31eb72`.
- R1 result directory:
  `/mnt/scratch/z/zitong/hsgr-obligation-ledger/results/witness_interference_p0g_r1_751577`

R1 controls passed:

| Control | Result | Frozen threshold |
|---|---:|---:|
| parse validity | 100% | >=95% |
| positive accuracy | 97.9% | >=95% |
| negative accuracy | 100% | >=95% |

## Automatic results before manual review

| Arm | Target repair | Successful repairs with regression | Regression among successful repairs | Median edit ratio |
|---|---:|---:|---:|---:|
| full rewrite | 15/24 | 0 | 0% | 0.062 |
| local patch | 8/24 | 2 | 25% | 0.173 |
| obligation-only patch | 8/24 | 2 | 25% | 0.133 |
| witness patch | 12/24 | 3 | 25% | 0.091 |

Across all arms, the verifier found 43 successful repairs and seven successful-repair regressions.
For successful patch repairs:

- regression with witness overlap: 7/26 node events, 26.9%;
- regression without witness overlap: 0/58 node events, 0%;
- selective invalidation of the repair target plus touched witnesses recalled 100% of detected
  regressions while avoiding 33.7% of all obligation checks.

## Manual review

All seven automatically detected regressions and ten frozen non-regression controls were read.

### Six clear successful-repair regressions

1. Transit ordering, witness patch: fixed order, deleted the five-year operating-cost result.
2. River ordering, witness patch: fixed order, deleted the protected-hectares result.
3. Data-center ordering, local patch: fixed order, deleted the peak-temperature result.
4. Data-center ordering, obligation-only patch: fixed order, deleted the peak-temperature result.
5. Data-center ordering, witness patch: fixed order, deleted the peak-temperature result.
6. Housing coverage, obligation-only patch: inserted the missing temperature result, but deleted the
   final retrofit recommendation.

The seventh automatic row, hospital ordering with a local patch, was rejected.  Its answer began
with only `[A].` and did not state the required event order, so the verifier had falsely marked the
target as recovered.

Among the ten frozen non-regression samples, no hidden regression was found.  One hospital coverage
row was another target-recovery false positive and was rejected; the remaining nine were clean
successful repairs.

The manual result is therefore at least six unambiguous `target: 0 -> 1` plus
`preserved obligation: 1 -> 0` transitions in this controlled suite.  This passes the existence
count of five without relying on the rejected verifier row.

## Gate decisions

### Gate 1: at least 15 successful repairs

**Pass.**  Forty-three were automatic; two target false positives were found in the frozen manual
sample, leaving ample margin above 15.

### Gate 2: at least five unambiguous successful-repair regressions

**Pass after manual review: 6.**

### Gate 3: at least 3x enrichment under witness overlap

**Numerically passes, but is not a clean causal result.**  Every detected regression occurred in an
edited witness and none occurred outside one.  However, the patch operator replaces selected source
sentences, while each satisfied obligation has a frozen single-sentence witness.  If that sentence
is not edited, its supporting text remains literally present.  Some of the observed enrichment is
therefore mechanically induced by the edit interface.  The result supports conservative
invalidation, not a strong causal claim that semantic witness sharing explains natural repair
interference.

### Gate 4: witness-preserving intervention

**Fail.**  Compared with obligation-only patches, witness patches improved target repair by 16.7
points, but did not lower successful-repair regression: both were 25%.  The frozen requirement was
at least a 50% relative regression reduction with no more than a 10-point target loss.

More strikingly, three of the six manually confirmed regressions came from the witness-patch arm.
Showing the model the protected text is not sufficient to make it preserve that text.

### Overall frozen decision

**Fail: not all four gates pass.**  P0g must not be reported as validation of a complete
witness-aware repair method.

## What is now supported

1. Targeted repair can be non-monotonic even when the saved answer has exactly one seeded failure.
2. A sentence-level witness ledger can conservatively identify which obligations need
   re-verification after a local edit.
3. Prompting alone is too weak to enforce preservation.
4. Typed relation graphs and hidden states remain unnecessary at this stage.

## What is not supported

- natural-data regression prevalence;
- witness overlap as an independently identified causal mechanism;
- witness-preserving prompting as a mitigation;
- superiority over full rewriting—in this deliberately short controlled suite, full rewrite had
  no detected successful-repair regression and the highest number of safe automatic successes;
- repair-influence prediction, planning, or hidden-state augmentation.

## Next falsifiable step

A defensible follow-up must change the experiment, not relax Gate 4:

1. **Paired layout intervention:** express identical content with the target and preserved witness
   either colocated in one sentence or separated across sentences.  Hold the repair target and
   semantics fixed.  This tests whether shared textual substrate itself raises regression.
2. **Algorithmic witness locking:** compare prompt-only preservation with a copy-constrained
   clause/insertion editor that is mechanically prevented from deleting protected spans.  This
   tests an actual intervention rather than another reminder.
3. **Longer natural answers:** the current three-sentence controlled answers favor full rewrite.
   Only after the paired mechanism passes should the test move to audited natural long-form cases.

Hidden states stay gated until observable edit/witness features fail to explain remaining risk.

