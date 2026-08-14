# ASQA fixed-support structure audit P0 — job 733199

## Outcome

`BORDERLINE`.  ASQA has a non-collapsed answer-facet structure and enough clean,
support-complete examples for a separate exploratory screen, but the frozen P0
did **not** pass.  The formal P1 protocol therefore remains unrun.

## Reproducibility

- Protocol: `EXPERIMENT_PROTOCOL_ASQA_FIXED_SUPPORT_P0.md`
- Code commit: `71f0ad0`
- Slurm job: `733199`, `xcnc17`, normal CPU, 2 CPUs, 8 GB, 5 seconds, exit `0:0`
- ASQA SHA-256: `8de0c824824372243b73b10314727dac1a02d9ecc729c34c752c30f8a3c1e58a`
- ALCE oracle top-5 SHA-256: `aaab90bee9b0d3e53050326b4c4d05077a929046996ce4a9cfe9bab1dc9ee75a`
- Scratch report: `/mnt/scratch/z/zitong/dch-hsgr/results/asqa_fixed_support_p0/report.json`
- Report SHA-256: `670a759c85c3a3f00eb5eeb39469cd868e2c8cf7e7bf29fb65b3edbb06664dc4`

## Frozen-gate results

| Gate | Result | Observed |
|---|---:|---:|
| at least 900 aligned | pass | 948 |
| at least 99% uniquely aligned | pass | 100% |
| at least 90% have 2+ unique facets | pass | 98.63% |
| at least 30% have 3+ unique facets | pass | 51.37% |
| duplicate-facet examples below 2% | **fail** | 22.47% |
| five non-empty fixed docs in at least 95% | pass | 100% |
| median fixed context at least 300 words | pass | 521 |
| passage STR-EM at least 80% and STR-HIT at least 50% | **fail** | 78.55%, 59.70% |
| best human STR-EM at least 80% and STR-HIT at least 50% | pass | 99.92%, 99.79% |
| verbatim long-answer leakage below 1% | pass | 0.84% |
| at least 192 baseline-eligible examples | pass | 547 |

Unique-facet counts were 13/448/224/152/62/49 examples for 1/2/3/4/5/6
facets.  Fixed contexts contained a median 521 words.  The stricter score over
all human annotations, rather than the best annotation per example, was still
98.36% STR-EM and 96.26% STR-HIT.

## Failure diagnosis

The 213 duplicate-facet examples contain 391 extra answer-alias groups.  Only 21
of those examples repeat an exactly identical disambiguated question.  Many of
the rest are paraphrased subquestions or distinct qualifications that happen to
share an answer.  Consequently, an answer-value node alone cannot always retain
the semantic identity of an ASQA facet.

The passage-coverage miss is not a normalization bug.  Across 3,184 raw facets,
the released ALCE `answers_found` flags and the frozen exact-presence scorer had
2,467 joint positives, 706 joint negatives, zero cases where ALCE was positive
but the scorer was negative, and 11 scorer-only positives.

After excluding every example with a duplicate normalized answer group, 735
examples remain.  Their fixed-passage STR-EM/STR-HIT are 78.28%/58.91%, and 427
meet the full clean, support-complete eligibility rule.  This supports a new
exploratory clean-subset experiment, not a retrospective relaxation of P0.

## Technical implication

The corrected claim is conditional: ASQA does provide explicit, diverse answer
facets on a large clean subset, but the full dev set is not a valid value-only
hierarchy under the frozen apparatus.  A subsequent screen may use the 427
clean, support-complete cases to test model headroom and oracle facet utility.
It must remain labeled exploratory and cannot establish hidden-state or causal
Guide value by itself.
