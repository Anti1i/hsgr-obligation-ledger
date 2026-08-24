# P0n0: RefineBench revision audit result

## Bottom line

The external-dataset case study confirms a real but narrower phenomenon:
**while fixing requested weaknesses, a model often deletes coverage that the
old answer already had**. After conservative review, 11 of 65 unique revised
answers that successfully fixed at least one target also contained at least one
clear direct omission (16.9%).

However, P0n0 found **zero nonlocal semantic regressions**. It therefore does not
support the dependency-recomputation or stale-verdict mechanism, and the frozen
rule says not to scale this line to 200 examples. The result supports a basic
coverage-preservation motivation, not a relation-aware mechanism claim.

## Run record

| Item | Value |
|---|---|
| Slurm job | `753693` |
| Job state | `COMPLETED`, exit `0:0` |
| Runtime | `01:08:32` |
| Node/resource | `xgpi13`, one H100-47 allocation |
| Code commit used by job | `592ea7a` |
| Dataset | `RefineBench/RefineBench` |
| Frozen dataset revision | `2777137e7c489f5049608f41d2432326429ea619` |
| Sample | 40 problems; 8 each from math/statistics, STEM, law, humanities, and other |
| Generator | `Qwen/Qwen3-8B`, non-thinking |
| Independent local judge | `Qwen/Qwen2.5-14B-Instruct` |
| Official GPT-4.1 evaluator | unavailable; this is not an official RefineBench score |

The two revision arms were:

1. `guided_failed`: show every checklist item currently judged failed;
2. `targeted_partial_failed`: show a deterministic half of the failed items.

Both arms intentionally hid previously passed items. This tests whether ordinary
failure-focused revision preserves them without an explicit ledger.

## Automatic screen

All pre-registered apparatus gates passed:

- parse validity: 98.7%;
- reference-answer judgment validity: 100%;
- reference-answer Yes rate: 95.9%;
- mixed initial answers: 36/40, above the required 20.

| Arm | Valid revisions | Target fixes | Target fix rate | Raw Yes→No | Raw prior-success regression rate | Successful revisions with any raw regression |
|---|---:|---:|---:|---:|---:|---:|
| guided failed | 37/37 | 134/166 | 80.7% | 23 | 11.6% | 12/36 (33.3%) |
| targeted partial failed | 37/37 | 69/78 | 88.5% | 36 | 18.1% | 11/33 (33.3%) |

These raw rates are not final findings because a Yes→No transition can be a
judge error.

## Researcher/Codex audit

The 59 raw candidate criterion transitions were conservatively labeled:

| Label | Count | Meaning |
|---|---:|---|
| direct | 36 | previously covered content was plainly deleted or overwritten |
| invalid | 22 | the old Yes or new No was not supported on review |
| uncertain | 1 | subjective overall-clarity criterion |
| local | 0 | no clearly localized edit-region collateral case |
| nonlocal | 0 | no preserved conclusion became false after an upstream semantic change |

The 36 direct omissions occurred in 14 arm-specific revisions. Twelve of those
14 revisions also fixed at least one target. After collapsing byte-identical
answers across arms, 11 of 65 unique successful revisions contained a confirmed
direct omission (16.9%). These cases span all five sampling strata.

The judge also showed meaningful instability:

- 2 of 40 sampled Yes-to-Yes controls were confirmed missed regressions;
- one further control was uncertain;
- one byte-identical answer received different verdicts on four criteria across
  the two arms.

Consequently, P0n0 is suitable as a case-study screen but not a prevalence or
benchmark-performance estimate. Detailed identifier-only labels are in
`MANUAL_AUDIT_REFINEBENCH_REVISION_P0N.md`.

## What the result does and does not establish

Supported:

1. The coverage-loss phenomenon is not confined to our synthetic setup. It also
   appears in a frozen external revision benchmark.
2. Fixing the named problem does not guarantee preserving requirements that were
   already satisfied.
3. A mechanism that explicitly carries forward passed obligations has a
   plausible target failure to address.

Not supported:

1. There is no evidence here for dependency-aware recomputation, semantic stale
   verdicts, or a relation-aware ledger.
2. There is no causal evidence yet that a ledger prevents the omissions.
3. The local-judge numbers are not official RefineBench results and must not be
   reported as paper-level prevalence.
4. This experiment says nothing about hidden-state control.

## Frozen decision and best next test

The pre-registered scale condition required at least three nonlocal regressions
across two domains and at least 5% of successful revisions. The observed count
is zero. Decision: **keep the stale-verdict/dependency-recomputation line closed
and do not scale P0n to 200 examples**.

If the broader obligation-ledger direction is retained, the most informative
small follow-up is a mechanism test on the confirmed direct-omission cases:

- failed-only feedback (current baseline);
- full flat ledger showing both `preserve` and `repair` obligations;
- shuffled-status ledger control;
- optional local-edit instruction.

The outcome should be measured with independent human or official-strength
judging: target-fix retention, previously-passed preservation, and answer quality.
This would test whether explicit obligation carry-forward causally prevents the
observed omissions. It would not, by itself, justify relation-aware or
hidden-state claims.

## Integrity records

- `p0n_report.json`: `08facd232cfca2148c276e79e32273e60b3256b2ba59d0255bf586318625fad1`
- `p0n_transition_rows.jsonl`: `a280e16bbcfb3f6e37f20cc5253f2bab361302e3fe601a94167dfa3ccca090db`
- `p0n_manual_review.jsonl`: `7f406758ce748499d121cd3d01faa2249c2284cc4bf223894c4f6df2b3dc94d8`

Raw RefineBench text and model outputs remain only in the project scratch result
directory because the dataset is CC BY-NC-ND 4.0; they are not committed to the
public repository.
