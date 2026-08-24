# RefineBench P0n0 researcher/Codex audit

This audit records only case identifiers, arms, criterion identifiers, and
aggregate judgments. It does not reproduce RefineBench questions, answers, or
checklist text.

## Label policy

- `direct`: content that supported a previously met criterion was plainly
  deleted or overwritten in the revision.
- `local`: collateral damage inside a clearly localized edit region.
- `nonlocal`: a preserved conclusion/support statement became false because
  semantically upstream content changed.
- `invalid`: the initial Yes or revised No was not supported on manual review.
- `uncertain`: the criterion was too subjective to resolve conservatively.

This is a researcher/Codex case-study review, not independent expert annotation.

## Candidate labels

| Case | Arm | Direct criterion IDs | Invalid criterion IDs | Uncertain IDs |
|---|---|---:|---:|---:|
| `000010` | guided failed | 8 | — | — |
| `000010` | targeted partial failed | 1, 2, 5, 6, 7, 8 | 9 | — |
| `000045` | guided failed | — | 6 | — |
| `000045` | targeted partial failed | — | 8 | — |
| `000100` | guided failed | 2, 3, 6, 8 | — | — |
| `000100` | targeted partial failed | — | 8 | — |
| `000147` | guided failed | 3, 8 | — | — |
| `000212` | targeted partial failed | — | 2, 3, 4, 5, 7, 13 | — |
| `000218` | targeted partial failed | — | 1, 2, 5, 7 | — |
| `000221` | guided failed | — | 8 | — |
| `000347` | guided failed | 3, 4, 6, 7 | — | — |
| `000347` | targeted partial failed | 3, 4, 6, 7 | 5 | — |
| `000531` | guided failed | — | 3 | — |
| `000577` | guided failed | 2 | — | — |
| `000577` | targeted partial failed | 3 | — | — |
| `000581` | guided failed | 2 | 5 | — |
| `000581` | targeted partial failed | — | 5 | — |
| `000584` | targeted partial failed | 1, 3, 4 | — | 5 |
| `000827` | guided failed | 2, 6, 11 | — | — |
| `000827` | targeted partial failed | 6, 11 | — | — |
| `000833` | guided failed | 1, 2 | — | — |
| `000833` | targeted partial failed | 1, 2 | — | — |
| `000926` | targeted partial failed | — | 4 | — |
| `000930` | guided failed | — | 5 | — |
| `000930` | targeted partial failed | — | 5 | — |

Totals over the 59 machine-flagged criterion transitions:

| Arm | Direct | Invalid | Uncertain | Local | Nonlocal |
|---|---:|---:|---:|---:|---:|
| guided failed | 18 | 5 | 0 | 0 | 0 |
| targeted partial failed | 18 | 17 | 1 | 0 | 0 |
| **total** | **36** | **22** | **1** | **0** | **0** |

Collapsing identical revised answers within a case leaves 56 unique
case/criterion/revision events: 34 direct, 21 invalid, and one uncertain.

## Yes-to-Yes controls and judge consistency

The frozen bundle included 40 stable Yes-to-Yes controls. Criterion-focused
review found two confirmed missed direct regressions:

- case `000147`, guided-failed arm, criterion 4;
- case `000124`, targeted-partial-failed arm, criterion 3.

Case `000584`, guided-failed arm, criterion 5 remained uncertain because the
criterion asks for subjective overall clarity. No missed regression was
confirmed in the other 37 controls from the criterion-focused evidence.

Five cases happened to produce byte-identical revised answers in both arms. In
case `000584`, the local judge nevertheless disagreed across the two identical
answers on four criteria (1, 3, 4, and 5). This is direct evidence that the local
judge is not stable enough for a paper-level prevalence estimate.

## Audit conclusion

The audit confirms ordinary revision-induced requirement loss, including cases
where a requested failure was fixed while previously covered material was
deleted. It confirms no preserved-support/upstream-change (`nonlocal`) event.
Therefore the frozen scale gate is not met.
