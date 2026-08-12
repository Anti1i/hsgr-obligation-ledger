# MathQA executable-DAG audit P0-B result (job 728934)

## Outcome

**GATE FAIL.**  MathQA has ample root-connected `deep+join+reuse` graphs and
high execution coverage, but the native programs agree with their annotated
correct options far below the frozen 95% requirement.

The corrected final run used commit `ce9333c` on CPU node `xcnf27`, completed
in 42s, and passed all 14 unit tests.  The official MathQA zip SHA-256 was:

`7344f30456a7aef3176d4866cc953b35b41bec44eda6b00cdbcfde2876b2f07a`

The raw report is at:

`/mnt/scratch/z/zitong/dch-hsgr/logs/mathqa_executor_audit_report_728934.json`

## Frozen semantics and correction

Execution used the MathQA semantics in fixed Google Trax commit
`220a62303ebf4ad18871aa5607b4dda2f064f2d2` and its 1% relative answer
tolerance.  Only the annotated correct option was compared; there was no
nearest-option search.

Initial job `728930` exposed a local option-parser issue for spaced negatives
such as `- 49` and ratios such as `1 : 729`.  The parser was corrected before
the final run and two regression tests were added.  This raised test target
agreement from 56.92% to 59.27%, but did not change the gate outcome.

## Final results

Every structural target below is fully connected to the final operation
(`dead_nodes == 0`) and has valid backward references.

| Split | Connected target | Executed | Execution coverage | Matched correct option | Agreement among executed |
|---|---:|---:|---:|---:|---:|
| train | 5,321 | 5,249 | 98.65% | 3,228 | **61.50%** |
| dev | 791 | 780 | 98.61% | 464 | **59.49%** |
| test | 535 | 523 | 97.76% | 310 | **59.27%** |

All-program agreement was also low: 66.80% train, 65.90% dev, and 65.08%
test.  Thus the target result is not explained by a target-only parser crash.

The two size checks and both execution-coverage checks passed.  The train and
test answer-agreement checks failed by more than 33 percentage points.

## Failure diagnosis

Unsupported operations were rare in test (`min`: 2; `surface_cylinder`: 2),
so missing executor vocabulary cannot explain the roughly 40% disagreement.
Read-only sample inspection found clear native annotation/program errors, for
example programs that:

- compute one forward population increase for a question asking for a
  two-year backward population;
- subtract train length and bridge length where the requested crossing time
  requires adding them;
- omit required compounding or return a decimal fraction for a percentage
  option without the required conversion.

This agrees with the observed pattern: the executor can run almost every
target, but many executed programs do not represent the question's correct
calculation.

## Decision and boundary

Under the pre-registered P0-B stop rule, the MathQA route stops here.  No Qwen
generation, atomic-edit oracle, hidden-state extraction, or GPU job is
authorized by this result.

There are 3,228 train and 310 test examples that are simultaneously native,
root-connected, `deep+join+reuse`, executable, and answer-matching.  This is a
useful diagnostic but **not a post-hoc PASS**: selecting them makes agreement
100% by construction.  Any future attempt to use that subset must be a new
study with an explicit data-cleaning rule, template/near-duplicate control, a
fresh split, and new untouched evaluation data.  It cannot be reported as the
outcome of P0-B.
