# P0o-R1 protocol: obligation carry-forward rescue screen

## Question

When a revision prompt identifies only failed requirements, can an explicit
state that separates `PRESERVE` from `REPAIR` reduce loss of previously met
requirements without sacrificing repair?

P0o-R1 is a selected-case rescue experiment. It is not a prevalence estimate.
The eight cases were selected because P0n0 found a manually confirmed direct
omission in a successful all-failed revision. They are therefore unsuitable for
claims about expected performance on fresh RefineBench data.

## Correct scope of the P0n observation

P0n0 found 11 unique successful revised answers with direct omission, arising
from eight distinct benchmark problems. It showed that the failure mode can be
elicited on public benchmark tasks under our generated revision process. It did
not observe official RefineBench revision logs or estimate real-world frequency.

## Frozen cases and state source

Cases: `000010`, `000100`, `000147`, `000347`, `000577`, `000581`, `000827`,
and `000833`.

P0o reuses the frozen Qwen3-8B initial answers from P0n0. A deterministic,
single-example Qwen2.5-14B judge re-evaluates the initial answers and reference
answers. The re-evaluated initial state defines:

- `PRESERVE`: every criterion judged Yes;
- `REPAIR`: every criterion judged No.

All cases must remain mixed (at least one criterion in each state), or the
apparatus fails.

## Four arms

Each case, arm, and replicate uses the same previous answer and the same true
repair set.

1. `failed_only`: only the failed requirements are shown.
2. `all_checklist_no_status`: all requirements are shown without state labels.
3. `full_ledger`: all requirements are shown under the correct `PRESERVE` and
   `REPAIR` headings.
4. `shuffled_status`: all requirements are shown with a deterministic incorrect
   assignment that preserves the exact numbers of `PRESERVE` and `REPAIR`
   labels and flips at least one item in each direction.

The shuffled arm is a negative control for correct state-to-obligation mapping.
It does not by itself prove a learned latent state mechanism.

## Generation

- generator: Qwen3-8B, non-thinking;
- 5 stochastic replicates per case/arm;
- temperature 0.7, top-p 0.9;
- common case/replicate random seed across arms;
- 8 cases x 4 arms x 5 replicates = 160 revisions.

## Evaluation

The independent local judge is Qwen2.5-14B-Instruct. Exact evaluation prompts
are deduplicated, judged one example at a time with greedy decoding, and cached.
This removes the identical-input/batch-composition inconsistency observed in
P0n0, but it does not make the local judge equivalent to the official GPT-4.1
RefineBench evaluator.

Primary revision-level metrics:

- `AnyRegression`: at least one initial Yes becomes No;
- `AnyFix`: at least one initial No becomes Yes;
- `JointSuccessAny`: `AnyFix AND NOT AnyRegression`;
- `StrictJointSuccess`: every initial No becomes Yes and every initial Yes stays
  Yes.

Supporting metrics:

- criterion-level target FixRate;
- criterion-level PreserveRate;
- revision-level AllPreserved and AllTargetsFixed;
- per-case values and paired case/replicate differences.

## Apparatus gates

- all evaluation prompts parse validly;
- reference-answer Yes rate is at least 90%;
- all eight initial answers have mixed states;
- the failed-only arm reproduces at least one regression in at least 3/8 cases
  and in at least 20% of its 40 sampled revisions.

Failure of an apparatus gate is `APPARATUS_FAILURE`, not evidence against the
carry-forward hypothesis.

## Mechanism gates

All four must pass before a fresh prospective P0o-R2 is justified:

1. Full ledger reduces revision-level regression by at least 50% relative to
   failed-only and by at least 10 percentage points absolute.
2. Full-ledger criterion FixRate is no more than 10 percentage points below
   failed-only.
3. Full-ledger `JointSuccessAny` exceeds both all-checklist-no-status and
   shuffled-status by at least 10 percentage points.
4. Full-ledger `JointSuccessAny` exceeds failed-only by at least 15 percentage
   points.

Automated passage of these gates remains provisional until researcher review of
the criterion transitions. Failure means do not launch P0o-R2.

## Interpretation guard

P0o-R1 can test whether explicit carry-forward state rescues selected direct
omissions. It cannot establish benchmark-wide effectiveness, method novelty,
relation-aware reasoning, hidden-state control, or an RL contribution.
