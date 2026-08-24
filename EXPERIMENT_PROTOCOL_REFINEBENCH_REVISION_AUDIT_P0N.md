# P0n: RefineBench natural-revision external-validity screen

## Question and scope

P0n asks whether criterion regressions arise when models revise answers on an
existing public refinement benchmark. It is an external-dataset stress test,
not proof of real-world prevalence: RefineBench revisions are still generated
inside a benchmark protocol.

P0n does not try to rescue the P0m method. It tests whether the controlled
stale-verdict line should remain closed or whether a larger natural audit is
warranted.

## Official-versus-custom boundary

RefineBench supplies the questions, contexts, reference answers, and binary
checklists. Its official evaluation uses GPT-4.1 and up to five turns.

No OpenAI or OpenRouter API credential is available in the current project.
P0n0 therefore uses a local cross-model judge and must be called a case-study
screen, not an official RefineBench score.

The official paper's partial-guided setting reveals a subset of checklist
criteria. Our `targeted_partial_failed` condition instead reveals a
deterministic half of the criteria judged failed at the previous turn. This is
a custom targeted-repair stress test and is never labelled as the official
partial-guided result.

## Dataset and licensing

- Dataset: `RefineBench/RefineBench`.
- Frozen Hugging Face revision:
  `2777137e7c489f5049608f41d2432326429ea619`.
- Dataset license: CC BY-NC-ND 4.0.
- Dataset rows, questions, checklists, and model-answer bundles remain in the
  project-specific scratch directory and are not committed to Git.

## P0n0 apparatus pilot

Select 40 instances independently of model outcomes: eight stable-hash
instances from each stratum below.

1. Math and Statistics;
2. STEM: CS, Physics, Chemistry, Engineering, Biology/Medicine;
3. Law;
4. Humanities/Social Science;
5. Economics/Business and Other.

Only technical input-integrity filters are allowed: nonempty question, at
least two checklist items, and input below the frozen character limit. Do not
resample based on initial model success or failure.

### Models

- generators: Qwen3-8B non-thinking and Qwen2.5-14B-Instruct;
- local checklist judge: Qwen2.5-14B-Instruct.

The Qwen2.5 generator is judged by the same checkpoint and is descriptive
only. Cross-generator robustness cannot be claimed from that cell.

### Revision arms

Both arms share the same initial answer and its blind checklist evaluation.

- `guided_failed`: reveal every criterion judged `No`;
- `targeted_partial_failed`: reveal a deterministic half of judged-`No`
  criteria, with at least one target when failures exist.

The model receives the original conversation, its previous answer, and the
feedback, and must return a complete revised answer. Each old and revised
answer is judged independently without its previous verdict or arm label.

## Evaluator safeguards

Official code maps an unparseable evaluator output to all `No`. P0n never does
this. A malformed or incomplete checklist judgment is invalid and excluded
from transition denominators.

The local judge is calibrated on the selected instances' reference answers.
The apparatus passes only if:

1. at least 95% of all judge outputs parse completely;
2. at least 90% of reference-answer checklist decisions are `Yes`;
3. each generator has at least 20 valid revision opportunities containing at
   least one prior `Yes` and one prior `No`.

Failure of any gate means `APPARATUS_FAILURE`, not evidence that regressions
are absent.

## Criterion transitions

For every valid answer revision and checklist item, record:

- `Yes -> Yes`: preserved;
- `No -> Yes`: fixed;
- `Yes -> No`: candidate regression;
- `No -> No`: still failed.

Primary metrics are:

- prior-success regression rate: `Yes->No / prior Yes`;
- target fix rate: targeted `No->Yes / targeted prior No`;
- successful-fix regression rate: fraction of revisions with at least one
  target fix that also contain at least one `Yes->No`;
- answer-level and domain-level candidate counts.

## Manual audit

Automatic `Yes->No` is only a candidate. P0n writes a review bundle containing
every candidate plus a stable sample of 20 `Yes->Yes` controls per
generator/arm. A reviewer labels each candidate:

- `direct`: criterion-bearing content is directly deleted or overwritten;
- `local`: collateral damage occurs in the edited local region;
- `nonlocal`: a preserved conclusion/support statement becomes false because
  semantically upstream content changes;
- `invalid`: one of the two checklist verdicts is wrong;
- `uncertain`.

SequenceMatcher ratios and exact-sentence preservation are review aids, not a
semantic diff baseline and not automatic category labels.

## Frozen scale decision

P0n0 cannot establish prevalence. Expand to a stratified 200-instance audit
only if all apparatus gates pass and researcher review confirms:

- at least three nonlocal regressions;
- across at least two domains;
- and nonlocal regression in at least 5% of successful-fix revisions.

If there are zero confirmed nonlocal regressions, or all valid regressions are
direct/local, keep the stale-verdict line closed. One or two isolated cases are
case studies, not a reopened main method.

Any 200-instance follow-up requires an official-strength evaluator or an
independent human validation sample before paper-level claims.
