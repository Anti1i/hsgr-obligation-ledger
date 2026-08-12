# Dependency-role activation patching P0: frozen protocol

Frozen on 2026-08-13 before any model feature or patch outcome from this
experiment was observed.

## 1. Question and claim boundary

This is a mechanistic falsification experiment, not a Guide, selector, repair
policy, or performance method.  It asks two separate questions:

1. Can a frozen LLM hidden state distinguish a checkpoint variable that is an
   ancestor of the program output from a matched unused variable?
2. Does restoring that dependency-relevant state have more downstream causal
   effect than restoring a value-matched unused state?

Passing would establish dependency-role-specific causal sufficiency under the
tested checkpoint intervention.  It would not by itself establish endogenous
use during unrestricted generation, an edge-level causal graph, or a paper
method.  Failure is local to the tested model, layers, task, and intervention.

## 2. Controlled executable programs

Generate 288 deterministic straight-line programs with seed `20260813`, split
before model use into 96 calibration and 192 held-out examples.  Each program
contains two value- and topology-matched branches.  Only one is routed to the
printed output:

```text
P ----> R ----\
|       ^      \
|       |       U ----> ROOT
|       Q -----/          ^
|       |                 |
\-----> S <---- T         Q

X ----> R' ---\
|       ^      \
|       |       U' ---> DECOY_ROOT  (not printed)
|       Q' ----/          ^
|       |                 |
\-----> S' <--- T'        Q'
```

Every internal operation is `+`, `-`, or `*` modulo 10.  The two branches use
the same operations and matched checkpoint values, so corresponding internal
values are identical.  `P` and `X` have the same in/out degree, reuse pattern,
depth, value, and local computation; only `P`'s branch reaches `print`.  Labels
are randomly permuted, same-level program lines are counterbalanced, and the
adjacent checkpoint order of `P` and `X` is counterbalanced.

For each clean program, replace only `P`'s checkpoint digit with another digit
that changes the exact executed root.  Clean and corrupted prompts must have
identical token length and differ at exactly one one-token digit position.
The exact executor, not the language model, supplies all labels.

## 3. Model and feature extraction

- Frozen `Qwen/Qwen2.5-7B-Instruct`.
- Candidate residual-stream layers: 7, 14, and 21 (one-indexed block output).
- No sampling and no generated reasoning trace.
- The model receives the program, checkpoint, and an instruction to answer
  with one digit.  Primary scores are the digit-restricted next-token log
  probability and digit accuracy.
- Clean hidden states are extracted at the checkpoint digit tokens for `P`
  and `X`.  A fixed seeded random projection to 128 dimensions precedes a
  fixed linear probe.

## 4. Representation test

On calibration only, five problem-disjoint folds select the layer with the
highest paired `P > X` accuracy, breaking ties by pooled AUROC and then the
earlier layer.  The selected probe is fitted on all calibration examples and
evaluated once on held-out examples.

A metadata-only control uses checkpoint position, program mention position,
and label identity, but no hidden state.  Labels and order are randomized, so
this control measures remaining serialization leakage.

The representation gate requires all of:

1. held-out paired accuracy at least 0.70 and one-sided exact sign-test
   `p < 0.01`;
2. paired accuracy at least 0.65 in both fixed ID-hash halves;
3. hidden paired accuracy exceeds metadata-only by at least 0.10.

## 5. Causal interventions

All causal arms start from the same corrupted prompt.  At each candidate
layer, a residual state is transplanted once during the prompt forward pass:

- `correct_role`: clean `P` state -> corrupted `P` digit position;
- `wrong_route`: clean same-value `X` state -> the same corrupted `P` position;
- `cross_problem`: another example's clean `P` state, matched on `P` value and
  clean root answer -> the same corrupted `P` position;
- `root_positive`: clean gold-root digit state -> the corrupted prompt's final
  position, as an intervention-sensitivity positive control.

The primary causal layer is selected by the representation probe only.  Patch
results from all three layers are reported; no layer is reselected from causal
outcomes.

## 6. Frozen causal gates and outcome classes

### Apparatus and task validity

1. clean digit accuracy is at least 0.50;
2. corrupted-state digit accuracy against its own executable answer is at
   least 0.40;
3. corruption lowers the clean-answer log probability by at least 0.20 nats,
   with paired-bootstrap 95% CI lower bound above zero;
4. `root_positive` raises clean-answer accuracy by at least 10 percentage
   points over corrupt and raises its log probability by at least 0.20 nats,
   with CI lower bound above zero.

### Dependency-specific causal utilization

At the representation-selected layer, `correct_role` must:

1. exceed `wrong_route` by at least 0.10 nats in mean clean-answer log
   probability, with paired-bootstrap CI lower bound above zero;
2. exceed it by at least 3 percentage points in clean-answer accuracy;
3. have a non-negative log-probability difference in both ID-hash halves;
4. recover at least 20% of the clean-versus-corrupt log-probability gap.

All four conditions are required for a causal PASS.  `cross_problem` is a
secondary coupling diagnostic and cannot substitute for `wrong_route`.

### Representation-utilization gap

P0 may be called a *gap candidate* only if the representation and apparatus
gates pass and the 95% CIs for `correct_role - wrong_route` lie entirely
inside both equivalence margins: `[-0.10, +0.10]` nats and `[-3, +3]`
percentage points.  A non-significant difference without equivalence is
`INCONCLUSIVE`, not evidence of a gap.

## 7. Stop rules

- Representation fail: do not claim that the tested states encode dependency
  role; stop before an edge-level paper claim.
- Apparatus fail: the patch experiment is invalid, not negative evidence.
- Causal pass: freeze a separate path/edge-patching protocol with matched
  parent/non-parent routes and join interaction controls.
- Gap candidate: replicate on another model and a distinct logic/code task
  before any paper-level gap claim.
- Inconclusive: do not tune layers, thresholds, or the same 192 outcomes.
