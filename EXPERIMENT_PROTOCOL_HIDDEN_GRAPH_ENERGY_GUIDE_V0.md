# Hidden Graph-Energy Guide V0: frozen conditional protocol

Frozen on 2026-08-12 while structural-hardness screen job 728006 was still
pending and before any hidden feature, assignment-root outcome, or reader
metric for this experiment was observed.  Thresholds may not be relaxed after
results are read.

## 1. Question

On a fixed two-parent join graph with a fixed sampled value domain at each
parent, can frozen-LM hidden states guide selection of a graph assignment that
has high downstream root utility?  The experiment separates three questions:

1. does the AEO dual view (`last response state`, `mean response state`) add
   information beyond either view alone;
2. does AEO's asymmetric entropy objective improve uncertainty/selection over
   ordinary BCE with the same reader; and
3. does a structure-tied pairwise energy reader improve over flat hidden
   verification and non-hidden candidate selection?

The experiment starts only if the frozen structural-hardness screen passes.
A failure of that screen stops hidden extraction rather than redefining the
benchmark after seeing model outcomes.

## 2. Candidate graph and labels

- Model: frozen `Qwen/Qwen2.5-7B-Instruct`.
- Calibration graphs: the rule-selected subset of `gsm_join_train`.
- Confirmation graphs: the same frozen rule applied once to `gsm_join_test`.
- Parent domains are the normalized unique values among the already generated
  one-greedy plus three temperature-0.8 parent completions.  Gold values are
  never inserted.  Empty/unparseable samples collapse into one explicit
  `UNKNOWN` class so no graph is silently dropped.
- An assignment is every Cartesian-product pair from the two parent domains.
  The root is greedily executed once with that pair bound into the root
  question.  Duplicate normalized pairs are removed.
- The assignment label is normalized exact match of that root execution, not
  parent correctness.  Therefore a value can be useful or harmful only through
  its observed downstream root consequence.

No root outcome, gold answer, substituted answer, symbolic program, or label
enters a prompt or feature.  Labels are used only inside training folds and
for evaluation.

## 3. Hidden views and readers

For each cached parent and assignment-root completion, teacher-force the exact
prompt plus response through the frozen model.  At layers 14, 21, and 28,
extract and fixed-random-project to 256 dimensions:

```text
h_last = hidden state at the last non-padding response token
h_mean = mean hidden state over response tokens only
h_dual = concat(h_last, h_mean)
```

The proposed permutation-invariant join reader has a shared edge potential and
a root-local potential:

```text
edge_i = phi(h_parent_i, h_root, h_parent_i * h_root,
             abs(h_parent_i - h_root), log_frequency_i)
E(graph, assignment) = psi(h_root) + sum_i edge_i
```

Lower energy is better.  `phi` is shared across the two incoming edges; parent
order is randomized during training and swapping siblings must leave the score
unchanged up to numerical tolerance.  The proposed loss is the within-graph
pairwise softplus ranking loss over every available positive/negative pair.

The same graph reader is also trained with:

- ordinary binary cross-entropy (BCE); and
- AEO-style asymmetric entropy: positive assignments receive valid-class
  cross-entropy, while negative assignments are pushed toward a uniform
  two-class output by negative entropy regularization.  Selection uses the
  valid-class probability.

The AEO arm tests a loss and uncertainty hypothesis.  It is not described as
energy minimization.  The energy arm tests graph-global ranking and is not
described as an implementation of the Energy-Based Transformer architecture.

## 4. Frozen baselines and controls

All learned readers use identical problem-disjoint outer folds, inner model
selection, layers, projections, parameter cap, epochs, and candidate pools.

1. modal/frequency assignment;
2. non-hidden listwise reader using value frequency, answer length, domain
   size, and surface numeric features;
3. root-only last-state BCE verifier;
4. root-only dual-view BCE verifier;
5. flat dual-view MLP over the ordered three nodes with matched parameter cap;
6. per-parent ordinary correctness probes followed by a deterministic router;
7. graph reader with BCE;
8. graph reader with AEO asymmetric entropy;
9. proposed graph reader with pairwise energy ranking;
10. root-local-only (all parent edges deleted), cross-problem root mismatch,
    last-only, and mean-only controls.

Sibling swap is an invariance audit, not a degradation control: the proposed
reader must be unchanged, whereas the flat ordered reader need not be.

## 5. Evaluation and selection discipline

- Development is five-fold problem-disjoint nested OOF on calibration graphs.
- Every graph has equal total training weight.  On mixed-label graphs, positive
  and negative assignments each receive half of that graph's weight; pairwise
  energy losses are averaged within a graph before averaging across graphs.
- Layer, hidden width, regularization, AEO entropy weight, energy margin, and
  frequency-prior weight are chosen only on inner training/validation graphs.
- The method family is fixed in advance: both AEO and energy are reported.  A
  winner is not renamed as the sole preregistered method after results.
- Primary endpoint is root normalized exact match after selecting one
  assignment per graph.  Assignment AUROC, within-graph pairwise AUROC, ECE,
  selective risk, and high-confidence-negative rate are diagnostics.
- Accuracy uses exact paired McNemar tests and paired bootstrap 95% confidence
  intervals, with the graph as the unit.  The three primary comparisons of
  energy against modal, flat hidden, and graph-BCE use Holm correction.
- Report prompt tokens, generated tokens, teacher-forced forward tokens, and
  calls separately.  Direct SC@8 is contextual, not compute-matched unless an
  equal-total-token baseline is actually run.

## 6. Frozen development gates

Hidden extraction is licensed only by a full structural-screen pass.  The
energy route then passes development only if every condition holds:

1. at least 100 OOF graphs, at least 20 graphs where modal is wrong but some
   assignment is correct, and at least 40 graphs containing both positive and
   negative assignments;
2. at least +3.0 percentage points over modal assignment selection, with
   Holm-adjusted paired `p < 0.05`;
3. at least +1.0 point over the non-hidden reader, flat hidden reader, and
   ordinary per-parent correctness-probe router;
4. at least +1.0 point over graph-BCE; otherwise energy ranking has not earned
   a distinct mechanism claim even if hidden verification helps;
5. at least +1.0 point over the root-local-only reader and at least +2.0 points
   over the cross-problem root-mismatch control;
6. positive gain over modal in at least four of five folds and both fixed
   ID-hash halves;
7. sibling-swap score difference below `1e-6`, with complete compute counters.

The AEO mechanism is considered supported only if, relative to graph-BCE, it
simultaneously reduces high-confidence-negative rate by at least 20% relative,
does not worsen ECE, and improves selection accuracy by at least +1.0 point.
Dual-view support additionally requires at least +1.0 point over both last-only
and mean-only readers.  These auxiliary passes do not substitute for failed
energy gates.

Only a full development pass permits one fixed run on confirmation graphs.
Confirmation requires at least +2.0 points over modal and +1.0 point over flat
hidden and graph-BCE, with non-negative gains in both hash halves.  A paper
claim still requires a second naturally occurring structured benchmark and a
fresh related-work audit.

## 7. Non-overlap boundary and stop rules

This experiment does not claim novelty for hidden correctness probes,
best-of-N reranking, uncertainty-triggered regeneration, graph neural networks,
pairwise ranking, or energy-based modelling separately.  The only prospective
HSGR contribution is a hidden-state energy tied to dependency edges and
trained on downstream assignment utility, used as a Guide over a graph-valued
candidate space.

- If flat hidden matches the graph reader, the result is ordinary verification.
- If edge deletion or root mismatch does not hurt, the score is not tied to
  the hierarchy.
- If BCE matches energy, the EBT-inspired ranking route adds no demonstrated
  value.
- If AEO improves calibration but not selection, retain it only as an optional
  compute trigger, not as the main Guide.
- If candidate actionability is insufficient, change the benchmark/generation
  regime prospectively; do not manufacture wrong candidates or filter by
  observed method failures.
