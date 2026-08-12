# Dependency-error provenance Guide: frozen oracle action protocol v1

Frozen on 2026-08-12 before generating any result for this route.  This is a
development experiment on the already constructed 400-example GSM-chain set.
It does not read the sealed MuSiQue final377 set.

## 1. Narrow claim and novelty boundary

The proposed HSGR Guide does **not** rank a fixed answer pool, perturb decoder
residuals, allocate conformal risk, discover a graph, or perform generic
step-level error detection.  It asks a narrower graph-specific question:

> When a downstream node is wrong, did the error arrive through its dependency
> edge, or was it introduced by the downstream computation itself?

The hierarchy defines two different interventions.  A future hidden-state
observer may choose between them, but hidden features are forbidden in this
oracle ceiling:

- `UPSTREAM_REPAIR`: recompute Question 1 and propagate the corrected value;
- `LOCAL_REPAIR`: hold the Question-1 value fixed and recompute Question 2.

The source oracle uses only the known synthetic data construction:
`UPSTREAM` iff the base Question-1 answer is wrong, otherwise `LOCAL`.  It does
not choose an arm by looking at repair outcomes.

This boundary is narrower than generic mistake localization, targeted
re-prompting, or adaptive test-time compute.  A literature overlap audit must
still be repeated before any paper novelty claim.

## 2. Data and labels

- Data: all 400 rows of `data/gsm_chain_test.jsonl`, development only.
- Each row contains an exact `hop1_answer` and final `answer` produced by the
  symbolic composition builder.
- `NONE`: base Question 2 is correct.
- Among base Question-2 errors, `UPSTREAM`: base Question 1 is wrong;
  `LOCAL`: base Question 1 is correct.
- The problem index is the statistical unit.  There is no candidate-level
  pseudo-replication.

## 3. Frozen generation arms

All generations use `Qwen/Qwen2.5-7B-Instruct`, greedy decoding, identical
full problem text, identical base trace, one repair call, and the same output
cap.  Only the action instruction differs.

1. Base: solve Question 1, then solve Question 2 with that predicted value.
2. Generic repair: inspect both stages and correct the final answer.
3. Upstream repair: recompute Question 1 and propagate it to Question 2.
4. Local repair: keep the predicted Question-1 value fixed and redo only
   Question 2.

For ceiling policies, base-correct examples are kept.  Base-error examples are
sent to generic, always-upstream, always-local, or the source-oracle arm.  A
hindsight `best_of_repairs` value may be reported only as an unattainable upper
bound and is not a gate.

## 4. Frozen gates

The action space has graph-specific headroom only if **all** gates pass:

1. At least 30 base-error examples in each of the `UPSTREAM` and `LOCAL`
   strata.
2. Source-oracle policy is at least +3.0 percentage points over generic repair
   on all 400 problems, with exact paired McNemar `p < 0.05`.
3. Source-oracle policy is at least +3.0 points over the better of the two
   fixed-focus policies, with exact paired McNemar `p < 0.05`.
4. On `UPSTREAM` errors, upstream repair beats local repair by at least 5
   points; on `LOCAL` errors, local repair beats upstream repair by at least 5
   points.
5. Every repair arm uses one model call and the same generation cap; actual
   prompt and generated tokens are reported.  The source-oracle policy may not
   use gold answers in a prompt.

## 5. Stop rule and next stage

If any gate fails, do not train a hidden-state provenance classifier: the
hierarchy-specific action is not useful enough even with perfect source
knowledge.  If all gates pass, freeze a second protocol before extracting
hidden states.  That protocol must include text-only, confidence-only,
ordinary correctness, shuffled-source, and answer-blinded controls, nested
problem-disjoint OOF, and an end-to-end KEEP/UPSTREAM/LOCAL policy.
