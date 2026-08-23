# Causal support-entanglement study P0h

## Question

P0h asks one narrow causal question:

> Holding propositions, the seeded target error, model, and repair instruction fixed, does placing
> the failed and preserved obligations in the same sentence make a sentence-level repair more
> likely to damage a previously satisfied obligation?

This is a controlled mechanism study.  It does not estimate natural-task prevalence and does not
test a provenance-ledger method, graph, planner, or hidden state.

## Matched-pair construction

Eighteen content blocks are frozen: six numeric targets, six attribution targets, and six ordering
targets.  Each block contains three logically atomic clauses:

- `O_LEFT`: satisfied and to be preserved;
- `O_TARGET`: contains the only seeded error;
- `O_RIGHT`: satisfied and to be preserved.

Every block is rendered in two layouts:

1. `entangled`: the three clauses are coordinated inside one sentence;
2. `disentangled`: the same three clauses appear as three separate sentences in the same order.

The evidence, clause propositions, citations, failed target, and repair instructions are identical
within a pair.  Only punctuation and the minimum connective wording needed by the layout differ.
The experiment therefore identifies support co-location under the frozen editing interface, not a
general semantic notion of entanglement.

## Operators

Two operators are crossed with both layouts:

1. `sentence_patch`: return a JSON patch replacing exactly one source sentence;
2. `full_rewrite`: return a complete minimally revised answer.

The sentence patch is the primary operator.  In the entangled layout, its smallest editable unit
contains all three obligations; in the disentangled layout, the target occupies its own unit.  Full
rewrite is an operator control: an equally large layout effect there would show that layout itself,
not specifically the mismatch between obligation and edit granularity, drives the result.

The chosen source sentence, its character share, frozen-witness overlap, answer edit ratio, target
repair, and preserved-obligation regressions are logged for every output.

## Verification and controls

Qwen2.5-14B-Instruct independently verifies all three obligations and returns answer-sentence
witnesses.  Before candidate results are interpreted, all clean and seeded answers are judged.
Parse validity, positive-control accuracy, and negative-control accuracy must each be at least 95%.

Every claimed regression must be manually reviewed.  A frozen sample of successful
non-regressions is also reviewed for verifier false negatives.

## Primary paired estimand

For each operator, the primary analysis retains content blocks where the target was successfully
repaired in both layouts.  It compares:

`Pr(any preserved-obligation regression | entangled)`

against:

`Pr(any preserved-obligation regression | disentangled)`.

The report includes the paired risk difference, entangled-only and disentangled-only discordant
counts, and the exact one-sided sign-test probability.  Conditioning on joint repair success is
reported explicitly; unconditional repair and safe-repair rates are also reported to expose any
selection trade-off.

## Frozen decision gates

P0h supports proceeding to a hard constrained-editor P0i only if all conditions survive manual
review:

1. the verifier controls pass;
2. at least 12 sentence-patch pairs repair the target in both layouts;
3. at least five discordant pairs regress only in the entangled layout, no more than one regresses
   only in the disentangled layout, and the exact one-sided sign-test probability is below 0.05;
4. the paired sentence-patch regression risk difference is at least 20 percentage points;
5. among at least ten content blocks successfully repaired in all four operator-by-layout cells,
   the sentence-patch layout effect exceeds the full-rewrite layout effect by at least 15 points.

If Gate 2 fails, the study is inconclusive rather than negative.  If Gates 3--5 fail with adequate
joint repair success, support co-location under this interface is not retained as the proposed
repair-interference mechanism.  Passing all gates motivates P0i but does not establish natural-data
generality.
