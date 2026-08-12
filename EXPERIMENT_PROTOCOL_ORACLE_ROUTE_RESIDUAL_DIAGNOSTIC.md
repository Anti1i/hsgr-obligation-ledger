# Oracle-structure route residual: frozen root-cause diagnostic

Frozen on 2026-08-12 after jobs `726572` and `726645`, and before running this
diagnostic.  It pools only the already consumed 200+320+320 problem feature
sets.  The remaining final377 examples stay sealed.

## Question

Did the structure-de-oracled route residual fail because the predicted
hierarchy collapsed, or because a route contrast has no predictive value
beyond an ordinary absolute hidden-state verifier even when the hierarchy is
oracle supplied?

This is a labelled oracle mechanism audit.  It cannot establish an end-to-end
method, reopen the stopped hidden selection route, or authorize final377.

## Frozen data and models

- `structured_hidden_features_726228.pt`: 200 problems;
- `dual_route_hidden_features_726354.pt`: 320 problems;
- `transition_fresh_features_726389.pt`: 320 problems.

All 840 problems are pooled for five-fold problem-disjoint nested OOF.  The
cached `correct` and `wrong` route states use oracle decomposition, verified
predecessor values, and gold support routing.

Per layer, the primary model contains:

```
absolute_l = Enc_abs(h_l(correct route))
route_l = Enc_route([
    h_l(correct route) - h_l(wrong route),
    h_l(correct route) * h_l(wrong route)
])
```

The parameter-matched `ordinary-wide` control receives only
`h_l(correct route)` and must have at least as many trainable parameters as the
primary model.  Both use identical scalar features, labels, outer folds, inner
checkpoint selection, listwise loss, and policy-weight tuning.

Route swap and within-problem wrong-state mismatch reuse the trained primary
model.  SC@8 and an inner-tuned explicit oracle-route policy are also reported.
The old cache has no same-prompt start state, so this diagnostic makes no claim
beyond an activation-delta verifier.

## Frozen gates

The oracle residual has mechanism headroom only if every gate passes:

1. at least +1.0pp over `ordinary-wide`, Holm-adjusted paired `p<0.05`;
2. at least +2.0pp over SC@8, Holm-adjusted paired `p<0.05`;
3. at least +1.0pp over the explicit oracle-route policy;
4. at least +2.0pp over route swap and +1.0pp over route mismatch, with both
   paired `p<0.05`;
5. gain over ordinary-wide is positive in at least four of five outer folds
   and both predeclared ID-hash halves;
6. gain over ordinary-wide is non-negative for 2/3/4-hop strata, and the
   4-hop gain is no more than 1pp below the 2-hop gain;
7. candidate/problem counts are exact, IDs do not overlap across source
   payloads, and ordinary-wide has at least the primary parameter count.

Holm correction covers the comparisons against SC@8, the explicit oracle
policy, and ordinary-wide.

## Stop rule

- If gate 1 fails, do not invest in a stronger hierarchy predictor for this
  selection mechanism: oracle structure itself does not create residual value.
- If gate 1 passes but any other gate fails, record the mechanism as
  inconclusive and keep final377 sealed.
- Even a full pass only localizes the bottleneck to structure induction.  A
  new end-to-end protocol would still need predicted structure, an
  activation-delta control, exact compute accounting, and unconsumed data.

