# Route-augmented hidden Guide: frozen development protocol v2

Frozen after observing job 726572 and before training the route-augmented
models.  It reuses the exact feature payload from commit `1b074fc`; no new
prompts, candidates, labels, or final377 examples are read.

## Hypothesis

The matched absolute hidden state carries generic correctness information.
The predecessor-role counterfactual response may add a smaller, complementary
structural residual.  The candidate model therefore encodes, per layer:

```
absolute_l = Enc_abs(h_l(matched))
route_l = Enc_route([
    h_l(matched) - h_l(counterfactual),
    h_l(matched) * h_l(counterfactual)
])
```

and predicts candidate outcome from scalar features plus all `absolute_l` and
`route_l`.  This is still a selection-time Guide, not generation steering.

## Parameter-matched controls

- `ordinary-wide`: same matched hidden state, with encoder width chosen to
  match or exceed the route-augmented model's trainable parameter count;
- `activation-wide`: same-prompt candidate-boundary-to-verdict hidden delta,
  also parameter matched;
- route swap and cross-problem route mismatch use the trained augmented model
  and the same inner-selected policy weight;
- SC@8 and inner-tuned explicit predicted-state policies remain unchanged.

Exact trainable parameter counts must be reported.  A capacity advantage is
not accepted as structural evidence.

## Frozen gates on nested OOF over the same 840 problems

All gates must pass:

1. route-augmented is at least +1.0pp over `ordinary-wide` with Holm-adjusted
   paired `p < 0.05`;
2. route-augmented is at least +1.0pp over `activation-wide` with
   Holm-adjusted paired `p < 0.05`;
3. route-augmented is at least +2.0pp over SC@8 with Holm-adjusted
   `p < 0.05`;
4. at least +2.0pp over route swap and +1.0pp over route mismatch;
5. gain over SC@8 is non-negative for 2/3/4-hop strata and the 4-hop gain is
   no more than 1pp below the 2-hop gain;
6. gain over `ordinary-wide` is positive in at least four of five outer folds
   and both predeclared ID-hash halves;
7. all hyperparameters are selected inside each outer fold and the original
   equal-token/no-truncation audit remains valid.

Even a full pass does not open final377 until exact frozen SC completion-token
accounting is reconstructed.  Any failure of gates 1 or 2 ends the
hierarchy-specific hidden Guide selection route.
