# One-hop dependency-role activation patching P1: frozen protocol

Frozen after apparatus-screen job 731104 and before observing any P1 hidden
feature or patch outcome.

## Question and claim boundary

The screen selected `chain1_copy`, the only family satisfying the fixed task
gate.  P1 asks whether a frozen LLM hidden state distinguishes and causally uses
the checkpoint value on a one-edge printed branch rather than a topology- and
value-matched unprinted branch.

A pass establishes only a minimal one-hop dependency-role mechanism.  It does
not establish a multi-level hierarchy, edge graph, HSGR Guide, or performance
method.  The three-edge screen failure remains a separate negative result.

## Data and model

- Frozen `Qwen/Qwen2.5-7B-Instruct`.
- New seed `20260816`, disjoint from P0 and the apparatus screen.
- 96 calibration and 192 held-out cases, split before model use.
- Each prompt contains two matched one-edge identity branches:

```text
P --(+0)--> ROOT  --> print
X --(+0)--> DECOY      (not printed)
```

- `P == X` in every clean prompt.  Labels, same-level program order, auxiliary
  checkpoint order, and adjacent P/X order are randomized and counterbalanced.
- Corruption changes only P's single digit, producing a different exact root.
- No generated reasoning trace; digit-restricted next-token scoring.

## Representation test

- Candidate residual layers: 7, 14, 21.
- Extract P and X checkpoint-digit residuals from clean prompts.
- Use the same fixed 128-dimensional random projection and linear probe as P0,
  preventing a post-P0 increase in probe capacity.
- Five case-disjoint calibration folds select the layer by paired P>X accuracy,
  then pooled AUROC, then the earlier layer.
- Metadata control: normalized checkpoint position, normalized program-mention
  position, and randomized label identity.

The frozen representation gate is unchanged from P0:

1. held-out paired accuracy >=0.70 and one-sided sign-test p<0.01;
2. paired accuracy >=0.65 in both fixed ID-hash halves;
3. hidden paired accuracy exceeds metadata-only by >=0.10.

## Causal interventions

Every arm starts from the same corrupted prompt and patches the corrupted P
digit position once at a candidate layer:

- `correct_role`: same-case clean P state;
- `wrong_route`: same-case clean X state, which carries the same clean digit;
- `cross_problem`: another clean P state matched on the clean digit.

The representation-selected layer is primary; all layers are reported without
causal re-selection.

## Task and patch apparatus gate

At the selected layer all conditions are required:

1. clean accuracy >=0.90;
2. corrupt accuracy against its own executable answer >=0.50;
3. corruption lowers clean-answer logp by >=0.20 nats with CI lower bound >0;
4. `correct_role` raises clean-answer logp over corrupt by >=0.20 nats with CI
   lower bound >0 and raises clean-answer accuracy by >=10 percentage points.

## Dependency-specific causal gate

At the selected layer `correct_role` must:

1. exceed `wrong_route` by >=0.10 nats with CI lower bound >0;
2. exceed it by >=3 percentage points in clean-answer accuracy;
3. have non-negative logp difference in both ID-hash halves;
4. recover >=20% of the clean-versus-corrupt logp gap.

The representation-utilization gap definition is unchanged: representation and
apparatus must pass, and both correct-minus-wrong 95% CIs must lie wholly inside
[-0.10,+0.10] nats and [-3,+3] percentage points.  Mere non-significance is
`INCONCLUSIVE`.

## Stop decisions

- `REPRESENTATION_FAIL`: stop this hidden dependency-role route.
- `APPARATUS_FAIL`: repair the intervention only; do not treat it as evidence
  about structural use.
- `CAUSAL_PASS`: freeze a separate depth-progression protocol; do not claim P1
  itself demonstrates hierarchy.
- `GAP_CANDIDATE`: replicate on another model before a gap claim.
- `INCONCLUSIVE`: do not tune on the 192 held-out cases.

