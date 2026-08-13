# Same-position dependency route-swap P2: frozen protocol

Frozen after P1 job 731174 and before observing any P2 model result.

## Question and claim boundary

P1 found that a clean printed-branch donor was more effective than an
equal-valued unprinted-branch donor at residual layer 21.  The donors came from
different checkpoint positions, so source-position compatibility remains an
alternative explanation.

P2 asks a narrower confirmatory question: does the effect remain when the two
donor states come from the **same token, value, absolute token position, label,
and local checkpoint context**, with only the earlier print route swapped?

A pass removes the specific P1 source-position confound.  It establishes at
most a one-hop route-conditioned hidden-state mechanism.  It does not establish
multi-level hierarchy, an HSGR Guide, or a downstream performance method.

## Frozen data construction

- Model: `Qwen/Qwen2.5-7B-Instruct`.
- New seed `20260817`; 96 calibration and 192 held-out cases.
- Fixed primary residual layer: **21**, inherited from P1 calibration.  P2 does
  not select a layer.
- Every case contains two one-edge identity branches, randomized single-letter
  labels, counterbalanced program-line order, and counterbalanced P/X
  checkpoint order:

```text
P --(+0)--> ROOT
X --(+0)--> DECOY
```

- Route-on donor prompt prints `ROOT`; route-off donor prompt prints `DECOY`.
- `P == X` in both donor prompts, so both prompts have the same executable
  answer and the checkpoint digit is identical.
- The corrupted receiver is the route-on prompt with only P's checkpoint digit
  changed, producing a different root.
- The route-on and route-off donor strings must have equal length and differ in
  exactly the one single-letter print target.
- At model tokenization time, the pair must have equal sequence length, the P
  checkpoint digit must have the same token index and token ID, and the input
  IDs must differ at exactly one earlier token.  Violation aborts the run.
- No generated reasoning trace; digit-restricted next-token scoring.

## Fixed representation diagnostic

At layer 21, extract the same P checkpoint digit from the route-on and route-off
donor prompts.  Reuse P0/P1's fixed 128-dimensional random projection and
linear probe.  Calibration cases fit the probe and held-out cases are untouched
until final evaluation.  Metadata vectors are identical within each pair.

The unchanged representation gate is reported as corroboration:

1. held-out paired accuracy >=0.70 and one-sided sign-test p<0.01;
2. paired accuracy >=0.65 in both fixed ID-hash halves;
3. hidden paired accuracy exceeds metadata-only by >=0.10.

P2's primary verdict is causal rather than probe-gated: a nonlinear or
distributed causal route effect should not be rejected solely because this
fixed linear probe misses it.

## Same-position causal intervention

Every arm starts from the same corrupted route-on receiver and patches its P
checkpoint digit once at locked layer 21:

- `correct_route_same_position`: route-on clean P state from the same case;
- `wrong_route_same_position`: route-off clean P state from the same case;
- `cross_problem`: route-on clean P state from another case matched on digit.

The first two source states are taken from the same absolute token index and
carry the same token/value.  Their prompts differ only in the earlier print
target.

## Frozen gates

Integrity gate: every paired donor satisfies all string/token invariants above.

Task and patch apparatus gate:

1. route-on and route-off clean accuracy are each >=0.90;
2. corrupted receiver accuracy against its own executable answer is >=0.50;
3. corruption lowers route-on clean-answer logp by >=0.20 nats with CI lower
   bound >0;
4. the correct-route patch raises clean-answer logp over corrupt by >=0.20 nats
   with CI lower bound >0 and raises accuracy by >=10 percentage points.

Same-position causal gate:

1. correct-route exceeds wrong-route by >=0.10 nats with CI lower bound >0;
2. correct-route exceeds wrong-route by >=3 percentage points in accuracy;
3. the logp difference is non-negative in both fixed ID-hash halves;
4. the correct-route patch recovers >=20% of the clean-versus-corrupt logp gap.

Equivalence requires both 95% CIs to lie wholly inside [-0.10,+0.10] nats and
[-3,+3] percentage points.  Mere non-significance is not equivalence.

## Frozen verdicts

- `INVALID_CONTROL`: any string/token invariant fails.
- `APPARATUS_FAIL`: the task or correct-route patch is invalid.
- `ROUTE_SWAP_CONFIRM`: apparatus and same-position causal gates pass.  This
  removes the P1 source-position explanation, but remains a one-hop claim.
- `POSITION_CONFOUND_SUPPORTED`: apparatus passes and the same-position contrast
  satisfies equivalence; P1's different-position effect should not be used as
  dependency-role evidence.
- `INCONCLUSIVE`: neither causal nor equivalence gate passes.  Do not tune on
  the held-out cases.

Only `ROUTE_SWAP_CONFIRM` permits freezing a separate depth-progression
protocol.  The representation diagnostic is reported separately in all cases.
