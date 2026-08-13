# Value-orthogonal route-subspace intervention P3: frozen protocol

Frozen after P2 job 731363 and before observing any P3 model outcome.

## Question and claim boundary

P2 established a strong same-position route-dependent hidden-state signal, but
a full-state route swap had a practically equivalent answer effect.  That does
not by itself distinguish an unused signal from a small causal component masked
by compensating changes elsewhere in the state.

P3 asks whether the calibration-estimated **route subspace itself** can switch
the model from the printed branch toward a distinct decoy branch while holding
checkpoint token, value, position, and linear digit-value subspace fixed.

A pass would establish only a controlled one-hop route mechanism.  It would
not establish multi-level hierarchy, an HSGR Guide, or benchmark improvement.

## Data and locked analysis choices

- Model: `Qwen/Qwen2.5-7B-Instruct`.
- New seed `20260818`; 96 calibration and 192 held-out cases.
- Two one-edge identity branches with randomized labels, program order, and
  checkpoint order.
- P, X, and corrupted P are three distinct digits.  Therefore route-on clean,
  route-off decoy, and corrupted-receiver answers are distinguishable.
- Route-on and route-off source prompts differ only in the single-letter
  `print(...)` target.  Their P checkpoint token/value/position and all other
  text are identical.
- The corrupted receiver is route-on and changes only P's digit.
- At tokenization time every pair must have equal length, identical P token
  index/ID, and exactly one earlier differing route token.  Any violation stops
  the run as `INVALID_CONTROL`.
- Primary intervention: layer 21, inherited from P1/P2.
- Secondary, non-selective diagnostic: fixed late window layers 19–21.  It may
  motivate replication but cannot rescue a failed primary result.
- No generated trace; digit-restricted next-token scoring.

## Calibration-only route direction

For each intervention layer, calibration cases only are used:

1. average the route-on/off source state for each case;
2. form a conservative linear digit-value basis from source-token digit
   centroids and route-off answer-digit centroids;
3. project every paired route-on minus route-off difference outside that value
   basis;
4. normalize each paired difference, average them, project outside the value
   basis again, and unit-normalize to obtain direction `d`;
5. construct a deterministic seeded sham direction `q`, orthogonal to both the
   value basis and `d`.

Held-out labels or model outcomes are not used to choose directions, layers,
windows, scales, or thresholds.  Direction integrity requires value-basis
overlap <=1e-5.  Layer-21 held-out paired direction accuracy must be >=0.70,
with both fixed hash halves >=0.65, for a causal verdict.

## Interventions

All arms patch the same corrupted route-on receiver at its P checkpoint.  At
each active layer, let clean route-on/off source states be `h_on` and `h_off`,
and `s = dot(h_off - h_on, d)`.

- `correct_full`: use `h_on`;
- `wrong_full`: use `h_off`;
- `route_swap`: use `h_on + s*d`, replacing only the route projection;
- `sham_plus`: use `h_on + abs(s)*q`;
- `sham_minus`: use `h_on - abs(s)*q`.

The two sham arms have the same edit norm as `route_swap`; their per-case
log-probabilities and correctness values are averaged before comparison.  The
same five arms are run at layer 21 and at all layers 19–21.

## Frozen gates

The apparatus gate requires, for each evaluated mode:

1. route-on and route-off clean accuracies each >=0.90;
2. corrupt accuracy against its own executable answer >=0.50;
3. route-on clean minus corrupt clean-answer logp >=0.20 nats with CI lower >0;
4. `correct_full` improves clean-answer logp over corrupt by >=0.20 nats with
   CI lower >0 and improves clean-answer accuracy by >=10 points.

The strict route-switch gate requires all of:

1. sham-average minus `route_swap` clean-answer logp >=0.10 nats, CI lower >0;
2. corresponding clean-answer accuracy difference >=3 points;
3. `route_swap` minus sham-average decoy-answer logp >=0.10 nats, CI lower >0;
4. corresponding decoy-answer accuracy difference >=3 points;
5. correct-full versus sham-average is practically equivalent: both 95% CIs
   lie within [-0.10,+0.10] nats and [-3,+3] points;
6. sham-minus-route clean-logp difference is non-negative in both fixed halves.

Route-null equivalence requires route-swap versus sham-average CIs for both
clean and decoy logp to lie within [-0.10,+0.10] nats, and both accuracy CIs
within [-3,+3] points.

## Verdicts

- `INVALID_CONTROL`: source/token/value-direction integrity fails.
- `DIRECTION_FAIL`: layer-21 held-out route-direction gate fails.
- `APPARATUS_FAIL`: primary layer-21 task/patch apparatus fails.
- `SUBSPACE_ROUTE_SWITCH`: the primary layer-21 strict switch gate passes.
- `WINDOW_ONLY`: primary fails but fixed 19–21 window passes; exploratory only,
  requiring a new-seed confirmation before a causal claim.
- `ROUTE_SUBSPACE_NULL`: both modes satisfy route-null equivalence.  Stop the
  present activation-steering line and move to external Guide utility.
- `INCONCLUSIVE`: no causal or equivalence verdict; do not tune on held-out data.

Depth progression is prohibited unless the primary result is
`SUBSPACE_ROUTE_SWITCH` or a later preregistered replication confirms a
`WINDOW_ONLY` result.
