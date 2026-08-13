# Distinct-value route apparatus-only screen P3A: frozen protocol

Frozen after P3 job 731461 and before observing any P3A model result.

## Purpose and claim boundary

P3's route directions were valid, but the base model reached only 76.0% and
71.4% clean accuracy on route-on and route-off prompts with distinct branch
values.  P3A performs one bounded prompt-apparatus repair before deciding
whether to stop the causal mechanism line.

P3A extracts no hidden states, fits no probe, and performs no activation patch.
It may select only a prompt rendering.  It is not evidence for route causality,
hierarchy, or HSGR Guide effectiveness.

## Frozen setup

- Model: `Qwen/Qwen2.5-7B-Instruct`.
- New seed `20260819`; 96 matched cases per rendering.
- P, X, and corrupted P are three distinct digits.
- Every route-on/off pair differs only in its single-letter `print(...)` target;
  every corruption changes only P's checkpoint digit.
- No generated reasoning trace; digit-restricted next-token scoring.

Three renderings are evaluated in fixed least-to-most-explicit order:

1. `original`: the exact P3 wording;
2. `explicit_select`: retains the full wording and states that only the branch
   selected by `print(...)` determines the answer;
3. `concise_select`: removes extraneous prose and directly states that the
   unprinted branch is irrelevant.

The examples, order, thresholds, and rendering order are fixed before model
use.  Selection is the first passing rendering, not the highest-scoring one.

## Frozen gate

A rendering passes only if all conditions hold:

1. route-on clean accuracy >=0.90;
2. route-off clean accuracy against its distinct decoy value >=0.90;
3. corrupted route-on accuracy against its own executable answer >=0.50;
4. route-on clean minus corrupted clean-target logp >=0.20 nats with bootstrap
   95% CI lower bound >0.

## Stop rule

- If one rendering passes, freeze it and use another disjoint seed for a single
  confirmatory rerun of the unchanged P3 subspace intervention.
- If none passes, stop the present hidden-state causal intervention line.  Do
  not add more prompt variants, relax the 0.90 threshold, or use P3 held-out
  outcomes to choose an intervention.  The next route must be external Guide
  utility on a separately defined task.
