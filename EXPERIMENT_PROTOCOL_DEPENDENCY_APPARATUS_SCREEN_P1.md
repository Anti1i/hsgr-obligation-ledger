# Dependency-role P1 apparatus-only screen: frozen protocol

Frozen after P0 job 731009 and before observing any outcome from this screen.

## Purpose and claim boundary

P0 was not a valid utilization test because Qwen2.5-7B-Instruct achieved only
9.9% clean accuracy on the direct-answer program task.  This screen determines
whether a simpler controlled program family supports a valid next-token causal
apparatus.  It does not extract hidden states, train a probe, perform activation
patching, or test a hierarchy claim.

## Fixed model and data

- Frozen `Qwen/Qwen2.5-7B-Instruct`.
- Seed `20260815` and 96 generated examples per family.
- Exact modulo-10 executor labels every clean and corrupted prompt.
- Clean and corrupt prompts differ only in one single-digit `P` checkpoint.
- `P` and `X` have the same checkpoint value, local operators, depth, degree,
  and corresponding auxiliary values.  Only the branch from `P` is printed.
- Labels, same-level line order, and adjacent `P`/`X` checkpoint order are
  randomized and counterbalanced.
- The model receives no generated reasoning trace and is scored on its
  digit-restricted next-token distribution.

## Pre-specified families, hardest first

1. `dag_add`: the P0 depth/reuse/join DAG, with every operator fixed to `+`.
2. `chain3_add`: matched three-edge P/X chains with random additive constants.
3. `chain3_copy`: matched three-edge chains whose constants are zero, so the
   printed root is an identity copy through a nontrivial dependency path.
4. `chain1_copy`: a matched one-edge identity control.

The order is fixed before outcomes.  It prioritizes structural complexity, not
the highest observed accuracy.

## Family gate and selection

A family passes only if all conditions hold:

1. clean executable digit accuracy >= 0.60;
2. corrupt accuracy against its own executable digit >= 0.50;
3. corruption lowers the clean-answer log probability by at least 0.20 nats;
4. the paired-bootstrap 95% CI lower bound for that decrease is above zero.

Select the first passing family in the fixed hardest-to-easiest order.  If no
family passes, stop direct-next-token dependency patching for this model.

## Use of the result

The screen examples cannot be used to fit or evaluate a structural probe.  If
a family passes, P1 must use a new seed and newly generated calibration and
held-out examples.  P1 representation and causal gates must be frozen before
their outcomes are observed.  P0's 192 held-out examples cannot be reused for
layer, projection, probe, or threshold selection.

