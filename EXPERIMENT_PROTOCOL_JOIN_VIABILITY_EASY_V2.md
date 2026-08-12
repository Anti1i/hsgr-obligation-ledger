# Easy-Join Viability V2: frozen protocol

Frozen on 2026-08-12 after Join Viability V1 Phase A failed and before any
model outcome was generated on the new easy-join datasets.

## Claim under test

This is still P0 benchmark/backbone selection, not an HSGR method experiment.
It tests whether a controlled easy compositional join has a learnable regime:
parents are usually solvable, clean end-to-end accuracy is away from floor and
ceiling, and correct parent values expose additional root-composition ability.

A pass licenses a separately frozen counterfactual edge-conditioned latent
Guide experiment.  A fail stops `Qwen2.5-7B-Instruct` on this join family; it
does not license threshold changes, harder/easier post-hoc filtering, hidden
training, structural edits, or prompt repair.

## Frozen data construction

The builder is `data_prep.py --which gsm_join_easy`.

- Parent source problems: annotated GSM8K solutions with at most 3 executable
  arithmetic steps, positive integral answers, and the pre-existing unitless
  exclusion.
- Root source problems: at most 2 executable arithmetic steps.
- Existing join constraints remain unchanged: two distinct numeric literals,
  independently causal replacements, integer positive recomputed answer,
  unambiguous one-time textual replacement, and replacement ratio in
  `[1/3, 3]`.
- Train: GSM8K train split, seed 41, first 400 constructed rows.
- Test: GSM8K test split, seed 43, all 184 constructible rows (limit 200).
- Frozen canonical-LF SHA256:
  - train `576fcbf7d6cee0d0d3c9e4a1cf059c0d474b17f675da183c1f47e41df30ae129`
  - test `5d274c136e149481e47b7ebe534599f7f899efc8e9ae073cce757b3c95c4e1ad`

The caps were selected from V1 calibration metadata/outcomes.  They are
therefore benchmark-development choices, not held-out evidence.  Test outcomes
may not change them.

## Frozen subsets

- Calibration: 96 train rows with smallest SHA256 rank of
  `join-viability-easy-v2|calibration|<problem>`.
- Confirmation: 128 test rows with smallest SHA256 rank of
  `join-viability-easy-v2|confirmation|<problem>`.
- Selection never uses answers or model outcomes.

## Frozen calls

Model: `Qwen/Qwen2.5-7B-Instruct`, deterministic greedy decoding.

For each graph:

1. one clean direct whole-graph call, `max_new=512`;
2. one independent call for each parent, `max_new=192` each;
3. one root call with both annotated gold parent values, `max_new=192`.

Prompts, system message, binding function, and answer parser are exactly those
used by `structural_hardness_screen.py`.  This stage has no sampled candidates,
repair prompts, hidden extraction, edits, or answer-conditioned routing.

## Calibration gate

Run calibration first.  Continue to confirmation only if all conditions hold:

1. exactly 96 rows and complete `96 * 4` call accounting;
2. direct accuracy in `[0.30, 0.70]`;
3. mean parent accuracy at least `0.70`;
4. gold-bound root accuracy at least `0.70`;
5. gold-bound root minus direct accuracy at least `0.10`;
6. all answer parse-validity rates at least `0.95`.

## Confirmation gate

If calibration passes, run the frozen 128-row confirmation once.  The final
gate requires:

1. exactly 128 rows and complete `128 * 4` call accounting;
2. direct accuracy in `[0.25, 0.75]`;
3. mean parent accuracy at least `0.65`;
4. gold-bound root accuracy at least `0.65`;
5. gold-bound root minus direct accuracy at least `0.08`;
6. all answer parse-validity rates at least `0.95`.

## Decision after a pass

Do not train immediately.  First freeze an SC@8 headroom addendum on the same
subsets and compare direct SC@8/oracle@8 with greedy performance.  Only if the
regime remains non-ceiling and useful after this stronger baseline may a
counterfactual edge-conditioned latent Guide protocol be written.
