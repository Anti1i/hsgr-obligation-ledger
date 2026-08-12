# Easy-Join Viability V3: frozen 14B backbone screen

Frozen on 2026-08-12 after the length-repaired 7B V2.1 calibration failed and
before any 14B generation.  This remains P0 benchmark/backbone selection, not
an HSGR method experiment.

## Evidence motivating the only change

On the frozen 96-row calibration, V2.1 measured:

- clean direct accuracy `0.53125`;
- mean parent accuracy `0.96875`;
- gold-bound root accuracy `0.625`;
- gold-bound-root minus direct `0.09375`;
- parse validity `1.0` and token-cap rate `0.0` for every arm.

Thus the 7B model is in the desired non-floor/non-ceiling direct regime and
solves nearly all parents, but fails the pre-registered root `0.70` and gap
`0.10` gates.  V3 tests the remaining pre-method alternative: whether a larger
backbone in the same model family makes this benchmark compositionally viable.

## Frozen model

- Model family: `Qwen/Qwen2.5-14B-Instruct`.
- Hugging Face snapshot:
  `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`.
- Inference must use the project-local snapshot path with Hub and Transformers
  offline modes enabled.  A floating `main` revision is forbidden.
- Deterministic greedy decoding, as in V2.1.

## Frozen invariants

The backbone snapshot is the only experimental change from V2.1.  V3 retains:

- the identical easy-join train/test files and canonical hashes;
- the identical 96 calibration and 128 confirmation IDs and ranking rule;
- the same system, direct, parent, and gold-bound-root prompts;
- the same binding function and boxed-answer parser;
- generation limits: direct `1536`; each parent and gold root `512`;
- one call per arm, with no sampling, repair, routing, hidden extraction,
  answer-conditioned behavior, or reuse of any 7B call cache;
- all V2.1 calibration and confirmation thresholds, including token-cap rate.

V2/V2.1 never evaluated confirmation, so confirmation remains untouched.  The
known 7B calibration is benchmark/backbone-development evidence; a 14B
calibration pass by itself is not final evidence.

## Progressive gate

Run 14B calibration first.  Continue automatically only if all hold:

1. exactly 96 rows and `96 * 4` one-call accounting;
2. direct accuracy in `[0.30, 0.70]`;
3. mean parent accuracy at least `0.70`;
4. gold-bound root accuracy at least `0.70`;
5. gold-bound root minus direct accuracy at least `0.10`;
6. parse validity at least `0.95` for every arm;
7. token-cap rate at most `0.05` for every arm.

Only after a complete calibration pass, run the frozen confirmation exactly
once.  It must satisfy:

1. exactly 128 rows and `128 * 4` one-call accounting;
2. direct accuracy in `[0.25, 0.75]`;
3. mean parent accuracy at least `0.65`;
4. gold-bound root accuracy at least `0.65`;
5. gold-bound root minus direct accuracy at least `0.08`;
6. parse validity at least `0.95` for every arm;
7. token-cap rate at most `0.05` for every arm.

## Decision rule

- Confirmation pass: freeze and run the SC@8 headroom addendum before any
  hidden-state or Guide training.
- Calibration or confirmation fail: stop this synthetic easy-join benchmark.
  No larger backbone, post-hoc filtering, threshold relaxation, prompt repair,
  or hidden/Guide training on this benchmark is licensed.
