# Easy-Join Viability V2.1: frozen length-repair protocol

Frozen on 2026-08-12 after V2 calibration exposed systematic truncation and
before any V2.1 generation.  This is a measurement repair, not a new method or
an independent calibration result.

## Why V2.1 exists

V2 used `max_new=512` for the direct arm and `max_new=192` for each parent and
gold-bound root arm.  Its parse-validity rates were only 0.833, 0.771, 0.698,
and 0.865 respectively.  Inspection showed that the missing boxed answers were
truncated generations: 54/96 direct calls and 15/96 gold-root calls reached
their token limits.  In particular, even the adversarial upper bound that
counts every invalid gold-root answer as correct was 0.698, immediately below
the pre-registered 0.70 gate.  V2 therefore cannot distinguish a failed join
regime from an inadequate generation budget.

## Frozen invariants

V2.1 keeps all of the following exactly as V2:

- `Qwen/Qwen2.5-7B-Instruct` with deterministic greedy decoding;
- easy-join train/test files and their canonical hashes;
- the 96 calibration and 128 confirmation subsets and ranking rule;
- system, direct, parent, and gold-bound-root prompts;
- binding function and boxed-answer parser;
- one call per arm with no sampling, repair, routing, hidden extraction, or
  answer-conditioned behavior;
- all accuracy, gap, sample-count, and parse-validity thresholds.

No V2 call cache may be reused.  V2.1 writes a fresh complete call set.

## The only decoding change

- direct whole-graph call: `max_new=1536`;
- each parent call: `max_new=512`;
- gold-bound root call: `max_new=512`.

In addition to V2's gates, every arm must have token-cap rate at most 0.05.
This stricter diagnostic gate ensures that a parsed intermediate box in a
truncated response cannot make the run appear valid.

## Progressive gate

Calibration runs first with the original V2 thresholds:

1. exactly 96 rows and `96 * 4` one-call accounting;
2. direct accuracy in `[0.30, 0.70]`;
3. mean parent accuracy at least `0.70`;
4. gold-bound root accuracy at least `0.70`;
5. gold-bound root minus direct accuracy at least `0.10`;
6. parse validity at least `0.95` for every arm;
7. token-cap rate at most `0.05` for every arm.

Because V2 calibration outcomes are already known, a V2.1 calibration pass is
only a license to evaluate; it is not final evidence.  The confirmation split
was never run by V2 and remains untouched.  It is evaluated exactly once only
after a complete calibration pass, with V2's original confirmation thresholds
plus the same token-cap gate:

1. exactly 128 rows and `128 * 4` one-call accounting;
2. direct accuracy in `[0.25, 0.75]`;
3. mean parent accuracy at least `0.65`;
4. gold-bound root accuracy at least `0.65`;
5. gold-bound root minus direct accuracy at least `0.08`;
6. parse validity at least `0.95` for every arm;
7. token-cap rate at most `0.05` for every arm.

Failure stops this 7B easy-join regime.  Passing confirmation licenses only the
separately frozen SC@8 headroom check described in V2, not hidden-state or Guide
training.
