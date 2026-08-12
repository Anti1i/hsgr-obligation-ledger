# MathQA executable-DAG audit P0-B: frozen protocol

Frozen on 2026-08-12 after MathQA passed structural P0 and before inspecting
the formulas or execution outcomes of individual MathQA examples.  This is a
zero-model, zero-GPU verification stage.

## Claim under test

The structural pass is genuine only if MathQA contains enough examples whose
native operation DAG is fully connected to the final result and whose gold
program actually executes to the annotated correct option.

## Eligible graph

An eligible example must:

1. parse under the operation-reference definition frozen in structural P0;
2. be `deep_join_reuse`;
3. have no operation outside the ancestry of the final operation
   (`dead_nodes == 0`); and
4. have valid backward references.

This stricter set prevents an unused branch from manufacturing a reuse signal.

## Execution semantics

- Resolve `nK` from numbers in the problem in textual order, `#K` from the
  K-th previous operation result, and `const_*` from MathQA constants.
- Use the operation semantics in Google Trax commit
  `220a62303ebf4ad18871aa5607b4dda2f064f2d2`, the fixed implementation used
  by its MathQA-Python generation notebook.
- Unsupported operators, malformed arguments, non-finite values, and runtime
  errors are execution failures, not silently dropped successes.
- Parse only the option identified by the annotated `correct` letter.  A
  nonnumeric correct option is unscorable.
- Agreement uses `math.isclose` with relative tolerance `0.01`, matching the
  fixed Trax implementation.  No best-option search is allowed.

Report execution coverage and answer agreement for both all programs and the
eligible target set, plus unsupported-operation and failure histograms.

## Frozen gate

P0-B passes only if all are true:

1. train eligible target count is at least 500;
2. test eligible target count is at least 100;
3. test eligible-target execution coverage is at least 95%;
4. test executable-target answer agreement is at least 95%;
5. train eligible-target execution coverage is at least 95%; and
6. train executable-target answer agreement is at least 95%.

Passing licenses a later difficulty/baseline audit and atomic intervention
oracle.  It does not license a hidden-state model by itself.  Failure stops the
MathQA route; thresholds are not relaxed after observing results.
