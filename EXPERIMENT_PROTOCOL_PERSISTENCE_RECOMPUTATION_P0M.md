# P0m: persistence versus recomputation diagnostic

## Frozen question

Why does a verifier retain an unchanged conclusion after its upstream facts
make it false?

P0m separates two factors:

1. `Cache`: whether an old `SAT` verdict is shown;
2. `Execution`: whether the model returns only a verdict or must externalize a
   typed execution record.

This is a diagnostic, not a proposed structured-verification method. A typed
trace or deterministic checker is not claimed as novel.

## Related-work boundary

VeryTrace already compiles natural-language reasoning into explicit
dependencies and executable expressions. VPRM already applies deterministic
intermediate-step verifiers to process supervision. Contract-checked graph
editing validates structured graph runnability, and GenProve studies
fine-grained provenance including inference. P0m therefore asks only whether
revision/cache causes a distinct propagation failure; it does not claim that
structured traces, process verifiers, or provenance are new.

## Matrix

Reuse all 40 P0k-R1 cases and both revised states:

- 40 dependency edits where the unchanged conclusion is false;
- 40 matched harmless edits where it remains true.

Harmless controls are mandatory: a condition cannot improve stale recall by
merely answering `FAIL` more often.

Four arms form a 2 x 2 design:

| Arm | Prior-verdict context | Output |
|---|---|---|
| `fresh_free` | no mention of any prior verdict | one Boolean verdict |
| `cached_free` | old-version verdict was SAT | one Boolean verdict |
| `fresh_structured` | no mention of any prior verdict | typed execution record |
| `cached_structured` | old-version verdict was SAT | typed execution record |

Within each output mode, Cached adds exactly one historical-verdict line to
the otherwise identical Fresh prompt. Fresh does not mention prior verdicts at
all. Both conditions explicitly require using only the current document facts.

Total: 40 cases x 2 states x 4 arms x 2 models = 640 generations.

## Typed record

The structured arms must emit:

```json
{
  "source_ids": [3],
  "operator": "less_than",
  "operands": [40, 30],
  "computed_met": false,
  "met": false
}
```

Allowed operators are:

- `less_than(lhs, rhs)`;
- `claim_matches(source_claim, asserted_claim)`;
- `subtract_equals(revenue, cost, stated_result)`;
- `before(first_event_year, second_event_year)`;
- `above_threshold(score, threshold)`.

The prompt lists the operator vocabulary but does not identify the correct
operator, source IDs, operands, or answer for a case.

The model reports `computed_met` and its final `met`. A deterministic program
independently executes the selected operator on the model-extracted operands
to produce `checker_met`.

## Stage labels

Executable construction supplies oracle source IDs, operator, operands, and
current truth value. Each structured output is scored for:

1. source-record localization;
2. dependency-source coverage;
3. operator recognition;
4. operand extraction;
5. external executability;
6. checker correctness;
7. reported computation versus checker;
8. final verdict versus checker.

A `final_override` requires the source record, operator, and operands to be
correct, the external computation and reported computation to be correct, but
the final `met` to contradict that result.

## Models

- Qwen3-8B, non-thinking;
- Qwen2.5-14B-Instruct.

Generation is deterministic and prompt order is deterministically shuffled.

## Primary metrics

- stale recall and harmless specificity of the model's final verdict;
- the same metrics for `checker_met` in structured arms;
- paired Cached-versus-Fresh differences;
- paired Structured-versus-Free differences;
- per-stage and per-mechanism failure counts.

## Frozen diagnostic thresholds

### D1: apparatus validity

Every model-arm cell must have at least 95% parse validity.

### D2: cache anchoring

For each model, `fresh_free` must exceed `cached_free` stale recall by at
least 20 points with exact paired McNemar `p <= 0.05`, while fresh harmless
specificity is no more than 5 points below cached.

Anchoring is robust only if both models pass. A one-model effect is explicitly
model-specific.

### D3: model-level execution rescue

Within each cache condition and model, the structured **final model verdict**
must reach 75% stale recall, improve at least 20 points over the matching Free
arm with `p <= 0.05`, and retain at least 95% harmless specificity.

### D4: checker-level execution rescue

Apply the same thresholds using external `checker_met`. This measures the
combined extraction-plus-program system, not the model's own final verdict.

### D5: strong final override

Report the count of cases where extraction and deterministic computation are
all correct but final `met` contradicts them. This is descriptive; no claim is
made from isolated examples. A cache-specific override claim additionally
requires more overrides in Cached than Fresh under paired comparison.

## Decision branches

- A: D2 passes robustly -- study revision-time verdict anchoring. Structured
  execution remains a diagnostic tool, not the novelty.
- B: D2 fails but D3 or D4 passes robustly -- the main issue is spontaneous
  recomputation; any method must be revision-triggered and compared directly
  with structured-verification current methods.
- C: neither cache nor execution effect is reliable -- stop this controlled
  HSGR/incremental-verification line. Do not add hidden states or RL.

Passing P0m still does not establish natural prevalence. A natural revision
audit would be required before a paper-level phenomenon claim.
