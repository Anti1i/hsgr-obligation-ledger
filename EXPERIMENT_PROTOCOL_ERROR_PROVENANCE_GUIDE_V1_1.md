# Dependency-error provenance Guide: corrective oracle protocol v1.1

Frozen on 2026-08-12 after job 727421 exposed severe 192-token truncation and
before generating any v1.1 output.  This is an explicitly post-diagnostic,
development-only corrective replication, not a confirmatory result.  It uses
the same 400-example `data/gsm_chain_test.jsonl` set and never reads the sealed
MuSiQue final377 set.

## 1. Allowed correction

V1.1 changes exactly one generation setting: every base node and every repair
arm receives a common maximum of 512 new tokens instead of 160/192.  Model,
greedy decoding, data, prompts, source labels, one-call repair budget, answer
checker, policies, effect sizes, and paired tests are unchanged from v1.

The purpose is to remove action-dependent right censoring, not to rescue a
particular action.  Job 727421 remains a failed v1 result regardless of v1.1.

## 2. Labels and actions

- `NONE`: the base Question-2 answer is correct.
- On base errors, `UPSTREAM`: the base Question-1 answer is wrong.
- On base errors, `LOCAL`: the base Question-1 answer is correct.
- Generic repair inspects both nodes.
- Upstream repair recomputes Question 1 and propagates it to Question 2.
- Local repair holds the predicted Question-1 value fixed and recomputes only
  Question 2.

The source oracle selects the action from synthetic gold provenance, never
from repair outcomes.  All repair arms use one call and the same 512-token
cap.  Hindsight best-of-repairs remains descriptive only.

## 3. Frozen gates

All gates must pass:

1. At least 30 base-error examples in both `UPSTREAM` and `LOCAL` strata.
2. Source oracle is at least +3.0 percentage points over generic repair on all
   400 problems, with exact paired McNemar `p < 0.05`.
3. Source oracle is at least +3.0 points over the better fixed-focus policy,
   with exact paired McNemar `p < 0.05`.
4. On `UPSTREAM` errors, upstream repair beats local by at least 5 points; on
   `LOCAL` errors, local repair beats upstream by at least 5 points.
5. Every repair arm uses one call and the same 512-token cap; actual prompt and
   generated tokens are reported.
6. Answer parse rate is at least 95% separately for base Question 1, base
   Question 2, generic repair, upstream repair, and local repair.

No threshold may be weakened after seeing v1.1 results.

## 4. Decision

If any validity or action-value gate fails, stop this route and do not submit
the join-provenance or hidden-state reader experiments.  If all pass, v1.1
establishes only action headroom on a two-node synthetic chain.  It does not
establish novelty.  The already frozen join-v2 oracle must then pass before a
hidden-state, permutation-equivariant incoming-edge scorer is considered.

