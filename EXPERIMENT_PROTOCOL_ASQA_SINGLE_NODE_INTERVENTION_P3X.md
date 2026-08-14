# ASQA fresh single-node intervention Oracle P3x (frozen protocol)

Date frozen: 2026-08-14 (Asia/Shanghai), after P1x job 733287 and P2x job
733309, but before any P3x model output was generated.

## 1. Claim under test

P1x established that showing all gold facet questions improves long-answer
coverage.  It did not establish that individual facet nodes have differentiated
causal utility or that selecting one node is a meaningful problem.

P3x tests the necessary action-space claim:

> On fresh clean fixed-support ASQA cases, different single-facet textual
> interventions produce complementary strict repairs, so a per-problem
> single-node Oracle materially exceeds both a no-node/all-node hindsight
> Oracle and any fixed candidate-position policy.

This is a textual intervention Oracle and target audit.  It is not HSGR, does
not test hidden states, and cannot be presented as a novel answer-generation
method.  Its only purpose is to determine whether a later node-specific latent
Guide has a non-trivial action space.

## 2. Fresh subset

Reconstruct the 427 clean/support-complete cases using the exact P1x rules and
released ASQA/ALCE joins.  Reconstruct the 192 P1x IDs using
`20260815-clean-p1x|released_id` and exclude all of them.  The expected fresh
pool therefore contains 235 cases.

Select 192 cases from that pool by ascending SHA-256 of
`20260815-asqa-single-node-p3x|released_id`.  Selection is independent of all
P1x/P2x outputs and all P3x generations.  Every selected case must retain 2--6
unique facet nodes, exactly five fixed documents, full alias support in those
documents, a strict human answer, and no verbatim long-answer leakage.

## 3. Model and frozen interventions

- Model: `Qwen/Qwen2.5-7B-Instruct`, bfloat16, greedy decoding.
- Maximum 192 new tokens; no sampling, retrieval, reranking, judge model,
  regeneration, or answer-dependent prompt modification.
- Every arm uses the same five ALCE oracle-reranked fixed documents.
- Gold short-answer aliases are used only for scoring.  They never enter a
  prompt, candidate representation, or model feature.

For every selected case generate:

1. `fixed_direct`: ambiguous question plus fixed documents, with no checklist;
2. `all_true`: the same prompt plus all released facet questions;
3. `single_true_i`: one generation for each facet `i`, with only that released
   facet question in the checklist;
4. `single_decoy`: one generation with a single facet question from another
   selected case, deterministically matched to the nearest question word count.

The decoy excludes the same problem and is selected by the tie-break hash
`asqa-p3x-decoy|source_id|decoy_id|facet_index`.  Candidate order is the
released QA-pair order and may not be changed after outputs are observed.

## 4. Scores and policies

Use the frozen ALCE-normalized alias-substring scorer from P0/P1x.

- `STR-EM`: mean facet coverage per problem.
- `STR-HIT`: fraction of problems for which every facet is covered.
- A strict node repair occurs when `fixed_direct` fails STR-HIT but a given
  `single_true_i` succeeds.
- A mixed intervention problem is a direct failure for which at least one
  single-node arm succeeds and at least one other single-node arm fails.

Report the following policies:

- `uniform_single`: within each problem, average the metrics over all its
  single-node arms, then average across problems;
- `best_fixed_position`: choose one released candidate index globally; use that
  single-node arm whenever it exists and otherwise use `fixed_direct`; choose
  the index only by aggregate P3x reporting, not as a deployable learned policy;
- `keep_or_all_oracle`: per problem, hindsight success of either
  `fixed_direct` or `all_true`;
- `keep_or_single_oracle`: per problem, hindsight success of
  `fixed_direct` or any one `single_true_i`.

For STR-EM Oracles, take the maximum per-problem coverage among the specified
arms.  For node specificity, compare each single-node arm's change in its
injected facet against its average change in non-injected facets, and compare
`uniform_single` against `single_decoy`.

## 5. Frozen gates

### Apparatus and fresh replication

1. exactly 427 eligible cases, 192 old P1x IDs, 235 fresh-pool cases, and 192
   selected P3x cases, with zero old/new ID overlap;
2. every selected problem has 2--6 nodes and all expected generations are
   present exactly once;
3. `fixed_direct` STR-HIT is 35--75%, its median length is 30--160 words, and
   at least 60 selected problems fail direct STR-HIT;
4. `all_true - fixed_direct` is at least +5 points STR-HIT, replicating static
   structure usefulness on the fresh subset.

### Single-node actionability

5. `keep_or_single_oracle - keep_or_all_oracle` is at least +5 points STR-HIT;
6. the exact paired two-sided McNemar p-value for those two Oracle success
   vectors is below 0.05, with more single-only than all-only successes;
7. at least 24 direct-failure problems are mixed intervention problems;
8. strict repair prevalence among single-node rows belonging to direct-failure
   problems is between 5% and 50%;
9. `keep_or_single_oracle - best_fixed_position` is at least +10 points
   STR-HIT;
10. `keep_or_single_oracle - keep_or_all_oracle` is at least +2 points STR-HIT
    in each of two fixed ID-hash halves;
11. `uniform_single - single_decoy` is at least +2 points STR-EM;
12. the mean injected-facet coverage change exceeds the mean non-injected-facet
    coverage change by at least 3 points.

All thresholds are fixed before P3x generation.  Report absolute metrics even
if a gate fails.  Do not retune the subset, candidate order, prompt, decoding,
or thresholds after observing outputs.

## 6. Interpretation and stop rule

- `SINGLE_NODE_ACTIONABILITY_PASS`: all 12 gates pass.  This licenses a new,
  separately frozen hidden-target screen using intervention-measured marginal
  node utility.  It does not license a novelty or steering claim by itself.
- `STATIC_REPLICATES_SELECTION_FAIL`: gates 1--4 pass but any actionability
  gate 5--12 fails.  Facet structure remains useful, but per-node selection is
  not supported; stop the candidate-node latent-selector route.
- `FRESH_REPLICATION_FAIL`: any gate 1--4 fails.  Do not build a new hidden
  target on this apparatus.

If P3x passes, a later method must predict node-specific intervention utility
from information available before intervention and must compare against
prompt-only, generic hidden-quality routing, fixed/random/wrong-node steering,
all-node scaffolding, and no-Guide controls.  The final method may not claim
novelty from disambiguation trees, outline prompting, latent verification, a
steering-vector library, or dynamic steering alone.  Gold facet text remains
an Oracle label/apparatus; a deployable method must construct nodes without
gold test annotations.
