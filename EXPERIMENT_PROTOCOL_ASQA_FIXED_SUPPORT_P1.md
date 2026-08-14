# ASQA fixed-support direct/scaffold headroom P1 (frozen protocol)

Date frozen: 2026-08-14 (Asia/Shanghai), before any P1 model output was generated.

## Question

On a valid fixed-support ASQA subset, does Qwen2.5-7B leave meaningful strict all-facet headroom, and does the gold facet structure change answer coverage beyond an equally sized irrelevant checklist?

P1 is an oracle textual-structure ceiling. It is not the proposed HSGR method and cannot support a hidden-state claim.

## Data and model

- Run only if ASQA P0 passes.
- Use the first 192 P0-eligible examples after frozen SHA-256 ordering with seed `20260815`.
- Fixed evidence is always the same first five ALCE oracle-reranked docs.
- Model: `Qwen/Qwen2.5-7B-Instruct`, bfloat16, greedy decoding, maximum 192 new tokens.
- No sampling, retrieval, citation requirement, post-hoc regeneration, or judge model.
- Score with official ALCE-normalized STR-EM and STR-HIT.

## Four paired arms

1. `closedbook`: ambiguous question only.
2. `fixed_direct`: ambiguous question plus the five fixed docs.
3. `true_facets`: same fixed docs plus the released disambiguated facet questions as a coverage checklist; no short answers are shown.
4. `decoy_facets`: same fixed docs plus facet questions from another example, matched on facet count and closest total whitespace-word count; no answers are shown.

The true and decoy arms use identical formatting and facet counts. The decoy mapping is frozen before generation and excludes the same released ID.

## Frozen gates

### Baseline difficulty and evidence dependence

1. `fixed_direct` STR-HIT is between 40% and 75%, inclusive;
2. `fixed_direct - closedbook` is at least +10 percentage points in STR-HIT **or** +10 points in STR-EM;
3. median generated length is between 30 and 160 whitespace words in `fixed_direct`.

### Actionable facet structure

4. `true_facets - decoy_facets` is at least +8 points in STR-HIT;
5. `true_facets - decoy_facets` is at least +5 points in STR-EM;
6. paired exact McNemar p-value for strict all-facet success, true versus decoy, is below 0.05;
7. `true_facets - fixed_direct` is at least +5 points in STR-HIT;
8. `decoy_facets - fixed_direct` is no more than +3 points in STR-HIT.

Report paired changes, fixed ID-hash halves, answer lengths, facet-count strata, and all four absolute metrics. Do not change thresholds after observing outputs.

## Outcomes

- `HEADROOM_AND_STRUCTURE_PASS`: all eight gates pass. Freeze a new disjoint split for an answer-prefix hidden-state uncovered-facet Guide.
- `HEADROOM_PASS_STRUCTURE_FAIL`: the benchmark is difficult and evidence-dependent, but explicit facets do not provide actionable benefit beyond the decoy checklist.
- `STRUCTURE_PASS_DIFFICULTY_FAIL`: facets help, but this subset/model is too easy or too hard for the main experiment.
- `FAIL`: neither group passes or the apparatus is invalid.

## Boundary from current methods

P1 itself is an oracle prompt ceiling. Any later HSGR experiment must keep retrieval fixed and must not select passages, rank evidence, enumerate complete answer paths, run tree search, or use iterative prompt-action routing. The proposed state is the low-dimensional covered/uncovered facet set inferred from the current answer-prefix hidden state.

