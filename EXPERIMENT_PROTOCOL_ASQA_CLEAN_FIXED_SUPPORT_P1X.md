# ASQA clean fixed-support direct/scaffold screen P1x (frozen protocol)

Date frozen: 2026-08-14 (Asia/Shanghai), after P0 job 733199 and its structural
diagnosis, but before any P1x model output was generated.

## Status and question

P0 was `BORDERLINE`, so this is a separately frozen **exploratory** screen.  It
does not retroactively pass P0 and is not the formal P1 described in
`EXPERIMENT_PROTOCOL_ASQA_FIXED_SUPPORT_P1.md`.

Question: on ASQA examples whose answer values are unique and whose fixed
passages contain every released facet, does Qwen2.5-7B retain meaningful strict
all-facet headroom, and does the true facet checklist help beyond an equally
sized irrelevant checklist?

## Frozen subset

Align official ASQA dev with ALCE oracle-reranked data exactly as in P0.  Keep an
example only if:

- every normalized short-answer alias group is non-empty and unique;
- it has 2--6 facet groups;
- it has exactly five non-empty fixed docs;
- all facet groups occur in those docs under official ALCE normalization;
- at least one released human long answer covers every facet;
- no released human long answer occurs verbatim in the context after whitespace
  normalization.

P0 diagnosis found 427 such cases.  Select 192 by ascending SHA-256 of
`20260815-clean-p1x|released_id`.  No model-result-dependent filtering is
allowed.

## Model and arms

- Model: `Qwen/Qwen2.5-7B-Instruct`, bfloat16, greedy decoding.
- Maximum 192 new tokens; no sampling, retrieval, citations, judge model, or
  regeneration.
- Fixed evidence is always the same five ALCE documents.

Four paired arms:

1. `closedbook`: ambiguous question only.
2. `fixed_direct`: ambiguous question plus the five fixed docs.
3. `true_facets`: same docs plus the released disambiguated questions as a
   checklist; short answers are never shown.
4. `decoy_facets`: same docs plus questions from another selected example,
   matched on facet count and nearest total checklist word count; short answers
   are never shown.

The decoy mapping is deterministic, excludes the same ID, and is frozen before
generation.

## Frozen gates

Baseline difficulty and evidence dependence:

1. `fixed_direct` STR-HIT is 40--75%, inclusive;
2. `fixed_direct - closedbook` is at least +10 points in STR-HIT or STR-EM;
3. median `fixed_direct` answer length is 30--160 whitespace words.

Actionable facet structure:

4. `true_facets - decoy_facets` is at least +8 points in STR-HIT;
5. `true_facets - decoy_facets` is at least +5 points in STR-EM;
6. paired exact McNemar p-value for true versus decoy STR-HIT is below 0.05;
7. `true_facets - fixed_direct` is at least +5 points in STR-HIT;
8. `decoy_facets - fixed_direct` is no more than +3 points in STR-HIT.

Report all absolute metrics, paired changes, fixed ID-hash halves, output lengths,
and facet-count strata.  Thresholds may not change after model outputs appear.

## Interpretation and method boundary

- `HEADROOM_AND_STRUCTURE_PASS`: all eight gates pass.
- `HEADROOM_PASS_STRUCTURE_FAIL`: the model has headroom but the facet scaffold
  is not specifically useful.
- `STRUCTURE_PASS_DIFFICULTY_FAIL`: facets help but the benchmark/model operating
  point is unsuitable.
- `FAIL`: neither group passes.

P1x is an oracle textual ceiling, not HSGR.  It cannot support a hidden-state
claim.  Any later method must keep retrieval fixed and must not rank passages,
enumerate complete answer paths, run tree search, or use prompt-action routing.
