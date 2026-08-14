# ASQA fixed/oracle-support structure audit P0 (frozen protocol)

Date frozen: 2026-08-14 (Asia/Shanghai), before dataset-wide ASQA/ALCE statistics were computed.

## Claim under test

ASQA is a viable HSGR long-answer benchmark only if its explicit disambiguated QA pairs form a non-collapsed facet domain **and** a reproducible fixed passage set contains the information needed to answer those facets.

Long answers or ambiguous questions alone are insufficient.

## Data and fixed-support definition

- Labels: official ASQA dev split (948 examples) from `ASQA.json`.
- Passages: official ALCE `asqa_eval_gtr_top100_reranked_oracle.json`.
- Fixed context: the first five ALCE oracle-reranked `docs` for each example, matching ALCE's `ndoc: 5` oracle configuration.
- Join by released question/ID fields, with QA-pair consistency checks.
- Do **not** use `annotations[*].knowledge` as fixed support: it may be empty or contain `null` content and is not a complete reproducible passage set.
- Do not retrieve, rerank, regenerate, or repair passages after seeing audit outcomes.

## Facet and scoring definitions

- One structural facet is one released `qa_pairs` item.
- Its accepted values are that item's `short_answers` aliases.
- Normalize exactly as ALCE: lowercase, remove ASCII punctuation and articles `a/an/the`, then normalize whitespace.
- A facet is present when any normalized alias is a substring of normalized text.
- `STR-EM` is mean per-example facet coverage.
- `STR-HIT` is the fraction of examples with every facet present.
- Duplicate facets are QA pairs with identical normalized short-answer alias sets within one example.

These are human-provided answer facets, not sampled model values or complete reasoning paths.

## Frozen apparatus gates

P0 passes only if all conditions hold after aligning original ASQA dev with ALCE oracle data:

1. at least 900 aligned examples;
2. at least 99% of original dev examples align uniquely;
3. at least 90% have two or more non-duplicate facets;
4. at least 30% have three or more non-duplicate facets;
5. fewer than 2% of examples contain duplicate normalized facet groups;
6. at least 95% have five non-empty fixed docs;
7. median fixed-context length is at least 300 whitespace words;
8. fixed passages achieve at least 80% STR-EM and 50% STR-HIT against the facet aliases;
9. released human long answers achieve at least 80% STR-EM and 50% STR-HIT;
10. fewer than 1% of fixed contexts contain a released human long answer verbatim after whitespace normalization.

Gate 10 guards against simply embedding the target answer in the support.

## Frozen baseline-eligible subset

An aligned example is eligible for P1 only if it has:

- 2--6 non-duplicate facets;
- exactly five non-empty fixed docs;
- every facet alias group present in the fixed context;
- at least one released human long answer with every facet present;
- no released human long answer copied verbatim into the fixed context.

At least 192 examples must be eligible. Selection for P1 is by SHA-256 of `20260815|released_id`; no result-dependent filtering is allowed.

## Interpretation

- `PASS`: all ten gates and the 192-example eligibility requirement pass.
- `BORDERLINE`: labels/joins are valid but one or more diversity or passage-coverage gates fail.
- `FAIL`: alignment, fixed support, leakage, or eligible sample size fails.

A P0 pass shows only that ASQA supplies a valid structural apparatus. It does not establish model difficulty, actionable structure, hidden-state signal, causal Guide utility, or generation gains.

