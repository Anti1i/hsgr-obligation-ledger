# Long-form grounded-QA structure audit P0 (frozen protocol)

Date frozen: 2026-08-14 (Asia/Shanghai)

## Decision claim

Long context by itself does **not** solve candidate-domain collapse. A benchmark is useful for the next HSGR experiment only when a single example contains several human-distinguishable, partially useful evidence or answer units whose composition changes the quality of the final answer.

This P0 is a zero-model, zero-GPU apparatus audit. It selects a benchmark; it does not claim an HSGR gain.

## Candidate benchmarks and intended role

1. **CLAPnQ GOLD (primary empirical audit).** Use only the official answerable train and development records. Each question is paired with one fixed gold passage, a long answer, and human-selected supporting sentences. Retrieval is therefore held fixed.
2. **ASQA fixed support (secondary).** Treat disambiguated QA pairs as answer-facet nodes and annotation knowledge passages as fixed support. Its schema and published statistics are assessed here; empirical local audit is reported only if the official data file is available.
3. **FACTS Grounding QA-like subset (stress test only).** Context is fixed and long, but the public benchmark has no human node-level gold answer facets or supporting spans. It cannot pass the present apparatus gate without adding new annotation.

"Long-content answer" is treated as a task family, not as a named dataset, unless a concrete dataset is later specified.

## Unit definitions

For CLAPnQ:

- one record = one question with one fixed passage;
- one atomic evidence unit = one unique human-selected support sentence;
- a multi-support record has at least two atomic evidence units;
- a non-consecutive record uses support sentences marked non-consecutive by the released annotation;
- structural headroom means either at least three evidence units or a non-consecutive multi-support composition;
- the observable candidate state capacity is `2^k` evidence-subset states for `k` atomic units. These are partial-support states, not fabricated numerical answers and not search paths.

Exact duplicates are removed before counting units. Missing selected sentences, selected sentences absent from the fixed passage, or malformed records are integrity failures.

## Frozen no-model gates

CLAPnQ GOLD passes the structural apparatus only if all conditions hold on the combined official answerable train+dev set:

1. at least 500 valid examples;
2. 100% have a fixed passage, a gold long answer, and an explicit selected-sentence list;
3. at least 50% have two or more unique support sentences;
4. at least 20% have three or more unique support sentences;
5. at least 25% are both multi-support and non-consecutive;
6. at least 30% satisfy the structural-headroom definition;
7. median long-answer length is at least 30 whitespace-delimited words;
8. fewer than 1% have selected sentences missing verbatim from the released passage sentence list.

These thresholds are fixed before computing the dataset-wide distributions.

## Collapse and diversity diagnostics

The report must include:

- number of valid/malformed records by split;
- support-sentence count distribution and percentiles;
- fixed-passage sentence and word counts;
- long-answer sentence and word counts;
- exact duplicate support rate;
- multi-support, three-plus-support, non-consecutive multi-support, and structural-headroom rates;
- evidence-subset state capacity distribution (`2^k`, reported with a safe cap for display only);
- pairwise token-Jaccard overlap among selected sentences as a lexical redundancy diagnostic;
- the fraction of records that pass every per-example structural filter for a later model pilot.

The per-example pilot filter is frozen as: answer has at least 30 words, at least two unique selected sentences, all selected sentences occur in the fixed passage, and either the support is non-consecutive or there are at least three selected sentences.

## Interpretation rules

- **PASS:** all eight apparatus gates pass. Proceed to a small baseline/headroom experiment on the frozen structurally eligible subset.
- **BORDERLINE:** integrity gates pass but one or more diversity gates fail. CLAPnQ may be useful as a control, not as the main HSGR benchmark.
- **FAIL:** fixed support or node-level gold is absent, integrity failures exceed the gate, or multi-unit composition is too rare.

A PASS shows that the benchmark supplies non-collapsed structural supervision. It does not prove that hidden states encode the structure, that the Guide can exploit it, or that the baseline has enough errors. Those require separate held-out causal and end-task gates.

## Next model gate if P0 passes

Freeze a 300-example pilot split before generation. Compare a direct fixed-passage baseline against HSGR under the same model, prompt budget, decoding, and evidence. Score atomic evidence coverage, unsupported-claim rate, answer completeness, and final answer quality separately. Continue only if the direct baseline leaves meaningful headroom (target 40-75% strict all-facet coverage) and HSGR changes decisions on at least 10% of examples.

