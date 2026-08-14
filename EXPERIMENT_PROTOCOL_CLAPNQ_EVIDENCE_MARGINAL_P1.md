# CLAPnQ fixed-support evidence-marginal P1 (frozen protocol)

Date frozen: 2026-08-14 (Asia/Shanghai), before any model score was computed.

## Question

Does CLAPnQ GOLD provide a model-relevant, non-collapsed domain of evidence nodes, and can a contextual hidden-state Guide read out the marginal usefulness of each node on held-out questions?

P1 is a headroom and representation screen. It cannot establish that a Guide intervention improves generated answers.

## Data

- Official `clapnq_dev_answerable.jsonl` only.
- Fixed gold passage; retrieval is not run.
- Deterministically sort records by SHA-256 of the released ID and seed `20260814`.
- Keep records with 2--6 unique selected support sentences, a 30--120-word gold answer, every selected sentence present in the passage sentence list, and either a non-consecutive annotation or at least three support sentences.
- Take the first 192 eligible records: 96 calibration, 96 held-out. If fewer than 192 are eligible, stop without model interpretation.

No result-dependent record filtering is allowed.

## Model and prompt

- Model: `Qwen/Qwen2.5-7B-Instruct`, bfloat16, greedy scoring only.
- Use the released passage sentences in their original order, each with a stable sentence index.
- Ask for a concise cohesive answer using only the fixed passage.
- Score the released long answer by mean teacher-forced token negative log likelihood (NLL); include no answer tokens in the prompt-side hidden representation.
- Reject, rather than truncate, any sequence exceeding the model context limit.

## Controlled arms

For every record compute:

1. `full`: all passage sentences;
2. `drop_all`: delete all selected support sentences but retain passage distractors;
3. `support_only`: keep only the selected support sentences in original order;
4. one `drop_i` arm for every selected support sentence `i`.

Define node marginal utility as `u_i = NLL(drop_i) - NLL(full)`. Positive values mean that removing the node makes the gold answer less likely. Define total evidence sensitivity as `NLL(drop_all) - NLL(full)`.

The arms are evidence subsets, not generated answer paths. No tree search, beam search, or iterative prompt action is used.

## Frozen structural/model gates

The evidence domain is `MODEL_RELEVANT` only if all conditions hold on held-out records/nodes:

1. all scores are finite and every answer has at least one scored token;
2. median total evidence sensitivity is at least `+0.020` nats/token;
3. at least 60% of records have positive total evidence sensitivity;
4. median `NLL(drop_all) - NLL(support_only)` is at least `+0.020` nats/token;
5. at least 35% of individual nodes have `u_i >= +0.005` nats/token;
6. at least 25% of records have two or more such active nodes;
7. at least 30% of records have a within-record marginal range of at least `0.010` nats/token.

Conditions 5--7 directly test whether useful node effects collapse to one indistinguishable value.

## Hidden-state Guide readout

From the `full` prompt only, mean-pool the contextual hidden states over each selected support-sentence span at transformer blocks 13, 20, and 27 (zero-indexed). Randomly project each vector to 64 dimensions using fixed Rademacher matrices and seed `20260814`.

Surface controls contain only quantities available without the gold answer: sentence word count, sentence position, passage sentence count, number of support nodes, question-to-sentence token Jaccard, and mean token Jaccard to the other selected sentences.

Fit ridge regressors on calibration questions only. Choose ridge strength from `{0.1, 1, 10, 100, 1000}` by five-fold question-grouped calibration RMSE. Select the hidden layer by the same calibration-only RMSE. Refit once on all calibration questions and evaluate once on held-out questions.

The hidden Guide is `READABLE` only if, on held-out nodes:

1. Spearman correlation with `u_i` is at least `0.20`;
2. Spearman exceeds the surface-control model by at least `0.08`;
3. RMSE is at least 5% lower than the surface-control model;
4. Spearman is positive in both fixed ID-hash halves.

## Outcomes

- `MODEL_RELEVANT_AND_READABLE`: all structural/model gates and all readout gates pass. Proceed to a separately frozen causal Guide intervention/generation experiment.
- `MODEL_RELEVANT_NOT_READABLE`: evidence nodes matter, but this hidden-state Guide does not generalize. Redesign the representation target; do not claim hidden guidance.
- `MODEL_IRRELEVANT`: the annotations are structurally rich but the model largely ignores their differences under this apparatus. Do not proceed to steering on this model/prompt.
- `APPARATUS_FAIL`: data, tokenization, context, or numerical integrity fails.

All probe results remain correlational. No hidden-state causal claim is permitted in P1.

