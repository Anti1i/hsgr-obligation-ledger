# Parent Multi-View Action Space V0: frozen calibration pilot

Frozen on 2026-08-12 after structural-hardness V0 failed, before generating
any multi-view completion.  The earlier four-sample parent cache showed that
96.9% of nodes collapsed to at most one parsed value and candidate gold
coverage exceeded modal accuracy by only 0.7pp.  This pilot tests a new
candidate generator; it is not a Hidden Guide result.

## Data and fixed budget

- Calibration only: all 800 parent nodes from the 400 `gsm_join_train` graphs.
- Confirmation `gsm_join_test` is not run or inspected by this pilot.
- Model remains frozen `Qwen/Qwen2.5-7B-Instruct`.
- Both generators use four calls per node.

Existing stochastic generator:

1. one ordinary greedy solve;
2. three temperature-0.8 samples of the same prompt;
3. 192 maximum generated tokens per call.

Proposed action-space generator:

1. the same cached ordinary greedy solve;
2. deterministic equation-first solve;
3. deterministic independent solve with explicit arithmetic verification;
4. deterministic solve-from-scratch check of the ordinary proposed answer;
5. 384 maximum generated tokens for each new arm.

The check arm sees only the ordinary proposed answer, not its gold label.  All
nodes receive every arm; no outcome-based triggering or per-item filtering is
allowed.  Every arm must end with one `\\boxed{...}` answer and uses the same
normalizer as the old cache.

## Metrics and frozen gate

Report parse validity by arm; normalized unique-value distribution; collapsed
node and graph rates; ordinary/modal accuracy; old and new candidate gold
coverage; recovered nodes/graphs where the old modal is wrong but the new pool
contains gold; and added-wrong-value counts when the old modal is correct.

The generator passes only if all conditions hold:

1. exactly 800 nodes and all three new arms are complete;
2. each new arm has at least 95% parse validity;
3. the new four-call pool improves candidate gold coverage over the old
   four-call pool by at least 10.0 percentage points;
4. at least 15% of graphs have a non-collapsed new parent domain;
5. at least 40 nodes and at least 20 graphs are recovered relative to the old
   modal assignment;
6. improvement over the old pool is non-negative in both fixed graph-ID hash
   halves and in parent slots 0 and 1;
7. prompt, generated-token, and call accounting is complete.

Passing licenses a separately frozen graph-root assignment audit.  It does not
license hidden extraction by itself.  Failure means prompting/verification at
this model scale still does not create a usable node action space; adding
known-wrong values is not an allowed rescue.

## Relation to the proposed method

Multi-prompt generation, self-correction, and candidate diversification are
baselines/infrastructure, not claimed contributions.  Any later HSGR claim
must use the identical frozen candidate pool for frequency, ordinary hidden
verification, flat readers, and the structure-tied Guide.  Candidate-generation
gain must be reported separately from Guide-selection gain.
