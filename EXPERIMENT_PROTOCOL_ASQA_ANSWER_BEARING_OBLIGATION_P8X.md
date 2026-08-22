# ASQA answer-bearing obligation ledger P8x (frozen protocol)

Date frozen: 2026-08-22 (Asia/Shanghai), after P7r job 749095 and before any
P8x model output was generated.

## Hypothesis

P7r showed that four freely induced subquestions have only a 26.03% action
Oracle, versus 45.21% with gold facet questions. Case inspection showed that
many nodes ask for background or document-specific details and do not carry the
answer content needed to repair the missing facet.

P8x changes the node type, not the selector. It induces a flat set of
evidence-anchored answer obligations. Each node is a pair:

- `scope`: the meaning, entity, version, event, or interpretation being covered;
- `claim`: one self-contained factual sentence that answers the ambiguous
  question for that scope using the fixed documents.

The saved answer, gold facets, aliases, and coverage labels are excluded from
node induction. The unchanged P6x `logit(B)-logit(A)` coverage readout selects
the most missing node. Its claim is appended directly, without a second repair
generation call.

## Frozen apparatus

- Exact 73 P1x exact-one-missing cases and frozen saved answers.
- Model: `Qwen/Qwen2.5-7B-Instruct`, bfloat16, SDPA, greedy.
- Induction sees only the question and five fixed documents. Request two to
  five distinct JSON objects, prioritizing different answer interpretations
  rather than several details of one interpretation. Maximum 512 new tokens.
- A valid set has 2--5 unique nonempty `(scope, claim)` pairs; every claim is at
  most 45 words. The parser predeclares the P7r code-fence, invalid-apostrophe,
  and singleton-array-line serialization repairs. Invalid outputs receive one
  deterministic fallback node whose claim is the original question and remain
  in the denominator.
- Coverage scoring sees the question, saved answer, and rendered scope/claim,
  but no documents. Maximum score wins; lower index breaks ties.
- `claim_oracle_append`: post-hoc best direct claim by strict hit then STR-EM;
  `claim_logit_append`: maximum coverage score; `claim_random_append`: frozen
  hash index.
- Replay P7r question-node Oracle/logit and P6x gold Oracle/logit/generic rows.

## Gates and outcomes

Apparatus gates require exact counts/rescoring/replay, at least 90% valid node
sets, median node count in 2--5, finite scores, full denominator retention, and
100% prior-facet preservation for the automatic direct append.

Answer-bearing ledger gates require:

1. claim-set Oracle at least 30% and at least 65% of gold-facet Oracle;
2. automatic claim append at least 20% and at least half gold-facet logit;
3. automatic claim append beats generic by at least 10 points with paired
   two-sided exact McNemar `p<0.05`;
4. automatic claim append beats claim-random by at least 10 points with paired
   `p<0.05`;
5. automatic claim append retains at least 60% of claim-set Oracle;
6. claim-set Oracle exceeds the P7r subquestion-set Oracle by at least five
   points (reported with a paired test, but significance is not a hard gate).

Outcomes:

- `APPARATUS_FAIL`;
- `ANSWER_BEARING_OBLIGATION_PASS` if every ledger gate passes;
- `ANSWER_BEARING_ACTION_ONLY` if the claim Oracle gate passes but an automatic
  gate fails;
- `ANSWER_BEARING_OBLIGATION_FAIL` otherwise.

P8x is not hierarchical and does not establish factual grounding, multi-step
updates, end-to-end training, or novelty. No prompt, parser, node count, arm,
metric, threshold, or outcome rule may change after P8x output is observed.
