# ASQA automatic obligation-set induction P7x (frozen protocol)

Date frozen: 2026-08-22 (Asia/Shanghai), after P6r job 749055 and before any
P7x model output was generated.

## Claim under test

P6x showed that, given the gold facet-question set, the frozen model's explicit
missingness readout selects the missing facet on 80.82% of exact-one-missing
cases and recovers 39.73% end to end. P7x removes the gold facet set.

For each of the same 73 P1x repair cases, induce four answer obligations from
only the ambiguous question and five fixed support documents. The induction
prompt must not include the saved answer, gold facet questions, gold aliases,
or gold coverage labels. Then:

1. score each induced obligation against the saved answer with the unchanged
   P6x A/B coverage prompt and `logit(B)-logit(A)` readout;
2. generate one bounded append for every induced obligation;
3. select the maximum-score obligation automatically;
4. compute a post-hoc action Oracle over the four generated appends to diagnose
   whether the induced set contained a useful repair target.

This is a set-valued obligation ledger, not path search: no partial trajectory,
branch expansion, backtracking, or tree search is used. It still does not test
hierarchical edges or multi-step ledger updates.

## Frozen apparatus

- Model: `Qwen/Qwen2.5-7B-Instruct`, bfloat16, SDPA, greedy decoding.
- Evaluation: the exact 73 P1x cases used by P6x with exactly one missing and
  at least one covered gold facet. Gold aliases are final metrics only.
- Induction: exactly four concise standalone subquestions, JSON-array request,
  maximum 256 new tokens. A predeclared numbered-list parser is the only format
  fallback. An invalid output receives one deterministic fallback obligation
  equal to the original ambiguous question and remains in the denominator.
- Coverage scoring: exact P6x A/B prompt, no documents, one forward pass.
- Repair: one 96-token maximum append per induced candidate using the same five
  fixed documents and append instruction as P6x.
- Automatic tie-break: lower candidate index. Random control: one frozen hash
  index. Action Oracle: first strict-success candidate; if none succeeds, the
  candidate with greatest STR-EM, then lower index.
- Frozen P6x `oracle_append`, `logit_append`, and `generic_append` rows are
  replayed rather than regenerated.

## Gates and outcomes

Apparatus gates require exact source/P6x counts, exact saved-answer rescoring,
finite scores, at least 90% valid four-node induction, median four unique nodes,
100% denominator retention, and exact replay of P6x Oracle/logit/generic rates.

Induction-ledger gates require:

1. induced-set action Oracle strict success at least 30% and at least 65% of
   the gold-facet Oracle rate;
2. automatic induced-logit success at least 20%, at least half the gold-facet
   logit rate, and at least 98% prior-facet preservation;
3. automatic induced-logit beats the frozen generic append by at least 10
   points with paired two-sided exact McNemar `p<0.05`;
4. automatic induced-logit beats induced-random by at least 10 points with
   paired `p<0.05`;
5. automatic induced-logit retains at least 60% of the induced action Oracle.

Outcomes:

- `APPARATUS_FAIL`: an apparatus/replay gate fails;
- `AUTO_OBLIGATION_LEDGER_PASS`: every induction-ledger gate passes;
- `INDUCED_ACTION_ONLY`: the induced-set Oracle gate passes but automatic
  selection/action gates do not;
- `AUTO_OBLIGATION_FAIL`: even the induced-set action Oracle gate fails.

No prompt, parser, fallback, case set, number of obligations, selector, arm,
metric, threshold, or outcome rule may change after P7x output is observed.
