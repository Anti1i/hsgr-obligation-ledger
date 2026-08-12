# Executable-DAG benchmark audit P0 result (job 728924)

## Outcome

Only MathQA passed the frozen structural gate.  MultiHiertt and HiTab failed.
This result licenses an executor audit for MathQA only; it does not license a
model or intervention experiment.

The audit used commit `1a1a7d4` on CPU node `xcnf27` and completed in 2m16s.
All eight unit tests and the parser self-test passed.  The raw report is at:

`/mnt/scratch/z/zitong/dch-hsgr/logs/executable_dag_audit_report_728924.json`

## Fixed sources

- MultiHiertt code commit: `45bd9ccdf3142ea059bd5e69c0afb83437fa539c`
- MultiHiertt official linked dataset folder was downloaded directly.
- MathQA official zip SHA-256:
  `7344f30456a7aef3176d4866cc953b35b41bec44eda6b00cdbcfde2876b2f07a`
- HiTab commit: `d179602662b490249baf068a76fbe4137029126e`
- HiTab used only `train/dev/test_samples.jsonl` gold `answer_formulas`, not
  weakly supervised searched programs.

## Results

| Dataset / held-out | Programs | Parse | Deep | Join | Reuse | Deep+join | Deep+join+reuse | Diamond | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MultiHiertt / dev | 842 | 100% | 187 | 18 | 0 | 16 | 0 | 0 | FAIL |
| MathQA / test | 2,985 | 100% | 2,083 | 1,576 | 535 | 1,386 | 535 | 108 | **PASS** |
| HiTab / dev | 1,671 | 12.51% | 14 | 4 | 0 | 0 | 0 | 0 | FAIL |

Additional MathQA test diagnostics:

- reconvergence: 397;
- `P(join | deep) = 0.6654`;
- `P(reuse | deep+join) = 0.3860`;
- median connected internal operations in `deep+join+reuse`: 6;
- invalid/forward reference rate: 0%.

MathQA passed every pre-registered structural check.  Its train split contains
5,321 `deep+join+reuse` examples, so this is not a held-out-only accident.

## Objective interpretation

- MultiHiertt has sufficient depth but remains almost entirely chain-like at
  the operation-dependency level.  Its hierarchical tables do not create the
  required computational hierarchy.
- MathQA genuinely contains native non-chain operation DAGs.  The reuse signal
  is not produced by treating problem numbers or constants as graph nodes.
- HiTab fails this protocol, but its low parse coverage limits the conclusion:
  the result is “not a usable native executable-DAG benchmark under the frozen
  gold-formula parser,” not proof that every unsupported formula is shallow.

Per protocol, only MathQA advanced to P0-B executor verification.  No GPU or
model inference was run in this stage.
