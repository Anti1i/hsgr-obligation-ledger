# Join viability results (2026-08-12)

## Decision

Stop the synthetic GSM easy-join benchmark.  Do not run confirmation, larger
backbones, post-hoc filters, prompt repair, SC@8, hidden-state training, or a
Guide on this benchmark.

The hierarchy signal is real but does not provide a sufficiently reliable
compositional state or enough routing headroom.  This falsifies the benchmark
as an HSGR development environment; it does not by itself falsify hierarchical
reasoning in every task.

## Frozen-screen sequence

### V1: metadata-only reuse of existing calibration calls

The pre-registered V1 screen reused the incomplete structural-hardness caches;
it made no new model calls and never touched confirmation.

| Rule | n | Direct SC@1 | Direct SC@8 | Greedy parent |
|---|---:|---:|---:|---:|
| all | 400 | 12.2% | 13.8% | 57.5% |
| total annotated steps <= 8 | 110 | 21.8% | 26.4% | 70.5% |
| max parent <= 3, root <= 2 | 36 | 36.1% | 44.4% | 75.0% |

No frozen rule simultaneously achieved `n >= 100`, direct accuracy in
`[0.30, 0.70]`, and greedy parent accuracy at least `0.70`.  V1 failed.

### V2: easy-join construction, initial decoding limits

V2 froze a new train/test construction before model outcomes:

- each parent has at most 3 annotated arithmetic steps;
- the root has at most 2 annotated arithmetic steps;
- all pre-existing independent-causality and substitution constraints remain;
- calibration is a fixed 96-row train subset;
- confirmation is a fixed 128-row test subset.

Job `728784` ran `Qwen/Qwen2.5-7B-Instruct`.  It produced direct `25.0%`, mean
parent `73.4%`, and gold-bound root `56.2%`, but the decoding limits truncated
many responses.  Parse validity was only `83.3%`, `77.1%`, `69.8%`, and `86.5%`
for direct, parent 0, parent 1, and root.  This run was treated as a measurement
failure rather than benchmark evidence.

### V2.1: length-repaired 7B screen

V2.1 changed only the generation limits: direct `1536`; each parent and root
`512`.  Data, IDs, model, prompts, parser, and all accuracy thresholds remained
unchanged.  A new token-cap-rate gate of at most `5%` was added.

Job `728816`, `xgpi13`, one H100-47/NVL, commit `2e59ff2`:

| Metric | Result | Frozen calibration gate |
|---|---:|---:|
| Direct | 53.1% | 30--70% (pass) |
| Mean parent | 96.9% | >=70% (pass) |
| Gold-bound root | 62.5% | >=70% (**fail**) |
| Root minus direct | +9.4pp | >=10pp (**fail**) |
| Parse validity | 100% all arms | >=95% (pass) |
| Token-cap rate | 0% all arms | <=5% (pass) |

This established a reliable 7B failure and left confirmation untouched.

### V3: same-family 14B backbone screen

V3 changed only the backbone.  It used the project-local, offline snapshot:

`Qwen/Qwen2.5-14B-Instruct@cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`

The 28 GB snapshot was cached by CPU job `728850`.  Job `728852` ran on
`xgpi13`, one H100-47/NVL, commit `438ae15`:

| Metric | Result | Frozen calibration gate |
|---|---:|---:|
| Direct | 57.3% | 30--70% (pass) |
| Mean parent | 97.9% | >=70% (pass) |
| Gold-bound root | 64.6% | >=70% (**fail**) |
| Root minus direct | +7.3pp | >=10pp (**fail**) |
| Parse validity | 100% all arms | >=95% (pass) |
| Token-cap rate | 0% all arms | <=5% (pass) |

V3 failed and therefore did not touch confirmation.

## Paired diagnosis

The direct and gold-root calls were compared per calibration example.

| Model | Both correct | Direct only | Root only | Neither | Direct/root union |
|---|---:|---:|---:|---:|---:|
| 7B | 48 | 3 | 12 | 33 | 65.6% |
| 14B | 54 | 1 | 8 | 33 | 65.6% |

Gold parent information has a real paired effect (two-sided exact sign tests:
`p=0.0352` at 7B and `p=0.0391` at 14B).  However, a perfect chooser between
the direct and gold-root calls reaches only `65.6%`; at 14B it adds just 1.0pp
over always using the gold-root call.

Root correctness was also scale-stable: 60 examples were correct at both
scales, 0 were 7B-only, 2 were 14B-only, and 34 were wrong at both scales.
Among wrong root outputs, only one 7B and one 14B output matched the original
unsubstituted root; only one 7B output matched a single-edge counterfactual.
The persistent failures are therefore not mainly simple missing-edge errors.

## What the evidence says

Verified facts:

1. Benchmark difficulty is no longer the problem: clean direct accuracy is in
   the intended middle regime at both 7B and 14B.
2. Parent solvability is not the bottleneck: it is 97--98%.
3. Output parsing and generation length are not the bottleneck after V2.1.
4. Gold hierarchical inputs help on paired examples, but root composition
   remains below the frozen reliability floor and barely scales from 7B to 14B.
5. The remaining direct-vs-root routing oracle is only 1.0pp at 14B.

Inference:

The synthetic join produces a low-ceiling compositional interface.  Training a
hidden-state Guide here would mostly learn around systematic benchmark/root
failures, not select among interventions with substantial causal headroom.

## Guarded next technical P0

A new domain is not licensed by the join protocols.  If separately approved,
the least weak candidate is a diagnostic-only FinQA program-intervention oracle:

- use FinQA's native expert-annotated numerical reasoning programs and exact
  executor, rather than stitching unrelated questions;
- generate one predicted program per example, not multiple paths;
- operate on a fixed program with finite atomic edits such as `replace_op`,
  `rebind_arg`, and `replace_evidence`, always including `NOOP`;
- execute every edit deterministically and measure downstream answer utility;
- compare no edit, every best fixed edit, one-edit oracle, node-only oracle,
  node-by-edit oracle, and a two-edit diagnostic oracle;
- do not train or inspect hidden states until the oracle gates pass.

Pre-registered minimum gates should be:

1. gold-program executor accuracy at least `95%`;
2. predicted-program execution accuracy in `[30%, 70%]`;
3. one-edit oracle minus no edit at least `10pp`;
4. node-by-edit oracle minus best fixed edit at least `5pp`;
5. two-edit oracle no more than `3pp` above one-edit oracle;
6. at least `60%` of correctable failures covered by the frozen atomic edit
   vocabulary on a disjoint audit subset.

Failure stops this HSGR technical route.  Passing licenses a held-out oracle
confirmation, not hidden/energy training.

## Novelty boundary

FinQA already defines mapping financial text/tables to executable gold programs
([FinQA](https://arxiv.org/abs/2109.00122)).  ProofWriter and LogicGuide already
cover generated or constrained logical deductions
([ProofWriter](https://arxiv.org/abs/2012.13048),
[LogicGuide](https://arxiv.org/abs/2306.04031)); CLUTRR already diagnoses graph
relation composition ([CLUTRR](https://arxiv.org/abs/1908.06177)).  Therefore
program generation, deterministic execution, graph reasoning, verification, and
fault localization cannot be claimed as the contribution.

The only defensible HSGR claim to test would be:

> Given one fixed hierarchical computation state, predict which minimal local
> intervention has positive downstream causal utility before paying to execute
> it, using hierarchy-conditioned hidden state.

