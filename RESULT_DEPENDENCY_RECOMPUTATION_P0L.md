# P0l causal dependency-recomputation Guide screen: result

## Decision

P0l fails its frozen causal Guide gate. A verbal relation-aware Guide does not
reliably cause either tested model to recompute the current dependency. This
Guide formulation must stop; the result does not justify hidden-state
prediction, training, or RL.

## Claim boundary

P0l evaluates a realistic reuse context because every arm is shown a cached
old `SAT` verdict. It shows severe failure **in the presence of** that verdict.
It does not isolate whether the cached verdict itself causes the failure,
because there is no otherwise-identical no-cache arm. `Verdict persistence`
is therefore a useful behavioral description here, not yet a causal account
of the internal mechanism.

## Run record

- code commit: `a0864af`
- Slurm job: `753484` (`COMPLETED`, exit `0:0`, 2m27s)
- compute host: `xgpi14`
- allocated device: `CUDA_VISIBLE_DEVICES=0`, H100 NVL MIG 3g.47gb
- models: Qwen3-8B non-thinking and Qwen2.5-14B-Instruct
- matrix: 40 cases x 2 revision states x 5 arms x 2 models = 800 generations
- result directory:
  `/mnt/scratch/z/zitong/hsgr-obligation-ledger/results/dependency_recomputation_p0l_753484`

All 17 server tests passed. Output validity was 100% for every model-arm cell.

## Five-arm result

### Qwen3-8B

| Arm | Stale recall | Harmless specificity |
|---|---:|---:|
| Flat | 0% (0/40) | 100% |
| Source-only | 2.5% (1/40) | 100% |
| Relation-only | 0% (0/40) | 100% |
| Source + Relation | 0% (0/40) | 100% |
| Shuffled Guide | 0% (0/40) | 100% |

The primary Guide is exactly identical to Flat on all 40 stale cases. Its
paired difference is zero, with McNemar `p=1.0` and bootstrap 95% interval
`[0, 0]`.

### Qwen2.5-14B-Instruct

| Arm | Stale recall | Harmless specificity |
|---|---:|---:|
| Flat | 45% (18/40) | 97.5% |
| Source-only | 47.5% (19/40) | 97.5% |
| Relation-only | 50% (20/40) | 97.5% |
| Source + Relation | 50% (20/40) | 97.5% |
| Shuffled Guide | 45% (18/40) | 100% |

The primary Guide adds two correct stale cases over both Flat and Shuffled:
`+5pp`, two paired wins and zero losses, exact McNemar `p=0.5`. Its paired
bootstrap 95% interval is `[0, 12.5pp]`, so the effect is not distinguishable
from zero. Relation contributes only `2.5pp` beyond Source-only, below the
frozen `10pp` requirement.

## Mechanism audit

Qwen3's primary Guide catches `0/8` in all five mechanisms. Qwen2.5 gives:

| Mechanism | Flat | Source-only | Relation-only | Source + Relation | Shuffled |
|---|---:|---:|---:|---:|---:|
| comparison | 4/8 | 4/8 | 4/8 | 4/8 | 4/8 |
| attribution/source binding | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| derived arithmetic | 3/8 | 3/8 | 3/8 | 3/8 | 3/8 |
| temporal ordering | 3/8 | 4/8 | 5/8 | 5/8 | 3/8 |
| definition/threshold | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |

Only temporal ordering responds at all. Attribution, arithmetic, comparison,
and definition are unchanged by the Guide; definition was already saturated.
The primary Guide reaches at least 6/8 in only one mechanism, not the required
four.

## Manual audit

All 60 primary stale misses were reviewed: 40 from Qwen3 and 20 from Qwen2.5.
The correct source IDs and mechanism-specific relation are present in each
primary prompt. The output is nevertheless `met=true`.

No invalid generations occurred. There is one additional informative safe
case, `hospital-derived`: Qwen2.5 rejects the harmless document under Flat,
Source-only, Relation-only, and Source+Relation, but accepts it under the wrong
Shuffled Guide. The correct arithmetic is unambiguous. This is inconsistent
with stable execution of the supplied operation and is not evidence for the
wrong Guide.

## Frozen gates

| Gate | Qwen3-8B | Qwen2.5-14B | Overall |
|---|---|---|---|
| G1 output validity | PASS | PASS | PASS |
| G2 causal rescue | FAIL | FAIL | FAIL |
| G3 harmless specificity | PASS | PASS | PASS |
| G4 relation beyond source | FAIL | FAIL | FAIL |
| G5 mechanism breadth | FAIL | FAIL | FAIL |
| all P0l gates | FAIL | FAIL | **FAIL** |

## Interpretation

What the result supports:

1. Merely telling a verifier where to look and naming the relation is not
   enough to make it execute that relation.
2. Qwen3-8B has an extreme SAT/conclusion-acceptance behavior in this setup:
   the correct full Guide changes none of its 40 stale decisions.
3. Qwen2.5-14B is stronger, but the Guide effect is only two temporal cases and
   is neither statistically significant nor mechanism-general.

What it does not support:

1. relation-aware verbal Guide causally restores recomputation;
2. source localization is the missing component;
3. hidden state contains a learnable Guide or stale-verdict signal;
4. adding a learned router, verifier, or RL objective will rescue this design.

## Only justified follow-up

If this problem is pursued once more, the intervention must change from
**describing** a computation to **requiring an externally checkable execution**:
the model must emit typed operands, the selected source IDs, the operator, and
the computed intermediate result before a verdict. A deterministic checker
then verifies the trace.

That would answer whether the current Guide fails because it is ignored or
because the model cannot perform the operation reliably. It is a different
method family, closer to structured verification/program execution, and its
novelty must be reassessed against those current methods. It should be a small
bounded diagnostic, not an automatic continuation into a larger HSGR system.

## Result hashes

- report: `a7802068f42cd7e2112652ab08e821e31bbc0fea4571394a9f3f7877f35e5cc5`
- rows: `af45ecac69fafd4068ff591584d888195e375238dd1ead74414a7081b9353cae`
- review: `91e78e50279a4bd04b5acf7fa944a9acd7334f3ef35337a6f0b5a74053413ee7`
