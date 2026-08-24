# P0j result: stale verdicts are predictable here, but for a trivial reason

## Decision

Do **not** proceed from this apparatus to a learned invalidator, hidden-state
controller, or RL. The controlled study confirms that a multi-sentence repair
can delete previously correct findings, but every manually confirmed failure
is already caught by a simple text-diff / old-witness rule. The learned model
does not provide the pre-registered improvement over that rule and misses one
confirmed stale verdict.

This is a useful negative result: it preserves the version-bound-verdict
problem, while ruling out the present synthetic apparatus as evidence for a
new learned mechanism.

## Run record

- Code: `f09dc27`; frozen manual audit/finalizer: `62d7774`.
- Slurm job: `753073`, completed successfully in 24:08 on `xgpi17`.
- GPU: Slurm-visible device 0, NVIDIA H100 NVL.
- Generators: Qwen2.5-7B-Instruct and Qwen3-8B, deterministic; Qwen3 thinking
  explicitly disabled.
- Verifier: Qwen2.5-14B-Instruct.
- Matrix: 8 scenarios, 3 repair-target types, 4 edit operators, 2 generators;
  192 attempted revisions.
- Eligible committed revisions: 178. A revision was eligible only if its edit
  parsed and the mandatory target recheck confirmed recovery.
- Non-target old-SAT transitions: 1,958.
- Scratch result: `/mnt/scratch/z/zitong/hsgr-obligation-ledger/results/stale_verdict_p0j_753073`.

## Apparatus control

The verifier control gate passed:

| Control | Result |
|---|---:|
| JSON parse validity | 100% |
| Old/clean positive accuracy | 99.72% |
| Deliberate target-negative accuracy | 100% |

Passing aggregate controls did not remove the need for manual review. The
manual audit found a localized verifier failure described below.

## Automatic result before manual review

The 14B verifier proposed 36 stale non-target verdicts. It appeared that all
36 were caught by witness overlap and witness-text similarity. A scenario-
grouped learned classifier also caught all 36 while checking fewer nodes, but
it still failed the frozen gate because it improved verification saving over
the strongest simple heuristic by only 0.19 percentage points, far below the
required 5 points.

These automatic numbers are not the final scientific result.

## Mandatory manual audit

All 36 candidate positives and the frozen 36-row candidate-negative sample
were inspected.

- 26/36 candidate positives were real SAT-to-FAIL transitions (72.22%
  candidate-positive precision).
- 10/36 were verifier false positives.
- All 36 sampled candidate negatives were confirmed non-stale.

The ten false positives had the same form: Qwen3 fused two findings without a
terminal period, but the audited fact remained explicitly present. The 14B
verifier treated the sentence-boundary defect as if the fact had disappeared.
This explains why aggregate control accuracy was high while one structured
output pattern still failed.

The 26 confirmed stale verdicts covered six scenarios and both generators,
but only two operators:

| Operator | Confirmed stale verdicts |
|---|---:|
| Free patch of up to four sentences | 14 |
| Forced four-sentence section patch | 12 |
| Forced target-sentence replacement | 0 |
| Complete rewrite | 0 |

They were direct omissions: the multi-sentence replacement swallowed an old
fact and failed to put it back. They were not hidden semantic state changes
outside the edited/witness region.

## Metrics after manual correction

The repair target is excluded from stale prediction and is always charged as
a mandatory check. Therefore “target-only” has zero non-target stale recall by
definition.

| Policy | Stale recall | Verification saving |
|---|---:|---:|
| Full recheck | 100% | 0% |
| Target only | 0% | 91.67% |
| Witness overlap | 100% | 85.63% |
| Edit proximity | 100% | 70.97% |
| Old-witness text similarity | **100%** | **89.14%** |
| Frozen union heuristic | 100% | 85.35% |
| Learned invalidator, scenario-grouped OOF | 96.15% | 90.50% |
| Matched-budget random (mean recall) | 20.52% | same learned budget |

The learned model is clearly better than random, but that is not enough. It
saves only 1.36 points more verification than witness similarity while missing
one real stale verdict. The frozen requirement was at least a 5-point saving
gain at at least 90% recall.

## Frozen gates

| Gate | Result | Reason |
|---|---|---|
| G1: verifier apparatus | Pass | All three aggregate controls at least 95% |
| G2: phenomenon support | **Fail** | 26 positives and 6 scenarios, but only 2 operators; 3 were required |
| G3: useful predictability | **Fail** | Learned model does not beat the best simple heuristic by 5 saving points |
| All gates | **Fail** | Stop learned invalidator/RL on this apparatus |

## What this does and does not establish

Established:

1. Criterion-level verdicts are version-bound: a sufficiently broad patch can
   delete an old correct item.
2. A target-only recheck is unsafe after a multi-sentence patch.
3. On this controlled list-like answer, actual diff and old-witness continuity
   are sufficient to invalidate safely and cheaply.
4. Qwen3-8B repaired the target more often than Qwen2.5-7B (95.83% versus
   89.58% before manual stale auditing), but also exposed a sentence-boundary
   failure mode in the 14B verifier. It is not an unqualified replacement.

Not established:

1. That naturally occurring long-form revisions have the same stale rate.
2. That hidden states carry a useful dynamic invalidation signal.
3. That revision risk requires learning or RL.
4. That witness overlap is a general causal mechanism. Here it succeeds for
   the mechanical reason that the old fact was directly deleted.

## Next-step constraint

A new predictability study is worthwhile only if its data first contains at
least 20 manually confirmed **semantic** stale transitions that are not
explained by direct deletion of the old witness, span at least three revision
operators, and cause a simple diff/witness rule to miss a material fraction.
Natural long-form revisions with paraphrase, cross-sentence consistency,
updated conclusions, and dependent obligations are the appropriate screen.

Until that prerequisite passes, adding Qwen3-14B, hidden-state features, a
larger classifier, or RL would increase complexity without addressing the
observed bottleneck.
