# Witness-interference mechanism study P0g

## Research question

P0g tests one mechanism, not a benchmark claim:

> Does a targeted edit regress already-satisfied obligations when it touches the answer spans that
> support them, and can explicitly preserving those spans reduce the regression?

Typed relation graphs and hidden states are excluded.  They remain gated behind this mechanism.

## Controlled blocked design

Six realistic evidence-grounded scenarios are crossed with four single seeded failures:

1. event or procedure order;
2. evidence attribution;
3. requested-result coverage;
4. recommendation consistency.

This produces 24 instances.  Each has exactly one failed target, three satisfied obligations, no
incidental seeded factual error, and frozen sentence-level witnesses for every satisfied
obligation.  The six scenarios are experimental blocks; the 24 rows are not claimed to be 24
independent natural tasks.

For every scenario/failure pair, the clean all-satisfied answer and the single-failure baseline are
created deterministically before model generation.  Witnesses are also frozen before generation.

## Repair arms

All arms see the same evidence, saved answer and failed target.

1. `full_rewrite`: minimally rewrite the complete answer.
2. `local_patch`: replace at most two consecutive sentences and generically preserve other content.
3. `obligation_patch`: the same patch interface plus the three already-satisfied obligations, but
   without their locations.
4. `witness_patch`: the same obligations plus their exact frozen sentence witnesses.

The primary causal comparison is `witness_patch` versus `obligation_patch`.  This distinguishes
localization information from merely reminding the model what must remain true.  Full rewrite is an
operator comparison, not a prompt-matched causal control.

## Verification

Qwen2.5-14B-Instruct independently returns YES/NO and answer-sentence witnesses for all four
obligations.  Before candidate interpretation it must pass deterministic controls:

- every clean answer satisfies all obligations;
- every seeded baseline fails only its frozen target;
- parse validity, positive accuracy and negative accuracy are each at least 95%.

Invalid candidate parses are unknown, not failures.  All candidates contributing a claimed
regression require manual review; the study also samples non-regression successes for false-negative
inspection.

## Measures

- target repair rate;
- preservation and satisfied-to-violated regression rate;
- sentence-level overlap between the edit set and each frozen witness set;
- regression enrichment under overlap (the frozen Gate 3 uses patch arms only, because a full
  rewrite mechanically touches every original witness);
- selective-reverification recall and judge-call saving;
- target-repair and regression differences between `obligation_patch` and `witness_patch`.

Witness overlap invalidates an obligation to `unknown`; it never directly labels it failed.

## Frozen gates

P0g supports continuing the repair-dynamics main line only if all of the following survive manual
review:

1. at least 15 successful targeted repairs across the four arms;
2. at least five unambiguous successful repairs contain a satisfied-to-violated regression;
3. regression under witness overlap is at least three times regression without overlap;
4. compared with `obligation_patch`, `witness_patch` lowers successful-repair regression by at least
   50% relative while reducing target repair by no more than 10 percentage points.

If the verifier control fails, no mechanism gate is interpreted.  Passing P0g would justify a
larger natural-data replication, not benchmark generalization, a planner, or a hidden-state claim.
