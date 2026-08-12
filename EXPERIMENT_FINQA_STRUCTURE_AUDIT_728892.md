# FinQA program-intervention P0 Stage-A result

Date: 2026-08-12  
Job: `728892` (`normal`, 2 CPU, 8 GB, 45 seconds, host `xcnf27`)  
Code: `9e96ad0`  
Frozen protocol: `EXPERIMENT_PROTOCOL_FINQA_PROGRAM_INTERVENTION_P0.md`

## Decision

**Stage A failed the pre-registered structural gate.**  Do not run predicted
program generation, atomic-edit oracle enumeration, hidden-state extraction,
or a FinQA Hidden Guide under this protocol.

The failure is not parser or executor failure.  The independent executor
reproduced every public gold execution result.  The benchmark is instead too
short and too nearly linear to support the intended hierarchical HSGR claim.

## Provenance

- Official source: `https://github.com/czyssrs/FinQA.git`
- Source commit: `0f16e2867befa6840783e58be38c9efb9229d742`
- Train SHA-256:
  `49f237eb9779b569473b26b08048867d04635a7cc39ad6a7a5664c55bb428db6`
- Dev SHA-256:
  `a847fb7e0d61a3125a1e2909852df6b89f1ee64d2c5ff1bf689e332214deee51`
- Public-test SHA-256:
  `831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc`
- Full report:
  `/mnt/scratch/z/zitong/dch-hsgr/logs/finqa_structure_audit_report_728892.json`
- Slurm log:
  `/mnt/scratch/z/zitong/dch-hsgr/logs/dch-finqa-a0-728892.out`

## Results

Every split had 100% program parse coverage and 100% agreement between the
independent executor and `qa.exe_ans`.

| Split | n | 1-step | 2-step | Deep | Join | Deep + join | Branch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 6,251 | 3,717 | 2,013 | 447 | 182 | 114 | 26 |
| Dev | 883 | 523 | 287 | 59 | 31 | 17 | 6 |
| Public test | 1,147 | 654 | 409 | 71 | 33 | 21 | 8 |

Frozen definitions:

- `deep`: at least three operations and dependency depth at least three;
- `join`: at least one operation consumes two distinct prior references;
- `branch`: at least one program result has out-degree greater than one.

Public-test step-length distribution:

| Steps | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| Count | 654 | 409 | 55 | 10 | 19 |

Public-test dependency-depth distribution:

| Depth | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| Count | 657 | 419 | 60 | 8 | 3 |

## Frozen checks

| Check | Result |
|---|---:|
| Gold parse coverage >=95% on every split | PASS (100%) |
| Gold execution agreement >=95% on every split | PASS (100%) |
| Public-test `deep` count >=150 | **FAIL (71)** |
| Public-test `join` count >=100 | **FAIL (33)** |

## Interpretation boundary

Verified facts:

1. FinQA's public gold programs and executor are clean enough for exact atomic
   program-repair experiments.
2. In public test, 92.7% of examples contain at most two operations.
3. Only 2.9% contain a two-reference join, and only 0.7% contain a branching
   intermediate result.

Inference:

FinQA could support a short program-repair paper, but a positive node/edit
oracle on this distribution would mainly be a one- or two-step localization
effect.  It would not provide credible evidence that a hierarchy-conditioned
Guide is needed.  Continuing to the GPU stages would therefore answer a weaker
question than the HSGR mechanism requires and would create unnecessary overlap
with current counterfactual-repair methods.

## Next benchmark requirement

Any replacement benchmark must be audited before model calls and should have:

1. native executable gold programs, not synthetic stitched questions;
2. at least 150 held-out examples with dependency depth >=3;
3. at least 100 held-out examples with two-reference joins or an equivalent
   multi-parent operation;
4. non-trivial branching/reuse, not only a flat sequence;
5. an exact executor and a gold-free finite atomic edit interface;
6. a frozen model baseline in the 30--70% range on the structural subset.

