# ASQA fresh single-node intervention Oracle P3x — job 733401

## Verdict

`STATIC_REPLICATES_SELECTION_FAIL`.

The fresh subset strongly replicates the usefulness of showing the complete
facet structure, but rejects the proposed single-node selection action space.
The hindsight `KEEP-or-single-node` Oracle is 7.81 points **worse** than the
`KEEP-or-all-node` Oracle, with a significant paired advantage in the opposite
direction (6 single-only versus 21 all-only successes; exact two-sided McNemar
`p = 0.005925`).

Do not train a hidden selector on single-node intervention utility under this
apparatus.  The result supports set-level facet coverage, not choosing one
facet to emphasize.

## Reproducibility

- Frozen protocol: `EXPERIMENT_PROTOCOL_ASQA_SINGLE_NODE_INTERVENTION_P3X.md`
- Code commit: `cf9249b`
- Slurm job: `733401`, `xgpi20`, H100 NVL MIG 3g.47gb, 12m33s, exit `0:0`
- Model: `Qwen/Qwen2.5-7B-Instruct`, bfloat16, greedy, 192 new-token cap
- Data: 427 clean eligible cases; exclude all 192 P1x cases; select 192 of the
  remaining 235 using the frozen P3x hash; old/new overlap 0
- Generation rows: 1,108 observed and 1,108 expected
- Result directory:
  `/mnt/scratch/z/zitong/dch-hsgr/results/asqa_single_node_intervention_p3x_733401`
- Report SHA-256:
  `78ce985f64d7f1f4d0887afed581e5ab429c3a83b635463996e31fed73065943`
- Generations SHA-256:
  `70889391f29760d0b173748f0beef56af424bbf9e0753a903d5e5d88b508fbdc`
- Selected IDs SHA-256:
  `55f458fdd4ab5d2a62bd7d8c45a3cba247471e855b9057055a70679f8e89fab4`
- Decoy mapping SHA-256:
  `b72080b06b2133d2f310cb6f26fce7e4849204cf55c4e7be50b25f94a5bd5bfd`

All protocol-match and artifact-completeness checks pass.  The selected facet
histogram is 118/32/22/8/12 problems with 2/3/4/5/6 nodes.

## Absolute results

| Arm or policy | STR-EM | STR-HIT | Median words |
|---|---:|---:|---:|
| fixed direct | 68.70% | 39.58% | 102.5 |
| all true nodes | 78.17% | 55.21% | 101.5 |
| uniform single node | 61.69% | 31.04% | 66.8 |
| matched single decoy | 57.44% | 30.73% | 82.0 |
| KEEP-or-all-node Oracle | 80.59% | 58.33% | — |
| KEEP-or-single-node Oracle | 77.27% | 50.52% | — |
| best fixed released position | 68.26% | 39.06% | — |

Showing all true nodes improves over direct by +9.47 points STR-EM and +15.63
points STR-HIT.  This fresh result confirms the P1x static-structure finding.

In contrast, a uniformly chosen single true node is 7.00 points worse than
direct in STR-EM and 8.54 points worse in STR-HIT.  It remains 4.25 points
better than a matched single decoy in STR-EM, so the node content has a real
effect, but that effect is not a useful complete-answer policy.

## Frozen gates

| Gate group | Result | Observation |
|---|---:|---|
| exact fresh counts and zero overlap | pass | 427 / 192 / 235 / 192; overlap 0 |
| complete 2--6-node apparatus | pass | 1,108 / 1,108 rows |
| direct operating point | pass | 39.58% HIT, 102.5 words, 116 failures |
| all-node static replication | pass | +15.63 points HIT |
| single Oracle beats all Oracle by 5 points | **fail** | **−7.81 points** |
| Oracle paired test favors single | **fail** | 6 single-only / 21 all-only; p=0.005925 |
| at least 24 mixed intervention problems | **fail** | 18 |
| repair prevalence 5--50% | pass | 24 / 352 = 6.82% |
| single Oracle beats best fixed by 10 points | pass | +11.46 points |
| single Oracle beats all Oracle in both halves | **fail** | −8.08 / −7.53 points |
| uniform single beats decoy by 2 points EM | pass | +4.25 points |
| injected-facet specificity gap at least 3 points | pass | +26.69 points |

All four apparatus/static gates pass.  Four of eight actionability gates fail,
including both primary Oracle gates and the fixed-half stability gate.

## Mechanism diagnosis

The single-node intervention is highly node-specific but compositionally
harmful:

- the injected facet's coverage increases by an average of **+9.02 points**;
- non-injected facets' coverage decreases by an average of **−17.67 points**;
- the injected-minus-non-injected specificity gap is **+26.69 points**.

Thus the prompt does not merely add noise.  It successfully makes the model
focus on the named facet, but that focus crowds out the other facets required
for strict long-answer success.  The hidden target proposed after P2x would
therefore optimize the wrong action: selecting one node is not equivalent to
improving global multi-node coverage.

There are only 24 strict repair rows among 352 single-node interventions on
direct-failure problems.  Only 18 problems contain both a repairing and a
non-repairing single-node action.  Repairs occur at released positions 1/2/3
with counts 10/12/2 and none at positions 4--6.

The result is stable in the two frozen ID halves:

| Half | N | KEEP/all HIT | KEEP/single HIT | Difference |
|---|---:|---:|---:|---:|
| 0 | 99 | 61.62% | 53.54% | −8.08 points |
| 1 | 93 | 54.84% | 47.31% | −7.53 points |

## Post-hoc facet-count diagnosis

These strata were not additional frozen gates and are diagnostic only.

| Facets | N | Direct HIT | All-node HIT | KEEP/all | KEEP/single | Single−all |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 118 | 50.85% | 65.25% | 69.49% | 65.25% | −4.24 points |
| 3 | 32 | 25.00% | 50.00% | 53.13% | 34.38% | −18.75 points |
| 4 | 22 | 22.73% | 36.36% | 36.36% | 27.27% | −9.09 points |
| 5 | 8 | 25.00% | 25.00% | 25.00% | 25.00% | 0.00 points |
| 6 | 12 | 8.33% | 25.00% | 25.00% | 8.33% | −16.67 points |

Single-node selection never exceeds all-node/KEEP in any facet-count stratum.
The disadvantage becomes especially large once three or more facets must be
composed, which is consistent with cross-facet crowd-out rather than a missed
global selector.

## Objective conclusion and route decision

The candidate-node proposal is now narrowed further:

1. ASQA clean fixed support supplies a valid, non-collapsed facet-node domain.
2. Exposing the complete true facet set consistently improves strict long-form
   coverage on two disjoint 192-example subsets.
3. Individual node prompts causally increase attention to their own facet.
4. That local gain is purchased by a larger loss on other facets, so choosing
   one node is not a useful global action space.

Under the frozen stop rule, do not proceed to a hidden marginal-utility reader
or single-node latent steering.  A future HSGR route would have to control a
**set-level coverage state**—preserving already represented facets while
recovering missing ones—rather than route generation toward one facet.  P3x
does not by itself establish that such a latent set controller exists, and a
textual all-node checklist remains an oracle scaffold that overlaps existing
clarification-tree/outline methods rather than a final novelty claim.
