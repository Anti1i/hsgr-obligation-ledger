# ASQA candidate-node hidden Guide P2x — job 733309

## Verdict

`APPARATUS_FAIL`; the hidden readout also fails every frozen predictive gate.
Do not run causal hidden steering for this target.

P1x established that the true static facet-node checklist is strongly useful.
P2x shows that a different and necessary claim does not follow: the current
answer-prefix hidden state does not reliably identify which uncovered node will
still be omitted by ordinary continuation.

## Reproducibility

- Protocol: `EXPERIMENT_PROTOCOL_ASQA_CANDIDATE_NODE_HIDDEN_P2X.md`
- Code commit: `40670d9`
- Slurm job: `733309`, `xgpi14`, H100 NVL MIG 3g.47gb, 2m01s, exit `0:0`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Split: 96 calibration / 96 untouched held-out problems
- Layers: 13, 20, 27; calibration selected block 20 and ridge alpha 1000
- Node sequences/tokens: 502 / 9,888
- Prefix sequences/tokens: 576 / 544,543
- Result directory: `/mnt/scratch/z/zitong/dch-hsgr/results/asqa_candidate_node_hidden_p2x_733309`
- Report SHA-256: `d248f5f07f046d1f212946053f8ca4368cfb367b7da35c96c7b94b27049555ae`
- Score rows SHA-256: `cf19a6766801785a9bc1b0e2d11f9042a76190b0d8a8579ef6e79cfdd55b5733`

## Apparatus

The held-out set contains 96 problems, 341 retained candidate-state rows, and
56 problems whose final direct answer misses at least one facet.  Those size
gates pass.  Two actionability gates fail:

- only 42 states contain both a future-missing and a future-covered candidate,
  below the frozen minimum of 60;
- 66.86% of retained nodes remain missing, above the allowed 60% ceiling.

The node domain itself did not collapse: every problem still has 2--6 unique
facets.  What collapsed is the dynamic target among currently uncovered nodes:
most candidates are either already covered or will never be covered without an
external scaffold.

## Held-out readout results

| Reader/control | AUROC | Recall@1 | MRR |
|---|---:|---:|---:|
| surface only | 0.653 | 0.500 | 0.740 |
| prefix hidden only | 0.572 | 0.452 | 0.704 |
| node hidden only | 0.649 | 0.548 | 0.756 |
| full prefix + node hidden | 0.582 | 0.548 | 0.752 |
| full with wrong node | 0.559 | 0.357 | 0.665 |

The full reader is 0.071 AUROC worse than surface-only and improves Recall@1
by only 0.048, below the required +0.10.  The node-only result is almost as
strong as surface-only and stronger than the full hidden reader in AUROC.  The
predictable signal is primarily that some facet questions are intrinsically
hard, not a useful interaction between the current prefix state and a specific
candidate node.

Wrong-node replacement reduces Recall@1 substantially, but reduces AUROC by
only 0.023 rather than the required 0.05.  This isolated effect cannot rescue
the failed absolute, surface-gain, stability, and apparatus gates.

## Stability and prefix diagnosis

The full reader loses to surface AUROC in both fixed halves (0.601 versus 0.686,
and 0.553 versus 0.624).  Recall improves in only one half.

The failure is not caused solely by late prefixes:

| Prefix | Rows | Mixed states | Omission rate | Surface AUROC | Hidden AUROC | Surface R@1 | Hidden R@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 25% | 140 | 20 | 54.29% | 0.557 | 0.528 | 0.550 | 0.500 |
| 50% | 111 | 14 | 68.47% | 0.547 | 0.488 | 0.429 | 0.500 |
| 75% | 90 | 8 | 84.44% | 0.522 | 0.588 | 0.500 | 0.750 |

The apparent 75% improvement rests on only eight mixed states and does not
generalize.  On calibration OOF data, the best AUROCs at layers 13/20/27 were
only 0.455/0.470/0.489, so the failure is not a held-out accident or a single
bad layer choice.

## Objective conclusion

The candidate-node proposal is **partly supported**:

1. ASQA supplies a non-collapsed facet-node domain on the clean subset.
2. Showing all true nodes as an oracle checklist causally improves strict
   coverage over matched decoys (P1x).
3. The tested hidden-state mechanism cannot selectively predict future omitted
   nodes beyond surface controls (P2x).

Therefore the static node scaffold is a valid positive result and benchmark
property, but the current hidden Guide route is not supported.  Proceeding to
activation steering would violate the frozen stop rule.  A future route must
change the state/target mechanism prospectively rather than tune prefixes,
layers, or thresholds on this failed test.
