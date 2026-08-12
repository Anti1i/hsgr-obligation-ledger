# Marginal Root-Utility Guide V0 result

## Verdict

**FAIL / route stopped before hidden-state extraction.**

The graph-level value-class credit is faithful to the exact counterfactual and
can identify useful decisions when a disagreement exists. However, the current
candidate domains produce too few such decisions for a valid hidden-state
learning experiment, and the GSM primary test misses both the frozen effect
size and significance gates.

This result follows the thresholds frozen in
`EXPERIMENT_PROTOCOL_MARGINAL_ROOT_UTILITY_GUIDE_V0.md`; no threshold was
changed after seeing the result.

## Reproduction

```bash
python -X utf8 -m unittest -v test_marginal_root_utility_guide.py
python -X utf8 audit_marginal_root_utility_guide.py
```

Machine-readable output: `marginal_root_utility_v0_report.json`.

## Primary results

| Split | N | Multi-class nodes | Actionable problems | LOO/CF argmax | Best baseline | LOO | Delta | W/L | Exact p | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MATH `outputs` | 137 | 138/361 | 9 | 0.986 | frequency 0.657 | 0.701 | +0.044 | 6/0 | 0.03125 | FAIL |
| GSM `outputs_gsm_test` | 248 | 99/646 | 8 | 0.990 | frequency 0.871 | 0.891 | +0.020 | 5/0 | 0.06250 | FAIL |

On the actionable subsets, LOO improves by 0.667 (MATH) and 0.625 (GSM), but
those subsets contain only 9 and 8 problems. The frozen minimum was 20 per
primary split. Candidate-domain collapse is 0.618 on MATH and 0.847 on GSM.

## Descriptive training-split results

| Split | N | Actionable | Frequency | LOO | Delta | Exact p | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| GSM train | 796 | 23 | 0.907 | 0.933 | +0.026 | 9.54e-7 | FAIL |
| MATH train | 190 | 22 | 0.563 | 0.647 | +0.084 | 3.05e-5 | PASS |

The larger GSM split confirms that rare disagreements are often decisive, but
its all-problem gain still falls below the frozen +3 pp threshold.

## Interpretation

### Established

- Value-class LOO is a structurally faithful proxy here: node-level argmax
  agreement with exact counterfactual credit is at least 0.979 on every split.
- It is not merely candidate frequency. When LOO and frequency differ, LOO
  nearly always wins in these cached assignments.
- The useful action is sparse because most node candidate domains collapse to
  one normalized value, especially on GSM.

### Not established

- The oracle LOO policy uses root gold outcomes. Its high accuracy is an action
  ceiling, not a deployable method.
- The result does not show that hidden states can predict marginal utility.
- The result does not justify GPU feature extraction, threshold relaxation, or
  end-to-end training on this route.

## Decision

Do not run the planned node-position hidden-state extraction on these data.
Continuing would train and evaluate on too few effective decisions and would
risk turning a sparse oracle observation into an unsupported method claim.
A new route must create a denser intervention/action space rather than merely
relabeling the same collapsed candidate domains.

