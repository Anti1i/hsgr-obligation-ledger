# TransitionGuard second fresh held-out (job 726389)

## Frozen setup

- Code: `86c5c89`
- Test problems: 320 untouched problems selected with seed `20260819`
- Frozen ID SHA-256: `927b1048c0df3f31efd42290859d105e7c90dbf67a2dc70ca90b8f1e023ddd12`
- Hop counts: 2-hop 183, 3-hop 80, 4-hop 57
- Model and decoding: Qwen2.5-7B-Instruct, SC@8
- Claim boundary: oracle decomposition, verified predecessor values, and gold support routing; this is a mechanism test, not an end-to-end result.

## Result

The pre-registered decision is **GATE FAIL**. The primary TransitionGuard signal was strong, but two secondary gates failed and must not be changed after seeing the result.

| Policy | Accuracy | Delta vs SC@8 |
|---|---:|---:|
| SC@8 | 44.0625% | - |
| Explicit predecessor guard | 48.7500% | +4.6875 pp |
| Length-only control | 49.6875% | +5.6250 pp |
| Route delta | 49.6875% | +5.6250 pp |
| Hidden predecessor-copy guard | 50.3125% | +6.2500 pp |
| TransitionGuard | **50.9375%** | **+6.8750 pp** |
| Direction-swapped TransitionGuard | 47.1875% | +3.1250 pp |

TransitionGuard made 27 fixes and 5 breaks relative to SC@8 (`p=1.13e-4`). Relative to the explicit predecessor guard it gained 2.1875 pp, with 13 fixes and 6 breaks (`p=0.167`). It exceeded the length-only control by 1.25 pp and the direction-swapped control by 3.75 pp.

## Pre-registered gates

| Gate | Result |
|---|---|
| Reader transfer | PASS |
| Transition headroom | PASS |
| Safe net vs explicit | PASS |
| Beyond length | PASS |
| Predecessor-route specificity | **FAIL**: 0.625 pp, required at least 1 pp |
| Transition directionality | PASS |
| Depth signature | **FAIL**: 4-hop delta was 5.2632 pp, 0.294 pp below the threshold implied by the 2-hop delta |

## Technical interpretation

The repeatable part of the signal is the directed difference between correct-route and predecessor-route support states. The predecessor-copy probe is weaker: its wrong-route policy reached 50.3125%, while the correct-route control reached 49.6875%. This indicates that part of the copy score is a route-agnostic candidate-form or answer-length signal.

The next diagnostic therefore uses one paired reader over `h_correct - h_predecessor`. It removes the independent copy probe and tests the same trained reader under sign reversal and within-problem predecessor-state mismatch controls. The 320-problem result above is development evidence for that diagnostic; it is not reused as a fresh confirmation.

