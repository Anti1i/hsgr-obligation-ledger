# Answer-state hidden Commit Guide result (job 726201)

## Boundary

Commit `724bdba`; Qwen2.5-7B-Instruct; 400 previously consumed MuSiQue
dependency units; five-fold nested OOF.  A completed base answer's hidden
state selected either `KEEP` or one fixed generic `REPAIR` action.  Supervision
was counterfactual action utility (`repair_EM - base_EM`), not candidate rank
or answer correctness.  The reserved 320-example split was not decoded.

## Action ceiling

The repair action itself had insufficient utility:

| Utility | Count |
|---|---:|
| repair fixes base | 15 |
| no outcome change | 356 |
| repair breaks base | 29 |

Base EM was 48.50%; always-repair EM was 45.00%.  Even the infeasible oracle
that repairs exactly the 15 positive-utility cases could improve by only
3.75pp, leaving almost no margin above the frozen +3pp policy gate.

## Nested-OOF result

| Policy | EM | Delta vs base | Action rate |
|---|---:|---:|---:|
| KEEP / base | 48.50% | -- | 0% |
| confidence-only | 48.00% | -0.50pp | 28.75% |
| hidden Commit Guide | 48.75% | +0.25pp | 12.25% |

The hidden policy selected 49/400 repairs, capturing five positive-utility and
four negative-utility cases.  It made five fixes and four breaks relative to
base (`p=1.0`).  Its gain over confidence-only was +0.75pp (10 vs 7 discordant,
`p=0.629`), below the frozen +2pp requirement.  Fixed-half gains were 0 and
+0.50pp, so sign stability alone passed.

## Decision

`OOF GATE FAIL.  RESERVED 320 NOT TOUCHED.`

The primary bottleneck is the generic repair action, not hidden-state policy
capacity.  Replacing the ridge observer, widening its projection, or tuning
the action threshold cannot create missing positive action utility.  This
closes `KEEP` versus generic repair as an HSGR main-method route.

