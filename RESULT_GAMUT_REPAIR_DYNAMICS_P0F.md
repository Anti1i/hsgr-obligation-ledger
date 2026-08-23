# GAMUT repair-dynamics case study P0f — result

## Verdict

**Partly supported, not established.**  The four-case audit gives clear evidence that targeted
repair attempts can damage obligations that were previously satisfied.  It gives one
benchmark-valid but pre-declared borderline example of "fix one, break another."  It does not yet
provide a clean, unambiguous demonstration strong enough to anchor the paper.

The useful result is therefore narrower:

> Repair is a state-changing intervention, and even an explicitly local patch can require
> re-verifying obligations touched by the edit.  A typed relation graph alone does not solve this.

## Frozen run

- Generator: Qwen2.5-7B-Instruct.
- Independent extractor: Qwen2.5-14B-Instruct, calibrated in P0c.
- Cases: four manually defensible P0d relation-only failures.
- Arms: flat versus typed guide, crossed with full rewrite versus at-most-four-sentence patch.
- Slurm job: `751359`, completed on `xgpi2` in 6m13s with exit code 0.
- P0e code snapshot used by the job: `dc13758`.
- P0f analysis snapshot: `2bebf1e`.
- P0e result directory:
  `/mnt/scratch/z/zitong/hsgr-obligation-ledger/results/gamut_manual_repair_p0e_751359`
- P0f result directory:
  `/mnt/scratch/z/zitong/hsgr-obligation-ledger/results/gamut_repair_dynamics_p0f_751359`

This is a mechanism case study, not a benchmark result.

## Automatic structural observations

| Repair arm | Complete target recovery | Attempts losing an existing process component | Median edit ratio |
|---|---:|---:|---:|
| flat full rewrite | 2/4 | 0/4 | 0.107 |
| typed full rewrite | 1/4 | 1/4 | 0.103 |
| flat sentence patch | 1/4 | 3/4 | 0.083 |
| typed sentence patch | 0/4 | 1/4 | 0.090 |

- The oracle over the four arms structurally recovered 3/4 cases.
- Five of sixteen repair attempts removed an existing process component.
- Four of the five observed negative edges were `R_ORDER -> P2`; the other was
  `R_ORDER -> P4`.
- The nominally local flat patch was the most destructive arm: it lost a component in 3/4 cases.
- Typed guides did not beat matched flat guides.  In both full-rewrite and patch comparisons there
  was one flat-only success and no typed-only success.

These counts show that a repair operator can have collateral effects.  They do not by themselves
show "fix one, break another": losing a required process component means the composite process
target was not fully repaired.

## Manual review of all sixteen outputs

### Clear negative side effects

1. **Ferrule, both patch arms.**  The edited span covered sentences 2--4 and deleted the statement
   that ferrules were nailed onto handles.  The remaining mentioned steps became sorted only
   because a necessary component disappeared.  This is a destructive repair attempt, not a
   successful repair.
2. **USDA, typed full rewrite.**  It omitted a distinct continuing-annual-inspection node and leaked
   the process guide into the answer.  The target was not completely recovered.
3. **USDA, flat patch.**  It changed text inside the process witness region but lost the extractor's
   on-site-inspection node.  Again, the target was not completely recovered.
4. **Egyptian multiplication, flat patch.**  It rewrote a multi-sentence process region and lost the
   successive-doubling node.  The target remained failed.

### Closest "fix one, break another" example

The Iron Man flat sentence patch changed the process surface order to the required
`P1 -> P2 -> P3 -> P4`.  At the same time it deleted:

- the first requested essential story arc, *Demon in a Bottle*;
- Tony Stark's engineer/businessman context;
- Ho Yinsen's role;
- the transition-to-Iron-Man and character-development explanation.

The question explicitly asked for both the origin and a comprehensive list of essential arcs, so
this is a real preservation regression.  However, the case was already marked borderline during
P0d because injury and capture form a tightly coupled event, and the patch changed 34.2% of the
answer.  It is therefore **suggestive existence evidence, not a clean flagship result**.

### Automatic successes that do not survive as clean evidence

- **USDA flat full rewrite:** the extractor marked the process complete, but the answer retained
  "the plan was reviewed and approved" before the on-site inspection and then added another
  certifier decision afterward.  The original ordering ambiguity/contradiction was not cleanly
  removed.  This is an automatic-evaluation false positive for the intended relation.
- **Egyptian multiplication, both full rewrites:** both put the named stages in surface order, but
  the numerical explanation is wrong.  For example, one output states `4 x 55 = 88` and
  `2 x 55 = 44`; the other states `4 x 55 = 176`.  Their displayed sums also do not equal 2695.
  Because the baseline already contained arithmetic errors, these are factual-safety warnings, not
  clean satisfied-to-violated ledger transitions.

## What the case study says about the three hypotheses

### H1: checklist repair is non-monotonic

**Partly supported.**  Negative state changes are clear in 5/16 attempts, and the Iron Man case is a
benchmark-valid example in which the targeted surface relation is recovered while requested content
is lost.  Because that example is borderline and the other automatic successes are contaminated,
the strong claim is not yet established.

### H2: repair interference is structured

**Preliminary support only.**  Damage is concentrated by repair operator and node: the flat local
patch loses a component in 3/4 cases, and P2 accounts for four of five lost-node events.  In the
readable examples, the lost obligation's text lies inside the edited sentence span.  This motivates
preservation witnesses and selective invalidation, but four cases cannot establish predictability.

### H3: intervention planning beats myopic repair

**Not tested.**  P0e/P0f compares prompts and edit scopes, not a learned influence model or planner.
No planner, hidden-state, efficiency, or benchmark-level performance claim is warranted.

## Method decision

The experiment rejects two shortcuts:

1. Do not use "typed graph beats flat checklist" as the main claim; this run shows the opposite
   descriptive pattern.
2. Do not treat a small edit as automatically safe.  The flat local patch was the arm most likely to
   remove an existing process component.

The defensible next case study is a **clean preservation-witness test**:

- freeze 6--8 answers with exactly one unambiguous failed target;
- require at least three independently satisfied obligations with manually marked witness spans;
- exclude answers with pre-existing factual contradictions;
- repair only the failed target with full-rewrite and local-patch operators;
- re-evaluate every ledger node and record `satisfied -> violated` transitions;
- test whether edit/witness overlap recalls every regression and reduces unnecessary rechecks.

Only after that gate passes should work proceed to asymmetric repair matrices, intervention
planning, or hidden-state prediction.

