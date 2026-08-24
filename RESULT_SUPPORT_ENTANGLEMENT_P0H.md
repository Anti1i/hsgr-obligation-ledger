# Support-entanglement study P0h — result

## Verdict

**P0h does not validate support co-location as the mechanism of repair regression.**  The original
run was an apparatus failure.  The calibrated R1 run passed verifier controls, but did not reach
the frozen minimum number of jointly successful matched pairs and found no preserved-obligation
regression in either layout.

The strict frozen decision is therefore **inconclusive / do not proceed to P0i**, not a proof that
entanglement can never matter.  The observed direction is nevertheless important: in this short
controlled setting, sentence-level co-location by itself was not sufficient to cause regression.

## Runs

### Original P0h: apparatus failure

- Job: `752920`, commit `43563e6`, node `xgpi17`, exit `0`.
- Result directory:
  `/mnt/scratch/z/zitong/hsgr-obligation-ledger/results/support_entanglement_p0h_752920`
- Verifier controls: parse 95.8%, positive 95.6%, negative 91.7%; the negative control failed the
  frozen 95% threshold.
- All 18 entangled patches returned an out-of-range source-sentence index (normally 2) although the semicolon answer
  had only numbered sentence `S1`.  The model treated the second clause as the second sentence.
- Only one content block had successful sentence patches in both layouts.  No mechanism result from
  this run is interpreted.

### P0h-R1: usable apparatus

- Job: `752961`, commit `191af7a`, node `xgpi17`, exit `0`.
- Result directory:
  `/mnt/scratch/z/zitong/hsgr-obligation-ledger/results/support_entanglement_p0h_r1_752961`
- R1 fixed the target source unit before generation and required the model to output only the whole
  replacement sentence.  It also made the verifier's JSON types and visible-answer comparison
  explicit.

R1 controls passed:

| Control | Result | Frozen threshold |
|---|---:|---:|
| parse validity | 100% | >=95% |
| positive accuracy | 98.9% | >=95% |
| negative accuracy | 100% | >=95% |

The two positive-control errors were false negatives on clean entangled attribution examples.  They
did not create regression evidence.

## Automatic R1 results

| Operator | Layout | Target repair | Safe repair | Regression among successful repairs |
|---|---|---:|---:|---:|
| sentence patch | entangled | 16/18 (88.9%) | 16/18 (88.9%) | 0/16 |
| sentence patch | disentangled | 10/18 (55.6%) | 10/18 (55.6%) | 0/10 |
| full rewrite | entangled | 15/18 (83.3%) | 15/18 (83.3%) | 0/15 |
| full rewrite | disentangled | 16/18 (88.9%) | 16/18 (88.9%) | 0/16 |

Sentence patches succeeded in both layouts for nine matched content blocks.  There were zero
entangled-only and zero disentangled-only regression pairs, so the paired risk difference was zero.

The unexpectedly lower disentangled target-repair rate did not come from preservation failures.
Inspection showed that the generator often repeated a wrong before/after relation or omitted the
required source attribution inside the isolated target sentence.  It preserved the untouched left
and right sentences.

## Manual review

- All 16 automatically successful entangled sentence patches were inspected.  Every answer still
  expressed both `O_LEFT` and `O_RIGHT`; no missed regression was found.
- The 12 deterministically sampled successful non-regression controls were inspected.  No hidden
  left/right regression was found.
- One sampled entangled full rewrite was a target-recovery false positive: its attribution sentence
  was malformed and did not state source B.  This lowers confidence in the automatic full-rewrite
  success count but does not create a regression or change the P0h decision.

## Frozen gate decisions

1. **Verifier controls: pass.**
2. **At least 12 jointly successful patch pairs: fail, 9.**  By protocol this makes the primary
   paired test formally inconclusive.
3. **At least five entangled-only regressions with exact one-sided p < 0.05: fail, 0.**
4. **At least +20 point paired regression difference: fail, 0 points.**
5. **At least +15 point patch-specific layout interaction on ten common four-cell successes: fail.**
   Only seven blocks were four-cell successes and the observed interaction was zero.

Not all gates pass.  They are not relaxed after seeing the data.

## Scientific interpretation

P0g established that a repair can fix its target while damaging a previous success.  P0h shows that
the proposed explanation was too broad:

> Logically atomic obligations sharing one sentence are not, by themselves, enough to make repair
> non-monotonic under this explicit whole-sentence replacement interface.

The six P0g regressions are therefore more plausibly tied to the combination of open-ended edit
scope, deletion/replacement behavior, task wording, and answer length than to sentence-level
co-location alone.  This is a corrected hypothesis, not a causal identification of those
alternative factors.

The short controlled answers also continue to favor ordinary rewriting: full rewrite repaired most
targets and produced no detected preserved-obligation regression.  P0h provides no basis for
claiming that a provenance-aware constrained editor is needed here.

## Route decision

- Do not promote `support entanglement -> regression` to the paper mechanism.
- Do not start P0i as the next main-line experiment; its proposed hard locks were motivated by a
  mechanism that P0h did not support in this setting.
- Keep the P0g phenomenon claim, restricted to controlled existence.
- The next separable question is P0j incremental verification: after an edit, can an explicit
  ledger retain unaffected verdicts while matching full re-verification recall at lower judge cost?
  That question does not require support entanglement to be true and needs its own larger design,
  random-budget control, and non-mechanical edit scopes.
