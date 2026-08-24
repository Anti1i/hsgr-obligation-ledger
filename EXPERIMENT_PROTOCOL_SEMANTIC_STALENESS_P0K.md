# P0k-R1: Semantic staleness outside the edited witness

## Revision record

The initial R0 construction placed the dependency edit and harmless edit in
different source sentences. A frozen preflight exposed label leakage: the
fixed surface union reached 95% stale recall with 35% verification saving
(token Jaccard: 90% recall, 42.5% saving). R0 therefore failed G2 before any
model run and is not used as evidence.

R1 is the single permitted apparatus repair. Both arms now edit the same
composite source sentence and concern the same entities. Where possible, the
harmless arm changes a nearby value or source that remains logically
consistent with the conclusion. Thresholds and gates below are unchanged.

## Frozen question

Can an old criterion verdict become invalid even though the sentence that
states the criterion's conclusion is byte-for-byte unchanged and outside the
edited span? If so, do cheap surface dependency rules already identify the
risk, or is there evidence for a genuinely semantic invalidation problem?

P0k is a controlled existence and apparatus screen. It does **not** estimate
how often semantic staleness occurs in natural long-form revisions.

## Definition

For old document `y0`, revised document `y1`, old-SAT obligation `Oi`, frozen
conclusion witness `Wi`, and actual edited old-sentence set `E`:

- `E` and `Wi` are disjoint;
- the text at `Wi` is exactly unchanged;
- `semantic_stale_i = 1` iff the unchanged conclusion is no longer logically
  supported or true under the revised document.

This is stricter than the P0j label. Direct deletion, rewriting, or sentence-
boundary corruption of the conclusion is excluded.

## Paired controlled matrix

Build 40 base documents: 8 application domains crossed with 5 dependency
mechanisms. Each document has ten sentences and one frozen conclusion
obligation.

Mechanisms:

1. premise/value update invalidates a comparison conclusion;
2. evidence content update invalidates an attribution conclusion;
3. input update invalidates a derived numerical conclusion;
4. date update invalidates a temporal-order conclusion;
5. definition threshold update invalidates a classification conclusion.

Each base document produces a matched pair:

- dependency edit: modify one upstream source so the unchanged conclusion is
  false;
- harmless edit: modify the same composite source sentence and the same local
  entities, but preserve the truth of the unchanged conclusion.

The two arms share the base document, obligation, mechanism, domain, edited
sentence position, local entity context, edit format, and conclusion witness.
Arm identity is never exposed to a predictor.
Four domains use direct naming; four use an explicit glossary alias to require
a two-hop dependency. Total revision rows: 80, balanced 40 stale / 40 safe.

## Ground truth and verification

Labels are determined by executable construction rules, not by an LLM judge.
Unit tests check the underlying comparison, arithmetic, ordering, attribution,
and threshold conditions.

Two independent frozen judges then test whether the natural-language
apparatus expresses those labels clearly:

- Qwen3-8B, non-thinking;
- Qwen2.5-14B-Instruct.

Each judge evaluates the old, dependency-edited, and harmless-edited document
for all 40 cases. It must return the conclusion sentence and upstream
dependency sentences as well as a Boolean verdict. Judge outputs never become
ground-truth labels.

## Leakage prevention

- Predictor features cannot use edit arm, expected label, mechanism name,
  domain ID, judge verdict, or judge rationale.
- Evaluation holds out an entire application domain. Both arms and all five
  mechanisms from a held-out domain remain together.
- Thresholds for any trained baseline are selected only inside the outer
  training domains.
- Positive and harmless edits modify the same single source sentence and the
  conclusion sentence is unchanged in both.

## Frozen surface baselines

All baselines see the old source sentence, revised source sentence, unchanged
conclusion witness, and obligation text:

- witness overlap / direct conclusion diff;
- token Jaccard between changed text and conclusion;
- character similarity;
- capitalized/entity-token overlap;
- citation-token overlap;
- a fixed union of the preceding cues;
- a small logistic model over only these numeric surface features;
- matched-budget random selection.

The screen does not yet use LLM hidden states. Hidden-state features are
permitted only after the semantic-staleness and simple-baseline gates pass.

Primary metrics are stale recall and verification saving. Accuracy is
secondary because deployment prioritizes not reusing stale judgments.

## Frozen gates

### G1: semantic apparatus

For each judge:

- parse validity at least 95%;
- old-SAT accuracy at least 95%;
- dependency-edit FAIL accuracy at least 95%;
- harmless-edit SAT accuracy at least 95%.

The two judges must agree on at least 95% of all 120 document states. Failure
means the natural-language construction is ambiguous; do not interpret
predictor results.

### G2: surface baselines are insufficient

No frozen single surface heuristic or their fixed union may achieve both:

- stale recall at least 90%; and
- verification saving at least 25%.

If a surface rule reaches that operating point, P0k does not justify a
semantic or hidden-state model.

### G3: mechanism coverage

Both judges must correctly recognize at least 7 of 8 dependency edits and at
least 7 of 8 harmless controls in at least four of the five mechanisms.

### G4: learned surface-feature screen

This is diagnostic, not required to establish semantic staleness. A
domain-grouped learned surface model is considered useful only if it reaches:

- stale recall at least 90%;
- verification saving at least 25%;
- at least 15 points more recall than matched-budget random; and
- at least 10 points more saving than the best frozen heuristic reaching 90%
  recall.

If G1 and G3 pass, G2 passes, and G4 fails, the next justified experiment is a
semantic/hidden-state representation screen. If G2 fails, stop: the controlled
task is still surface-solvable. RL remains out of scope in either case.

## Required manual audit

Before a final claim, inspect all judge disagreements and every case missed by
the strongest surface heuristic. Confirm that:

- the unchanged conclusion is unambiguously true in the old and harmless
  documents;
- it is unambiguously false after the dependency edit;
- no conclusion text or conclusion sentence boundary changed;
- the control edit is genuinely unrelated.

## Decision rule

- G1-G3 pass and G2 confirms surface insufficiency: semantic stale is a viable
  hard-case direction; only then test semantic/hidden-state representations.
- G2 fails: do not add learned or hidden-state components.
- G1 or G3 fails: repair the apparatus once; continued failure stops P0k.
- P0k says nothing about natural prevalence. A later natural long-form audit
  remains mandatory before a main-paper claim.
