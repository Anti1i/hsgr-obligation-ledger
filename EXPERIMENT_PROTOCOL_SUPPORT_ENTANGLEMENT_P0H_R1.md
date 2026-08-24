# P0h-R1 apparatus recalibration

## Why P0h job 752920 is not interpreted

The frozen P0h job completed normally but failed its apparatus gates:

- verifier parse validity was 95.8%, positive accuracy 95.6%, and negative accuracy 91.7%;
- 18/18 entangled sentence patches were rejected because the model returned sentence index 2 even
  though the semicolon-coordinated answer contained only numbered sentence `[S1]`;
- consequently, only one content block had successful sentence patches in both layouts.

Inspection showed that the entangled replacements were usually clauses corresponding to the second
semicolon item.  The model treated clause ordinal as sentence ordinal.  This confounds support
co-location with source-unit localization and leaves P0h below its frozen minimum of 12 jointly
successful pairs.  The job is an apparatus failure, not evidence for or against the mechanism.

The verifier also returned lists instead of JSON booleans on three controls and silently corrected
one visibly reversed event order from the evidence.  These are schema and answer-grounding errors.

## Frozen R1 changes

R1 changes only the failed apparatus components:

1. the target-containing source sentence is deterministically fixed before generation (`S1` for
   entangled and `S2` for disentangled);
2. the patch model returns only a `replacement` string for that whole source sentence, so unit
   selection is no longer learned or inferred;
3. the verifier prompt explicitly requires JSON booleans, integer witness IDs, and visible-answer
   comparison for wrong numbers, citations, and reversed order.

The 18 content blocks, two layouts, full-rewrite control, models, metrics, manual-review rule, and
all five P0h decision gates remain unchanged.  R1 generates fresh candidates because the original
patch prompts exposed a different output interface.  No result from job 752920 is used to loosen a
gate or select content.
