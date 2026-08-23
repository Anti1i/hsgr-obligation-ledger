# GAMUT manually confirmed repair case study P0e

## Purpose

P0d leaves only four manually defensible relation-only failures among 192 long-form answers. P0e
asks a narrow question: on these confirmed cases, does an explicit directed process guide help a 7B
model repair order while retaining every process component, compared with an ordinary ordered list?

This is a four-case mechanism check, not a benchmark performance result.

## Frozen cases

The four IDs were fixed after reading all seven P0d automatic relation-only candidates:

- `22cab127-2c4e-4b8e-9590-76d8d7ada2cd`: ferrule attachment chronology.
- `8dab7a6d-13e5-4081-884e-eced5b4cf615`: Iron Man injury/capture surface order (borderline).
- `91ed57a0-bdf6-4111-a689-955e47280cb2`: organic inspection before certification decision.
- `afe5977f-01b0-4f5d-acea-842349f3d37b`: Egyptian multiplication operation order.

The three rejected automatic candidates are excluded, not silently relabeled.

## Four repair arms

All arms receive the same saved answer, fixed evidence, process-node text and canonical node order.

1. `flat_full_rewrite`: ordinary numbered list; return a minimally revised complete answer.
2. `typed_full_rewrite`: typed nodes plus explicit directed edges; return a minimally revised answer.
3. `flat_sentence_patch`: ordinary numbered list; replace at most four consecutive sentences.
4. `typed_sentence_patch`: typed nodes plus explicit directed edges; replace at most four sentences.

The typed arms add representation, not new oracle facts: both flat and typed prompts contain the same
nodes in the same canonical order.

## Evaluation

- P0c's calibrated answer-only extraction followed by deterministic order checking.
- Structural safe success requires valid extraction, all process nodes present, and canonical order.
- Patch validity and character edit ratio are recorded.
- Every one of the 16 outputs must be manually inspected for non-process factual regressions before
  interpretation.

## Frozen interpretation

- The repair action space is viable if the per-case oracle over four arms safely repairs at least two
  of four cases.
- Typed representation is suggestive only if a typed arm succeeds where its matched flat arm fails
  on at least two manually verified cases, without the reverse happening as often.
- Minimal repair is viable only if a patch arm safely repairs at least two cases with edit ratio at
  most 0.25 and manual preservation.
- Four cases cannot justify significance, benchmark-level gains, hidden selection, or a main-paper
  claim. Hidden-state work remains gated off regardless of P0e outcome.

