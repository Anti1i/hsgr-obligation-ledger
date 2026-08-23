# P0g-R1 frozen verifier recalibration

P0g job `751560` is retained as an apparatus failure: parse validity and positive-control accuracy
were 100%, but negative-control accuracy was 83.3%, below the frozen 95% gate.

All four false negatives were attribution controls.  In each, the saved answer visibly attached
`[B]` to the primary fact, but the verifier inferred the correct source from fixed evidence and
marked the answer as if it had written `[A]`.

R1 makes one control-driven change before examining individual candidate outputs: the verifier is
explicitly told to judge the citation displayed in the answer and never silently repair it using
the evidence.  R1 reuses the exact 96 candidate texts from job `751560`; it does not regenerate,
filter or relabel candidates, change witnesses, alter gates, or change any repair arm.

The original P0g control thresholds and four mechanism gates remain unchanged.  If R1 controls do
not pass, P0g remains apparatus-failed and no candidate mechanism result is interpreted.

