# Executable-DAG benchmark audit P0: frozen protocol

Frozen on 2026-08-12 before downloading or inspecting the candidate dataset
files.  This is a zero-model, zero-GPU screen.  It supersedes choosing a
benchmark from its name, reported average program length, or table hierarchy.

## 1. Testable claim

At least one of MultiHiertt, MathQA, or HiTab contains enough native,
gold-annotated **operation-dependency DAGs** to support a future experiment in
which a hierarchy-conditioned reader selects one local intervention.

Priority is descriptive only: MultiHiertt, then MathQA, then HiTab.  All three
are audited in one run under identical graph definitions.

## 2. What counts as a graph

Each executable operation is a node.  There is a directed edge `u -> v` only
when operation `v` consumes the intermediate result produced by operation
`u`.  Raw numbers, constants, cells, ranges, evidence facts, headers, and table
hierarchy nodes are leaves outside the operation graph.

Consequences:

- data hierarchy is not counted as reasoning hierarchy;
- a long straight-line program may be deep but has no internal join or reuse;
- an operation that combines one intermediate result and one raw number has
  operation indegree one, not two;
- disconnected/dead operations are retained and reported;
- HiTab's searched `saved_programs.json` is excluded because it is weakly
  supervised and may be spurious.  Only annotated `answer_formulas` are used.
- A HiTab row with multiple answer formulas has no single annotated executable
  root and is reported as unsupported rather than choosing the richest formula
  after inspection.

## 3. Metrics

Edges point from producer to consumer.

- `nodes`, `edges`, `max_depth`;
- `join_nodes`: operation nodes with indegree at least two;
- `reuse_nodes`: operation nodes with outdegree at least two;
- `deep`: at least three operation nodes and depth at least three;
- `join`: at least one join node;
- `reuse`: at least one reuse node;
- `deep_join`, `join_reuse`, and `deep_join_reuse`;
- `diamond`: distinct immediate nodes `u,v1,v2,r` with
  `u->v1`, `u->v2`, `v1->r`, and `v2->r`;
- `reconvergence`: two distinct children of one operation have any common
  downstream operation, a less brittle superset of immediate diamonds;
- `root_connected_nodes` and `connected_internal_nodes`: operations on the
  ancestry of the final operation, with the latter excluding the root;
- `dead_nodes`: parsed operations outside the final-operation ancestry.

Report distributions and conditional rates, especially
`P(join | deep)` and `P(reuse | deep_join)`.

## 4. Sources and splits

- MultiHiertt: official `psunlpgroup/MultiHiertt` repository plus its official
  linked Google Drive dataset.
- MathQA: the official `https://math-qa.github.io/math-QA/data/MathQA.zip`.
- HiTab: official `microsoft/HiTab` repository.

Record repository commits and SHA-256 hashes of every audited annotation file.
For each dataset use public test as held-out if it has program-bearing rows and
at least 95% parse coverage; otherwise use dev.  Empty/non-program answers
remain in annotation-coverage denominators but not parse-coverage denominators.

## 5. Frozen gate

A dataset passes structural P0 only if all are true:

1. held-out program-bearing examples at least 500;
2. held-out parse coverage at least 95%;
3. held-out invalid/forward-reference rate at most 1%;
4. held-out `deep >= 150`;
5. held-out `join >= 100`;
6. held-out `deep_join >= 100`;
7. held-out `reuse >= 50`;
8. held-out `deep_join_reuse >= 30`;
9. train `deep_join_reuse >= 100`;
10. median connected internal nodes within `deep_join_reuse` is at least two.

Diamond and reconvergence counts are diagnostic, not hard gates.  They may be
rare even in a useful non-chain DAG, whereas reuse plus a join already creates
multiple downstream paths somewhere in the graph.

## 6. Stop and progression rules

- Dataset-name hierarchy, table-header hierarchy, long formulas, or large
  operation vocabulary cannot compensate for a structural gate failure.
- A passing structure audit licenses an independent executor audit only.  Gold
  execution agreement must later reach at least 95% before any intervention
  oracle or model inference.
- If multiple datasets pass, prefer the one with higher `deep_join_reuse`
  coverage and cleaner native intermediate references, not the easiest code.
- If none pass, stop numerical-reasoning benchmark hopping.  The evidence would
  indicate that the required executable DAG intervention substrate is rare in
  these standard datasets; do not synthesize one and present it as native.
