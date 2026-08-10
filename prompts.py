"""Prompt templates for the DCH-HSGR pilot and the depth-3 (S2) pipeline."""

DECOMPOSE_SYSTEM = (
    "You are an expert at planning solutions to math problems by breaking them "
    "into independent subquestions."
)

DECOMPOSE_USER = """Problem:
{problem}

Break this problem into 2 or 3 subquestions that satisfy ALL of these rules:
1. Each subquestion must be SELF-CONTAINED: answerable using only the original problem statement, without knowing the answer to any other subquestion.
2. Each subquestion must have a single well-defined answer (a number or a short closed-form expression).
3. Knowing the answers to all subquestions should make the final answer easy to derive.
4. Do NOT include the original question itself as a subquestion.

Respond with ONLY a JSON object, no other text:
{{"subquestions": ["...", "..."]}}"""

# S2: split an existing subquestion one level further (root -> L1 -> L2)
DEEP_DECOMPOSE_USER = """Original problem (for context only):
{problem}

Your task is to break down this SUBQUESTION of the problem:
{subquestion}

Split the subquestion into 2 or 3 smaller steps that satisfy ALL of these rules:
1. Each step must be SELF-CONTAINED: answerable from the original problem statement alone, without knowing the answers to the other steps.
2. Each step must have a single well-defined answer (a number or a short closed-form expression).
3. Knowing the answers to all steps should make the subquestion easy to answer.
4. Do NOT restate the subquestion itself as one of the steps.
5. If the subquestion is already a single atomic computation, respond with {{"subquestions": []}}.

Respond with ONLY a JSON object, no other text:
{{"subquestions": ["...", "..."]}}"""

SUBQ_USER = """Consider this math problem as context:
{problem}

Now answer ONLY the following subquestion (not the original problem):
{subquestion}

Reason briefly (at most 120 words), then give the subquestion's answer as \\boxed{{...}}."""

ROOT_COT_USER = """Solve this math problem step by step.

{problem}

End your solution with the final answer as \\boxed{{...}}."""

AGGREGATE_USER = """Problem:
{problem}

A colleague has already worked out these intermediate results:
{facts}

Treat the intermediate results as given. Using them, derive the final answer to the problem with brief reasoning (at most 120 words). End with the final answer as \\boxed{{...}}."""

# S2: aggregate child answers into the value of an intermediate (L1) node
MID_AGGREGATE_USER = """Original problem (for context only):
{problem}

You must answer this SUBQUESTION of the problem:
{subquestion}

A colleague has already worked out these smaller steps:
{facts}

Treat those step results as given. Using them, answer the SUBQUESTION (not the original problem) with brief reasoning (at most 100 words). End with the subquestion's answer as \\boxed{{...}}."""

COMPAT_USER = """Problem:
{problem}

Proposed intermediate results:
{facts}

Proposed final answer: {final}

Evaluate this solution sketch:
- Is each intermediate result actually correct for the problem?
- Are the intermediate results consistent with each other?
- Does the final answer correctly follow from the problem and the intermediate results?

Respond with ONLY a single integer from 0 (definitely wrong/inconsistent) to 10 (definitely correct and consistent). No other text."""

VERIFY_USER = """Problem:
{problem}

Proposed intermediate results:
{facts}

Proposed final answer: {final}

Independently verify this solution sketch. Be skeptical: recompute each intermediate result yourself from the problem statement instead of trusting it, then check whether the final answer follows. Keep your check under 120 words.

After checking, output exactly one final line:
VERDICT: VALID
or
VERDICT: INVALID"""

SCORER_USER = """Problem:
{problem}

Proposed intermediate results:
{facts}

Proposed final answer: {final}

Is this solution sketch correct and internally consistent? Answer VALID or INVALID."""

# E0: structure-conditioned local execution (oracle hierarchy MVP serialization)
ORACLE_NODE_SYSTEM = (
    "You are solving one intermediate step of a math problem. "
    "Obey the structural state: use DEPENDS_ON values when provided, "
    "and answer ONLY the current GOAL."
)

ORACLE_NODE_USER = """Structural state for the current reasoning node:

{structure}

Reason briefly (at most 80 words) about the CURRENT goal only, then give that
intermediate quantity as \\boxed{{...}}. Do not solve the full original problem
unless the current goal is the final answer."""
