"""FinQA program parsing, execution, and dependency-structure utilities."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


NUMERIC_OPS = (
    "add", "subtract", "multiply", "divide", "exp", "greater",
)
TABLE_OPS = ("table_max", "table_min", "table_sum", "table_average")
ALL_OPS = NUMERIC_OPS + TABLE_OPS
COMMUTATIVE_OPS = {"add", "multiply"}
_STEP_RE = re.compile(
    r"\b(" + "|".join(sorted(ALL_OPS, key=len, reverse=True))
    + r")\s*\((.*?)\)",
    re.IGNORECASE | re.DOTALL,
)
_REF_RE = re.compile(r"^#(\d+)$")


@dataclass(frozen=True)
class Step:
    op: str
    args: Tuple[str, str]


@dataclass(frozen=True)
class Execution:
    valid: bool
    value: Any
    error: Optional[str]
    intermediates: Tuple[Any, ...]


def _clean_arg(value: str) -> str:
    value = value.strip().strip("`\"'")
    return re.sub(r"\s+", " ", value)


def parse_program(text: Any) -> List[Step]:
    """Parse the flat FinQA `op(arg1, arg2), ...` representation."""
    if isinstance(text, list):
        tokens = [str(piece) for piece in text if piece != "EOF"]
        if tokens and len(tokens) % 4 == 0:
            token_steps: List[Step] = []
            for index in range(0, len(tokens), 4):
                op_token, left, right, close = tokens[index:index + 4]
                op = op_token.strip().rstrip("(").strip().lower()
                if op not in ALL_OPS or close.strip() != ")":
                    break
                token_steps.append(Step(op, (_clean_arg(left), _clean_arg(right))))
            else:
                return token_steps
        text = " ".join(tokens)
    if not isinstance(text, str):
        raise ValueError("program must be text or a token list")
    matches = list(_STEP_RE.finditer(text))
    if not matches:
        raise ValueError("no FinQA operations found")
    steps: List[Step] = []
    for match in matches:
        op = match.group(1).lower()
        inside = match.group(2).strip()
        if "," not in inside:
            raise ValueError("operation does not have two arguments: %s" % inside)
        # Official table-row names may contain punctuation.  The second table
        # argument is the final dummy argument, so splitting from the right is
        # the least destructive deterministic rule.
        left, right = inside.rsplit(",", 1)
        args = (_clean_arg(left), _clean_arg(right))
        if not all(args):
            raise ValueError("empty operation argument")
        steps.append(Step(op=op, args=args))
    return steps


def canonical_program(steps: Sequence[Step]) -> str:
    return ", ".join(
        "%s(%s, %s)" % (step.op, step.args[0], step.args[1])
        for step in steps
    )


def parse_number(text: Any) -> float:
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text)
    value = str(text).strip().replace("$", "").replace(",", "")
    if value.lower().startswith("const_"):
        value = value[6:]
        if value.lower().startswith("m"):
            value = "-" + value[1:]
    negative_parentheses = value.startswith("(") and value.endswith(")")
    if negative_parentheses:
        value = "-" + value[1:-1]
    percent = value.endswith("%")
    if percent:
        value = value[:-1]
    result = float(value)
    return result / 100.0 if percent else result


def _table_numbers(row: Sequence[Any]) -> List[float]:
    numbers: List[float] = []
    for cell in row:
        cell_text = str(cell).strip()
        # Match the official executor: ignore parenthetical annotations after
        # the primary value, such as `12.3 (unaudited)`.
        cell_text = cell_text.split("(", 1)[0].strip()
        numbers.append(parse_number(cell_text))
    return numbers


def execute_program(steps: Sequence[Step], table: Sequence[Sequence[Any]]) -> Execution:
    table_map: Dict[str, Sequence[Any]] = {}
    table_map_folded: Dict[str, Sequence[Any]] = {}
    for row in table:
        if not row:
            continue
        key = str(row[0]).strip()
        table_map[key] = row[1:]
        table_map_folded[key.casefold()] = row[1:]
    results: List[Any] = []

    def resolve(arg: str, index: int) -> Any:
        ref = _REF_RE.fullmatch(arg)
        if ref:
            target = int(ref.group(1))
            if target >= index:
                raise ValueError("forward or self reference #%d at step %d" % (target, index))
            return results[target]
        return parse_number(arg)

    try:
        if not steps:
            raise ValueError("empty program")
        for index, step in enumerate(steps):
            if step.op not in ALL_OPS:
                raise ValueError("unknown operation %s" % step.op)
            if step.op in TABLE_OPS:
                row = table_map.get(step.args[0])
                if row is None:
                    row = table_map_folded.get(step.args[0].casefold())
                if row is None:
                    raise ValueError("unknown table row %s" % step.args[0])
                values = _table_numbers(row)
                if not values:
                    raise ValueError("empty numeric table row")
                if step.op == "table_max":
                    value = max(values)
                elif step.op == "table_min":
                    value = min(values)
                elif step.op == "table_sum":
                    value = sum(values)
                else:
                    value = sum(values) / len(values)
            else:
                left = resolve(step.args[0], index)
                right = resolve(step.args[1], index)
                if step.op == "add":
                    value = left + right
                elif step.op == "subtract":
                    value = left - right
                elif step.op == "multiply":
                    value = left * right
                elif step.op == "divide":
                    value = left / right
                elif step.op == "exp":
                    value = left ** right
                else:
                    value = "yes" if left > right else "no"
            if isinstance(value, complex):
                raise ValueError("complex result")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("non-finite result")
            results.append(value)
        final = results[-1]
        if isinstance(final, (int, float)) and not isinstance(final, bool):
            final = round(float(final), 5)
        return Execution(True, final, None, tuple(results))
    except Exception as exc:  # Invalid candidates are data, not fatal errors.
        return Execution(False, None, "%s: %s" % (type(exc).__name__, exc), tuple(results))


def execution_matches(value: Any, gold: Any) -> bool:
    if isinstance(value, str) or isinstance(gold, str):
        left, right = str(value).strip().casefold(), str(gold).strip().casefold()
        if left == right:
            return True
    try:
        left_num, right_num = parse_number(value), parse_number(gold)
        return round(left_num, 5) == round(right_num, 5)
    except (TypeError, ValueError, OverflowError):
        return False


def reference_indices(step: Step) -> List[int]:
    refs: List[int] = []
    for arg in step.args:
        match = _REF_RE.fullmatch(arg)
        if match:
            refs.append(int(match.group(1)))
    return refs


def structure_metrics(steps: Sequence[Step]) -> Dict[str, Any]:
    depths: List[int] = []
    out_degree = [0 for _ in steps]
    edge_count = 0
    two_ref_join = False
    valid_refs = True
    for index, step in enumerate(steps):
        refs = reference_indices(step)
        if any(ref >= index for ref in refs):
            valid_refs = False
        legal_refs = [ref for ref in refs if 0 <= ref < index]
        for ref in legal_refs:
            out_degree[ref] += 1
            edge_count += 1
        parent_depth = max((depths[ref] for ref in legal_refs), default=0)
        depths.append(parent_depth + 1)
        if len(set(legal_refs)) >= 2:
            two_ref_join = True
    branch_nodes = sum(degree > 1 for degree in out_degree)
    return {
        "n_steps": len(steps),
        "depth": max(depths, default=0),
        "reference_edges": edge_count,
        "two_ref_join": two_ref_join,
        "branch_nodes": branch_nodes,
        "has_branch": branch_nodes > 0,
        "table_steps": sum(step.op in TABLE_OPS for step in steps),
        "numeric_steps": sum(step.op in NUMERIC_OPS for step in steps),
        "valid_references": valid_refs,
        "deep": len(steps) >= 3 and max(depths, default=0) >= 3,
        "join": two_ref_join,
        "deep_join": (
            len(steps) >= 3 and max(depths, default=0) >= 3 and two_ref_join
        ),
    }


def table_to_text(table: Sequence[Sequence[Any]]) -> str:
    return "\n".join(" | ".join(str(cell) for cell in row) for row in table)


def evidence_text(row: Dict[str, Any]) -> str:
    chunks: List[str] = []
    chunks.extend(str(item) for item in row.get("pre_text", []))
    chunks.append(table_to_text(row.get("table", [])))
    chunks.extend(str(item) for item in row.get("post_text", []))
    return "\n".join(chunks)


def iter_records(paths: Iterable[str]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    import json
    import os

    for path in paths:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        split = os.path.splitext(os.path.basename(path))[0]
        for row in data:
            yield split, row
