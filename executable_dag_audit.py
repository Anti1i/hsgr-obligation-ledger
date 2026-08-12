"""Unified zero-model audit for operation-dependency DAG annotations."""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import itertools
import json
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


PROTOCOL = "EXPERIMENT_PROTOCOL_EXECUTABLE_DAG_BENCHMARK_AUDIT_P0.md"
REF_RE = re.compile(r"^\$?\\?#(\d+)\$?$")
NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


@dataclass
class ProgramGraph:
    nodes: List[str]
    edges: Set[Tuple[int, int]]
    valid_references: bool = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_top_level(text: str, delimiter: str = ",") -> List[str]:
    parts: List[str] = []
    start = 0
    depth = 0
    quote: Optional[str] = None
    for index, char in enumerate(text):
        if quote:
            if char == quote and (index == 0 or text[index - 1] != "\\"):
                quote = None
            continue
        if char in "'\"":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced closing parenthesis")
        elif char == delimiter and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    if quote or depth != 0:
        raise ValueError("unbalanced expression")
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def parse_call(text: str) -> Tuple[str, List[str]]:
    text = text.strip()
    match = NAME_RE.match(text)
    if not match:
        raise ValueError("missing operation name")
    name = match.group(0)
    cursor = match.end()
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text) or text[cursor] != "(":
        raise ValueError("missing opening parenthesis")
    depth = 0
    close = None
    for index in range(cursor, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                close = index
                break
    if close is None or text[close + 1:].strip():
        raise ValueError("malformed operation call")
    inside = text[cursor + 1:close]
    return name.lower(), split_top_level(inside) if inside.strip() else []


def scan_flat_calls(text: str) -> List[str]:
    """Read top-level op(...) calls separated by `|`, `,`, or whitespace."""
    text = text.strip().rstrip("|").strip()
    calls: List[str] = []
    cursor = 0
    while cursor < len(text):
        while cursor < len(text) and (text[cursor].isspace() or text[cursor] in "|,"):
            cursor += 1
        if cursor >= len(text):
            break
        match = NAME_RE.match(text, cursor)
        if not match:
            raise ValueError("unexpected token at %d" % cursor)
        open_index = text.find("(", match.end())
        if open_index < 0:
            raise ValueError("missing operation parenthesis")
        depth = 0
        close = None
        for index in range(open_index, len(text)):
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
                if depth == 0:
                    close = index
                    break
        if close is None:
            raise ValueError("unclosed operation")
        calls.append(text[cursor:close + 1].strip())
        cursor = close + 1
    if not calls:
        raise ValueError("no operations")
    return calls


def parse_straight_line(text: str) -> ProgramGraph:
    calls = scan_flat_calls(text)
    nodes: List[str] = []
    edges: Set[Tuple[int, int]] = set()
    valid = True
    for index, call in enumerate(calls):
        name, args = parse_call(call)
        nodes.append(name)
        for arg in args:
            normalized = re.sub(r"\s+", "", arg)
            match = REF_RE.fullmatch(normalized)
            if not match:
                continue
            source = int(match.group(1))
            if source < 0 or source >= index:
                valid = False
            else:
                edges.add((source, index))
    return ProgramGraph(nodes, edges, valid)


EXCEL_TOKEN_RE = re.compile(
    r"\s*(?:"
    r"(?P<string>\"(?:[^\"]|\"\")*\")|"
    r"(?P<ref>(?:'[^']+'|[A-Za-z_][A-Za-z0-9_.]*)?!?\$?[A-Za-z]{1,3}\$?\d+(?::\$?[A-Za-z]{1,3}\$?\d+)?)|"
    r"(?P<number>\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)|"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_.]*)|"
    r"(?P<op><=|>=|<>|[+\-*/^%=<>,():])"
    r")"
)


class ExcelParser:
    PRECEDENCE = {"=": 1, "<>": 1, "<": 1, ">": 1, "<=": 1, ">=": 1,
                  "+": 2, "-": 2, "*": 3, "/": 3, "^": 4}

    def __init__(self, formula: str):
        text = formula.strip()
        if text.startswith("="):
            text = text[1:]
        self.tokens: List[Tuple[str, str]] = []
        cursor = 0
        while cursor < len(text):
            match = EXCEL_TOKEN_RE.match(text, cursor)
            if not match:
                raise ValueError("unsupported Excel token at %d" % cursor)
            kind = next(key for key, value in match.groupdict().items() if value is not None)
            self.tokens.append((kind, match.group(kind)))
            cursor = match.end()
        self.pos = 0
        self.nodes: List[str] = []
        self.edges: Set[Tuple[int, int]] = set()

    def peek(self, value: Optional[str] = None) -> bool:
        if self.pos >= len(self.tokens):
            return False
        return value is None or self.tokens[self.pos][1] == value

    def take(self) -> Tuple[str, str]:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def add_op(self, name: str, children: Sequence[Optional[int]]) -> int:
        index = len(self.nodes)
        self.nodes.append(name.lower())
        for child in children:
            if child is not None:
                self.edges.add((child, index))
        return index

    def primary(self) -> Optional[int]:
        if self.peek("("):
            self.take()
            node = self.expression(0)
            if not self.peek(")"):
                raise ValueError("missing closing parenthesis")
            self.take()
            return node
        kind, value = self.take()
        if kind == "op" and value in "+-":
            return self.add_op("unary" + value, [self.primary()])
        if kind == "name" and self.peek("("):
            self.take()
            args: List[Optional[int]] = []
            if not self.peek(")"):
                while True:
                    args.append(self.expression(0))
                    if not self.peek(","):
                        break
                    self.take()
            if not self.peek(")"):
                raise ValueError("missing function parenthesis")
            self.take()
            return self.add_op(value, args)
        if kind in {"number", "ref", "string", "name"}:
            return None
        raise ValueError("unexpected Excel primary %s" % value)

    def expression(self, min_precedence: int) -> Optional[int]:
        left = self.primary()
        while self.pos < len(self.tokens):
            kind, value = self.tokens[self.pos]
            if kind != "op" or value not in self.PRECEDENCE:
                break
            precedence = self.PRECEDENCE[value]
            if precedence < min_precedence:
                break
            self.take()
            right = self.expression(precedence + (0 if value == "^" else 1))
            left = self.add_op(value, [left, right])
        if self.peek("%"):
            self.take()
            left = self.add_op("percent", [left])
        return left

    def parse(self) -> ProgramGraph:
        root = self.expression(0)
        if self.pos != len(self.tokens):
            raise ValueError("unconsumed Excel tokens")
        if root is None and not self.nodes:
            raise ValueError("formula has no operations")
        return ProgramGraph(self.nodes, self.edges, True)


def parse_excel_formula(text: str) -> ProgramGraph:
    return ExcelParser(text).parse()


def graph_metrics(graph: ProgramGraph) -> Dict[str, Any]:
    n = len(graph.nodes)
    parents: List[Set[int]] = [set() for _ in range(n)]
    children: List[Set[int]] = [set() for _ in range(n)]
    for source, target in graph.edges:
        if not (0 <= source < target < n):
            graph.valid_references = False
            continue
        parents[target].add(source)
        children[source].add(target)
    depths: List[int] = []
    for index in range(n):
        depths.append(1 + max((depths[parent] for parent in parents[index]), default=0))
    root = n - 1
    ancestry: Set[int] = set()
    stack = [root] if n else []
    while stack:
        node = stack.pop()
        if node in ancestry:
            continue
        ancestry.add(node)
        stack.extend(parents[node])
    join_nodes = sum(len(item) >= 2 for item in parents)
    reuse_nodes = sum(len(item) >= 2 for item in children)
    diamond = False
    reconvergence = False
    descendants: List[Set[int]] = [set() for _ in range(n)]
    for node in range(n - 1, -1, -1):
        for child in children[node]:
            descendants[node].add(child)
            descendants[node].update(descendants[child])
    for source in range(n):
        branch_children = sorted(children[source])
        for left, right in itertools.combinations(branch_children, 2):
            if children[left].intersection(children[right]):
                diamond = True
            if descendants[left].intersection(descendants[right]):
                reconvergence = True
    deep = n >= 3 and max(depths, default=0) >= 3
    join = join_nodes > 0
    reuse = reuse_nodes > 0
    return {
        "nodes": n,
        "edges": len(graph.edges),
        "max_depth": max(depths, default=0),
        "join_nodes": join_nodes,
        "reuse_nodes": reuse_nodes,
        "deep": deep,
        "join": join,
        "reuse": reuse,
        "deep_join": deep and join,
        "join_reuse": join and reuse,
        "deep_join_reuse": deep and join and reuse,
        "diamond": diamond,
        "reconvergence": reconvergence,
        "root_connected_nodes": len(ancestry),
        "connected_internal_nodes": max(0, len(ancestry) - 1),
        "dead_nodes": n - len(ancestry),
        "valid_references": graph.valid_references,
    }


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError("expected list dataset: %s" % path)
    return value


def choose_split_files(dataset: str, root: Path, names: Sequence[str]) -> Dict[str, Path]:
    chosen: Dict[str, Path] = {}
    for split in names:
        if dataset == "hitab":
            candidates = list(root.rglob("%s_samples.jsonl" % split))
        else:
            candidates = [
                path for path in root.rglob("%s.json" % split)
                if "reasoning_module_input" not in str(path)
                and "output" not in str(path)
            ]
        if candidates:
            chosen[split] = sorted(candidates, key=lambda p: (len(p.parts), str(p)))[0]
    return chosen


def formula_values(dataset: str, row: Dict[str, Any]) -> List[str]:
    if dataset == "multihiertt":
        value = row.get("qa", {}).get("program")
        return [value] if isinstance(value, str) and value.strip() else []
    if dataset == "mathqa":
        value = row.get("linear_formula")
        return [value] if isinstance(value, str) and value.strip() else []
    values = row.get("answer_formulas")
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, str) and value.strip()]


def audit_split(dataset: str, path: Path) -> Dict[str, Any]:
    rows = read_json_or_jsonl(path)
    cases: List[Dict[str, Any]] = []
    parse_errors: List[Dict[str, str]] = []
    program_bearing = 0
    multi_formula_rows = 0
    for position, row in enumerate(rows):
        values = formula_values(dataset, row)
        if not values:
            continue
        program_bearing += 1
        if len(values) > 1:
            multi_formula_rows += 1
            uid = str(row.get("uid", row.get("id", position)))
            parse_errors.append({
                "id": uid,
                "error": "multiple answer formulas have no single annotated root",
            })
            continue
        parsed_options: List[Tuple[Dict[str, Any], str]] = []
        option_errors: List[str] = []
        for value in values:
            try:
                graph = parse_excel_formula(value) if dataset == "hitab" else parse_straight_line(value)
                parsed_options.append((graph_metrics(graph), value))
            except Exception as exc:
                option_errors.append("%s: %s" % (type(exc).__name__, exc))
        if not parsed_options:
            uid = str(row.get("uid", row.get("id", position)))
            parse_errors.append({"id": uid, "error": " | ".join(option_errors)[:1000]})
            continue
        metrics, selected = parsed_options[0]
        cases.append({
            "id": str(row.get("uid", row.get("id", position))),
            "position": position,
            **metrics,
        })
    return summarize_cases(rows, program_bearing, multi_formula_rows, cases, parse_errors, path)


def histogram(cases: Sequence[Dict[str, Any]], field: str) -> Dict[str, int]:
    return {
        str(key): value for key, value in sorted(
            collections.Counter(int(case[field]) for case in cases).items()
        )
    }


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_cases(
    rows: Sequence[Dict[str, Any]], program_bearing: int, multi_formula_rows: int,
    cases: List[Dict[str, Any]], parse_errors: List[Dict[str, str]], path: Path,
) -> Dict[str, Any]:
    counts = {
        key: sum(bool(case[key]) for case in cases)
        for key in (
            "deep", "join", "reuse", "deep_join", "join_reuse",
            "deep_join_reuse", "diamond", "reconvergence",
        )
    }
    invalid_refs = sum(not case["valid_references"] for case in cases)
    target_cases = [case for case in cases if case["deep_join_reuse"]]
    connected = [case["connected_internal_nodes"] for case in target_cases]
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "program_bearing": program_bearing,
        "annotation_coverage": ratio(program_bearing, len(rows)),
        "parsed": len(cases),
        "parse_coverage": ratio(len(cases), program_bearing),
        "invalid_references": invalid_refs,
        "invalid_reference_rate": ratio(invalid_refs, len(cases)),
        "multi_formula_rows": multi_formula_rows,
        "counts": counts,
        "rates": {key: ratio(value, len(cases)) for key, value in counts.items()},
        "p_join_given_deep": ratio(counts["deep_join"], counts["deep"]),
        "p_reuse_given_deep_join": ratio(counts["deep_join_reuse"], counts["deep_join"]),
        "target_median_connected_internal": statistics.median(connected) if connected else 0.0,
        "node_histogram": histogram(cases, "nodes"),
        "depth_histogram": histogram(cases, "max_depth"),
        "join_node_histogram": histogram(cases, "join_nodes"),
        "reuse_node_histogram": histogram(cases, "reuse_nodes"),
        "dead_node_histogram": histogram(cases, "dead_nodes"),
        "parse_errors": parse_errors[:100],
        "cases": cases,
    }


def select_heldout(splits: Dict[str, Dict[str, Any]]) -> str:
    test = splits.get("test")
    if test and test["program_bearing"] > 0 and test["parse_coverage"] >= 0.95:
        return "test"
    if "dev" in splits:
        return "dev"
    raise ValueError("no usable test or dev split")


def gate_dataset(splits: Dict[str, Dict[str, Any]], heldout_name: str) -> Dict[str, bool]:
    heldout = splits[heldout_name]
    train = splits.get("train", {"counts": {"deep_join_reuse": 0}})
    counts = heldout["counts"]
    return {
        "heldout_program_bearing_ge_500": heldout["program_bearing"] >= 500,
        "heldout_parse_ge_095": heldout["parse_coverage"] >= 0.95,
        "heldout_invalid_refs_le_001": heldout["invalid_reference_rate"] <= 0.01,
        "heldout_deep_ge_150": counts["deep"] >= 150,
        "heldout_join_ge_100": counts["join"] >= 100,
        "heldout_deep_join_ge_100": counts["deep_join"] >= 100,
        "heldout_reuse_ge_50": counts["reuse"] >= 50,
        "heldout_deep_join_reuse_ge_30": counts["deep_join_reuse"] >= 30,
        "train_deep_join_reuse_ge_100": train["counts"]["deep_join_reuse"] >= 100,
        "target_median_connected_internal_ge_2": heldout["target_median_connected_internal"] >= 2,
    }


def audit_dataset(dataset: str, root: Path) -> Dict[str, Any]:
    files = choose_split_files(dataset, root, ("train", "dev", "test"))
    if not files:
        raise FileNotFoundError("no dataset split files under %s" % root)
    splits = {split: audit_split(dataset, path) for split, path in files.items()}
    heldout = select_heldout(splits)
    checks = gate_dataset(splits, heldout)
    return {
        "root": str(root.resolve()),
        "files": {key: str(value.resolve()) for key, value in files.items()},
        "heldout": heldout,
        "splits": splits,
        "checks": checks,
        "gate_pass": all(checks.values()),
    }


def print_report(report: Dict[str, Any]) -> None:
    for dataset, result in report["datasets"].items():
        print("DATASET=%s heldout=%s PASS=%s" % (dataset, result["heldout"], result["gate_pass"]))
        for split, metrics in result["splits"].items():
            count = metrics["counts"]
            print(
                "%s/%s rows=%d programs=%d parsed=%d parse=%.4f deep=%d join=%d "
                "reuse=%d deep_join=%d deep_join_reuse=%d diamond=%d reconv=%d "
                "Pjoin|deep=%.4f Preuse|deep_join=%.4f"
                % (
                    dataset, split, metrics["rows"], metrics["program_bearing"],
                    metrics["parsed"], metrics["parse_coverage"], count["deep"],
                    count["join"], count["reuse"], count["deep_join"],
                    count["deep_join_reuse"], count["diamond"], count["reconvergence"],
                    metrics["p_join_given_deep"], metrics["p_reuse_given_deep_join"],
                )
            )
        print("checks=%s" % json.dumps(result["checks"], sort_keys=True))
    print("PASSING_DATASETS=%s" % ",".join(report["passing_datasets"]))


def self_test() -> None:
    chain = parse_straight_line("add(1,2)|multiply(#0,3)|subtract(#1,4)|")
    chain_m = graph_metrics(chain)
    assert chain_m["deep"] and not chain_m["join"] and not chain_m["reuse"]
    diamond = parse_straight_line(
        "add(1,2), multiply(#0,3), subtract(#0,4), add(#1,#2)"
    )
    dm = graph_metrics(diamond)
    assert dm["deep_join_reuse"] and dm["diamond"] and dm["reconvergence"]
    tree = parse_excel_formula("=(A1+B1)/(C1-D1)")
    tm = graph_metrics(tree)
    assert tm["join"] and not tm["reuse"] and tm["nodes"] == 3
    function = parse_excel_formula("=SUM(A1:A3)+AVERAGE(B1:B3)")
    fm = graph_metrics(function)
    assert fm["nodes"] == 3 and fm["join"]
    print("SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multihiertt-root")
    parser.add_argument("--mathqa-root")
    parser.add_argument("--hitab-root")
    parser.add_argument("--source-meta", default="")
    parser.add_argument("--out", default="executable_dag_audit_report.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    roots = {
        "multihiertt": args.multihiertt_root,
        "mathqa": args.mathqa_root,
        "hitab": args.hitab_root,
    }
    if not all(roots.values()):
        raise ValueError("all three dataset roots are required")
    source_meta: Dict[str, Any] = {}
    if args.source_meta:
        with open(args.source_meta, encoding="utf-8") as handle:
            source_meta = json.load(handle)
    datasets = {
        name: audit_dataset(name, Path(root)) for name, root in roots.items()
    }
    report = {
        "protocol": PROTOCOL,
        "source_meta": source_meta,
        "datasets": datasets,
        "passing_datasets": [name for name, value in datasets.items() if value["gate_pass"]],
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    print_report(report)


if __name__ == "__main__":
    main()
