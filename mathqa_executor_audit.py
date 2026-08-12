"""P0-B: verify that MathQA's structural DAG target is natively executable."""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

from executable_dag_audit import graph_metrics, parse_call, parse_straight_line, scan_flat_calls


PROTOCOL = "EXPERIMENT_PROTOCOL_MATHQA_EXECUTOR_AUDIT_P0B.md"
TRAX_COMMIT = "220a62303ebf4ad18871aa5607b4dda2f064f2d2"
TRAX_SOURCE = (
    "https://raw.githubusercontent.com/google/trax/" + TRAX_COMMIT
    + "/trax/data/tf_inputs.py"
)
NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
)
REF_RE = re.compile(r"^\$?\\?#(\d+)\$?$")
INPUT_RE = re.compile(r"^n(\d+)$")


class ExecutionFailure(ValueError):
    pass


def problem_numbers(problem: str) -> List[float]:
    return [float(item.replace(",", "")) for item in NUMBER_RE.findall(problem)]


def constant_value(token: str) -> float:
    if token == "const_pi":
        return math.pi
    if token == "const_deg_to_rad":
        return math.pi / 180.0
    if not token.startswith("const_"):
        raise ExecutionFailure("unsupported argument: " + token)
    body = token[len("const_"):]
    sign = -1.0 if body.startswith("-") else 1.0
    body = body.lstrip("+-")
    parts = body.split("_")
    try:
        if len(parts) == 1:
            return sign * float(parts[0])
        if len(parts) == 2:
            return sign * (float(parts[0]) + float("0." + parts[1]))
    except ValueError as exc:
        raise ExecutionFailure("bad constant: " + token) from exc
    raise ExecutionFailure("bad constant: " + token)


def safe_divide(left: float, right: float) -> float:
    return left / right if right != 0 else 0.0


def execute_operation(name: str, a: Sequence[float]) -> float:
    """Independent numeric implementation of the fixed Trax MathQA semantics."""
    binary: Dict[str, Callable[[float, float], float]] = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": safe_divide,
        "speed": safe_divide,
        "max": max,
        "reminder": lambda x, y: x % y,
        "rectangle_area": lambda x, y: x * y,
        "rectangle_perimeter": lambda x, y: 2 * (x + y),
        "rhombus_area": lambda x, y: x * y / 2,
        "triangle_area": lambda x, y: x * y / 2,
        "diagonal": lambda x, y: math.sqrt(x * x + y * y),
        "stream_speed": lambda x, y: (x - y) / 2,
        "speed_in_still_water": lambda x, y: (x + y) / 2,
        "speed_ratio_steel_to_stream": lambda x, y: safe_divide(x + y, x - y),
        "percent": lambda x, y: x / 100 * y,
        "price_after_gain": lambda x, y: (1 + x / 100) * y,
        "p_after_gain": lambda x, y: (1 + x / 100) * y,
        "price_after_loss": lambda x, y: (1 - x / 100) * y,
    }
    unary: Dict[str, Callable[[float], float]] = {
        "floor": math.floor,
        "sqrt": lambda x: math.sqrt(max(0.0, x)),
        "inverse": lambda x: safe_divide(1.0, x),
        "negate": lambda x: -x,
        "log": lambda x: math.log(max(1e-5, x), 2),
        "square_area": lambda x: x**2,
        "square_edge_by_area": lambda x: math.sqrt(x),
        "square_edge_by_perimeter": lambda x: x / 4,
        "cube_edge_by_volume": lambda x: x ** (1 / 3),
        "volume_cube": lambda x: x**3,
        "surface_cube": lambda x: 6 * x**2,
        "surface_sphere": lambda x: 4 * math.pi * x**2,
        "volume_sphere": lambda x: 4 / 3 * math.pi * x**3,
        "circle_area": lambda x: math.pi * x**2,
        "circumface": lambda x: 2 * math.pi * x,
        "semi_circle_perimiter": lambda x: math.pi * x + 2 * x,
        "square_perimeter": lambda x: 4 * x,
        "rhombus_perimeter": lambda x: 4 * x,
        "from_percent": lambda x: x / 100,
        "gain_percent": lambda x: 100 + x,
        "loss_percent": lambda x: 100 - x,
        "negate_percent": lambda x: 100 - x,
        "negate_prob": lambda x: 1 - x,
        "sine": math.sin,
        "cosine": math.cos,
        "factorial": lambda x: float(math.factorial(min(15, int(x)))),
    }
    if name in binary and len(a) == 2:
        return float(binary[name](a[0], a[1]))
    if name in unary and len(a) == 1:
        return float(unary[name](a[0]))
    if name == "power" and len(a) == 2:
        return float(a[0] ** min(a[1], 5))
    if name == "choose" and len(a) == 2:
        return float(math.comb(int(a[0]), int(a[1])))
    if name == "permutation" and len(a) == 2:
        low, high = int(min(a)), int(max(a))
        return float(math.factorial(high) / math.factorial(high - low))
    if name == "gcd" and len(a) == 2:
        return float(math.gcd(int(a[0]), int(a[1])))
    if name == "lcm" and len(a) == 2:
        return float(math.lcm(int(a[0]), int(a[1])))
    if name == "circle_arc" and len(a) == 2:
        return a[0] / 360 * math.pi * 2 * a[1]
    if name == "circle_sector_area" and len(a) == 2:
        return a[1] / 360 * math.pi * a[0]**2
    if name == "combined_work" and len(a) == 2:
        return 1 / (min(a[0], 1 / a[0]) + min(a[1], 1 / a[1]))
    if name == "find_work" and len(a) == 2:
        left, right = min(a[0], 1 / a[0]), min(a[1], 1 / a[1])
        return 1 / (max(left, right) - min(left, right))
    if name == "count_interval" and len(a) == 2:
        return a[0] - a[1] + 1
    if name == "original_price_before_loss" and len(a) == 2:
        return a[1] * 100 / (100 + 1e-5 - a[0])
    if name == "original_price_before_gain" and len(a) == 2:
        return a[1] * 100 / (100 + a[0])
    if name in {"quadrilateral_area", "trapezium_area"} and len(a) == 3:
        return a[0] * (a[1] + a[2]) / 2
    if name == "triangle_perimeter" and len(a) == 3:
        return sum(a)
    if name == "triangle_area_three_edges" and len(a) == 3:
        semi = sum(a) / 2
        return math.sqrt(max(0.0, semi * (semi-a[0]) * (semi-a[1]) * (semi-a[2])))
    if name == "union_prob" and len(a) == 3:
        return a[0] + a[1] - a[2]
    if name == "volume_cone" and len(a) == 2:
        return math.pi * a[0]**2 * a[1] / 3
    if name == "volume_cylinder" and len(a) == 2:
        return math.pi * a[0]**2 * a[1]
    if name == "volume_rectangular_prism" and len(a) == 3:
        return a[0] * a[1] * a[2]
    if name == "surface_rectangular_prism" and len(a) == 3:
        return 2 * (a[0]*a[1] + a[0]*a[2] + a[1]*a[2])
    raise ExecutionFailure("unsupported_op:" + name)


def resolve_argument(token: str, inputs: Sequence[float], results: Sequence[float]) -> float:
    normalized = re.sub(r"\s+", "", token)
    ref = REF_RE.fullmatch(normalized)
    if ref:
        index = int(ref.group(1))
        if index >= len(results):
            raise ExecutionFailure("invalid_ref")
        return results[index]
    inp = INPUT_RE.fullmatch(normalized)
    if inp:
        index = int(inp.group(1))
        if index >= len(inputs):
            raise ExecutionFailure("invalid_input")
        return inputs[index]
    if normalized.startswith("const_"):
        return constant_value(normalized)
    try:
        return float(normalized.replace(",", ""))
    except ValueError as exc:
        raise ExecutionFailure("unsupported_arg:" + normalized) from exc


def execute_formula(problem: str, formula: str) -> float:
    inputs = problem_numbers(problem)
    results: List[float] = []
    for call in scan_flat_calls(formula):
        name, args = parse_call(call)
        values = [resolve_argument(arg, inputs, results) for arg in args]
        value = execute_operation(name, values)
        if not math.isfinite(value):
            raise ExecutionFailure("non_finite")
        results.append(value)
    if not results:
        raise ExecutionFailure("empty_program")
    return results[-1]


def correct_option_value(options: str, correct: str) -> float:
    labels = list(re.finditer(r"(?i)(?<![a-z])([a-e])\s*\)", options))
    wanted = correct.strip().lower()
    for position, label in enumerate(labels):
        if label.group(1).lower() != wanted:
            continue
        start = label.end()
        end = labels[position + 1].start() if position + 1 < len(labels) else len(options)
        segment = options[start:end]
        segment = re.sub(r"(?<!\d)([-+])\s+(?=\d)", r"\1", segment)
        numbers = NUMBER_RE.findall(segment)
        if not numbers:
            raise ExecutionFailure("nonnumeric_option")
        numerator = float(numbers[0].replace(",", ""))
        first_end = segment.find(numbers[0]) + len(numbers[0])
        separator = segment[first_end:segment.find(numbers[1])] if len(numbers) >= 2 else ""
        if len(numbers) >= 2 and ("/" in separator or ":" in separator or "∶" in separator):
            denominator = float(numbers[1].replace(",", ""))
            return numerator / denominator
        return numerator
    raise ExecutionFailure("missing_correct_option")


def audit_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counters: collections.Counter[str] = collections.Counter()
    failures: collections.Counter[str] = collections.Counter()
    mismatches: List[Dict[str, Any]] = []
    target_mismatches: List[Dict[str, Any]] = []
    target_failures: List[Dict[str, Any]] = []
    for position, row in enumerate(rows):
        formula = row.get("linear_formula")
        if not isinstance(formula, str) or not formula.strip():
            continue
        counters["programs"] += 1
        try:
            metrics = graph_metrics(parse_straight_line(formula))
        except Exception as exc:
            failures["parse:" + type(exc).__name__] += 1
            continue
        target = bool(metrics["deep_join_reuse"])
        connected_target = target and metrics["dead_nodes"] == 0 and metrics["valid_references"]
        counters["target"] += int(target)
        counters["connected_target"] += int(connected_target)
        try:
            expected = correct_option_value(str(row.get("options", "")), str(row.get("correct", "")))
            counters["scorable"] += 1
            counters["target_scorable"] += int(connected_target)
            actual = execute_formula(str(row.get("Problem", "")), formula)
            counters["executed"] += 1
            counters["target_executed"] += int(connected_target)
            matched = math.isclose(actual, expected, rel_tol=0.01)
            counters["matched"] += int(matched)
            counters["target_matched"] += int(connected_target and matched)
            if not matched and len(mismatches) < 30:
                mismatches.append({"position": position, "actual": actual, "expected": expected})
            if connected_target and not matched and len(target_mismatches) < 30:
                target_mismatches.append(
                    {"position": position, "actual": actual, "expected": expected}
                )
        except Exception as exc:
            key = str(exc) or type(exc).__name__
            failures[key] += 1
            if connected_target and len(target_failures) < 30:
                target_failures.append({"position": position, "error": key})
    def rate(num: str, den: str) -> float:
        return counters[num] / counters[den] if counters[den] else 0.0
    return {
        "rows": len(rows),
        **dict(counters),
        "execution_coverage": rate("executed", "programs"),
        "answer_agreement": rate("matched", "executed"),
        "target_execution_coverage": rate("target_executed", "connected_target"),
        "target_answer_agreement": rate("target_matched", "target_executed"),
        "failures": dict(failures.most_common()),
        "mismatch_examples": mismatches,
        "target_mismatch_examples": target_mismatches,
        "target_failure_examples": target_failures,
    }


def load_split(root: Path, name: str) -> List[Dict[str, Any]]:
    candidates = sorted(root.rglob(name + ".json"), key=lambda p: (len(p.parts), str(p)))
    if not candidates:
        raise FileNotFoundError(name + ".json")
    with candidates[0].open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError("split is not a list")
    return value


def audit(root: Path) -> Dict[str, Any]:
    splits = {name: audit_rows(load_split(root, name)) for name in ("train", "dev", "test")}
    train, test = splits["train"], splits["test"]
    checks = {
        "train_connected_target_ge_500": train.get("connected_target", 0) >= 500,
        "test_connected_target_ge_100": test.get("connected_target", 0) >= 100,
        "test_target_execution_coverage_ge_095": test["target_execution_coverage"] >= 0.95,
        "test_target_answer_agreement_ge_095": test["target_answer_agreement"] >= 0.95,
        "train_target_execution_coverage_ge_095": train["target_execution_coverage"] >= 0.95,
        "train_target_answer_agreement_ge_095": train["target_answer_agreement"] >= 0.95,
    }
    return {
        "protocol": PROTOCOL,
        "trax_commit": TRAX_COMMIT,
        "trax_source": TRAX_SOURCE,
        "splits": splits,
        "checks": checks,
        "gate_pass": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="mathqa_executor_audit_report.json")
    args = parser.parse_args()
    report = audit(Path(args.root))
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    for split, metrics in report["splits"].items():
        print(
            f"{split} programs={metrics.get('programs', 0)} target={metrics.get('target', 0)} "
            f"connected_target={metrics.get('connected_target', 0)} "
            f"target_exec={metrics['target_execution_coverage']:.4f} "
            f"target_agree={metrics['target_answer_agreement']:.4f} "
            f"all_exec={metrics['execution_coverage']:.4f} "
            f"all_agree={metrics['answer_agreement']:.4f}"
        )
        print("failures=" + json.dumps(metrics["failures"], sort_keys=True))
    print("checks=" + json.dumps(report["checks"], sort_keys=True))
    print("GATE_PASS=" + str(report["gate_pass"]))


if __name__ == "__main__":
    main()
