"""Calibrate and confirm an outcome-independent structural-hardness subset.

The selector may inspect model outcomes on gsm_join_train only.  It then
applies the selected metadata rule once to the disjoint gsm_join_test split.
No hidden states are extracted and no deployable Guide is trained.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Callable, Optional

from answer_check import answers_equal, extract_boxed, normalize_answer
from hsgr_join_provenance_ceiling import (
    BASE_SYSTEM,
    PARENT_USER,
    ROOT_USER,
    bind_root,
    split_questions,
)
from pilot import JWriter, Runner, jread


HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"
DIRECT_K = 8
PARENT_K = 4
TEMPERATURE = 0.8

DIRECT_USER = """Solve this complete three-question dependency problem.
Resolve Questions 1 and 2, use their answers in Question 3, and end with the
answer to Question 3 in the last \\boxed{{...}}.

{problem}"""


def read_rows(path: str, limit: int = 0) -> list[dict]:
    rows = jread(path)
    if limit:
        rows = rows[:limit]
    out = []
    for idx, row in enumerate(rows):
        required = (
            "problem", "answer", "parent_answers", "parent_step_counts",
            "root_step_count", "n_steps", "graph",
        )
        if not all(row.get(key) is not None for key in required):
            raise ValueError(f"row {idx} misses one of {required}")
        if row["graph"].get("edges") != [
            ["parent_0", "root"], ["parent_1", "root"]
        ]:
            raise ValueError(f"row {idx} has an unexpected graph")
        item = dict(row)
        item["id"] = idx
        out.append(item)
    return out


def generated_tokens(runner: Runner, text: str) -> int:
    return len(runner.tok.encode(text, add_special_tokens=False))


def candidates(greedy: str, sampled: list[str]) -> list[dict]:
    out = []
    for kind, text in [("greedy", greedy)] + [("sample", x) for x in sampled]:
        answer = extract_boxed(text)
        out.append({
            "kind": kind,
            "text": text,
            "answer": answer,
            "norm": normalize_answer(answer),
        })
    return out


def modal_candidate(cands: list[dict]) -> dict:
    valid = [c for c in cands if c["norm"] is not None]
    if not valid:
        return cands[0]
    counts = Counter(c["norm"] for c in valid)
    greedy_norm = cands[0]["norm"]
    best_norm = max(
        counts,
        key=lambda value: (
            counts[value], value == greedy_norm,
            -next(i for i, c in enumerate(cands) if c["norm"] == value),
        ),
    )
    return next(c for c in cands if c["norm"] == best_norm)


def sc_candidate(cands: list[dict], k: int) -> dict:
    return modal_candidate(cands[:k])


def run_direct(
    runner: Runner, rows: list[dict], out_dir: str, batch_size: int
) -> dict[int, dict]:
    path = os.path.join(out_dir, "direct.jsonl")
    done = {int(row["id"]): row for row in jread(path)}
    todo = [row for row in rows if row["id"] not in done]
    if todo:
        users = [DIRECT_USER.format(problem=row["problem"]) for row in todo]
        greedy = runner.chat_batch(
            users, system=BASE_SYSTEM, max_new=512, bs=batch_size
        )
        sampled = runner.chat_batch(
            users, system=BASE_SYSTEM, max_new=512, temperature=TEMPERATURE,
            n=DIRECT_K - 1, bs=batch_size,
        )
        writer = JWriter(path)
        for row, g, ss in zip(todo, greedy, sampled):
            cs = candidates(g[0], ss)
            record = {
                "id": row["id"],
                "candidates": cs,
                "generated_tokens": sum(generated_tokens(runner, c["text"]) for c in cs),
            }
            writer.write(record)
            done[row["id"]] = record
    return done


def run_parents(
    runner: Runner, rows: list[dict], out_dir: str, batch_size: int
) -> dict[tuple[int, int], dict]:
    path = os.path.join(out_dir, "parents.jsonl")
    done = {(int(row["id"]), int(row["slot"])): row for row in jread(path)}
    units = []
    for row in rows:
        q0, q1, _ = split_questions(row["problem"])
        for slot, question in enumerate((q0, q1)):
            if (row["id"], slot) not in done:
                units.append((row, slot, PARENT_USER.format(question=question)))
    if units:
        users = [unit[2] for unit in units]
        greedy = runner.chat_batch(
            users, system=BASE_SYSTEM, max_new=192, bs=batch_size
        )
        sampled = runner.chat_batch(
            users, system=BASE_SYSTEM, max_new=192, temperature=TEMPERATURE,
            n=PARENT_K - 1, bs=batch_size,
        )
        writer = JWriter(path)
        for (row, slot, _), g, ss in zip(units, greedy, sampled):
            cs = candidates(g[0], ss)
            record = {
                "id": row["id"], "slot": slot, "candidates": cs,
                "generated_tokens": sum(generated_tokens(runner, c["text"]) for c in cs),
            }
            writer.write(record)
            done[(row["id"], slot)] = record
    return done


def run_roots(
    runner: Runner,
    rows: list[dict],
    parents: dict[tuple[int, int], dict],
    out_dir: str,
    batch_size: int,
) -> dict[tuple[int, str], dict]:
    path = os.path.join(out_dir, "roots.jsonl")
    done = {(int(row["id"]), row["arm"]): row for row in jread(path)}
    units = []
    for row in rows:
        _, _, root = split_questions(row["problem"])
        modal = [modal_candidate(parents[(row["id"], slot)]["candidates"])["answer"]
                 or "UNKNOWN" for slot in (0, 1)]
        bindings = {
            "modal": modal,
            "gold": [str(x) for x in row["parent_answers"]],
        }
        for arm, values in bindings.items():
            if (row["id"], arm) in done:
                continue
            bound = bind_root(root, values[0], values[1])
            user = ROOT_USER.format(
                parent_0=values[0], parent_1=values[1], root=bound
            )
            units.append((row, arm, values, user))
    if units:
        texts = runner.chat_batch(
            [unit[3] for unit in units], system=BASE_SYSTEM,
            max_new=192, bs=batch_size,
        )
        writer = JWriter(path)
        for (row, arm, values, _), result in zip(units, texts):
            text = result[0]
            answer = extract_boxed(text)
            record = {
                "id": row["id"], "arm": arm, "bindings": values,
                "text": text, "answer": answer,
                "generated_tokens": generated_tokens(runner, text),
            }
            writer.write(record)
            done[(row["id"], arm)] = record
    return done


def build_cases(
    rows: list[dict], direct: dict[int, dict],
    parents: dict[tuple[int, int], dict], roots: dict[tuple[int, str], dict]
) -> list[dict]:
    cases = []
    for row in rows:
        pid = row["id"]
        dcs = direct[pid]["candidates"]
        direct_hits = {
            f"sc{k}": answers_equal(sc_candidate(dcs, k)["answer"], row["answer"])
            for k in (1, 3, 5, 8)
        }
        direct_any = any(answers_equal(c["answer"], row["answer"]) for c in dcs)
        parent_modal_hits, parent_any_hits, parent_collapsed = [], [], []
        for slot in (0, 1):
            cs = parents[(pid, slot)]["candidates"]
            gold = row["parent_answers"][slot]
            parent_modal_hits.append(answers_equal(modal_candidate(cs)["answer"], gold))
            parent_any_hits.append(any(answers_equal(c["answer"], gold) for c in cs))
            parent_collapsed.append(len({c["norm"] for c in cs if c["norm"] is not None}) <= 1)
        modal_root_hit = answers_equal(roots[(pid, "modal")]["answer"], row["answer"])
        gold_root_hit = answers_equal(roots[(pid, "gold")]["answer"], row["answer"])
        recoverable = all(parent_any_hits) and gold_root_hit
        actionable = recoverable and not modal_root_hit and not all(parent_modal_hits)
        cases.append({
            "id": pid,
            "n_steps": int(row["n_steps"]),
            "root_step_count": int(row["root_step_count"]),
            "parent_step_counts": [int(x) for x in row["parent_step_counts"]],
            **direct_hits,
            "direct_oracle8": direct_any,
            "parent_modal_hits": parent_modal_hits,
            "parent_any_hits": parent_any_hits,
            "parent_collapsed": parent_collapsed,
            "modal_root_hit": modal_root_hit,
            "gold_root_hit": gold_root_hit,
            "recoverable": recoverable,
            "actionable": actionable,
        })
    return cases


def rule_specs() -> list[tuple[str, Callable[[dict], bool]]]:
    specs = [("all", lambda _: True)]
    totals = (8, 10, 12, 14, 16)
    roots = (2, 3, 4)
    parents = (2, 3, 4, 5)
    for threshold in totals:
        specs.append((f"total_ge_{threshold}",
                      lambda c, t=threshold: c["n_steps"] >= t))
    for threshold in roots:
        specs.append((f"root_ge_{threshold}",
                      lambda c, t=threshold: c["root_step_count"] >= t))
    for threshold in parents:
        specs.append((f"min_parent_ge_{threshold}",
                      lambda c, t=threshold: min(c["parent_step_counts"]) >= t))
    for p in parents:
        for r in roots:
            specs.append((
                f"min_parent_ge_{p}__root_ge_{r}",
                lambda c, p=p, r=r: min(c["parent_step_counts"]) >= p
                and c["root_step_count"] >= r,
            ))
    for total in totals:
        for r in roots:
            specs.append((
                f"total_ge_{total}__root_ge_{r}",
                lambda c, total=total, r=r: c["n_steps"] >= total
                and c["root_step_count"] >= r,
            ))
    return specs


def mean(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(cases: list[dict]) -> dict:
    n = len(cases)
    direct = {f"sc{k}": mean([c[f"sc{k}"] for c in cases]) for k in (1, 3, 5, 8)}
    parent_modal = mean([
        hit for c in cases for hit in c["parent_modal_hits"]
    ])
    parent_any = mean([hit for c in cases for hit in c["parent_any_hits"]])
    noncollapsed = mean([
        not all(c["parent_collapsed"]) for c in cases
    ])
    modal_root = mean([c["modal_root_hit"] for c in cases])
    recoverable = mean([c["recoverable"] for c in cases])
    strong_baseline = max(direct["sc8"], modal_root)
    return {
        "n": n,
        "direct": direct,
        "direct_oracle8": mean([c["direct_oracle8"] for c in cases]),
        "parent_modal_accuracy": parent_modal,
        "parent_candidate_oracle": parent_any,
        "gold_bound_root_accuracy": mean([c["gold_root_hit"] for c in cases]),
        "modal_graph_accuracy": modal_root,
        "noncollapsed_problem_rate": noncollapsed,
        "recoverability": recoverable,
        "strong_baseline": strong_baseline,
        "recoverability_gap": recoverable - strong_baseline,
        "n_actionable": sum(c["actionable"] for c in cases),
        "actionable_rate": mean([c["actionable"] for c in cases]),
    }


def calibration_eligible(metrics: dict) -> bool:
    return (
        metrics["n"] >= 100
        and 0.40 <= metrics["direct"]["sc8"] <= 0.60
        and metrics["parent_modal_accuracy"] >= 0.70
        and metrics["gold_bound_root_accuracy"] >= 0.80
        and metrics["noncollapsed_problem_rate"] >= 0.15
        and metrics["recoverability_gap"] >= 0.10
        and metrics["n_actionable"] >= 20
    )


def confirmation_checks(metrics: dict) -> dict:
    return {
        "n_ge_100": metrics["n"] >= 100,
        "direct_sc8_between_035_065": 0.35 <= metrics["direct"]["sc8"] <= 0.65,
        "parent_modal_accuracy_ge_065": metrics["parent_modal_accuracy"] >= 0.65,
        "gold_bound_root_accuracy_ge_075": metrics["gold_bound_root_accuracy"] >= 0.75,
        "noncollapsed_problem_rate_ge_015": metrics["noncollapsed_problem_rate"] >= 0.15,
        "recoverability_gap_ge_010": metrics["recoverability_gap"] >= 0.10,
        "n_actionable_ge_20": metrics["n_actionable"] >= 20,
    }


def analyze(calibration_cases: list[dict], test_cases: list[dict]) -> dict:
    rules = []
    predicates = dict(rule_specs())
    for name, predicate in rule_specs():
        metrics = summarize([c for c in calibration_cases if predicate(c)])
        rules.append({"name": name, "eligible": calibration_eligible(metrics), "metrics": metrics})
    eligible = [r for r in rules if r["eligible"]]
    chosen = None
    if eligible:
        chosen = min(
            eligible,
            key=lambda r: (
                -r["metrics"]["n"],
                abs(r["metrics"]["direct"]["sc8"] - 0.50),
                -r["metrics"]["recoverability_gap"],
                r["name"],
            ),
        )
    chosen_name = chosen["name"] if chosen else None
    test_metrics = summarize(
        [c for c in test_cases if predicates[chosen_name](c)]
    ) if chosen_name else summarize(test_cases)
    checks = confirmation_checks(test_metrics) if chosen_name else {
        "calibration_rule_found": False
    }
    return {
        "protocol": "EXPERIMENT_PROTOCOL_STRUCTURAL_HARDNESS_SCREEN_V0.md",
        "calibration_rules": rules,
        "chosen_rule": chosen_name,
        "calibration_metrics": chosen["metrics"] if chosen else None,
        "confirmation_metrics": test_metrics,
        "confirmation_checks": checks,
        "gate_pass": bool(chosen_name) and all(checks.values()),
    }


def run_split(
    runner: Runner, data_path: str, out_dir: str, limit: int, batch_size: int
) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    rows = read_rows(data_path, limit)
    direct = run_direct(runner, rows, out_dir, batch_size)
    parents = run_parents(runner, rows, out_dir, batch_size)
    roots = run_roots(runner, rows, parents, out_dir, batch_size)
    cases = build_cases(rows, direct, parents, roots)
    with open(os.path.join(out_dir, "cases.json"), "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=1)
    return cases


def print_metrics(label: str, metrics: Optional[dict]) -> None:
    if metrics is None:
        print(f"{label}: no eligible rule")
        return
    print(
        f"{label}: n={metrics['n']} SC@8={metrics['direct']['sc8']:.3f} "
        f"oracle@8={metrics['direct_oracle8']:.3f} "
        f"parent_modal={metrics['parent_modal_accuracy']:.3f} "
        f"gold_root={metrics['gold_bound_root_accuracy']:.3f} "
        f"modal_graph={metrics['modal_graph_accuracy']:.3f} "
        f"noncollapse={metrics['noncollapsed_problem_rate']:.3f} "
        f"recoverable={metrics['recoverability']:.3f} "
        f"gap={metrics['recoverability_gap']:+.3f} "
        f"actionable={metrics['n_actionable']}"
    )


def self_test() -> None:
    cs = [
        {"kind": "greedy", "answer": "2", "norm": "2", "text": ""},
        {"kind": "sample", "answer": "3", "norm": "3", "text": ""},
        {"kind": "sample", "answer": "3", "norm": "3", "text": ""},
        {"kind": "sample", "answer": "2", "norm": "2", "text": ""},
    ]
    assert modal_candidate(cs)["norm"] == "2"
    specs = dict(rule_specs())
    case = {"n_steps": 14, "root_step_count": 3, "parent_step_counts": [4, 2]}
    assert specs["total_ge_12__root_ge_3"](case)
    assert not specs["min_parent_ge_3"](case)
    print("SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-data", default="data/gsm_join_train.jsonl")
    parser.add_argument("--test-data", default="data/gsm_join_test.jsonl")
    parser.add_argument("--out-dir", default="structural_hardness_screen")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-calibration", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    base = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(HERE, args.out_dir)
    runner = Runner(args.model)
    calibration = run_split(
        runner, args.calibration_data, os.path.join(base, "calibration"),
        args.limit_calibration, args.batch_size,
    )
    test = run_split(
        runner, args.test_data, os.path.join(base, "confirmation"),
        args.limit_test, args.batch_size,
    )
    report = analyze(calibration, test)
    with open(os.path.join(base, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"chosen_rule={report['chosen_rule']}")
    print_metrics("calibration", report["calibration_metrics"])
    print_metrics("confirmation", report["confirmation_metrics"])
    print(f"checks={report['confirmation_checks']}")
    print(f"OVERALL={'PASS' if report['gate_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
