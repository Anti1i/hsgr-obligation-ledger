"""CPU-only Phase A for the frozen Join Viability Screen V1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from typing import Callable

from answer_check import answers_equal, normalize_answer


PROTOCOL = "EXPERIMENT_PROTOCOL_JOIN_VIABILITY_SCREEN_V1.md"
TOTAL_THRESHOLDS = (8, 9, 10, 11, 12)
ROOT_THRESHOLDS = (2, 3, 4)
PARENT_THRESHOLDS = (3, 4, 5)


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(path: str) -> list[dict]:
    rows = read_jsonl(path)
    required = (
        "answer", "parent_answers", "n_steps", "root_step_count",
        "parent_step_counts", "graph",
    )
    for pid, source in enumerate(rows):
        if not all(key in source for key in required):
            raise ValueError(f"data row {pid} misses a required field")
        if source["graph"].get("edges") != [
            ["parent_0", "root"], ["parent_1", "root"]
        ]:
            raise ValueError(f"data row {pid} has an unexpected graph")
        source["id"] = pid
    return rows


def load_caches(
    direct_path: str, parent_path: str, expected: int,
) -> tuple[dict[int, dict], dict[tuple[int, int], dict]]:
    direct = {int(row["id"]): row for row in read_jsonl(direct_path)}
    parents = {
        (int(row["id"]), int(row["slot"])): row
        for row in read_jsonl(parent_path)
    }
    if len(direct) != expected:
        raise ValueError(f"expected {expected} direct rows, found {len(direct)}")
    if len(parents) != expected * 2:
        raise ValueError(
            f"expected {expected * 2} parent rows, found {len(parents)}"
        )
    for pid in range(expected):
        if pid not in direct or any((pid, slot) not in parents for slot in (0, 1)):
            raise ValueError(f"cache is incomplete at id={pid}")
        if len(direct[pid].get("candidates", [])) != 8:
            raise ValueError(f"direct id={pid} does not have 8 candidates")
        for slot in (0, 1):
            if len(parents[(pid, slot)].get("candidates", [])) != 4:
                raise ValueError(
                    f"parent id={pid} slot={slot} does not have 4 candidates"
                )
    return direct, parents


def candidate_norm(candidate: dict) -> str | None:
    value = candidate.get("norm")
    return value if value is not None else normalize_answer(candidate.get("answer"))


def modal_candidate(candidates: list[dict]) -> dict:
    valid = [
        (index, candidate, candidate_norm(candidate))
        for index, candidate in enumerate(candidates)
    ]
    valid = [item for item in valid if item[2] is not None]
    if not valid:
        return candidates[0]
    counts = Counter(item[2] for item in valid)
    greedy_norm = candidate_norm(candidates[0])
    best_norm = max(
        counts,
        key=lambda value: (
            counts[value],
            value == greedy_norm,
            -next(index for index, _, norm in valid if norm == value),
        ),
    )
    return next(candidate for _, candidate, norm in valid if norm == best_norm)


def rule_specs() -> list[tuple[str, Callable[[dict], bool]]]:
    rules: list[tuple[str, Callable[[dict], bool]]] = [("all", lambda _: True)]
    for threshold in TOTAL_THRESHOLDS:
        rules.append((
            f"total_le_{threshold}",
            lambda row, threshold=threshold: int(row["n_steps"]) <= threshold,
        ))
    for threshold in ROOT_THRESHOLDS:
        rules.append((
            f"root_le_{threshold}",
            lambda row, threshold=threshold: int(row["root_step_count"]) <= threshold,
        ))
    for threshold in PARENT_THRESHOLDS:
        rules.append((
            f"max_parent_le_{threshold}",
            lambda row, threshold=threshold: max(
                int(value) for value in row["parent_step_counts"]
            ) <= threshold,
        ))
    for total in TOTAL_THRESHOLDS:
        for root in ROOT_THRESHOLDS:
            rules.append((
                f"total_le_{total}__root_le_{root}",
                lambda row, total=total, root=root: (
                    int(row["n_steps"]) <= total
                    and int(row["root_step_count"]) <= root
                ),
            ))
    for parent in PARENT_THRESHOLDS:
        for root in ROOT_THRESHOLDS:
            rules.append((
                f"max_parent_le_{parent}__root_le_{root}",
                lambda row, parent=parent, root=root: (
                    max(int(value) for value in row["parent_step_counts"]) <= parent
                    and int(row["root_step_count"]) <= root
                ),
            ))
    return rules


def fraction(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(
    rows: list[dict], direct: dict[int, dict],
    parents: dict[tuple[int, int], dict],
) -> dict:
    direct_hits: dict[int, list[bool]] = {k: [] for k in (1, 3, 5, 8)}
    direct_oracle = []
    parent_greedy = []
    parent_oracle = []
    noncollapsed = []
    for row in rows:
        pid = int(row["id"])
        candidates = direct[pid]["candidates"]
        for k in direct_hits:
            selected = modal_candidate(candidates[:k])
            direct_hits[k].append(answers_equal(selected.get("answer"), row["answer"]))
        direct_oracle.append(any(
            answers_equal(candidate.get("answer"), row["answer"])
            for candidate in candidates
        ))
        collapsed_slots = []
        for slot in (0, 1):
            candidates = parents[(pid, slot)]["candidates"]
            gold = row["parent_answers"][slot]
            parent_greedy.append(answers_equal(candidates[0].get("answer"), gold))
            parent_oracle.append(any(
                answers_equal(candidate.get("answer"), gold)
                for candidate in candidates
            ))
            collapsed_slots.append(len({
                candidate_norm(candidate) for candidate in candidates
                if candidate_norm(candidate) is not None
            }) <= 1)
        noncollapsed.append(not all(collapsed_slots))
    return {
        "n": len(rows),
        "direct": {
            f"sc{k}": fraction(direct_hits[k]) for k in (1, 3, 5, 8)
        },
        "direct_oracle8": fraction(direct_oracle),
        "parent_greedy_accuracy": fraction(parent_greedy),
        "parent_oracle4": fraction(parent_oracle),
        "noncollapsed_problem_rate": fraction(noncollapsed),
    }


def is_preeligible(metrics: dict) -> bool:
    return (
        metrics["n"] >= 100
        and 0.30 <= metrics["direct"]["sc1"] <= 0.70
        and 0.30 <= metrics["direct"]["sc8"] <= 0.70
        and metrics["parent_greedy_accuracy"] >= 0.70
    )


def analyze(
    rows: list[dict], direct: dict[int, dict],
    parents: dict[tuple[int, int], dict],
    data_path: str, direct_path: str, parent_path: str,
) -> dict:
    rules = []
    for name, predicate in rule_specs():
        metrics = summarize([row for row in rows if predicate(row)], direct, parents)
        rules.append({
            "name": name,
            "preeligible": is_preeligible(metrics),
            "metrics": metrics,
        })
    preeligible = [rule["name"] for rule in rules if rule["preeligible"]]
    return {
        "protocol": PROTOCOL,
        "phase": "A_cached_cpu_prefilter",
        "inputs": {
            "data": os.path.abspath(data_path),
            "data_sha256": file_sha256(data_path),
            "direct_cache": os.path.abspath(direct_path),
            "direct_sha256": file_sha256(direct_path),
            "parent_cache": os.path.abspath(parent_path),
            "parent_sha256": file_sha256(parent_path),
        },
        "rule_count": len(rules),
        "rules": rules,
        "preeligible_rules": preeligible,
        "phase_a_pass": bool(preeligible),
    }


def print_report(report: dict) -> None:
    for rule in report["rules"]:
        metrics = rule["metrics"]
        direct = metrics["direct"]
        print(
            f"{rule['name']}: n={metrics['n']} sc1={direct['sc1']:.3f} "
            f"sc8={direct['sc8']:.3f} parent_greedy="
            f"{metrics['parent_greedy_accuracy']:.3f} "
            f"preeligible={rule['preeligible']}"
        )
    print(f"JOIN_VIABILITY_PHASE_A={'PASS' if report['phase_a_pass'] else 'FAIL'}")


def self_test() -> None:
    assert len(rule_specs()) == 36
    assert is_preeligible({
        "n": 100,
        "direct": {"sc1": 0.30, "sc8": 0.70},
        "parent_greedy_accuracy": 0.70,
    })
    assert not is_preeligible({
        "n": 99,
        "direct": {"sc1": 0.50, "sc8": 0.50},
        "parent_greedy_accuracy": 0.90,
    })
    print("SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gsm_join_train.jsonl")
    parser.add_argument("--direct-cache")
    parser.add_argument("--parent-cache")
    parser.add_argument("--out", default="join_viability_phase_a_v1.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.direct_cache or not args.parent_cache:
        parser.error("--direct-cache and --parent-cache are required")
    rows = load_data(args.data)
    direct, parents = load_caches(
        args.direct_cache, args.parent_cache, len(rows)
    )
    report = analyze(
        rows, direct, parents, args.data, args.direct_cache, args.parent_cache
    )
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    print_report(report)


if __name__ == "__main__":
    main()

