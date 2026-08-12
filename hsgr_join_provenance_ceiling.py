"""Oracle action ceiling for source routing on a three-node HSGR join.

Two independently solved parent nodes feed one root.  A wrong base root is
labelled by the first erroneous graph source: parent 0, parent 1, both parents,
or the local root when both parents are correct.  The source oracle chooses one
equal-cap repair action without inspecting repair outcomes.

This script is development-only.  It extracts no hidden states and trains no
classifier.  See EXPERIMENT_PROTOCOL_JOIN_PROVENANCE_GUIDE_V2.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict

from answer_check import answers_equal, extract_boxed
from hsgr_error_provenance_ceiling import (
    batched_generate,
    chat_token_count,
    exact_mcnemar,
    generated_token_count,
    mean,
    sha_ids,
)
from pilot import JWriter, Runner, jread


HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_BASE = 160
MAX_NEW_REPAIR = 224

BASE_SYSTEM = (
    "You solve one node of an arithmetic dependency graph accurately. Follow "
    "the requested node boundary and put the answer in the last \\boxed{...}."
)

PARENT_USER = """Solve only this parent-node problem:
{question}

Give a short calculation and end with the parent answer in \\boxed{{...}}."""

ROOT_USER = """Solve only the root node.  Its two parent values are fixed:
[parent_0] {parent_0}
[parent_1] {parent_1}

Root question after binding both dependencies:
{root}

Give a short calculation and end with the root answer in \\boxed{{...}}."""

REPAIR_SYSTEM = (
    "You repair one three-node arithmetic dependency graph. Obey the specified "
    "repair action. End with the repaired root answer in the last \\boxed{...}."
)

REPAIR_USER = """Full graph problem:
{problem}

Base node predictions:
[parent_0] {parent_0}
[parent_1] {parent_1}
[root] {root}

[REPAIR ACTION]
{action}

Show a short calculation and end with the repaired root answer in
\\boxed{{...}}."""

ACTIONS = {
    "generic": (
        "Inspect all three nodes, find any errors, and correct the root answer."
    ),
    "p0": (
        "Recompute parent_0. Keep the base parent_1 prediction fixed. Then "
        "propagate those two values into the root and recompute it."
    ),
    "p1": (
        "Recompute parent_1. Keep the base parent_0 prediction fixed. Then "
        "propagate those two values into the root and recompute it."
    ),
    "local": (
        "Keep both base parent predictions fixed. Recompute only the root with "
        "those two values."
    ),
    "both": (
        "Recompute both parents independently. Then propagate both recomputed "
        "values into the root and recompute it."
    ),
}


def read_rows(path: str, limit: int = 0) -> list[dict]:
    rows = jread(path)
    if limit:
        rows = rows[:limit]
    out = []
    for idx, row in enumerate(rows):
        required = ("problem", "answer", "parent_answers", "graph")
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


def split_questions(problem: str) -> tuple[str, str, str]:
    match = re.search(
        r"\AQuestion 1:\s*(.*?)\n\nQuestion 2:\s*(.*?)"
        r"\n\nQuestion 3:\s*(.*?)\n\nGive the answer to Question 3\.\s*\Z",
        problem,
        flags=re.S,
    )
    if not match:
        raise ValueError(f"cannot split join problem: {problem[:120]!r}")
    return tuple(match.group(i).strip() for i in (1, 2, 3))


def bind_root(root: str, parent_0: str, parent_1: str) -> str:
    markers = (
        "(the answer to Question 1)", "(the answer to Question 2)"
    )
    if any(marker not in root for marker in markers):
        raise ValueError("root is missing a parent dependency marker")
    root = root.replace(markers[0], str(parent_0), 1)
    return root.replace(markers[1], str(parent_1), 1)


def run_base(
    runner: Runner, rows: list[dict], out_dir: str, batch_size: int
) -> list[dict]:
    path = os.path.join(out_dir, "base.jsonl")
    done = {int(row["id"]): row for row in jread(path)}
    todo = [row for row in rows if row["id"] not in done]
    if todo:
        writer = JWriter(path)
        parent_units = []
        for row in todo:
            q0, q1, _ = split_questions(row["problem"])
            parent_units.extend(
                [(row, 0, PARENT_USER.format(question=q0)),
                 (row, 1, PARENT_USER.format(question=q1))]
            )
        parent_texts = batched_generate(
            runner,
            [unit[2] for unit in parent_units],
            BASE_SYSTEM,
            MAX_NEW_BASE,
            batch_size,
        )
        parents = defaultdict(dict)
        for (row, slot, user), text in zip(parent_units, parent_texts):
            parents[row["id"]][slot] = {
                "user": user, "text": text, "answer": extract_boxed(text)
            }

        root_units = []
        for row in todo:
            _, _, root = split_questions(row["problem"])
            p0 = parents[row["id"]][0]["answer"] or "UNKNOWN"
            p1 = parents[row["id"]][1]["answer"] or "UNKNOWN"
            bound = bind_root(root, p0, p1)
            user = ROOT_USER.format(parent_0=p0, parent_1=p1, root=bound)
            root_units.append((row, user))
        root_texts = batched_generate(
            runner,
            [unit[1] for unit in root_units],
            BASE_SYSTEM,
            MAX_NEW_BASE,
            batch_size,
        )

        for (row, root_user), root_text in zip(root_units, root_texts):
            ps = parents[row["id"]]
            root_answer = extract_boxed(root_text)
            p_answers = [ps[i]["answer"] for i in (0, 1)]
            p_hits = [
                bool(answer and answers_equal(answer, gold))
                for answer, gold in zip(p_answers, row["parent_answers"])
            ]
            record = {
                "id": row["id"],
                "gold": str(row["answer"]),
                "parent_gold": [str(x) for x in row["parent_answers"]],
                "parent_texts": [ps[i]["text"] for i in (0, 1)],
                "parent_answers": p_answers,
                "parent_hits": p_hits,
                "root_text": root_text,
                "root_answer": root_answer,
                "base_hit": bool(
                    root_answer and answers_equal(root_answer, row["answer"])
                ),
                "prompt_tokens": sum(
                    chat_token_count(runner, ps[i]["user"], BASE_SYSTEM)
                    for i in (0, 1)
                ) + chat_token_count(runner, root_user, BASE_SYSTEM),
                "generated_tokens": sum(
                    generated_token_count(runner, ps[i]["text"])
                    for i in (0, 1)
                ) + generated_token_count(runner, root_text),
                "calls": 3,
            }
            writer.write(record)
            done[row["id"]] = record
    return [done[row["id"]] for row in rows]


def source_label(base: dict) -> str:
    if base["base_hit"]:
        return "keep"
    p0, p1 = map(bool, base["parent_hits"])
    if not p0 and not p1:
        return "both"
    if not p0:
        return "p0"
    if not p1:
        return "p1"
    return "local"


def run_repairs(
    runner: Runner,
    rows: list[dict],
    base: list[dict],
    out_dir: str,
    batch_size: int,
) -> list[dict]:
    path = os.path.join(out_dir, "repairs.jsonl")
    done = {(int(row["id"]), row["arm"]): row for row in jread(path)}
    base_by_id = {row["id"]: row for row in base}
    units = []
    for row in rows:
        b = base_by_id[row["id"]]
        if b["base_hit"]:
            continue
        for arm, action in ACTIONS.items():
            if (row["id"], arm) in done:
                continue
            user = REPAIR_USER.format(
                problem=row["problem"],
                parent_0=b["parent_answers"][0] or "UNKNOWN",
                parent_1=b["parent_answers"][1] or "UNKNOWN",
                root=b["root_answer"] or "UNKNOWN",
                action=action,
            )
            units.append((row, arm, user))
    if units:
        writer = JWriter(path)
        texts = batched_generate(
            runner,
            [unit[2] for unit in units],
            REPAIR_SYSTEM,
            MAX_NEW_REPAIR,
            batch_size,
        )
        for (row, arm, user), text in zip(units, texts):
            answer = extract_boxed(text)
            record = {
                "id": row["id"],
                "arm": arm,
                "text": text,
                "answer": answer,
                "hit": bool(answer and answers_equal(answer, row["answer"])),
                "prompt_tokens": chat_token_count(runner, user, REPAIR_SYSTEM),
                "generated_tokens": generated_token_count(runner, text),
                "calls": 1,
                "max_new": MAX_NEW_REPAIR,
            }
            writer.write(record)
            done[(row["id"], arm)] = record
    error_ids = [row["id"] for row in base if not row["base_hit"]]
    return [done[(pid, arm)] for pid in error_ids for arm in ACTIONS]


def fixed_hash_half(pid: int) -> int:
    digest = hashlib.sha256(str(pid).encode()).digest()
    return digest[0] & 1


def analyze(rows: list[dict], base: list[dict], repairs: list[dict]) -> dict:
    base_by_id = {row["id"]: row for row in base}
    row_by_id = {row["id"]: row for row in rows}
    rep = defaultdict(dict)
    for record in repairs:
        rep[record["id"]][record["arm"]] = record

    sources = [source_label(base_by_id[row["id"]]) for row in rows]
    source_counts = Counter(sources)
    policy = defaultdict(list)
    cases = []
    for row, source in zip(rows, sources):
        pid = row["id"]
        b = base_by_id[pid]
        base_hit = bool(b["base_hit"])
        arm_hits = {
            arm: (base_hit or bool(rep[pid][arm]["hit"]))
            for arm in ACTIONS
        }
        routed = base_hit or bool(rep[pid][source]["hit"])
        policy["base"].append(base_hit)
        for arm, hit in arm_hits.items():
            policy[f"oracle_keep_{arm}"].append(hit)
        policy["oracle_keep_source_routed"].append(routed)
        policy["oracle_keep_best_of_repairs"].append(
            base_hit or any(arm_hits.values())
        )
        cases.append({
            "id": pid,
            "source": source,
            "base_hit": base_hit,
            "parent_hits": b["parent_hits"],
            "root_step_count": row["root_step_count"],
            "hash_half": fixed_hash_half(pid),
            **{f"{arm}_hit": hit for arm, hit in arm_hits.items()},
            "source_routed_hit": routed,
        })

    accuracy = {name: mean(values) for name, values in policy.items()}
    fixed_names = [f"oracle_keep_{arm}" for arm in ("p0", "p1", "local", "both")]
    best_fixed = max(fixed_names, key=lambda name: (accuracy[name], name))
    source_name = "oracle_keep_source_routed"
    generic_name = "oracle_keep_generic"
    delta_generic = accuracy[source_name] - accuracy[generic_name]
    delta_fixed = accuracy[source_name] - accuracy[best_fixed]
    mc_generic = exact_mcnemar(policy[source_name], policy[generic_name])
    mc_fixed = exact_mcnemar(policy[source_name], policy[best_fixed])

    strata = {}
    crossover_pass = True
    for source in ("p0", "p1", "local", "both"):
        idxs = [i for i, value in enumerate(sources) if value == source]
        values = {
            arm: mean([cases[i][f"{arm}_hit"] for i in idxs])
            for arm in ACTIONS
        }
        nonmatching = [arm for arm in ("p0", "p1", "local") if arm != source]
        margin = (
            values[source] - max(values[arm] for arm in nonmatching)
            if source in ("p0", "p1", "local") else None
        )
        if margin is not None:
            crossover_pass &= margin >= 0.05
        strata[source] = {"n": len(idxs), **values, "matching_margin": margin}

    median_root_steps = sorted(row["root_step_count"] for row in rows)[len(rows) // 2]
    stability = {}
    for name, selector in (
        ("hash_half_0", lambda c: c["hash_half"] == 0),
        ("hash_half_1", lambda c: c["hash_half"] == 1),
        ("root_steps_low", lambda c: c["root_step_count"] <= median_root_steps),
        ("root_steps_high", lambda c: c["root_step_count"] > median_root_steps),
    ):
        idxs = [i for i, case in enumerate(cases) if selector(case)]
        stability[name] = {
            "n": len(idxs),
            "source_minus_generic": mean([policy[source_name][i] for i in idxs])
            - mean([policy[generic_name][i] for i in idxs]),
            "source_minus_best_fixed": mean([policy[source_name][i] for i in idxs])
            - mean([policy[best_fixed][i] for i in idxs]),
        }
    stability_pass = all(
        values["source_minus_generic"] >= 0
        and values["source_minus_best_fixed"] >= 0
        for values in stability.values()
    )

    error_repairs = [record for record in repairs]
    token_costs = {}
    for arm in ACTIONS:
        arm_rows = [record for record in error_repairs if record["arm"] == arm]
        token_costs[arm] = {
            "n_error_calls": len(arm_rows),
            "calls_per_base_error": mean([record["calls"] for record in arm_rows]),
            "prompt_tokens_per_base_error": mean(
                [record["prompt_tokens"] for record in arm_rows]
            ),
            "generated_tokens_per_base_error": mean(
                [record["generated_tokens"] for record in arm_rows]
            ),
            "max_new": MAX_NEW_REPAIR,
        }

    parent_depth_means = [
        mean([row["parent_step_counts"][slot] for row in rows])
        for slot in (0, 1)
    ]
    integrity = {
        "unique_graphs": len({row["problem"] for row in rows}) == len(rows),
        "both_edges_symbolically_causal": all(
            all(value != row["original_root_answer"] for value in row["single_edge_answers"])
            for row in rows
        ),
        "parent_depth_difference_le_025": abs(
            parent_depth_means[0] - parent_depth_means[1]
        ) <= 0.25,
    }
    gates = {
        "p0_p1_local_n_ge_30": min(source_counts.get(s, 0) for s in ("p0", "p1", "local")) >= 30,
        "source_vs_generic_ge_3pp_p_lt_05": delta_generic >= 0.03 and mc_generic["p"] < 0.05,
        "source_vs_best_fixed_ge_3pp_p_lt_05": delta_fixed >= 0.03 and mc_fixed["p"] < 0.05,
        "three_way_crossover_ge_5pp": crossover_pass,
        "stability_nonnegative": stability_pass,
        "one_call_same_cap": all(
            cost["calls_per_base_error"] == 1.0 and cost["max_new"] == MAX_NEW_REPAIR
            for cost in token_costs.values()
        ),
        "data_integrity": all(integrity.values()),
    }
    return {
        "protocol": "EXPERIMENT_PROTOCOL_JOIN_PROVENANCE_GUIDE_V2.md",
        "n": len(rows),
        "id_sha256": sha_ids([row["id"] for row in rows]),
        "source_counts": dict(source_counts),
        "parent_depth_means": parent_depth_means,
        "accuracy": accuracy,
        "best_fixed_policy": best_fixed,
        "delta_source_vs_generic": delta_generic,
        "delta_source_vs_best_fixed": delta_fixed,
        "mcnemar_source_vs_generic": mc_generic,
        "mcnemar_source_vs_best_fixed": mc_fixed,
        "strata": strata,
        "stability": stability,
        "median_root_steps": median_root_steps,
        "token_costs": token_costs,
        "integrity": integrity,
        "gates": gates,
        "gate_pass": all(gates.values()),
        "cases": cases,
    }


def print_report(report: dict) -> None:
    print(f"== HSGR join provenance ceiling (n={report['n']}) ==")
    print(f"sources={report['source_counts']} parent_depth_means={report['parent_depth_means']}")
    for name, value in sorted(report["accuracy"].items()):
        print(f"  {name:30s} {value:.4f}")
    print(
        f"source vs generic {report['delta_source_vs_generic']:+.4f} "
        f"{report['mcnemar_source_vs_generic']}"
    )
    print(
        f"source vs {report['best_fixed_policy']} "
        f"{report['delta_source_vs_best_fixed']:+.4f} "
        f"{report['mcnemar_source_vs_best_fixed']}"
    )
    for source, values in report["strata"].items():
        print(f"  stratum {source}: {values}")
    for name, passed in report["gates"].items():
        print(f"GATE {name}: {'PASS' if passed else 'FAIL'}")
    print(f"OVERALL: {'PASS' if report['gate_pass'] else 'FAIL -> NO HIDDEN READER'}")


def self_test() -> None:
    problem = (
        "Question 1: What is 2 plus 3?\n\n"
        "Question 2: What is 4 plus 2?\n\n"
        "Question 3: Multiply (the answer to Question 1) by "
        "(the answer to Question 2).\n\nGive the answer to Question 3."
    )
    q0, q1, root = split_questions(problem)
    assert q0 == "What is 2 plus 3?" and q1 == "What is 4 plus 2?"
    assert bind_root(root, "5", "6") == "Multiply 5 by 6."
    rows = []
    base = []
    source_specs = [
        ("keep", [True, True], True),
        ("p0", [False, True], False),
        ("p1", [True, False], False),
        ("local", [True, True], False),
        ("both", [False, False], False),
    ]
    for pid, (source, parent_hits, base_hit) in enumerate(source_specs):
        rows.append({
            "id": pid,
            "problem": problem + str(pid),
            "answer": "30",
            "parent_step_counts": [2, 2],
            "root_step_count": 1,
            "original_root_answer": "12",
            "single_edge_answers": ["18", "20"],
        })
        base.append({"id": pid, "parent_hits": parent_hits, "base_hit": base_hit})
        assert source_label(base[-1]) == source
    repairs = []
    for pid, (source, _, base_hit) in enumerate(source_specs):
        if base_hit:
            continue
        for arm in ACTIONS:
            repairs.append({
                "id": pid,
                "arm": arm,
                "hit": arm == source,
                "calls": 1,
                "max_new": MAX_NEW_REPAIR,
                "prompt_tokens": 10,
                "generated_tokens": 2,
            })
    report = analyze(rows, base, repairs)
    assert report["source_counts"] == {
        "keep": 1, "p0": 1, "p1": 1, "local": 1, "both": 1
    }
    assert report["accuracy"]["oracle_keep_source_routed"] == 1.0
    assert not report["gate_pass"]
    print("SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gsm_join_test.jsonl")
    parser.add_argument("--out-dir", default="hsgr_join_provenance")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    data = args.data if os.path.isabs(args.data) else os.path.join(HERE, args.data)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(HERE, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rows = read_rows(data, args.limit)
    if args.analyze_only:
        base = jread(os.path.join(out_dir, "base.jsonl"))
        repairs = jread(os.path.join(out_dir, "repairs.jsonl"))
    else:
        runner = Runner(args.model)
        base = run_base(runner, rows, out_dir, args.batch_size)
        repairs = run_repairs(runner, rows, base, out_dir, args.batch_size)
    n_errors = sum(not record["base_hit"] for record in base)
    if len(base) != len(rows) or len(repairs) != n_errors * len(ACTIONS):
        raise RuntimeError(
            f"incomplete outputs rows={len(rows)} base={len(base)} "
            f"repairs={len(repairs)} expected_repairs={n_errors * len(ACTIONS)}"
        )
    report = analyze(rows, base, repairs)
    cases = report.pop("cases")
    report_path = os.path.join(out_dir, "hsgr_join_provenance_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with open(
        os.path.join(out_dir, "hsgr_join_provenance_cases.jsonl"),
        "w",
        encoding="utf-8",
    ) as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print_report(report)
    print(f"saved {report_path}")


if __name__ == "__main__":
    main()
