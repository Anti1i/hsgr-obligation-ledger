"""Frozen V0 oracle matrix for typed interventions on a fixed join DAG."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict

from answer_check import answers_equal, extract_boxed
from hsgr_error_provenance_ceiling import (
    exact_mcnemar,
    generated_token_count,
    mean,
)
from hsgr_join_provenance_ceiling import (
    BASE_SYSTEM,
    PARENT_USER,
    ROOT_USER,
    bind_root,
    split_questions,
)
from pilot import JWriter, Runner, jread


PROTOCOL = "EXPERIMENT_PROTOCOL_TYPED_NODE_ACTION_ORACLE_V0.md"
MODEL_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"
N_GRAPHS = 128
MAX_NEW = 512
NODES = ("parent_0", "parent_1", "root")
COMMON_ACTIONS = ("equation", "independent", "backward", "redecompose")
ARMS = tuple(
    [(node, action) for node in NODES for action in COMMON_ACTIONS]
    + [("root", "rebind")]
)

ACTION_TEXT = {
    "equation": (
        "Write the minimal equations for the TARGET node, check units, and "
        "recompute that node deterministically."
    ),
    "independent": (
        "Ignore the proposed TARGET value and independently recompute the "
        "TARGET node from its local question."
    ),
    "backward": (
        "Audit the proposed TARGET value backward against every quantity, "
        "constraint, and arithmetic relation in its local question; correct "
        "it if any check fails."
    ),
    "redecompose": (
        "Re-solve the TARGET node with a different local decomposition into "
        "smaller subproblems; do not reuse the proposed derivation."
    ),
    "rebind": (
        "Audit the ordered binding, meaning, and units of both fixed parent "
        "values in the root question, then recompute only the root."
    ),
}

INTERVENTION_SYSTEM = (
    "You execute one typed intervention on one fixed arithmetic dependency "
    "graph. Do not change non-target node assignments. Return the requested "
    "machine-readable final lines."
)

INTERVENTION_USER = """Fixed three-node dependency graph:

[parent_0 local question]
{q0}

[parent_1 local question]
{q1}

[root local question before binding]
{root}

Current assignments:
[parent_0] {p0}
[parent_1] {p1}
[root] {base_root}

TARGET NODE: {node}
TYPED ACTION: {action}
ACTION INSTRUCTION: {instruction}

Rules:
1. Change only the TARGET node. Keep every non-target parent assignment fixed.
2. If TARGET is a parent, propagate its repaired value together with the fixed
   other parent into the root and recompute the root.
3. If TARGET is root, keep both parent values fixed and recompute only root.
4. Do not propose an alternative full reasoning path or edit graph structure.
5. End with exactly these two lines and no later boxes:
TARGET: \\boxed{{target_value}}
ROOT: \\boxed{{propagated_root_value}}"""

GENERIC_SYSTEM = (
    "You repair one fixed three-node arithmetic dependency graph. Inspect all "
    "nodes once and end with the repaired root in the requested format."
)

GENERIC_USER = """Full graph problem:
{problem}

Current assignments:
[parent_0] {p0}
[parent_1] {p1}
[root] {base_root}

Inspect the graph once, correct any errors needed to obtain the final root,
and end with exactly one final line and no later box:
ROOT: \\boxed{{root_value}}"""


def stable_rank(pid: int) -> str:
    return hashlib.sha256(f"typed-node-action-v0|{pid}".encode()).hexdigest()


def stable_half(pid: int) -> int:
    digest = hashlib.sha256(f"typed-node-action-v0-half|{pid}".encode()).digest()
    return digest[0] & 1


def selected_rows(path: str, limit: int = N_GRAPHS) -> list[dict]:
    rows = jread(path)
    if len(rows) < limit:
        raise ValueError(f"need at least {limit} data rows, found {len(rows)}")
    ranked = sorted(enumerate(rows), key=lambda pair: stable_rank(pair[0]))[:limit]
    out = []
    for pid, row in sorted(ranked):
        if row.get("graph", {}).get("edges") != [
            ["parent_0", "root"], ["parent_1", "root"]
        ]:
            raise ValueError(f"unexpected graph at id={pid}")
        item = dict(row)
        item["id"] = pid
        out.append(item)
    return out


def load_parent_greedy(path: str) -> dict[tuple[int, int], dict]:
    rows = jread(path)
    cache = {(int(row["id"]), int(row["slot"])): row for row in rows}
    if len(cache) != 800:
        raise ValueError(f"expected 800 cached parent nodes, found {len(cache)}")
    return cache


def extract_boxed_all(text: str) -> list[str]:
    values = []
    start = 0
    while True:
        idx = text.find("\\boxed", start)
        if idx < 0:
            break
        left = text.find("{", idx)
        if left < 0:
            break
        depth = 0
        for right in range(left, len(text)):
            if text[right] == "{":
                depth += 1
            elif text[right] == "}":
                depth -= 1
                if depth == 0:
                    values.append(text[left + 1:right])
                    start = right + 1
                    break
        else:
            break
    return values


def parse_intervention(text: str) -> tuple[str | None, str | None]:
    boxes = extract_boxed_all(text)
    if len(boxes) < 2:
        return None, extract_boxed(text)
    return boxes[-2], boxes[-1]


def chat_prompt_tokens(runner: Runner, user: str, system: str) -> int:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    rendered = runner.tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return len(runner.tok.encode(rendered, add_special_tokens=False))


def base_units(rows: list[dict], parent_cache: dict) -> list[dict]:
    units = []
    for row in rows:
        q0, q1, root = split_questions(row["problem"])
        parents = []
        for slot in (0, 1):
            candidate = parent_cache[(row["id"], slot)]["candidates"][0]
            parents.append(candidate.get("answer") or "UNKNOWN")
        units.append({
            "id": row["id"],
            "problem": row["problem"],
            "gold": str(row["answer"]),
            "parent_gold": [str(value) for value in row["parent_answers"]],
            "questions": [q0, q1],
            "root_question": root,
            "parents": parents,
        })
    return units


def run_base_roots(
    runner: Runner, units: list[dict], path: str, batch_size: int
) -> dict[int, dict]:
    done = {int(row["id"]): row for row in jread(path)}
    todo = [unit for unit in units if unit["id"] not in done]
    writer = JWriter(path)
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        users = []
        for unit in batch:
            bound = bind_root(
                unit["root_question"], unit["parents"][0], unit["parents"][1]
            )
            users.append(ROOT_USER.format(
                parent_0=unit["parents"][0], parent_1=unit["parents"][1],
                root=bound,
            ))
        outputs = runner.chat_batch(
            users, system=BASE_SYSTEM, max_new=MAX_NEW, bs=batch_size
        )
        for unit, user, output in zip(batch, users, outputs):
            text = output[0]
            record = {
                "id": unit["id"], "text": text,
                "answer": extract_boxed(text),
                "prompt_tokens": chat_prompt_tokens(runner, user, BASE_SYSTEM),
                "generated_tokens": generated_token_count(runner, text),
                "calls": 1, "max_new": MAX_NEW,
            }
            writer.write(record)
            done[unit["id"]] = record
        print(f"[base] {min(start + batch_size, len(todo))}/{len(todo)}", flush=True)
    return done


def intervention_prompt(unit: dict, base_root: str, node: str, action: str) -> str:
    return INTERVENTION_USER.format(
        q0=unit["questions"][0], q1=unit["questions"][1],
        root=unit["root_question"], p0=unit["parents"][0],
        p1=unit["parents"][1], base_root=base_root,
        node=node, action=action, instruction=ACTION_TEXT[action],
    )


def run_generic(
    runner: Runner, units: list[dict], base: dict[int, dict], path: str,
    batch_size: int,
) -> dict[int, dict]:
    done = {int(row["id"]): row for row in jread(path)}
    todo = [unit for unit in units if unit["id"] not in done]
    writer = JWriter(path)
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        users = [GENERIC_USER.format(
            problem=unit["problem"], p0=unit["parents"][0],
            p1=unit["parents"][1], base_root=base[unit["id"]]["answer"] or "UNKNOWN",
        ) for unit in batch]
        outputs = runner.chat_batch(
            users, system=GENERIC_SYSTEM, max_new=MAX_NEW, bs=batch_size
        )
        for unit, user, output in zip(batch, users, outputs):
            text = output[0]
            record = {
                "id": unit["id"], "text": text, "answer": extract_boxed(text),
                "prompt_tokens": chat_prompt_tokens(runner, user, GENERIC_SYSTEM),
                "generated_tokens": generated_token_count(runner, text),
                "calls": 1, "max_new": MAX_NEW,
            }
            writer.write(record)
            done[unit["id"]] = record
        print(f"[generic] {min(start + batch_size, len(todo))}/{len(todo)}", flush=True)
    return done


def run_interventions(
    runner: Runner, units: list[dict], base: dict[int, dict], path: str,
    batch_size: int,
) -> dict[tuple[int, str, str], dict]:
    done = {
        (int(row["id"]), row["node"], row["action"]): row for row in jread(path)
    }
    all_units = [
        (unit, node, action)
        for node, action in ARMS
        for unit in units
        if (unit["id"], node, action) not in done
    ]
    writer = JWriter(path)
    for start in range(0, len(all_units), batch_size):
        batch = all_units[start:start + batch_size]
        users = [intervention_prompt(
            unit, base[unit["id"]]["answer"] or "UNKNOWN", node, action
        ) for unit, node, action in batch]
        outputs = runner.chat_batch(
            users, system=INTERVENTION_SYSTEM, max_new=MAX_NEW, bs=batch_size
        )
        for (unit, node, action), user, output in zip(batch, users, outputs):
            text = output[0]
            target, root = parse_intervention(text)
            record = {
                "id": unit["id"], "node": node, "action": action,
                "text": text, "target_answer": target, "root_answer": root,
                "prompt_tokens": chat_prompt_tokens(
                    runner, user, INTERVENTION_SYSTEM
                ),
                "generated_tokens": generated_token_count(runner, text),
                "calls": 1, "max_new": MAX_NEW,
            }
            writer.write(record)
            done[(unit["id"], node, action)] = record
        if start == 0 or (start // batch_size + 1) % 8 == 0:
            print(
                f"[interventions] {min(start + batch_size, len(all_units))}/"
                f"{len(all_units)}", flush=True,
            )
    return done


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def analyze(
    units: list[dict], base: dict[int, dict], generic: dict[int, dict],
    interventions: dict[tuple[int, str, str], dict],
) -> dict:
    ids = [unit["id"] for unit in units]
    unit_by_id = {unit["id"]: unit for unit in units}
    base_hit = {pid: answers_equal(base[pid]["answer"], unit_by_id[pid]["gold"])
                for pid in ids}
    generic_hit = {
        pid: answers_equal(generic[pid]["answer"], unit_by_id[pid]["gold"])
        for pid in ids
    }
    root_hit = {}
    target_valid = {}
    root_valid = {}
    for pid in ids:
        for node, action in ARMS:
            record = interventions[(pid, node, action)]
            key = (pid, node, action)
            root_hit[key] = answers_equal(
                record["root_answer"], unit_by_id[pid]["gold"]
            )
            target_valid[key] = record["target_answer"] is not None
            root_valid[key] = record["root_answer"] is not None

    n = len(ids)
    error_ids = [pid for pid in ids if not base_hit[pid]]
    correct_ids = [pid for pid in ids if base_hit[pid]]
    base_acc = mean(list(base_hit.values()))
    generic_raw = mean(list(generic_hit.values()))
    generic_keep_vec = [base_hit[pid] or generic_hit[pid] for pid in ids]
    generic_keep = mean(generic_keep_vec)

    arm_metrics = {}
    arm_keep_vectors = {}
    for node, action in ARMS:
        name = f"{node}:{action}"
        raw = [root_hit[(pid, node, action)] for pid in ids]
        keep = [base_hit[pid] or root_hit[(pid, node, action)] for pid in ids]
        arm_keep_vectors[name] = keep
        arm_metrics[name] = {
            "root_accuracy": mean(raw),
            "oracle_keep_accuracy": mean(keep),
            "delta_from_base": mean(raw) - base_acc,
            "base_error_repair_rate": mean([
                root_hit[(pid, node, action)] for pid in error_ids
            ]),
            "base_correct_corruption_rate": mean([
                not root_hit[(pid, node, action)] for pid in correct_ids
            ]),
            "target_parse_validity": mean([
                target_valid[(pid, node, action)] for pid in ids
            ]),
            "root_parse_validity": mean([
                root_valid[(pid, node, action)] for pid in ids
            ]),
            "mean_prompt_tokens": mean([
                interventions[(pid, node, action)]["prompt_tokens"] for pid in ids
            ]),
            "mean_generated_tokens": mean([
                interventions[(pid, node, action)]["generated_tokens"] for pid in ids
            ]),
        }

    best_fixed = max(
        arm_metrics,
        key=lambda name: (arm_metrics[name]["oracle_keep_accuracy"], name),
    )
    best_fixed_acc = arm_metrics[best_fixed]["oracle_keep_accuracy"]
    strongest_name, strongest_acc, strongest_vec = (
        ("generic", generic_keep, generic_keep_vec)
        if generic_keep >= best_fixed_acc
        else (best_fixed, best_fixed_acc, arm_keep_vectors[best_fixed])
    )

    joint_vec = [
        base_hit[pid] or any(root_hit[(pid, node, action)] for node, action in ARMS)
        for pid in ids
    ]
    joint_acc = mean(joint_vec)

    fixed_node = {}
    fixed_node_vectors = {}
    for node in NODES:
        actions = list(COMMON_ACTIONS) + (["rebind"] if node == "root" else [])
        vec = [
            base_hit[pid] or any(root_hit[(pid, node, action)] for action in actions)
            for pid in ids
        ]
        fixed_node[node] = mean(vec)
        fixed_node_vectors[node] = vec
    best_action_oracle_node = max(fixed_node, key=lambda x: (fixed_node[x], x))
    action_oracle_acc = fixed_node[best_action_oracle_node]

    fixed_action = {}
    fixed_action_vectors = {}
    for action in COMMON_ACTIONS:
        vec = [
            base_hit[pid] or any(root_hit[(pid, node, action)] for node in NODES)
            for pid in ids
        ]
        fixed_action[action] = mean(vec)
        fixed_action_vectors[action] = vec
    best_node_oracle_action = max(fixed_action, key=lambda x: (fixed_action[x], x))
    node_oracle_acc = fixed_action[best_node_oracle_action]

    raw_arm_values = [
        root_hit[(pid, node, action)] for pid in ids for node, action in ARMS
    ]
    uniform_expected = mean(raw_arm_values)
    majority_vec = []
    for pid in ids:
        answers = [
            interventions[(pid, node, action)]["root_answer"]
            for node, action in ARMS
            if interventions[(pid, node, action)]["root_answer"] is not None
        ]
        norms = [re.sub(r"\s+", "", answer.strip()).lower() for answer in answers]
        if not norms:
            majority_vec.append(False)
            continue
        counts = Counter(norms)
        chosen_norm = max(
            counts,
            key=lambda value: (counts[value], -norms.index(value)),
        )
        chosen_answer = answers[norms.index(chosen_norm)]
        majority_vec.append(answers_equal(chosen_answer, unit_by_id[pid]["gold"]))

    action_repair_sets = {}
    for action in tuple(COMMON_ACTIONS) + ("rebind",):
        eligible_nodes = [node for node, candidate in ARMS if candidate == action]
        action_repair_sets[action] = {
            pid for pid in error_ids
            if any(root_hit[(pid, node, action)] for node in eligible_nodes)
        }
    node_repair_sets = {
        node: {
            pid for pid in error_ids
            if any(root_hit[(pid, n, action)] for n, action in ARMS if n == node)
        }
        for node in NODES
    }

    exclusive_actions = {
        action: len(repaired - set().union(*[
            values for other, values in action_repair_sets.items() if other != action
        ]))
        for action, repaired in action_repair_sets.items()
    }
    exclusive_nodes = {
        node: len(repaired - set().union(*[
            values for other, values in node_repair_sets.items() if other != node
        ]))
        for node, repaired in node_repair_sets.items()
    }

    pairwise = {}
    action_names = list(action_repair_sets)
    for i, left in enumerate(action_names):
        for right in action_names[i + 1:]:
            a, b = action_repair_sets[left], action_repair_sets[right]
            union = a | b
            pairwise[f"{left}|{right}"] = {
                "both_repair": len(a & b),
                "left_only": len(a - b),
                "right_only": len(b - a),
                "neither": len(set(error_ids) - union),
                "jaccard": _fraction(len(a & b), len(union)),
                "p_left_repairs_right_fails_given_base_error": _fraction(
                    len(a - b), len(error_ids)
                ),
                "p_right_repairs_left_fails_given_base_error": _fraction(
                    len(b - a), len(error_ids)
                ),
            }

    stability = {}
    for half in (0, 1):
        idxs = [i for i, pid in enumerate(ids) if stable_half(pid) == half]
        stability[str(half)] = {
            "n": len(idxs),
            "joint_accuracy": mean([joint_vec[i] for i in idxs]),
            "strongest_comparator_accuracy": mean([
                strongest_vec[i] for i in idxs
            ]),
            "joint_minus_strongest": mean([joint_vec[i] for i in idxs])
            - mean([strongest_vec[i] for i in idxs]),
        }

    validity = {
        "base_root": mean([base[pid]["answer"] is not None for pid in ids]),
        "generic_root": mean([generic[pid]["answer"] is not None for pid in ids]),
        "arms_target_min": min(
            metric["target_parse_validity"] for metric in arm_metrics.values()
        ),
        "arms_root_min": min(
            metric["root_parse_validity"] for metric in arm_metrics.values()
        ),
    }
    accounting_ok = all(
        base[pid]["calls"] == generic[pid]["calls"] == 1
        and base[pid]["max_new"] == generic[pid]["max_new"] == MAX_NEW
        for pid in ids
    ) and all(
        interventions[(pid, node, action)]["calls"] == 1
        and interventions[(pid, node, action)]["max_new"] == MAX_NEW
        for pid in ids for node, action in ARMS
    )

    gates = {
        "complete_128x13_plus_base_generic": (
            len(ids) == N_GRAPHS and len(base) == N_GRAPHS
            and len(generic) == N_GRAPHS
            and len(interventions) == N_GRAPHS * len(ARMS)
        ),
        "parse_validity": (
            validity["generic_root"] >= 0.95
            and validity["arms_target_min"] >= 0.90
            and validity["arms_root_min"] >= 0.95
        ),
        "base_errors_ge_40": len(error_ids) >= 40,
        "joint_vs_strongest_ge_8pp": joint_acc - strongest_acc >= 0.08,
        "joint_vs_action_oracle_ge_3pp": joint_acc - action_oracle_acc >= 0.03,
        "joint_vs_node_oracle_ge_3pp": joint_acc - node_oracle_acc >= 0.03,
        "two_actions_exclusive_ge_3": sum(
            value >= 3 for value in exclusive_actions.values()
        ) >= 2,
        "two_nodes_exclusive_ge_3": sum(
            value >= 3 for value in exclusive_nodes.values()
        ) >= 2,
        "nonnegative_both_hash_halves": all(
            values["joint_minus_strongest"] >= 0
            for values in stability.values()
        ),
        "one_call_equal_cap_complete_accounting": accounting_ok,
    }

    return {
        "protocol": PROTOCOL,
        "n": n,
        "ids": ids,
        "id_sha256": hashlib.sha256(
            ",".join(map(str, ids)).encode()
        ).hexdigest(),
        "n_base_errors": len(error_ids),
        "validity": validity,
        "accuracy": {
            "base": base_acc,
            "generic_raw": generic_raw,
            "generic_oracle_keep": generic_keep,
            "best_fixed_arm": best_fixed,
            "best_fixed_oracle_keep": best_fixed_acc,
            "strongest_equal_call_comparator": strongest_name,
            "strongest_equal_call_accuracy": strongest_acc,
            "uniform_one_intervention_expected": uniform_expected,
            "all_13_majority_vote": mean(majority_vec),
            "action_oracle_best_fixed_node": best_action_oracle_node,
            "action_oracle_accuracy": action_oracle_acc,
            "node_oracle_best_fixed_action": best_node_oracle_action,
            "node_oracle_accuracy": node_oracle_acc,
            "node_x_action_oracle": joint_acc,
            "joint_minus_strongest": joint_acc - strongest_acc,
            "joint_minus_action_oracle": joint_acc - action_oracle_acc,
            "joint_minus_node_oracle": joint_acc - node_oracle_acc,
        },
        "mcnemar_joint_vs_strongest": exact_mcnemar(joint_vec, strongest_vec),
        "fixed_node_action_oracles": fixed_node,
        "fixed_action_node_oracles": fixed_action,
        "arms": arm_metrics,
        "exclusive_action_repairs": exclusive_actions,
        "exclusive_node_repairs": exclusive_nodes,
        "action_pairwise_complementarity": pairwise,
        "stability": stability,
        "cost": {
            "base_calls_per_graph": 1,
            "generic_calls_per_graph": 1,
            "single_intervention_calls_per_graph": 1,
            "majority_vote_calls_per_graph": len(ARMS),
            "max_new_per_call": MAX_NEW,
            "total_calls": n * (2 + len(ARMS)),
            "total_prompt_tokens": sum(base[pid]["prompt_tokens"] for pid in ids)
            + sum(generic[pid]["prompt_tokens"] for pid in ids)
            + sum(
                interventions[(pid, node, action)]["prompt_tokens"]
                for pid in ids for node, action in ARMS
            ),
            "total_generated_tokens": sum(
                base[pid]["generated_tokens"] for pid in ids
            ) + sum(generic[pid]["generated_tokens"] for pid in ids) + sum(
                interventions[(pid, node, action)]["generated_tokens"]
                for pid in ids for node, action in ARMS
            ),
        },
        "gates": gates,
        "gate_pass": all(gates.values()),
    }


def print_report(report: dict) -> None:
    acc = report["accuracy"]
    print(
        f"n={report['n']} base_errors={report['n_base_errors']} "
        f"base={acc['base']:.3f} generic_keep={acc['generic_oracle_keep']:.3f} "
        f"best_fixed={acc['best_fixed_oracle_keep']:.3f} "
        f"action_oracle={acc['action_oracle_accuracy']:.3f} "
        f"node_oracle={acc['node_oracle_accuracy']:.3f} "
        f"joint={acc['node_x_action_oracle']:.3f}", flush=True,
    )
    print(
        f"joint deltas: strongest={acc['joint_minus_strongest']:+.3f} "
        f"action_oracle={acc['joint_minus_action_oracle']:+.3f} "
        f"node_oracle={acc['joint_minus_node_oracle']:+.3f}", flush=True,
    )
    print(f"exclusive_actions={report['exclusive_action_repairs']}", flush=True)
    print(f"exclusive_nodes={report['exclusive_node_repairs']}", flush=True)
    for name, passed in report["gates"].items():
        print(f"GATE {name}: {'PASS' if passed else 'FAIL'}", flush=True)
    print(
        f"TYPED_NODE_ACTION_ORACLE={'PASS' if report['gate_pass'] else 'FAIL'}",
        flush=True,
    )


def self_test() -> None:
    text = "work\nTARGET: \\boxed{5}\nROOT: \\boxed{11}"
    assert extract_boxed_all(text) == ["5", "11"]
    assert parse_intervention(text) == ("5", "11")
    assert len(ARMS) == 13
    assert stable_rank(7) == stable_rank(7)
    assert stable_half(7) in (0, 1)
    print("SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gsm_join_train.jsonl")
    parser.add_argument("--baseline-parents", required=True)
    parser.add_argument("--out-dir", default="typed_node_action_oracle_v0")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    os.makedirs(args.out_dir, exist_ok=True)
    rows = selected_rows(args.data)
    units = base_units(rows, load_parent_greedy(args.baseline_parents))
    base_path = os.path.join(args.out_dir, "base.jsonl")
    generic_path = os.path.join(args.out_dir, "generic.jsonl")
    intervention_path = os.path.join(args.out_dir, "interventions.jsonl")
    if args.analyze_only:
        base = {int(row["id"]): row for row in jread(base_path)}
        generic = {int(row["id"]): row for row in jread(generic_path)}
        interventions = {
            (int(row["id"]), row["node"], row["action"]): row
            for row in jread(intervention_path)
        }
    else:
        runner = Runner(args.model)
        base = run_base_roots(runner, units, base_path, args.batch_size)
        generic = run_generic(
            runner, units, base, generic_path, args.batch_size
        )
        interventions = run_interventions(
            runner, units, base, intervention_path, args.batch_size
        )

    expected = len(units) * len(ARMS)
    if len(base) != len(units) or len(generic) != len(units) or len(interventions) != expected:
        raise SystemExit(
            f"incomplete base={len(base)} generic={len(generic)} "
            f"interventions={len(interventions)} expected={expected}"
        )
    report = analyze(units, base, generic, interventions)
    report_path = os.path.join(args.out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print_report(report)


if __name__ == "__main__":
    main()
