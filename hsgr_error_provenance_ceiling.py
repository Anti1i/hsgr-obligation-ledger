"""Oracle action ceiling for an HSGR dependency-error provenance Guide.

This development-only experiment separates two causes of a wrong downstream
answer on the exact two-node graphs in ``data/gsm_chain_test.jsonl``:

* UPSTREAM: Question 1 is wrong and its value is propagated to Question 2;
* LOCAL: Question 1 is correct but Question 2 is wrong.

Three one-call repair prompts receive the same problem and base trace.  The
source oracle selects UPSTREAM_REPAIR or LOCAL_REPAIR from the exact hop-1
label, never from repair outcomes.  The experiment is an action-space ceiling;
it extracts no hidden states and trains no classifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict

from answer_check import answers_equal, extract_boxed
from pilot import JWriter, Runner, jread


HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_BASE = 160
MAX_NEW_REPAIR = 192

BASE_SYSTEM = (
    "You solve arithmetic word problems accurately. Follow the requested node "
    "boundary and put the requested answer in the last \\boxed{...}."
)

Q1_USER = """Solve only Question 1 from this composed problem.

{problem}

Give a short calculation and end with the Question-1 answer in \\boxed{{...}}."""

Q2_USER = """Solve only Question 2.  The dependency value from Question 1 is
fixed to: {hop1}

Question 2 after binding that dependency:
{question2}

Give a short calculation and end with the Question-2 answer in \\boxed{{...}}."""

REPAIR_SYSTEM = (
    "You repair one two-node arithmetic reasoning graph. Obey the specified "
    "repair action. Use the full problem and base trace below. End with only "
    "the repaired Question-2 answer in the last \\boxed{...}."
)

REPAIR_USER = """Full composed problem:
{problem}

Base trace:
[QUESTION 1 PREDICTION] {q1_answer}
[QUESTION 2 PREDICTION] {q2_answer}

[REPAIR ACTION]
{action}

Show a short calculation and end with the repaired Question-2 answer in
\\boxed{{...}}."""

ACTIONS = {
    "generic": (
        "Inspect both nodes, find any error, and correct the final answer."
    ),
    "upstream": (
        "Recompute Question 1 first. Then propagate that recomputed value into "
        "Question 2 and recompute the final answer."
    ),
    "local": (
        "Treat the base Question-1 prediction as fixed and do not recompute or "
        "change it. Recompute only Question 2 using that fixed value."
    ),
}


def read_rows(path: str, limit: int = 0) -> list[dict]:
    rows = jread(path)
    if limit:
        rows = rows[:limit]
    out = []
    for idx, row in enumerate(rows):
        required = ("problem", "answer", "hop1_answer")
        if not all(row.get(key) is not None for key in required):
            raise ValueError(f"row {idx} misses one of {required}")
        item = dict(row)
        item["id"] = idx
        out.append(item)
    return out


def split_questions(problem: str) -> tuple[str, str]:
    match = re.search(
        r"\AQuestion 1:\s*(.*?)\n\nQuestion 2:\s*(.*?)"
        r"\n\nGive the answer to Question 2\.\s*\Z",
        problem,
        flags=re.S,
    )
    if not match:
        raise ValueError(f"cannot split composed problem: {problem[:120]!r}")
    return match.group(1).strip(), match.group(2).strip()


def bind_question2(question2: str, hop1: str) -> str:
    marker = "(the answer to Question 1)"
    if marker not in question2:
        raise ValueError("Question 2 has no dependency marker")
    return question2.replace(marker, str(hop1), 1)


def answer_from(text: str) -> str | None:
    return extract_boxed(text)


def chat_token_count(runner: Runner, user: str, system: str) -> int:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    rendered = runner.tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return len(runner.tok(rendered, add_special_tokens=False)["input_ids"])


def generated_token_count(runner: Runner, text: str) -> int:
    return len(runner.tok(text, add_special_tokens=False)["input_ids"])


def batched_generate(
    runner: Runner,
    users: list[str],
    system: str,
    max_new: int,
    batch_size: int,
) -> list[str]:
    outputs = []
    for start in range(0, len(users), batch_size):
        chunk = users[start : start + batch_size]
        raw = runner.chat_batch(
            chunk, system=system, max_new=max_new, bs=batch_size
        )
        outputs.extend(x[0] for x in raw)
        print(f"[generate] {min(start + batch_size, len(users))}/{len(users)}", flush=True)
    return outputs


def run_base(runner: Runner, rows: list[dict], out_dir: str, batch_size: int) -> list[dict]:
    path = os.path.join(out_dir, "base.jsonl")
    done = {int(row["id"]): row for row in jread(path)}
    todo = [row for row in rows if row["id"] not in done]
    if todo:
        writer = JWriter(path)
        q1_users = [Q1_USER.format(problem=row["problem"]) for row in todo]
        q1_texts = batched_generate(
            runner, q1_users, BASE_SYSTEM, MAX_NEW_BASE, batch_size
        )
        q2_users = []
        interim = []
        for row, q1_user, q1_text in zip(todo, q1_users, q1_texts):
            q1_answer = answer_from(q1_text)
            _, q2 = split_questions(row["problem"])
            bound = bind_question2(q2, q1_answer or "UNKNOWN")
            q2_user = Q2_USER.format(hop1=q1_answer or "UNKNOWN", question2=bound)
            q2_users.append(q2_user)
            interim.append((row, q1_user, q1_text, q1_answer, q2_user))
        q2_texts = batched_generate(
            runner, q2_users, BASE_SYSTEM, MAX_NEW_BASE, batch_size
        )
        for (row, q1_user, q1_text, q1_answer, q2_user), q2_text in zip(
            interim, q2_texts
        ):
            q2_answer = answer_from(q2_text)
            record = {
                "id": row["id"],
                "gold": str(row["answer"]),
                "hop1_gold": str(row["hop1_answer"]),
                "q1_text": q1_text,
                "q1_answer": q1_answer,
                "q1_hit": bool(q1_answer and answers_equal(q1_answer, row["hop1_answer"])),
                "q2_text": q2_text,
                "q2_answer": q2_answer,
                "base_hit": bool(q2_answer and answers_equal(q2_answer, row["answer"])),
                "prompt_tokens": (
                    chat_token_count(runner, q1_user, BASE_SYSTEM)
                    + chat_token_count(runner, q2_user, BASE_SYSTEM)
                ),
                "generated_tokens": (
                    generated_token_count(runner, q1_text)
                    + generated_token_count(runner, q2_text)
                ),
            }
            writer.write(record)
            done[row["id"]] = record
    return [done[row["id"]] for row in rows]


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
        for arm, action in ACTIONS.items():
            if (row["id"], arm) in done:
                continue
            user = REPAIR_USER.format(
                problem=row["problem"],
                q1_answer=b.get("q1_answer") or "UNKNOWN",
                q2_answer=b.get("q2_answer") or "UNKNOWN",
                action=action,
            )
            units.append((row, arm, user))
    if units:
        writer = JWriter(path)
        users = [unit[2] for unit in units]
        texts = batched_generate(
            runner, users, REPAIR_SYSTEM, MAX_NEW_REPAIR, batch_size
        )
        for (row, arm, user), output in zip(units, texts):
            answer = answer_from(output)
            record = {
                "id": row["id"],
                "arm": arm,
                "text": output,
                "answer": answer,
                "hit": bool(answer and answers_equal(answer, row["answer"])),
                "prompt_tokens": chat_token_count(runner, user, REPAIR_SYSTEM),
                "generated_tokens": generated_token_count(runner, output),
                "calls": 1,
                "max_new": MAX_NEW_REPAIR,
            }
            writer.write(record)
            done[(row["id"], arm)] = record
    return [done[(row["id"], arm)] for row in rows for arm in ACTIONS]


def exact_mcnemar(a: list[bool], b: list[bool]) -> dict:
    a_only = sum(bool(x) and not bool(y) for x, y in zip(a, b))
    b_only = sum(bool(y) and not bool(x) for x, y in zip(a, b))
    n = a_only + b_only
    if not n:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(0, min(a_only, b_only) + 1))
        p = min(1.0, 2.0 * tail / (2**n))
    return {"a_only": a_only, "b_only": b_only, "discordant": n, "p": p}


def mean(values) -> float:
    return sum(values) / max(1, len(values))


def sha_ids(ids: list[int]) -> str:
    return hashlib.sha256("\n".join(map(str, sorted(ids))).encode()).hexdigest()


def analyze(rows: list[dict], base: list[dict], repairs: list[dict]) -> dict:
    base_by_id = {row["id"]: row for row in base}
    rep = defaultdict(dict)
    for row in repairs:
        rep[row["id"]][row["arm"]] = row

    policy = defaultdict(list)
    sources = []
    case_rows = []
    for row in rows:
        pid = row["id"]
        b = base_by_id[pid]
        base_hit = bool(b["base_hit"])
        source = "none" if base_hit else ("local" if b["q1_hit"] else "upstream")
        sources.append(source)
        arms = {name: bool(rep[pid][name]["hit"]) for name in ACTIONS}
        policy["base"].append(base_hit)
        for arm in ACTIONS:
            policy[f"oracle_keep_{arm}"].append(base_hit or arms[arm])
        routed = base_hit or arms[source]
        policy["oracle_keep_source_routed"].append(routed)
        policy["oracle_keep_best_of_repairs"].append(base_hit or any(arms.values()))
        case_rows.append({
            "id": pid,
            "source": source,
            "q1_hit": bool(b["q1_hit"]),
            "base_hit": base_hit,
            **{f"{arm}_hit": hit for arm, hit in arms.items()},
            "source_routed_hit": routed,
        })

    accuracy = {name: mean(values) for name, values in policy.items()}
    fixed_names = ("oracle_keep_upstream", "oracle_keep_local")
    best_fixed = max(fixed_names, key=lambda name: (accuracy[name], name))
    source_name = "oracle_keep_source_routed"
    generic_name = "oracle_keep_generic"

    source_counts = Counter(sources)
    stratum = {}
    for source in ("upstream", "local"):
        idxs = [i for i, value in enumerate(sources) if value == source]
        stratum[source] = {
            "n": len(idxs),
            "upstream": mean([case_rows[i]["upstream_hit"] for i in idxs]),
            "local": mean([case_rows[i]["local_hit"] for i in idxs]),
            "generic": mean([case_rows[i]["generic_hit"] for i in idxs]),
        }
    cross_up = stratum["upstream"]["upstream"] - stratum["upstream"]["local"]
    cross_local = stratum["local"]["local"] - stratum["local"]["upstream"]

    mcnemar_generic = exact_mcnemar(policy[source_name], policy[generic_name])
    mcnemar_fixed = exact_mcnemar(policy[source_name], policy[best_fixed])
    delta_generic = accuracy[source_name] - accuracy[generic_name]
    delta_fixed = accuracy[source_name] - accuracy[best_fixed]

    token_costs = {}
    for arm in ACTIONS:
        arm_rows = [row for row in repairs if row["arm"] == arm]
        token_costs[arm] = {
            "calls_per_problem": mean([row["calls"] for row in arm_rows]),
            "prompt_tokens_per_problem": mean([row["prompt_tokens"] for row in arm_rows]),
            "generated_tokens_per_problem": mean(
                [row["generated_tokens"] for row in arm_rows]
            ),
            "max_new": MAX_NEW_REPAIR,
        }

    gates = {
        "both_error_strata_n_ge_30": min(
            source_counts.get("upstream", 0), source_counts.get("local", 0)
        ) >= 30,
        "source_vs_generic_ge_3pp_p_lt_05": (
            delta_generic >= 0.03 and mcnemar_generic["p"] < 0.05
        ),
        "source_vs_best_fixed_ge_3pp_p_lt_05": (
            delta_fixed >= 0.03 and mcnemar_fixed["p"] < 0.05
        ),
        "source_specific_crossover_ge_5pp": cross_up >= 0.05 and cross_local >= 0.05,
        "one_call_same_cap": all(
            cost["calls_per_problem"] == 1.0 and cost["max_new"] == MAX_NEW_REPAIR
            for cost in token_costs.values()
        ),
    }
    report = {
        "protocol": "EXPERIMENT_PROTOCOL_ERROR_PROVENANCE_GUIDE_V1.md",
        "n": len(rows),
        "id_sha256": sha_ids([row["id"] for row in rows]),
        "source_counts": dict(source_counts),
        "accuracy": accuracy,
        "best_fixed_policy": best_fixed,
        "delta_source_vs_generic": delta_generic,
        "delta_source_vs_best_fixed": delta_fixed,
        "mcnemar_source_vs_generic": mcnemar_generic,
        "mcnemar_source_vs_best_fixed": mcnemar_fixed,
        "strata": stratum,
        "crossover": {"upstream": cross_up, "local": cross_local},
        "token_costs": token_costs,
        "gates": gates,
        "gate_pass": all(gates.values()),
        "cases": case_rows,
    }
    return report


def print_report(report: dict) -> None:
    print(f"== HSGR dependency-error provenance ceiling (n={report['n']}) ==")
    print(f"sources={report['source_counts']}")
    for name, value in sorted(report["accuracy"].items()):
        print(f"  {name:30s} {value:.4f}")
    print(
        "source-routed vs generic: "
        f"{report['delta_source_vs_generic']:+.4f}  "
        f"McNemar={report['mcnemar_source_vs_generic']}"
    )
    print(
        f"source-routed vs {report['best_fixed_policy']}: "
        f"{report['delta_source_vs_best_fixed']:+.4f}  "
        f"McNemar={report['mcnemar_source_vs_best_fixed']}"
    )
    for source, values in report["strata"].items():
        print(
            f"  {source:8s} n={values['n']} upstream={values['upstream']:.4f} "
            f"local={values['local']:.4f} generic={values['generic']:.4f}"
        )
    print(f"crossover={report['crossover']}")
    for name, passed in report["gates"].items():
        print(f"GATE {name}: {'PASS' if passed else 'FAIL'}")
    print(f"OVERALL: {'PASS' if report['gate_pass'] else 'FAIL -> STOP'}")


def self_test() -> None:
    sample = (
        "Question 1: What is 2 plus 3?\n\n"
        "Question 2: Add 4 to (the answer to Question 1).\n\n"
        "Give the answer to Question 2."
    )
    q1, q2 = split_questions(sample)
    assert q1 == "What is 2 plus 3?"
    assert bind_question2(q2, "5") == "Add 4 to 5."
    assert answer_from("x \\boxed{5}") == "5"
    assert exact_mcnemar([True, False], [False, False])["p"] == 1.0
    rows = [
        {"id": 0, "answer": "9", "hop1_answer": "5"},
        {"id": 1, "answer": "9", "hop1_answer": "5"},
        {"id": 2, "answer": "9", "hop1_answer": "5"},
    ]
    base = [
        {"id": 0, "base_hit": True, "q1_hit": True},
        {"id": 1, "base_hit": False, "q1_hit": False},
        {"id": 2, "base_hit": False, "q1_hit": True},
    ]
    synthetic = {
        0: {"generic": True, "upstream": True, "local": True},
        1: {"generic": False, "upstream": True, "local": False},
        2: {"generic": False, "upstream": False, "local": True},
    }
    repairs = [
        {
            "id": pid,
            "arm": arm,
            "hit": hit,
            "calls": 1,
            "max_new": MAX_NEW_REPAIR,
            "prompt_tokens": 10,
            "generated_tokens": 2,
        }
        for pid, arms in synthetic.items()
        for arm, hit in arms.items()
    ]
    report = analyze(rows, base, repairs)
    assert report["source_counts"] == {"none": 1, "upstream": 1, "local": 1}
    assert report["accuracy"]["oracle_keep_source_routed"] == 1.0
    assert not report["gate_pass"]  # sample-count gate must prevent a toy pass
    print("SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gsm_chain_test.jsonl")
    parser.add_argument("--out-dir", default="hsgr_error_provenance")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    data_path = args.data if os.path.isabs(args.data) else os.path.join(HERE, args.data)
    out_dir = (
        args.out_dir if os.path.isabs(args.out_dir) else os.path.join(HERE, args.out_dir)
    )
    os.makedirs(out_dir, exist_ok=True)
    rows = read_rows(data_path, args.limit)
    if args.analyze_only:
        base = jread(os.path.join(out_dir, "base.jsonl"))
        repairs = jread(os.path.join(out_dir, "repairs.jsonl"))
    else:
        runner = Runner(args.model)
        base = run_base(runner, rows, out_dir, args.batch_size)
        repairs = run_repairs(runner, rows, base, out_dir, args.batch_size)
    if len(base) != len(rows) or len(repairs) != len(rows) * len(ACTIONS):
        raise RuntimeError(
            f"incomplete outputs: rows={len(rows)} base={len(base)} repairs={len(repairs)}"
        )
    report = analyze(rows, base, repairs)
    report_path = os.path.join(out_dir, "hsgr_error_provenance_report.json")
    cases = report.pop("cases")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with open(
        os.path.join(out_dir, "hsgr_error_provenance_cases.jsonl"),
        "w",
        encoding="utf-8",
    ) as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print_report(report)
    print(f"saved {report_path}")


if __name__ == "__main__":
    main()
