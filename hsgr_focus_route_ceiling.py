"""Held-out oracle ceiling for relation-specific HSGR evidence focus.

This is a causal action test, not an end-to-end hidden-state method.  Every arm
uses the same programmatically compiled final-hop goal, verified predecessor
values, original question, and complete set of gold support paragraphs.  Only
the repair instruction and a same-format GUIDE-FOCUS marker differ:

* neutral: no evidence block is selected;
* correct_focus: the final-hop support block is selected by an oracle;
* wrong_focus: a length-matched predecessor-hop support block is selected.

The sample excludes both the original Random(0), n=200 slice and the subsequent
seed-20260811 n=400 edge-repair slice.  The official MuSiQue alias-aware answer
normalization, exact match, and token F1 are used after balanced boxed-answer
extraction.

Pre-registered gates on the new n=400 slice:

1. route_headroom: correct_focus beats neutral by >=3pp normalized EM and
   >=0.03 mean official answer F1, recovers >=20% of base normalized-EM errors,
   and both paired tests have p<.05;
2. route_specificity: correct_focus beats wrong_focus by >=3pp normalized EM
   with exact paired McNemar p<.05;
3. depth_non_decreasing: the correct_focus-minus-neutral EM delta does not
   decrease from 2-hop to 3-hop to 4-hop.

Only all three permit a hidden-state routing observer/controller experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import string
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mh_ceiling import evidence_from_row  # noqa: E402
from mh_e0 import hop_deps, load_rows, resolve_goal  # noqa: E402
from pilot import Runner  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

SYSTEM = (
    "You execute exactly one current node in a typed multi-hop reasoning "
    "hierarchy. Use only the supplied compiled goal, verified predecessor "
    "mappings, and evidence. Return the current-node answer in \\boxed{}."
)

BASE_USER = """Original question (context only): {question}

Typed hierarchy state:
[CURRENT] hop {hop}/{n_hops}
[COMPILED CURRENT GOAL] {compiled_goal}
[VERIFIED PREDECESSORS]
{dependencies}

[ALL SUPPORT EVIDENCE]
{evidence_blocks}

Execute only the compiled current goal. Put its answer in \\boxed{{}}."""

REPAIR_USER = """Original question (context only): {question}

Typed hierarchy state:
[CURRENT] hop {hop}/{n_hops}
[COMPILED CURRENT GOAL] {compiled_goal}
[VERIFIED PREDECESSORS]
{dependencies}

[ALL SUPPORT EVIDENCE]
{evidence_blocks}

[PREVIOUS CANDIDATE]
{candidate}

{instruction}
Return only the repaired current-node answer in \\boxed{{}}."""

NEUTRAL_INSTRUCTION = (
    "[GENERIC RETRY] No evidence block is selected. Re-evaluate the compiled "
    "current goal against all evidence blocks and repair the candidate."
)

FOCUS_INSTRUCTION = (
    "[GUIDE ROUTE] The block marked GUIDE-FOCUS is selected for the current "
    "relation. Ground the repaired answer in that block; use other blocks "
    "only as context."
)


def unit_id(row: dict) -> str:
    return str(row.get("id") or row.get("_uid"))


def support_for_step(row: dict, step: dict) -> str:
    sp = step.get("support_paragraph")
    if isinstance(sp, dict) and sp.get("paragraph_text"):
        return f"[{sp.get('title', '')}] {sp['paragraph_text']}".strip()
    idx = step.get("paragraph_support_idx")
    for i, paragraph in enumerate(row.get("paragraphs") or []):
        if not isinstance(paragraph, dict):
            continue
        para_idx = paragraph.get("idx", i)
        if para_idx == idx and paragraph.get("paragraph_text"):
            return (
                f"[{paragraph.get('title', '')}] "
                f"{paragraph['paragraph_text']}"
            ).strip()
    return ""


def render_evidence(blocks: list[str], focus_idx: int | None) -> str:
    rendered = []
    for i, block in enumerate(blocks):
        marker = " | GUIDE-FOCUS" if i == focus_idx else ""
        rendered.append(f"[E{i + 1}{marker}]\n{block}")
    return "\n\n".join(rendered)


def make_unit(row: dict) -> dict | None:
    decomp = row.get("question_decomposition") or []
    if len(decomp) < 2:
        return None
    hop = len(decomp) - 1
    deps = hop_deps(decomp)[hop]
    if not deps:
        return None
    blocks = [support_for_step(row, step) for step in decomp]
    if any(not block for block in blocks):
        return None

    pred_vals = {j: str(decomp[j]["answer"]) for j in deps}
    compiled_goal = resolve_goal(str(decomp[hop]["question"]), pred_vals)
    dependencies = "\n".join(
        f"  - #{j + 1} = {pred_vals[j]} (verified)" for j in deps
    )
    wrong_candidates = list(range(hop))
    wrong_focus = min(
        wrong_candidates,
        key=lambda j: (abs(len(blocks[j]) - len(blocks[hop])), j),
    )
    fields = {
        "question": str(row["question"]),
        "hop": hop + 1,
        "n_hops": len(decomp),
        "compiled_goal": compiled_goal,
        "dependencies": dependencies,
    }
    return {
        "id": unit_id(row),
        "n_hops": len(decomp),
        "n_dependencies": len(deps),
        "gold": str(row["answer"]),
        "aliases": list(row.get("answer_aliases") or []),
        "fields": fields,
        "blocks": blocks,
        "correct_focus": hop,
        "wrong_focus": wrong_focus,
        "focus_collision": blocks[hop] == blocks[wrong_focus],
    }


def select_fresh_units(
    data: str,
    prior_cases: str,
    original_limit: int,
    prior_limit: int,
    prior_seed: int,
    new_limit: int,
    seed: int,
) -> tuple[list[dict], dict]:
    original_ids = {
        unit_id(row) for row in load_rows(data, original_limit, seed=0)
    }
    all_units = []
    skipped_incomplete_support = 0
    for row in load_rows(data, 0, seed=0):
        if unit_id(row) in original_ids:
            continue
        unit = make_unit(row)
        if unit is None:
            skipped_incomplete_support += 1
            continue
        all_units.append(unit)

    if len(all_units) < prior_limit + new_limit:
        raise SystemExit(
            f"need >= {prior_limit + new_limit} dependency units after original "
            f"exclusion; found {len(all_units)}"
        )
    prior_source = "deterministic_reconstruction"
    if prior_cases and os.path.isfile(prior_cases):
        with open(prior_cases, encoding="utf-8") as handle:
            prior_ids = {
                str(json.loads(line)["id"]) for line in handle if line.strip()
            }
        if len(prior_ids) != prior_limit:
            raise SystemExit(
                f"expected {prior_limit} unique prior case ids in {prior_cases}; "
                f"found {len(prior_ids)}"
            )
        prior_source = os.path.abspath(prior_cases)
    else:
        prior_units = random.Random(prior_seed).sample(all_units, prior_limit)
        prior_ids = {unit["id"] for unit in prior_units}
    fresh_pool = [unit for unit in all_units if unit["id"] not in prior_ids]
    if len(fresh_pool) < new_limit:
        raise SystemExit(f"need {new_limit} fresh units; found {len(fresh_pool)}")
    chosen = random.Random(seed).sample(fresh_pool, new_limit)
    meta = {
        "original_excluded": len(original_ids),
        "prior_edge_slice_excluded": len(prior_ids),
        "eligible_before_prior_exclusion": len(all_units),
        "eligible_fresh_pool": len(fresh_pool),
        "skipped_incomplete_support": skipped_incomplete_support,
        "selection_seed": seed,
        "prior_seed": prior_seed,
        "prior_exclusion_source": prior_source,
        "prior_id_sha256": hashlib.sha256(
            "\n".join(sorted(prior_ids)).encode("utf-8")
        ).hexdigest(),
        "hop_counts": dict(Counter(u["n_hops"] for u in chosen)),
        "focus_text_collisions": sum(u["focus_collision"] for u in chosen),
        "id_sha256": hashlib.sha256(
            "\n".join(sorted(u["id"] for u in chosen)).encode("utf-8")
        ).hexdigest(),
    }
    return chosen, meta


def extract_boxed_balanced(text: str) -> str | None:
    text = text or ""
    idx = text.rfind("\\boxed")
    if idx >= 0:
        start = text.find("{", idx)
        if start >= 0:
            depth = 0
            for j in range(start, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start + 1 : j].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def strip_latex_wrappers(answer: str | None) -> str:
    value = (answer or "").strip()
    previous = None
    while value != previous:
        previous = value
        value = re.sub(
            r"\\(?:text|mbox|textbf|mathrm)\s*\{([^{}]*)\}", r"\1", value
        )
    return value.strip()


def parsed_answer(text: str) -> str:
    return strip_latex_wrappers(extract_boxed_balanced(text))


def normalize_answer(answer: str) -> str:
    """Official MuSiQue/SQuAD-style normalization."""
    answer = (answer or "").lower()
    answer = "".join(ch for ch in answer if ch not in set(string.punctuation))
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    return " ".join(answer.split())


def answer_em(pred: str, golds: list[str]) -> bool:
    pred_norm = normalize_answer(pred)
    return any(pred_norm == normalize_answer(gold) for gold in golds if gold)


def answer_f1_one(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    if not same:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_f1(pred: str, golds: list[str]) -> float:
    return max((answer_f1_one(pred, gold) for gold in golds if gold), default=0.0)


def score_text(unit: dict, text: str) -> tuple[bool, float, str]:
    answer = parsed_answer(text)
    golds = [unit["gold"]] + unit["aliases"]
    return answer_em(answer, golds), answer_f1(answer, golds), answer


def exact_mcnemar(a: list[bool], b: list[bool]) -> dict:
    a_only = sum(x and not y for x, y in zip(a, b))
    b_only = sum(y and not x for x, y in zip(a, b))
    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        k = min(a_only, b_only)
        tail = sum(math.comb(n, j) for j in range(k + 1)) / (2**n)
        p = min(1.0, 2.0 * tail)
    return {"a_only": a_only, "b_only": b_only, "discordant": n, "p": p}


def paired_signflip_p(
    a: list[float], b: list[float], seed: int, samples: int
) -> dict:
    diffs = [x - y for x, y in zip(a, b)]
    observed = sum(diffs) / max(1, len(diffs))
    nonzero = [value for value in diffs if value != 0]
    if not nonzero:
        return {"mean_delta": observed, "nonzero": 0, "samples": samples, "p": 1.0}
    rng = random.Random(seed)
    extreme = 0
    target = abs(observed)
    for _ in range(samples):
        randomized = sum(value if rng.random() < 0.5 else -value for value in nonzero)
        randomized /= len(diffs)
        if abs(randomized) >= target - 1e-15:
            extreme += 1
    return {
        "mean_delta": observed,
        "nonzero": len(nonzero),
        "samples": samples,
        "p": (extreme + 1) / (samples + 1),
    }


def flatten(outputs: list[list[str]]) -> list[str]:
    return [row[0] for row in outputs]


def prompt_tokens(runner: Runner, prompts: list[str]) -> int:
    return sum(len(runner.tok.encode(p, add_special_tokens=False)) for p in prompts)


def mean(values: list[float | bool]) -> float:
    return sum(values) / max(1, len(values))


def build_base_prompt(unit: dict) -> str:
    fields = dict(unit["fields"])
    fields["evidence_blocks"] = render_evidence(unit["blocks"], None)
    return BASE_USER.format(**fields)


def build_repair_prompt(unit: dict, candidate: str, arm: str) -> str:
    if arm == "neutral":
        focus_idx = None
        instruction = NEUTRAL_INSTRUCTION
    elif arm == "correct_focus":
        focus_idx = unit["correct_focus"]
        instruction = FOCUS_INSTRUCTION
    elif arm == "wrong_focus":
        focus_idx = unit["wrong_focus"]
        instruction = FOCUS_INSTRUCTION
    else:
        raise ValueError(arm)
    fields = dict(unit["fields"])
    fields.update(
        evidence_blocks=render_evidence(unit["blocks"], focus_idx),
        candidate=candidate[:800],
        instruction=instruction,
    )
    return REPAIR_USER.format(**fields)


def dry_run(units: list[dict], meta: dict) -> None:
    assert parsed_answer(r"\boxed{\text{The Hateful Eight}}") == "The Hateful Eight"
    assert normalize_answer("The Charles University,") == "charles university"
    assert abs(answer_f1("Charles University in Prague", ["Charles University"]) - 2 / 3) < 1e-12
    print(json.dumps(meta, indent=1, ensure_ascii=False))
    for unit in units[:2]:
        print("\n== unit ==")
        print(
            json.dumps(
                {
                    "id": unit["id"],
                    "n_hops": unit["n_hops"],
                    "correct_focus": unit["correct_focus"],
                    "wrong_focus": unit["wrong_focus"],
                    "compiled_goal": unit["fields"]["compiled_goal"],
                },
                ensure_ascii=False,
            )
        )
        print(build_repair_prompt(unit, r"\\boxed{wrong}", "correct_focus")[:1800])


def main(args) -> None:
    data = args.data if os.path.isabs(args.data) else os.path.join(HERE, args.data)
    units, selection_meta = select_fresh_units(
        data=data,
        prior_cases=args.prior_cases,
        original_limit=args.original_exclude_limit,
        prior_limit=args.prior_exclude_limit,
        prior_seed=args.prior_seed,
        new_limit=args.new_limit,
        seed=args.seed,
    )
    print(f"[data] {json.dumps(selection_meta, ensure_ascii=False)}", flush=True)
    if args.dry_run:
        dry_run(units, selection_meta)
        return

    runner = Runner(args.model)
    token_counts = {}
    base_prompts = [build_base_prompt(unit) for unit in units]
    before = runner.n_new_tokens
    base_texts = flatten(
        runner.chat_batch(base_prompts, system=SYSTEM, max_new=args.max_new, bs=args.bs)
    )
    token_counts["base_prompt"] = prompt_tokens(runner, base_prompts)
    token_counts["base_generated"] = runner.n_new_tokens - before

    base_em, base_f1, base_answers = [], [], []
    for unit, text in zip(units, base_texts):
        em, f1, answer = score_text(unit, text)
        base_em.append(em)
        base_f1.append(f1)
        base_answers.append(answer)
    wrong = [i for i, hit in enumerate(base_em) if not hit]
    print(
        f"[base] official_em={mean(base_em):.4f} "
        f"official_f1={mean(base_f1):.4f} wrong={len(wrong)}/{len(units)}",
        flush=True,
    )

    arm_texts = {arm: list(base_texts) for arm in ("neutral", "correct_focus", "wrong_focus")}
    arm_em = {arm: list(base_em) for arm in arm_texts}
    arm_f1 = {arm: list(base_f1) for arm in arm_texts}
    arm_answers = {arm: list(base_answers) for arm in arm_texts}

    for arm in ("neutral", "correct_focus", "wrong_focus"):
        prompts = [build_repair_prompt(units[i], base_texts[i], arm) for i in wrong]
        before = runner.n_new_tokens
        repaired = flatten(
            runner.chat_batch(prompts, system=SYSTEM, max_new=args.max_new, bs=args.bs)
        )
        token_counts[f"{arm}_prompt"] = prompt_tokens(runner, prompts)
        token_counts[f"{arm}_generated"] = runner.n_new_tokens - before
        for idx, text in zip(wrong, repaired):
            em, f1, answer = score_text(units[idx], text)
            arm_texts[arm][idx] = text
            arm_em[arm][idx] = em
            arm_f1[arm][idx] = f1
            arm_answers[arm][idx] = answer
        print(
            f"[{arm}] official_em={mean(arm_em[arm]):.4f} "
            f"official_f1={mean(arm_f1[arm]):.4f} "
            f"recovered_em={sum(arm_em[arm][i] for i in wrong)}/{len(wrong)}",
            flush=True,
        )

    comparisons = {
        "correct_vs_neutral_em": exact_mcnemar(
            arm_em["correct_focus"], arm_em["neutral"]
        ),
        "correct_vs_wrong_em": exact_mcnemar(
            arm_em["correct_focus"], arm_em["wrong_focus"]
        ),
        "correct_vs_neutral_f1": paired_signflip_p(
            arm_f1["correct_focus"],
            arm_f1["neutral"],
            seed=args.seed,
            samples=args.permutation_samples,
        ),
    }
    deltas = {
        "correct_vs_neutral_em": mean(arm_em["correct_focus"]) - mean(arm_em["neutral"]),
        "correct_vs_wrong_em": mean(arm_em["correct_focus"]) - mean(arm_em["wrong_focus"]),
        "correct_vs_neutral_f1": mean(arm_f1["correct_focus"]) - mean(arm_f1["neutral"]),
    }
    recovery = sum(arm_em["correct_focus"][i] for i in wrong) / max(1, len(wrong))
    depth_delta = {}
    for hop in sorted({unit["n_hops"] for unit in units}):
        idxs = [i for i, unit in enumerate(units) if unit["n_hops"] == hop]
        depth_delta[str(hop)] = mean([arm_em["correct_focus"][i] for i in idxs]) - mean(
            [arm_em["neutral"][i] for i in idxs]
        )
    depth_keys = [str(hop) for hop in (2, 3, 4) if str(hop) in depth_delta]
    depth_non_decreasing = all(
        depth_delta[b] >= depth_delta[a]
        for a, b in zip(depth_keys, depth_keys[1:])
    )

    gates = {
        "route_headroom": (
            deltas["correct_vs_neutral_em"] >= 0.03
            and deltas["correct_vs_neutral_f1"] >= 0.03
            and recovery >= 0.20
            and comparisons["correct_vs_neutral_em"]["p"] < 0.05
            and comparisons["correct_vs_neutral_f1"]["p"] < 0.05
        ),
        "route_specificity": (
            deltas["correct_vs_wrong_em"] >= 0.03
            and comparisons["correct_vs_wrong_em"]["p"] < 0.05
        ),
        "depth_non_decreasing": depth_non_decreasing,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    cases_path = os.path.join(args.out_dir, "hsgr_focus_route_cases.jsonl")
    with open(cases_path, "w", encoding="utf-8") as handle:
        for i, unit in enumerate(units):
            record = {
                "id": unit["id"],
                "n_hops": unit["n_hops"],
                "compiled_goal": unit["fields"]["compiled_goal"],
                "gold": unit["gold"],
                "correct_focus": unit["correct_focus"],
                "wrong_focus": unit["wrong_focus"],
                "focus_collision": unit["focus_collision"],
                "base": {
                    "answer": base_answers[i], "em": base_em[i], "f1": base_f1[i],
                    "text": base_texts[i][:1000],
                },
            }
            for arm in arm_texts:
                record[arm] = {
                    "answer": arm_answers[arm][i],
                    "em": arm_em[arm][i],
                    "f1": arm_f1[arm][i],
                    "text": arm_texts[arm][i][:1000],
                    "reused_base": base_em[i],
                }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "experiment": "HSGR oracle relation-focus action ceiling",
        "claim_boundary": (
            "Gold final-hop focus measures oracle action headroom only. It does not "
            "establish a deployable focus selector, hidden-state observer, or controller."
        ),
        "data": {"n": len(units), **selection_meta},
        "scorer": {
            "primary": "official MuSiQue alias-aware answer F1",
            "binary": "official MuSiQue/SQuAD-style normalized EM",
            "output_parser": "balanced boxed extraction plus LaTeX text-wrapper stripping",
        },
        "official_em": {
            "base": mean(base_em),
            **{arm: mean(values) for arm, values in arm_em.items()},
        },
        "official_f1": {
            "base": mean(base_f1),
            **{arm: mean(values) for arm, values in arm_f1.items()},
        },
        "base_wrong_em": len(wrong),
        "correct_focus_recovery_rate": recovery,
        "deltas": deltas,
        "paired_tests": comparisons,
        "depth_correct_vs_neutral_em": depth_delta,
        "gates": gates,
        "advance_to_hidden_route_guide": all(gates.values()),
        "token_counts": token_counts,
        "model": args.model,
    }
    report_path = os.path.join(args.out_dir, "hsgr_focus_route_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, ensure_ascii=False)

    print("\n== HSGR oracle relation-focus action ceiling ==")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"saved {report_path} and {cases_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/musique_ans_val.jsonl")
    parser.add_argument("--prior-cases", default="")
    parser.add_argument("--out-dir", default="hsgr_focus_route")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--original-exclude-limit", type=int, default=200)
    parser.add_argument("--prior-exclude-limit", type=int, default=400)
    parser.add_argument("--prior-seed", type=int, default=20260811)
    parser.add_argument("--new-limit", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--permutation-samples", type=int, default=50000)
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
