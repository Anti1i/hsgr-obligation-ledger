"""Build structure-de-oracled route-counterfactual hidden features.

The planner sees only the original question and the same full evidence used by
SC@8.  It predicts required predecessor questions plus one answerable foil.
All nodes are executed without gold annotations.  For every frozen SC@8 answer
candidate, matched and counterfactual verifier prompts contain exactly the
same question, evidence, node texts, predicted values, candidate, and multiset
of role labels.  The intervention swaps PARENT/NONPARENT roles between the
first predicted parent and the foil.

This script does not evaluate candidate correctness.  Gold labels carried by
the frozen candidate payloads are copied only for the later OOF evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict

from hsgr_structured_hidden_verifier import LAYERS, PROJECTION_DIM, projectors
from mh_ceiling import SYSTEM as SC_SYSTEM
from mh_ceiling import USER as SC_USER
from mh_ceiling import evidence_from_row, extract_boxed, normalize
from mh_e0 import load_rows
from pilot import JWriter, Runner, jread


PLANNER_SYSTEM = (
    "You predict a small dependency hierarchy for multi-hop question answering. "
    "Do not answer the original question and do not reveal any answer values."
)

PLANNER_USER = """Full evidence (the same evidence available to the answer generator):
{evidence}

Original question:
{question}

Predict a compact hierarchy that would help answer the original question.
Return exactly one JSON object with this schema:
{{
  "root_goal": "a short restatement of the original goal, without its answer",
  "parent_questions": ["one to three evidence-answerable predecessor questions"],
  "foil_question": "an evidence-answerable question that is related but not required by the root goal"
}}

Rules:
- Parent questions must be answerable from the full evidence and their answers
  must be useful inputs to the root goal.
- The foil must also be answerable from the evidence, but must not be needed.
- Do not include answers, gold labels, paragraph IDs, or support annotations.
- Do not repeat the original question as a parent or foil."""

EXECUTOR_SYSTEM = (
    "Answer exactly one predicted hierarchy node using only the supplied full "
    "evidence. Put the short answer in \\boxed{}."
)

EXECUTOR_USER = """Full evidence:
{evidence}

Original question (context only):
{question}

Predicted node question:
{node_question}

Answer only the predicted node question. Put the answer in \\boxed{{}}."""

VERIFY_SYSTEM = (
    "You are the HSGR Guide. Read a predicted hierarchy state and assess one "
    "proposed final answer. The role labels define which predicted nodes are "
    "predecessors of the root goal."
)

VERIFY_USER = """Original question:
{question}

Full evidence (identical in every intervention arm):
{evidence}

Predicted root goal:
{root_goal}

Predicted hierarchy state:
{state_block}

[PROPOSED FINAL ANSWER]
{candidate}

[GUIDE COMMITMENT CHECK]
Assess whether the proposed final answer is compatible with the predicted root
goal, the nodes marked PARENT, and the full evidence. The hidden state at the
assistant verdict slot will be read; do not generate a verdict."""

FALLBACK_PARENT = (
    "What intermediate entity or fact must be identified immediately before "
    "answering the original question?"
)
FALLBACK_FOIL = (
    "What other entity or fact is stated in the evidence but is not needed to "
    "answer the original question?"
)
ROUTES = ("matched", "counterfactual", "mismatch")


def sha_ids(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def json_object(text):
    decoder = json.JSONDecoder()
    for start, char in enumerate(text or ""):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def clean_question(value, fallback):
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    if not value:
        return fallback
    return value[:600]


def parse_plan(text, original_question):
    value = json_object(text) or {}
    parents = value.get("parent_questions")
    if not isinstance(parents, list):
        parents = []
    cleaned = []
    seen = set()
    original_norm = normalize(original_question)
    for item in parents:
        question = clean_question(item, "")
        key = normalize(question)
        if not question or not key or key == original_norm or key in seen:
            continue
        seen.add(key)
        cleaned.append(question)
        if len(cleaned) == 3:
            break
    if not cleaned:
        cleaned = [FALLBACK_PARENT]
    foil = clean_question(value.get("foil_question"), FALLBACK_FOIL)
    if normalize(foil) in seen or normalize(foil) == original_norm:
        foil = FALLBACK_FOIL
    return {
        "root_goal": clean_question(value.get("root_goal"), original_question),
        "parent_questions": cleaned,
        "foil_question": foil,
        "used_fallback_parent": cleaned == [FALLBACK_PARENT],
        "used_fallback_foil": foil == FALLBACK_FOIL,
    }


def load_candidate_metas(torch, paths):
    metas = []
    seen = set()
    source_sizes = []
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        local = payload.get("metas")
        if not isinstance(local, list):
            raise RuntimeError(f"missing metas in {path}")
        ids = {meta["id"] for meta in local}
        overlap = seen & ids
        if overlap:
            raise RuntimeError(f"candidate payload overlap: {len(overlap)} in {path}")
        seen.update(ids)
        source_sizes.append({"path": path, "problems": len(ids), "candidates": len(local)})
        for meta in local:
            metas.append(
                {
                    key: meta.get(key)
                    for key in (
                        "id",
                        "cand",
                        "ans",
                        "norm",
                        "label",
                        "n_hops",
                    )
                }
            )
    return metas, source_sizes


def rows_for_ids(data, ids):
    loaded = load_rows(data, 0, seed=0)
    by_id = {row["_uid"]: row for row in loaded}
    missing = set(ids) - set(by_id)
    if missing:
        raise RuntimeError(f"missing {len(missing)} consumed rows in data")
    # Strip answer/decomposition fields before any planner, executor, or hidden
    # feature prompt is built.  The open-book evidence string is frozen here.
    return {
        pid: {
            "_uid": pid,
            "question": by_id[pid]["question"],
            "_evidence": evidence_from_row(by_id[pid]),
        }
        for pid in ids
    }


def row_evidence(row):
    return row["_evidence"]


def planner_prompt(row):
    return PLANNER_USER.format(
        evidence=row_evidence(row), question=row["question"]
    )


def chat_prompt_token_count(tokenizer, system, user):
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def save_counters(path, counters):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(counters), handle, ensure_ascii=False, indent=2)


def plan_hierarchies(
    runner, rows, path, batch_size, counters, accounting_path
):
    completed = {item["id"]: item for item in jread(path)}
    writer = JWriter(path)
    ordered = [rows[pid] for pid in sorted(rows)]
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        missing = [row for row in batch if row["_uid"] not in completed]
        if missing:
            prompts = [planner_prompt(row) for row in missing]
            counters["planner_calls"] += len(prompts)
            counters["planner_prompt_tokens"] += sum(
                chat_prompt_token_count(runner.tok, PLANNER_SYSTEM, prompt)
                for prompt in prompts
            )
            before = runner.n_new_tokens
            outputs = runner.chat_batch(
                prompts,
                system=PLANNER_SYSTEM,
                max_new=320,
                bs=batch_size,
            )
            counters["planner_generated_tokens"] += runner.n_new_tokens - before
            for row, output in zip(missing, outputs):
                raw = output[0]
                plan = parse_plan(raw, row["question"])
                record = {"id": row["_uid"], **plan, "raw": raw[:2400]}
                completed[row["_uid"]] = record
                writer.write(record)
            save_counters(accounting_path, counters)
        print(f"[planner] {min(start + batch_size, len(ordered))}/{len(ordered)}", flush=True)
    return {pid: completed[pid] for pid in sorted(rows)}


def node_key(pid, kind, index):
    return f"{pid}\t{kind}\t{index}"


def execute_nodes(
    runner, rows, plans, path, batch_size, counters, accounting_path
):
    existing = {}
    for item in jread(path):
        existing[node_key(item["id"], item["kind"], item["index"])] = item
    units = []
    for pid in sorted(rows):
        plan = plans[pid]
        for index, question in enumerate(plan["parent_questions"]):
            units.append((pid, "parent", index, question))
        units.append((pid, "foil", 0, plan["foil_question"]))
    writer = JWriter(path)
    for start in range(0, len(units), batch_size):
        batch = units[start : start + batch_size]
        missing = [unit for unit in batch if node_key(*unit[:3]) not in existing]
        if missing:
            prompts = [
                EXECUTOR_USER.format(
                    evidence=row_evidence(rows[pid]),
                    question=rows[pid]["question"],
                    node_question=question,
                )
                for pid, _, _, question in missing
            ]
            counters["executor_calls"] += len(prompts)
            counters["executor_prompt_tokens"] += sum(
                chat_prompt_token_count(runner.tok, EXECUTOR_SYSTEM, prompt)
                for prompt in prompts
            )
            before = runner.n_new_tokens
            outputs = runner.chat_batch(
                prompts,
                system=EXECUTOR_SYSTEM,
                max_new=128,
                bs=batch_size,
            )
            counters["executor_generated_tokens"] += runner.n_new_tokens - before
            for unit, output in zip(missing, outputs):
                pid, kind, index, question = unit
                raw = output[0]
                answer = extract_boxed(raw) or "(no predicted value)"
                record = {
                    "id": pid,
                    "kind": kind,
                    "index": index,
                    "question": question,
                    "answer": str(answer)[:600],
                    "norm": normalize(answer),
                    "raw": raw[:1200],
                }
                existing[node_key(pid, kind, index)] = record
                writer.write(record)
            save_counters(accounting_path, counters)
        print(f"[executor] {min(start + batch_size, len(units))}/{len(units)}", flush=True)
    states = {}
    for pid in sorted(rows):
        parent_nodes = []
        for index, question in enumerate(plans[pid]["parent_questions"]):
            item = existing[node_key(pid, "parent", index)]
            parent_nodes.append({"question": question, "answer": item["answer"], "norm": item["norm"]})
        foil_item = existing[node_key(pid, "foil", 0)]
        states[pid] = {
            "root_goal": plans[pid]["root_goal"],
            "parents": parent_nodes,
            "foil": {
                "question": plans[pid]["foil_question"],
                "answer": foil_item["answer"],
                "norm": foil_item["norm"],
            },
            "predicted_depth": len(parent_nodes) + 1,
        }
    return states


def state_nodes(state):
    return list(state["parents"]) + [state["foil"]]


def state_block(state, mode):
    nodes = state_nodes(state)
    n_parent = len(state["parents"])
    if mode == "matched":
        roles = ["PARENT"] * n_parent + ["NONPARENT"]
    elif mode == "counterfactual":
        roles = ["PARENT"] * n_parent + ["NONPARENT"]
        roles[0], roles[-1] = roles[-1], roles[0]
    else:
        raise ValueError(mode)
    chunks = []
    for index, (node, role) in enumerate(zip(nodes, roles), 1):
        chunks.append(
            f"[NODE {index}] ROLE={role}\n"
            f"QUESTION: {node['question']}\n"
            f"PREDICTED VALUE: {node['answer']}"
        )
    return "\n\n".join(chunks)


def donor_map(states):
    grouped = defaultdict(list)
    for pid, state in states.items():
        grouped[state["predicted_depth"]].append(pid)
    donors = {}
    for _, pids in sorted(grouped.items()):
        ordered = sorted(
            pids,
            key=lambda pid: (
                len(state_block(states[pid], "counterfactual")),
                pid,
            ),
        )
        if len(ordered) == 1:
            donors[ordered[0]] = ordered[0]
        else:
            for index, pid in enumerate(ordered):
                donors[pid] = ordered[(index + 1) % len(ordered)]
    return donors


def verifier_user(row, state, candidate, mode, donor_state=None):
    if mode == "mismatch":
        block = state_block(donor_state, "counterfactual")
    else:
        block = state_block(state, mode)
    return VERIFY_USER.format(
        question=row["question"],
        evidence=row_evidence(row),
        root_goal=state["root_goal"],
        state_block=block,
        candidate=candidate or "(empty candidate)",
    )


def rendered_text(tokenizer, user):
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_route(runner, metas, rows, states, donors, route, batch_size, max_context, matrices, counters):
    torch = runner.torch
    tokenizer = runner.tok
    tokenizer.truncation_side = "left"
    end_parts = {layer: [] for layer in LAYERS}
    start_parts = {layer: [] for layer in LAYERS} if route == "matched" else None
    lengths = []
    processed = 0
    for start in range(0, len(metas), batch_size):
        batch = metas[start : start + batch_size]
        users = []
        texts = []
        boundary_chars = []
        raw_lengths = []
        prefix_lengths = []
        for meta in batch:
            pid = meta["id"]
            donor_state = states[donors[pid]] if route == "mismatch" else None
            user = verifier_user(
                rows[pid], states[pid], meta.get("ans") or "", route, donor_state
            )
            text = rendered_text(tokenizer, user)
            users.append(user)
            texts.append(text)
            raw_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            raw_lengths.append(len(raw_ids))
            if route == "matched":
                marker = text.rfind("[PROPOSED FINAL ANSWER]")
                if marker < 0:
                    raise RuntimeError("candidate boundary marker missing")
                prefix_lengths.append(
                    len(tokenizer(text[:marker], add_special_tokens=False)["input_ids"])
                )
                boundary_chars.append(marker)
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_context,
            add_special_tokens=False,
        ).to("cuda")
        seq_lengths = enc["attention_mask"].sum(dim=1).tolist()
        lengths.extend(int(value) for value in seq_lengths)
        processed += int(enc["input_ids"].numel())
        with torch.no_grad():
            result = runner.model(
                **enc,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        for layer in LAYERS:
            hidden = result.hidden_states[layer].float()
            end_hidden = hidden[:, -1, :]
            projected = end_hidden @ matrices[layer]
            projected = projected / (projected.norm(dim=1, keepdim=True) + 1e-8)
            end_parts[layer].append(projected.half().cpu())
            if route == "matched":
                positions = []
                padded_width = hidden.shape[1]
                for raw_len, prefix_len, seq_len in zip(raw_lengths, prefix_lengths, seq_lengths):
                    dropped = max(0, raw_len - max_context)
                    local = prefix_len - dropped - 1
                    if local < 0 or local >= seq_len:
                        raise RuntimeError("candidate boundary was truncated")
                    positions.append(padded_width - int(seq_len) + int(local))
                row_index = torch.arange(len(batch), device=hidden.device)
                pos_index = torch.tensor(positions, device=hidden.device)
                start_hidden = hidden[row_index, pos_index, :]
                start_projected = start_hidden @ matrices[layer]
                start_projected = start_projected / (
                    start_projected.norm(dim=1, keepdim=True) + 1e-8
                )
                start_parts[layer].append(start_projected.half().cpu())
        del enc, result
        torch.cuda.empty_cache()
        if start == 0 or (start + batch_size) % 320 == 0:
            print(f"[hidden-{route}] {min(start + batch_size, len(metas))}/{len(metas)}", flush=True)
    counters[f"{route}_attended_prompt_tokens"] = sum(lengths)
    counters[f"{route}_processed_forward_tokens"] = processed
    counters[f"{route}_truncated_candidates"] = sum(
        int(length == max_context) for length in lengths
    )
    output = {"end": {layer: torch.cat(parts) for layer, parts in end_parts.items()}}
    if route == "matched":
        output["start"] = {
            layer: torch.cat(parts) for layer, parts in start_parts.items()
        }
    return output, lengths


def candidate_scalars(metas, rows, states):
    counts = Counter((meta["id"], meta.get("norm")) for meta in metas if meta.get("norm"))
    grouped_indices = defaultdict(list)
    for index, meta in enumerate(metas):
        grouped_indices[meta["id"]].append(index)
    length_z = [0.0] * len(metas)
    for indices in grouped_indices.values():
        values = [len(metas[index].get("norm") or "") for index in indices]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        scale = math.sqrt(variance) if variance else 1.0
        for index, value in zip(indices, values):
            length_z[index] = (value - mean) / scale
    scalars = []
    summaries = {}
    for pid, state in states.items():
        summaries[pid] = {
            "predicted_depth": state["predicted_depth"],
            "n_parents": len(state["parents"]),
            "parent_values": [node["norm"] for node in state["parents"]],
            "foil_value": state["foil"]["norm"],
        }
    for index, meta in enumerate(metas):
        pid = meta["id"]
        norm = meta.get("norm") or ""
        evidence_norm = normalize(row_evidence(rows[pid]))
        parent_values = set(summaries[pid]["parent_values"])
        scalars.append(
            [
                counts[(pid, norm)] / max(1, len(grouped_indices[pid])) if norm else 0.0,
                float(bool(norm) and norm in evidence_norm),
                float(bool(norm) and norm in parent_values),
                float(bool(norm) and norm == summaries[pid]["foil_value"]),
                float(length_z[index]),
                min(4, summaries[pid]["predicted_depth"]) / 4.0,
                min(3, summaries[pid]["n_parents"]) / 3.0,
            ]
        )
    return scalars, summaries


def generation_accounting(tokenizer, rows, metas):
    # Exact prompt/model-call accounting can be reconstructed from the frozen
    # setup.  Full sampled completion text was not retained in feature payloads,
    # so answer-token counts are only an explicit lower bound.
    prompt_tokens = 0
    for pid in sorted(rows):
        user = SC_USER.format(evidence=row_evidence(rows[pid]), question=rows[pid]["question"])
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SC_SYSTEM},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens += 8 * len(tokenizer(text, add_special_tokens=False)["input_ids"])
    answer_lower_bound = sum(
        len(tokenizer(meta.get("ans") or "", add_special_tokens=False)["input_ids"])
        for meta in metas
    )
    return {
        "problems": len(rows),
        "samples_per_problem": 8,
        "model_calls": len(rows) * 8,
        "prompt_tokens_exact": prompt_tokens,
        "generated_answer_tokens_lower_bound": answer_lower_bound,
        "generated_completion_tokens_exact": None,
        "completion_accounting_complete": False,
    }


def main(args):
    import torch

    os.makedirs(args.out_dir, exist_ok=True)
    metas, source_sizes = load_candidate_metas(torch, args.source_features)
    ids = sorted({meta["id"] for meta in metas})
    if len(ids) != 840:
        raise RuntimeError(f"expected 840 consumed problems, got {len(ids)}")
    rows = rows_for_ids(args.data, ids)
    prepare_accounting_path = os.path.join(
        args.out_dir, "prepare_accounting.json"
    )
    counters = Counter()
    if os.path.isfile(prepare_accounting_path):
        with open(prepare_accounting_path, encoding="utf-8") as handle:
            counters.update(json.load(handle))
    runner = Runner(args.model)

    plans_path = os.path.join(args.out_dir, "predicted_hierarchies.jsonl")
    nodes_path = os.path.join(args.out_dir, "predicted_nodes.jsonl")
    plans = plan_hierarchies(
        runner,
        rows,
        plans_path,
        args.bs_planner,
        counters,
        prepare_accounting_path,
    )
    states = execute_nodes(
        runner,
        rows,
        plans,
        nodes_path,
        args.bs_executor,
        counters,
        prepare_accounting_path,
    )
    save_counters(prepare_accounting_path, counters)
    donors = donor_map(states)
    if any(donors[pid] == pid for pid in donors) and len(ids) > 1:
        singleton_depths = Counter(states[pid]["predicted_depth"] for pid in states)
        bad = [depth for depth, count in singleton_depths.items() if count == 1]
        if bad:
            print(f"[warning] mismatch self-donor for singleton predicted depths {bad}", flush=True)

    matrices = projectors(torch, runner.model.config.hidden_size, "cuda")
    route_features = {}
    route_lengths = {}
    for route in ROUTES:
        route_path = os.path.join(
            args.out_dir, f"route_counterfactual_{route}_features.pt"
        )
        if os.path.isfile(route_path):
            cached = torch.load(route_path, map_location="cpu")
            route_features[route] = cached["features"]
            route_lengths[route] = cached["lengths"]
            for key, value in cached["accounting"].items():
                counters[key] = value
            print(f"[hidden-{route}] reused {route_path}", flush=True)
        else:
            route_features[route], route_lengths[route] = extract_route(
                runner,
                metas,
                rows,
                states,
                donors,
                route,
                args.bs_hidden,
                args.max_context,
                matrices,
                counters,
            )
            prefix = f"{route}_"
            route_accounting = {
                key: value for key, value in counters.items() if key.startswith(prefix)
            }
            torch.save(
                {
                    "features": route_features[route],
                    "lengths": route_lengths[route],
                    "accounting": route_accounting,
                },
                route_path,
            )
            print(f"[hidden-{route}] checkpointed {route_path}", flush=True)
    unequal = sum(
        int(left != right)
        for left, right in zip(route_lengths["matched"], route_lengths["counterfactual"])
    )
    if unequal:
        raise RuntimeError(
            f"primary intervention token-length mismatch for {unequal} candidates"
        )

    scalars, summaries = candidate_scalars(metas, rows, states)
    accounting = {
        "hierarchy_and_predecessor_generation": dict(counters),
        "candidate_generation": generation_accounting(runner.tok, rows, metas),
        "matched_counterfactual_length_mismatches": unequal,
    }
    payload = {
        "protocol": "EXPERIMENT_PROTOCOL_ROUTE_COUNTERFACTUAL_GUIDE_V1.md",
        "claim_boundary": (
            "Structure-de-oracled predicted hierarchy and predecessor values; identical "
            "fixed open-book evidence "
            "in all primary arms; consumed 840-problem development pool only."
        ),
        "metas": metas,
        "scalars": torch.tensor(scalars, dtype=torch.float32),
        "state_summary": summaries,
        "donors": donors,
        "features": {
            "matched": route_features["matched"]["end"],
            "counterfactual": route_features["counterfactual"]["end"],
            "mismatch": route_features["mismatch"]["end"],
            "matched_start": route_features["matched"]["start"],
        },
        "data": {
            "problem_count": len(ids),
            "candidate_count": len(metas),
            "id_sha256": sha_ids(ids),
            "sources": source_sizes,
            "predicted_depth_counts": dict(Counter(state["predicted_depth"] for state in states.values())),
            "fallback_parent_count": sum(plan["used_fallback_parent"] for plan in plans.values()),
            "fallback_foil_count": sum(plan["used_fallback_foil"] for plan in plans.values()),
        },
        "accounting": accounting,
        "projection": {
            "layers": list(LAYERS),
            "dimension": PROJECTION_DIM,
        },
    }
    output_path = os.path.join(args.out_dir, "route_counterfactual_features.pt")
    torch.save(payload, output_path)
    with open(os.path.join(args.out_dir, "route_counterfactual_accounting.json"), "w", encoding="utf-8") as handle:
        json.dump({"data": payload["data"], "accounting": accounting}, handle, ensure_ascii=False, indent=2)
    print(f"[saved] {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--source-features", nargs=3, required=True)
    parser.add_argument("--out-dir", default="hsgr_route_counterfactual")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--bs-planner", type=int, default=8)
    parser.add_argument("--bs-executor", type=int, default=16)
    parser.add_argument("--bs-hidden", type=int, default=4)
    parser.add_argument("--max-context", type=int, default=8192)
    main(parser.parse_args())
