"""Conditional Hidden Graph-Energy Guide experiment.

The frozen protocol is in
EXPERIMENT_PROTOCOL_HIDDEN_GRAPH_ENERGY_GUIDE_V0.md.  This module deliberately
keeps candidate construction independent of hidden-reader training so the
action space can be audited before any learned score is inspected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from itertools import product
from typing import Iterable, Sequence


LAYERS = (14, 21, 28)
PROJECTION_DIM = 256
PROJECTION_SEED = 20260812


def read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", buffering=1, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_bucket(value: object, n: int, salt: str = "") -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    digest = hashlib.sha256(f"{salt}|{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n


def value_classes(candidates: Sequence[dict]) -> list[dict]:
    """Collapse duplicate normalized samples without discarding their mass.

    The first occurrence supplies the representative printable answer.  Every
    completion remains in ``members`` for later hidden-state averaging.
    Invalid/empty normalized values collapse into one explicit UNKNOWN class,
    preventing silent graph deletion.
    """
    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        raw_norm = candidate.get("norm")
        norm = (
            str(raw_norm)
            if raw_norm is not None and str(raw_norm).strip()
            else "__UNKNOWN__"
        )
        groups[norm].append((index, candidate))
    total = max(1, len(candidates))
    classes = []
    for norm, members in groups.items():
        first_index, first = members[0]
        classes.append({
            "norm": norm,
            "answer": first.get("answer") or "UNKNOWN",
            "first_index": first_index,
            "count": len(members),
            "frequency": len(members) / total,
            "member_indices": [index for index, _ in members],
            "members": [candidate for _, candidate in members],
        })
    classes.sort(key=lambda item: item["first_index"])
    return classes


def modal_norm(candidates: Sequence[dict]) -> str | None:
    classes = value_classes(candidates)
    if not classes:
        return None
    greedy_raw = candidates[0].get("norm") if candidates else None
    greedy_norm = (
        str(greedy_raw)
        if greedy_raw is not None and str(greedy_raw).strip()
        else "__UNKNOWN__"
    )
    return max(
        classes,
        key=lambda item: (
            item["count"],
            item["norm"] == greedy_norm,
            -item["first_index"],
        ),
    )["norm"]


def assignment_specs(parent_0: Sequence[dict], parent_1: Sequence[dict]) -> list[dict]:
    """Enumerate the normalized Cartesian product used by every reader."""
    domains = [value_classes(parent_0), value_classes(parent_1)]
    modal = [modal_norm(parent_0), modal_norm(parent_1)]
    assignments = []
    for index, (left, right) in enumerate(product(*domains)):
        assignments.append({
            "assignment_index": index,
            "norms": [left["norm"], right["norm"]],
            "bindings": [left["answer"], right["answer"]],
            "frequencies": [left["frequency"], right["frequency"]],
            "counts": [left["count"], right["count"]],
            "member_indices": [left["member_indices"], right["member_indices"]],
            "is_modal": [left["norm"], right["norm"]] == modal,
        })
    return assignments


def screen_selected_ids(screen_dir: str, split: str) -> tuple[str, set[int]]:
    """Apply the calibration-selected metadata rule without outcome filtering."""
    from structural_hardness_screen import rule_specs

    report = read_json(os.path.join(screen_dir, "report.json"))
    if not report.get("gate_pass"):
        raise SystemExit("structural-hardness gate failed; hidden extraction is forbidden")
    rule_name = report.get("chosen_rule")
    rules = dict(rule_specs())
    if rule_name not in rules:
        raise SystemExit(f"unknown frozen screen rule: {rule_name!r}")
    case_path = os.path.join(screen_dir, split, "cases.json")
    cases = read_json(case_path)
    selected = {int(case["id"]) for case in cases if rules[rule_name](case)}
    return rule_name, selected


def build_assignment_units(
    data_path: str,
    screen_dir: str,
    split: str,
) -> list[dict]:
    """Build root-execution prompts for every fixed parent-domain assignment."""
    from structural_hardness_screen import ROOT_USER, bind_root, split_questions

    _, selected = screen_selected_ids(screen_dir, split)
    rows = {int(row["id"]): row for row in read_jsonl(data_path)}
    parent_rows = read_jsonl(os.path.join(screen_dir, split, "parents.jsonl"))
    parents = {(int(row["id"]), int(row["slot"])): row for row in parent_rows}
    units = []
    for pid in sorted(selected):
        row = rows.get(pid)
        if row is None or (pid, 0) not in parents or (pid, 1) not in parents:
            raise SystemExit(f"missing data or parent cache for selected graph {pid}")
        _, _, root_question = split_questions(row["problem"])
        p0 = parents[(pid, 0)]["candidates"]
        p1 = parents[(pid, 1)]["candidates"]
        for spec in assignment_specs(p0, p1):
            values = [str(value) for value in spec["bindings"]]
            bound = bind_root(root_question, values[0], values[1])
            user = ROOT_USER.format(
                parent_0=values[0], parent_1=values[1], root=bound
            )
            units.append({
                "id": pid,
                "gold": str(row["answer"]),
                "split": split,
                "user": user,
                **spec,
            })
    return units


def run_assignment_roots(
    runner,
    units: Sequence[dict],
    output_path: str,
    batch_size: int,
) -> list[dict]:
    """Greedily execute missing assignment roots and checkpoint JSONL rows."""
    from structural_hardness_screen import (
        BASE_SYSTEM,
        answers_equal,
        extract_boxed,
        generated_tokens,
        normalize_answer,
    )

    existing = read_jsonl(output_path)
    done = {
        (int(row["id"]), tuple(row["norms"])): row
        for row in existing
    }
    todo = [
        unit for unit in units
        if (int(unit["id"]), tuple(unit["norms"])) not in done
    ]
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        outputs = runner.chat_batch(
            [unit["user"] for unit in batch],
            system=BASE_SYSTEM,
            max_new=192,
            bs=batch_size,
        )
        records = []
        for unit, generated in zip(batch, outputs):
            text = generated[0]
            answer = extract_boxed(text)
            record = {
                key: unit[key]
                for key in (
                    "id", "split", "assignment_index", "norms", "bindings",
                    "frequencies", "counts", "member_indices", "is_modal",
                    "gold", "user",
                )
            }
            record.update({
                "text": text,
                "answer": answer,
                "norm": normalize_answer(answer),
                "label": int(answers_equal(answer, unit["gold"])),
                "generated_tokens": generated_tokens(runner, text),
            })
            records.append(record)
            done[(int(unit["id"]), tuple(unit["norms"]))] = record
        append_jsonl(output_path, records)
        print(
            f"[assignment-root] {min(start + batch_size, len(todo))}/{len(todo)}",
            flush=True,
        )
    return [done[(int(unit["id"]), tuple(unit["norms"]))] for unit in units]


def response_spans(prompt_lengths: Sequence[int], full_lengths: Sequence[int], width: int):
    """Return left-padded [start, end) response spans for hidden pooling."""
    if len(prompt_lengths) != len(full_lengths):
        raise ValueError("length lists must align")
    spans = []
    for prompt_len, full_len in zip(prompt_lengths, full_lengths):
        if not (0 <= prompt_len < full_len <= width):
            raise ValueError(
                f"invalid response span prompt={prompt_len} full={full_len} width={width}"
            )
        offset = width - full_len
        spans.append((offset + prompt_len, offset + full_len))
    return spans


def projectors(torch, hidden_size: int, device: str):
    matrices = {}
    for layer in LAYERS:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(PROJECTION_SEED + layer)
        matrix = torch.randint(
            0, 2, (hidden_size, PROJECTION_DIM), generator=generator,
            dtype=torch.int8,
        ).float()
        matrix.mul_(2.0).sub_(1.0).div_(math.sqrt(PROJECTION_DIM))
        matrices[layer] = matrix.to(device)
    return matrices


def pool_dual_view(torch, hidden, spans, matrix):
    """Project normalized last-token and response-token-mean hidden views."""
    last_rows, mean_rows = [], []
    for row_index, (start, end) in enumerate(spans):
        last_rows.append(hidden[row_index, end - 1].float())
        mean_rows.append(hidden[row_index, start:end].float().mean(dim=0))
    last = torch.stack(last_rows) @ matrix
    mean = torch.stack(mean_rows) @ matrix
    last = last / (last.norm(dim=1, keepdim=True) + 1e-8)
    mean = mean / (mean.norm(dim=1, keepdim=True) + 1e-8)
    return last, mean


def teacher_force_ids(tokenizer, system: str, user: str, response: str, max_context: int):
    """Tokenize prompt and exact response while excluding chat end markers."""
    messages = (
        [{"role": "system", "content": system}] if system else []
    ) + [{"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    response_ids = tokenizer(
        response or "UNKNOWN", add_special_tokens=False
    )["input_ids"]
    if not response_ids:
        response_ids = tokenizer("UNKNOWN", add_special_tokens=False)["input_ids"]
    full = list(prompt) + list(response_ids)
    if len(response_ids) >= max_context:
        raise ValueError("response alone exceeds maximum context")
    cut = max(0, len(full) - max_context)
    full = full[cut:]
    response_start = max(0, len(prompt) - cut)
    if response_start >= len(full):
        raise ValueError("truncation removed the complete response")
    return full, (response_start, len(full)), len(prompt), len(response_ids)


def extract_dual_view_features(
    runner,
    units: Sequence[dict],
    batch_size: int,
    max_context: int = 2048,
) -> dict:
    """Teacher-force exact completions and return fixed projected dual views."""
    torch = runner.torch
    tokenizer = runner.tok
    matrices = projectors(torch, runner.model.config.hidden_size, "cuda")
    feature_chunks = {
        layer: {"last": [], "mean": []} for layer in LAYERS
    }
    prompt_tokens = response_tokens = processed_tokens = 0
    for start in range(0, len(units), batch_size):
        batch = units[start:start + batch_size]
        encoded = [
            teacher_force_ids(
                tokenizer, unit.get("system") or "", unit["user"],
                unit.get("response") or "", max_context,
            )
            for unit in batch
        ]
        width = max(len(item[0]) for item in encoded)
        input_rows, attention_rows, spans = [], [], []
        for ids, (response_start, response_end), raw_prompt, raw_response in encoded:
            padding = width - len(ids)
            input_rows.append([tokenizer.pad_token_id] * padding + ids)
            attention_rows.append([0] * padding + [1] * len(ids))
            spans.append((padding + response_start, padding + response_end))
            prompt_tokens += raw_prompt
            response_tokens += raw_response
            processed_tokens += width
        input_ids = torch.tensor(input_rows, device="cuda")
        attention = torch.tensor(attention_rows, device="cuda")
        with torch.no_grad():
            output = runner.model(
                input_ids=input_ids,
                attention_mask=attention,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        for layer in LAYERS:
            last, mean = pool_dual_view(
                torch, output.hidden_states[layer], spans, matrices[layer]
            )
            feature_chunks[layer]["last"].append(last.half().cpu())
            feature_chunks[layer]["mean"].append(mean.half().cpu())
        del input_ids, attention, output
        torch.cuda.empty_cache()
        print(
            f"[dual-view] {min(start + batch_size, len(units))}/{len(units)}",
            flush=True,
        )
    return {
        "layers": LAYERS,
        "projection_dim": PROJECTION_DIM,
        "features": {
            layer: {
                view: torch.cat(chunks)
                for view, chunks in views.items()
            }
            for layer, views in feature_chunks.items()
        },
        "keys": [unit["key"] for unit in units],
        "metas": [unit.get("meta", {}) for unit in units],
        "accounting": {
            "raw_prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "processed_forward_tokens": processed_tokens,
        },
    }


def hidden_feature_units(
    data_path: str,
    screen_dir: str,
    assignment_records: Sequence[dict],
) -> tuple[list[dict], list[dict]]:
    """Create unique parent-completion and assignment-root teacher-force units."""
    from structural_hardness_screen import (
        BASE_SYSTEM,
        PARENT_USER,
        answers_equal,
        split_questions,
    )

    rows = {int(row["id"]): row for row in read_jsonl(data_path)}
    parent_rows = read_jsonl(os.path.join(screen_dir, "calibration", "parents.jsonl"))
    parents = {(int(row["id"]), int(row["slot"])): row for row in parent_rows}
    selected_parent_keys = set()
    for record in assignment_records:
        pid = int(record["id"])
        for slot, member_indices in enumerate(record["member_indices"]):
            for member_index in member_indices:
                selected_parent_keys.add((pid, slot, int(member_index)))
    parent_units = []
    for pid, slot, member_index in sorted(selected_parent_keys):
        row = rows[pid]
        questions = split_questions(row["problem"])
        candidate = parents[(pid, slot)]["candidates"][member_index]
        parent_units.append({
            "key": ("parent", pid, slot, member_index),
            "system": BASE_SYSTEM,
            "user": PARENT_USER.format(question=questions[slot]),
            "response": candidate.get("text") or "UNKNOWN",
            "meta": {
                "id": pid,
                "slot": slot,
                "member_index": member_index,
                "norm": candidate.get("norm"),
                "label": int(answers_equal(
                    candidate.get("answer"), row["parent_answers"][slot]
                )),
            },
        })
    root_units = [{
        "key": ("root", int(record["id"]), *record["norms"]),
        "system": BASE_SYSTEM,
        "user": record["user"],
        "response": record.get("text") or "UNKNOWN",
        "meta": {
            "id": int(record["id"]),
            "norms": record["norms"],
            "label": int(record["label"]),
        },
    } for record in assignment_records]
    return parent_units, root_units


def graph_raw_features(torch, parent_0, parent_1, root, frequencies):
    """Permutation-invariant raw join representation used by all graph losses."""
    parent_mean = 0.5 * (parent_0 + parent_1)
    interaction = 0.5 * (parent_0 * root + parent_1 * root)
    disagreement = 0.5 * (
        torch.abs(parent_0 - root) + torch.abs(parent_1 - root)
    )
    freq_sum = frequencies.sum(dim=1, keepdim=True)
    freq_product = frequencies.prod(dim=1, keepdim=True)
    return torch.cat(
        [root, parent_mean, interaction, disagreement, freq_sum, freq_product],
        dim=1,
    )


def aeo_asymmetric_loss(
    torch, logits, labels, entropy_weight: float, weights=None
):
    """Positive CE plus maximum-entropy regularization on negative assignments."""
    positive = labels > 0.5
    negative = ~positive
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError("AEO loss requires both positive and negative assignments")
    positive_values = torch.nn.functional.softplus(-logits[positive])
    positive_ce = (
        (positive_values * weights[positive]).sum() / weights[positive].sum()
        if weights is not None else positive_values.mean()
    )
    probability = torch.sigmoid(logits[negative]).clamp(1e-6, 1.0 - 1e-6)
    entropy_values = -(
        probability * torch.log(probability)
        + (1.0 - probability) * torch.log(1.0 - probability)
    )
    entropy = (
        (entropy_values * weights[negative]).sum() / weights[negative].sum()
        if weights is not None else entropy_values.mean()
    )
    return positive_ce - entropy_weight * entropy


def pairwise_energy_loss(torch, scores, labels, problem_ids, margin: float):
    """Within-graph ranking loss; larger ``scores`` means lower graph energy."""
    by_problem = defaultdict(lambda: {0: [], 1: []})
    for index, (label, pid) in enumerate(zip(labels.tolist(), problem_ids)):
        by_problem[pid][int(label > 0.5)].append(index)
    graph_losses = []
    for groups in by_problem.values():
        positive_indices, negative_indices = [], []
        for positive in groups[1]:
            for negative in groups[0]:
                positive_indices.append(positive)
                negative_indices.append(negative)
        if positive_indices:
            pos = torch.tensor(positive_indices, device=scores.device)
            neg = torch.tensor(negative_indices, device=scores.device)
            graph_losses.append(torch.nn.functional.softplus(
                margin - (scores[pos] - scores[neg])
            ).mean())
    if not graph_losses:
        raise ValueError("pairwise energy loss requires a mixed-label graph")
    return torch.stack(graph_losses).mean()


def summarize_assignment_space(records: Sequence[dict]) -> dict:
    by_problem: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_problem[int(record["id"])].append(record)
    modal_correct = 0
    oracle_correct = 0
    mixed = 0
    actionable = 0
    sizes = []
    for rows in by_problem.values():
        labels = [int(row["label"]) for row in rows]
        modal = next((int(row["label"]) for row in rows if row["is_modal"]), 0)
        modal_correct += modal
        oracle_correct += int(any(labels))
        mixed += int(any(labels) and not all(labels))
        actionable += int(not modal and any(labels))
        sizes.append(len(rows))
    n = len(by_problem)
    return {
        "n_graphs": n,
        "n_assignments": sum(sizes),
        "mean_assignments": sum(sizes) / n if n else 0.0,
        "modal_accuracy": modal_correct / n if n else 0.0,
        "assignment_oracle": oracle_correct / n if n else 0.0,
        "oracle_gap": (oracle_correct - modal_correct) / n if n else 0.0,
        "n_mixed_label_graphs": mixed,
        "n_actionable": actionable,
    }


def structural_assignment_gate(summary: dict) -> dict:
    return {
        "n_graphs_ge_100": summary["n_graphs"] >= 100,
        "n_actionable_ge_20": summary["n_actionable"] >= 20,
        "n_mixed_ge_40": summary["n_mixed_label_graphs"] >= 40,
        "oracle_gap_ge_10pp": summary["oracle_gap"] >= 0.10,
    }


def self_test() -> None:
    candidates = [
        {"kind": "greedy", "answer": "2", "norm": "2", "text": "a"},
        {"kind": "sample", "answer": "2.0", "norm": "2", "text": "b"},
        {"kind": "sample", "answer": "3", "norm": "3", "text": "c"},
        {"kind": "sample", "answer": None, "norm": None, "text": "d"},
    ]
    classes = value_classes(candidates)
    assert [item["norm"] for item in classes] == ["2", "3", "__UNKNOWN__"]
    assert classes[0]["count"] == 2 and classes[0]["frequency"] == 0.5
    assert modal_norm(candidates) == "2"
    assignments = assignment_specs(candidates, candidates)
    assert len(assignments) == 9
    assert assignments[0]["is_modal"]
    assert response_spans([3, 2], [5, 4], 5) == [(3, 5), (3, 5)]
    assert stable_bucket("x", 5, "a") == stable_bucket("x", 5, "a")
    print("SELF_TEST_OK")


def assignment_audit(args) -> None:
    from pilot import Runner

    split = "calibration"
    units = build_assignment_units(args.data, args.screen_dir, split)
    if not units:
        raise SystemExit("no assignments built from the selected screen subset")
    runner = Runner(args.model)
    output_path = os.path.join(args.out_dir, split, "assignment_roots.jsonl")
    records = run_assignment_roots(runner, units, output_path, args.batch_size)
    summary = summarize_assignment_space(records)
    checks = structural_assignment_gate(summary)
    report = {
        "stage": "calibration_assignment_action_space",
        "screen_dir": os.path.abspath(args.screen_dir),
        "data": args.data,
        "model": args.model,
        "summary": summary,
        "checks": checks,
        "gate_pass": all(checks.values()),
        "generated_tokens": sum(int(row["generated_tokens"]) for row in records),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "assignment_audit_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    print(json.dumps(report, indent=1), flush=True)
    print(f"ASSIGNMENT_ACTION_SPACE={'PASS' if report['gate_pass'] else 'FAIL'}")


def hidden_extract(args) -> None:
    import torch
    from pilot import Runner

    audit_report = read_json(os.path.join(args.out_dir, "assignment_audit_report.json"))
    if not audit_report.get("gate_pass"):
        raise SystemExit("assignment action-space gate failed; hidden extraction stopped")
    assignment_path = os.path.join(
        args.out_dir, "calibration", "assignment_roots.jsonl"
    )
    records = read_jsonl(assignment_path)
    if not records:
        raise SystemExit(f"no assignment records under {assignment_path}")
    parent_units, root_units = hidden_feature_units(
        args.data, args.screen_dir, records
    )
    runner = Runner(args.model)
    parent_payload = extract_dual_view_features(
        runner, parent_units, args.batch_size, args.max_context
    )
    root_payload = extract_dual_view_features(
        runner, root_units, args.batch_size, args.max_context
    )
    payload = {
        "model": args.model,
        "screen_dir": os.path.abspath(args.screen_dir),
        "assignment_path": os.path.abspath(assignment_path),
        "parent": parent_payload,
        "root": root_payload,
        "metas": [{
            key: record[key]
            for key in (
                "id", "norms", "bindings", "frequencies", "counts",
                "member_indices", "is_modal", "label",
            )
        } for record in records],
    }
    feature_path = os.path.join(args.out_dir, "calibration_dual_features.pt")
    torch.save(payload, feature_path)
    print(f"[dual-view] saved {feature_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("assignment-audit", "hidden-extract", "self-test"),
        default="self-test",
    )
    parser.add_argument("--data", default="data/gsm_join_train.jsonl")
    parser.add_argument("--screen-dir", default="structural_hardness_screen")
    parser.add_argument("--out-dir", default="hidden_graph_energy_guide")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-context", type=int, default=2048)
    args = parser.parse_args()
    if args.stage == "self-test":
        self_test()
    elif args.stage == "assignment-audit":
        assignment_audit(args)
    else:
        hidden_extract(args)


if __name__ == "__main__":
    main()
