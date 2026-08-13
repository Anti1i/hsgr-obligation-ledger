"""Apparatus-only prompt screen for the distinct-value P3 route task."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass

from dependency_patch_p0 import ModelRunner, bootstrap_ci, json_scalar, mean
from dependency_route_subspace_p3 import _ordered, generate_case as generate_p3_case


TEMPLATE_ORDER = ("original", "explicit_select", "concise_select")


@dataclass
class ApparatusCase:
    id: str
    template: str
    labels: dict[str, str]
    clean_p: int
    decoy_x: int
    corrupt_p: int
    route_on_user: str
    route_off_user: str
    corrupt_user: str
    clean_root: int
    corrupt_root: int
    decoy_root: int


def render(template, labels, p_value, x_value, p_first, print_role):
    lines = {
        "root": f"{labels['root']} = ({labels['p']} + {labels['a0']}) % 10",
        "decoy": f"{labels['decoy']} = ({labels['x']} + {labels['ax0']}) % 10",
    }
    program = "\n".join(
        lines[role] for role in _ordered(("root", "decoy"), "program", labels)
    )
    checkpoint_roles = list(_ordered(("a0", "ax0"), "checkpoint", labels))
    checkpoint_roles.extend(("p", "x") if p_first else ("x", "p"))
    values = {"a0": 0, "ax0": 0, "p": p_value, "x": x_value}
    checkpoints = "\n".join(
        f"{labels[role]} = {values[role]}" for role in checkpoint_roles
    )
    if template == "original":
        lead = (
            "A straight-line Python-style program resumes from a checkpoint. "
            "All operations are modulo 10. Use the checkpoint values as the current "
            "state; do not recompute or replace them."
        )
        tail = "What single digit is printed? Answer with one digit only."
    elif template == "explicit_select":
        lead = (
            "A straight-line Python-style program resumes from a checkpoint. "
            "All operations are modulo 10. Use the checkpoint values as the current "
            "state; do not recompute or replace them. The final print statement "
            "selects the only branch that determines the answer; ignore the unprinted branch."
        )
        tail = "What single digit is printed? Answer with one digit only."
    elif template == "concise_select":
        lead = (
            "Execute this two-branch modulo-10 program from the checkpoint values. "
            "The final print statement selects the answer, and the unprinted branch is irrelevant."
        )
        tail = "Return exactly the one digit printed."
    else:
        raise ValueError(template)
    return (
        lead
        + "\n\nDownstream program:\n"
        + program
        + f"\nprint({labels[print_role]})\n\nCheckpoint values:\n"
        + checkpoints
        + "\n\n"
        + tail
    )


def generate_case(template: str, index: int, seed: int):
    base = generate_p3_case(index, seed)
    on = render(template, base.labels, base.clean_p, base.decoy_x, base.p_first, "root")
    off = render(template, base.labels, base.clean_p, base.decoy_x, base.p_first, "decoy")
    corrupt = render(template, base.labels, base.corrupt_p, base.decoy_x, base.p_first, "root")
    case_id = hashlib.sha256(
        f"dependency-route-apparatus-p3a|{seed}|{template}|{index}".encode()
    ).hexdigest()[:16]
    return ApparatusCase(
        id=case_id,
        template=template,
        labels=base.labels,
        clean_p=base.clean_p,
        decoy_x=base.decoy_x,
        corrupt_p=base.corrupt_p,
        route_on_user=on,
        route_off_user=off,
        corrupt_user=corrupt,
        clean_root=base.clean_root,
        corrupt_root=base.corrupt_root,
        decoy_root=base.decoy_root,
    )


def build_cases(template: str, n: int, seed: int):
    cases = [generate_case(template, index, seed) for index in range(n)]
    cases.sort(key=lambda case: case.id)
    return cases


def score_users(runner, users, batch_size):
    torch = runner.torch
    logps, preds = [], []
    texts = [runner.chat_text(user) for user in users]
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = runner.tok(chunk, padding=True, return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.cuda() for key, value in encoded.items()}
        with torch.no_grad():
            output = runner.model(**inputs, use_cache=False)
        last = inputs["attention_mask"].sum(1) - 1
        logp, pred = runner._digit_scores(output.logits[torch.arange(len(chunk), device="cuda"), last])
        logps.extend(logp.tolist())
        preds.extend(pred.tolist())
        del output, inputs, encoded
    return logps, preds


def evaluate(cases, route_on, route_off, corrupt):
    on_logp, on_pred = route_on
    off_logp, off_pred = route_off
    corrupt_logp, corrupt_pred = corrupt
    on_correct = [int(pred == case.clean_root) for pred, case in zip(on_pred, cases)]
    off_correct = [int(pred == case.decoy_root) for pred, case in zip(off_pred, cases)]
    corrupt_own = [int(pred == case.corrupt_root) for pred, case in zip(corrupt_pred, cases)]
    corruption = [
        float(clean[case.clean_root] - changed[case.clean_root])
        for case, clean, changed in zip(cases, on_logp, corrupt_logp)
    ]
    delta = mean(corruption)
    ci = bootstrap_ci(corruption, seed=20260819)
    gate = (
        mean(on_correct) >= 0.90
        and mean(off_correct) >= 0.90
        and mean(corrupt_own) >= 0.50
        and delta >= 0.20
        and ci[0] > 0.0
    )
    return {
        "route_on_clean_accuracy": mean(on_correct),
        "route_off_clean_accuracy": mean(off_correct),
        "corrupt_own_accuracy": mean(corrupt_own),
        "clean_minus_corrupt_clean_logp": {"mean": delta, "ci95": ci},
        "gate_pass": bool(gate),
    }


def select_template(reports):
    return next((template for template in TEMPLATE_ORDER if reports[template]["gate_pass"]), None)


def main(args):
    all_cases = {template: build_cases(template, args.n, args.seed) for template in TEMPLATE_ORDER}
    if args.dry_run:
        print(json.dumps({template: asdict(cases[0]) for template, cases in all_cases.items()}, indent=2))
        return
    runner = ModelRunner(args.model)
    reports = {}
    for template in TEMPLATE_ORDER:
        cases = all_cases[template]
        on = score_users(runner, [case.route_on_user for case in cases], args.batch_size)
        off = score_users(runner, [case.route_off_user for case in cases], args.batch_size)
        corrupt = score_users(runner, [case.corrupt_user for case in cases], args.batch_size)
        reports[template] = evaluate(cases, on, off, corrupt)
        print(f"[template] {template} {reports[template]}", flush=True)
    selected = select_template(reports)
    report = {
        "protocol": "EXPERIMENT_PROTOCOL_DEPENDENCY_ROUTE_APPARATUS_P3A.md",
        "model": args.model,
        "seed": args.seed,
        "n_per_template": args.n,
        "template_order": list(TEMPLATE_ORDER),
        "reports": reports,
        "selected_template": selected,
        "verdict": "APPARATUS_PASS" if selected else "APPARATUS_FAIL_STOP",
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=json_scalar)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=json_scalar))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--n", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="dependency_route_apparatus_p3a_report.json")
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
