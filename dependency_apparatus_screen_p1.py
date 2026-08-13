"""Apparatus-only screen for a solvable dependency-role patching task."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass

from dependency_patch_p0 import ModelRunner, bootstrap_ci, json_scalar, mean


FAMILY_ORDER = ("dag_add", "chain3_add", "chain3_copy", "chain1_copy")


@dataclass
class ScreenCase:
    id: str
    family: str
    clean_user: str
    corrupt_user: str
    clean_root: int
    corrupt_root: int
    p_first: bool


def add(left: int, right: int) -> int:
    return (left + right) % 10


def ordered(items, salt: str, labels: dict[str, str]):
    return tuple(
        sorted(
            items,
            key=lambda role: hashlib.sha256(
                f"{salt}|{labels[role]}".encode()
            ).hexdigest(),
        )
    )


def render_case(
    family: str,
    labels: dict[str, str],
    p: int,
    corrupt_p: int,
    aux: tuple[int, ...],
    p_first: bool,
) -> tuple[str, str, int, int]:
    if family == "dag_add":
        q, t = aux
        roles = ("r", "s", "rd", "sd", "u", "ud", "root", "decoy")
        lines = {
            "r": f"{labels['r']} = ({labels['p']} + {labels['q']}) % 10",
            "s": f"{labels['s']} = ({labels['p']} + {labels['t']}) % 10",
            "rd": f"{labels['rd']} = ({labels['x']} + {labels['qx']}) % 10",
            "sd": f"{labels['sd']} = ({labels['x']} + {labels['tx']}) % 10",
            "u": f"{labels['u']} = ({labels['r']} + {labels['s']}) % 10",
            "ud": f"{labels['ud']} = ({labels['rd']} + {labels['sd']}) % 10",
            "root": f"{labels['root']} = ({labels['u']} + {labels['q']}) % 10",
            "decoy": f"{labels['decoy']} = ({labels['ud']} + {labels['qx']}) % 10",
        }
        program_roles = (
            *ordered(("r", "s", "rd", "sd"), "level1", labels),
            *ordered(("u", "ud"), "level2", labels),
            *ordered(("root", "decoy"), "level3", labels),
        )
        checkpoints = {"q": q, "qx": q, "t": t, "tx": t}

        def execute(p_value):
            r, s = add(p_value, q), add(p_value, t)
            return add(add(r, s), q)

    else:
        depth = 1 if family == "chain1_copy" else 3
        constants = (0,) * depth if family.endswith("copy") else aux[:depth]
        lines = {}
        main_prev, decoy_prev = labels["p"], labels["x"]
        main_roles, decoy_roles = [], []
        checkpoints = {}
        for level in range(depth):
            main_role = "root" if level == depth - 1 else f"m{level}"
            decoy_role = "decoy" if level == depth - 1 else f"d{level}"
            a_role, ax_role = f"a{level}", f"ax{level}"
            value = constants[level]
            checkpoints[a_role] = value
            checkpoints[ax_role] = value
            lines[main_role] = (
                f"{labels[main_role]} = ({main_prev} + {labels[a_role]}) % 10"
            )
            lines[decoy_role] = (
                f"{labels[decoy_role]} = ({decoy_prev} + {labels[ax_role]}) % 10"
            )
            main_prev, decoy_prev = labels[main_role], labels[decoy_role]
            main_roles.append(main_role)
            decoy_roles.append(decoy_role)
        program_roles = tuple(
            role
            for level in range(depth)
            for role in ordered(
                (main_roles[level], decoy_roles[level]),
                f"level{level}",
                labels,
            )
        )

        def execute(p_value):
            value = p_value
            for constant in constants:
                value = add(value, constant)
            return value

    checkpoint_roles = list(checkpoints)
    checkpoint_roles.sort(
        key=lambda role: hashlib.sha256(f"checkpoint|{labels[role]}".encode()).hexdigest()
    )
    checkpoint_roles.extend(("p", "x") if p_first else ("x", "p"))

    def build(p_value: int) -> str:
        values = dict(checkpoints, p=p_value, x=p)
        checkpoint_text = "\n".join(
            f"{labels[role]} = {values[role]}" for role in checkpoint_roles
        )
        program_text = "\n".join(lines[role] for role in program_roles)
        return (
            "A straight-line Python-style program resumes from a checkpoint. "
            "All operations are modulo 10. Use the checkpoint values as the current "
            "state; do not recompute or replace them.\n\nDownstream program:\n"
            + program_text
            + f"\nprint({labels['root']})\n\nCheckpoint values:\n"
            + checkpoint_text
            + "\n\nWhat single digit is printed? Answer with one digit only."
        )

    return build(p), build(corrupt_p), execute(p), execute(corrupt_p)


def generate_case(family: str, index: int, seed: int) -> ScreenCase:
    rng = random.Random(f"dependency-apparatus|{seed}|{family}|{index}")
    if family == "dag_add":
        roles = ["p", "x", "q", "qx", "t", "tx", "r", "s", "rd", "sd", "u", "ud", "root", "decoy"]
        aux = (rng.randrange(10), rng.randrange(10))
    else:
        depth = 1 if family == "chain1_copy" else 3
        roles = ["p", "x", "root", "decoy"]
        for level in range(depth - 1):
            roles.extend((f"m{level}", f"d{level}"))
        for level in range(depth):
            roles.extend((f"a{level}", f"ax{level}"))
        aux = tuple(rng.randrange(10) for _ in range(depth))
    alphabet = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
    labels = dict(zip(roles, rng.sample(alphabet, len(roles))))
    p = rng.randrange(10)
    choices = list(range(10))
    rng.shuffle(choices)
    p_first = bool(rng.getrandbits(1))
    for corrupt_p in choices:
        if corrupt_p == p:
            continue
        clean, corrupt, clean_root, corrupt_root = render_case(
            family, labels, p, corrupt_p, aux, p_first
        )
        if clean_root != corrupt_root:
            case_id = hashlib.sha256(
                f"dependency-apparatus|{seed}|{family}|{index}".encode()
            ).hexdigest()[:16]
            return ScreenCase(
                id=case_id,
                family=family,
                clean_user=clean,
                corrupt_user=corrupt,
                clean_root=clean_root,
                corrupt_root=corrupt_root,
                p_first=p_first,
            )
    raise RuntimeError(f"no sensitive corruption for {family}/{index}")


def build_cases(family: str, n: int, seed: int):
    cases = [generate_case(family, i, seed) for i in range(n)]
    cases.sort(key=lambda case: case.id)
    return cases


def score_users(runner: ModelRunner, users: list[str], batch_size: int):
    torch = runner.torch
    all_logp, all_pred = [], []
    texts = [runner.chat_text(user) for user in users]
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = runner.tok(chunk, padding=True, return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.cuda() for key, value in encoded.items()}
        with torch.no_grad():
            output = runner.model(**inputs, use_cache=False)
        last = inputs["attention_mask"].sum(1) - 1
        logp, pred = runner._digit_scores(
            output.logits[torch.arange(len(chunk), device="cuda"), last]
        )
        all_logp.extend(logp.tolist())
        all_pred.extend(pred.tolist())
        del output, inputs, encoded
    return all_logp, all_pred


def summarize_family(cases, clean_logp, clean_pred, corrupt_logp, corrupt_pred):
    clean_gold = [case.clean_root for case in cases]
    corrupt_gold = [case.corrupt_root for case in cases]
    clean_correct = [int(pred == gold) for pred, gold in zip(clean_pred, clean_gold)]
    corrupt_correct = [int(pred == gold) for pred, gold in zip(corrupt_pred, corrupt_gold)]
    clean_scores = [row[gold] for row, gold in zip(clean_logp, clean_gold)]
    corrupt_clean_scores = [row[gold] for row, gold in zip(corrupt_logp, clean_gold)]
    deltas = [left - right for left, right in zip(clean_scores, corrupt_clean_scores)]
    delta_mean = mean(deltas)
    delta_ci = bootstrap_ci(deltas, seed=20260815)
    clean_accuracy = mean(clean_correct)
    corrupt_own_accuracy = mean(corrupt_correct)
    gate = (
        clean_accuracy >= 0.60
        and corrupt_own_accuracy >= 0.50
        and delta_mean >= 0.20
        and delta_ci[0] > 0.0
    )
    return {
        "n": len(cases),
        "clean_accuracy": clean_accuracy,
        "corrupt_own_accuracy": corrupt_own_accuracy,
        "mean_clean_answer_logp": mean(clean_scores),
        "clean_minus_corrupt_clean_answer_logp": {
            "mean": delta_mean,
            "ci95": delta_ci,
        },
        "gate_pass": bool(gate),
    }


def select_family(reports: dict[str, dict]):
    for family in FAMILY_ORDER:
        if reports[family]["gate_pass"]:
            return family
    return None


def main(args):
    all_cases = {
        family: build_cases(family, args.n_per_family, args.seed)
        for family in FAMILY_ORDER
    }
    if args.dry_run:
        print(json.dumps({family: asdict(cases[0]) for family, cases in all_cases.items()}, indent=2))
        return
    runner = ModelRunner(args.model)
    reports = {}
    for family in FAMILY_ORDER:
        cases = all_cases[family]
        clean_logp, clean_pred = score_users(
            runner, [case.clean_user for case in cases], args.batch_size
        )
        corrupt_logp, corrupt_pred = score_users(
            runner, [case.corrupt_user for case in cases], args.batch_size
        )
        reports[family] = summarize_family(
            cases, clean_logp, clean_pred, corrupt_logp, corrupt_pred
        )
        print(f"[family] {family} {reports[family]}", flush=True)
    report = {
        "protocol": "EXPERIMENT_PROTOCOL_DEPENDENCY_APPARATUS_SCREEN_P1.md",
        "model": args.model,
        "seed": args.seed,
        "family_order": list(FAMILY_ORDER),
        "families": reports,
        "selected_family": select_family(reports),
        "verdict": "APPARATUS_PASS" if select_family(reports) else "APPARATUS_FAIL",
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=json_scalar)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=json_scalar))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--n-per-family", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="dependency_apparatus_screen_p1_report.json")
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())

