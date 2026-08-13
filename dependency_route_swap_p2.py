"""Same-position route-swap control for the P1 dependency patch effect."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass

from dependency_patch_p0 import (
    ModelRunner,
    arm_metrics,
    hash_half,
    json_scalar,
    mean,
    paired_delta,
    run_probe,
)


LOCKED_LAYER = 21


@dataclass
class RouteSwapCase:
    id: str
    labels: dict[str, str]
    clean_p: int
    corrupt_p: int
    route_on_user: str
    route_off_user: str
    corrupt_user: str
    clean_root: int
    corrupt_root: int
    p_first: bool


def _ordered(items, salt: str, labels: dict[str, str]):
    return tuple(
        sorted(
            items,
            key=lambda role: hashlib.sha256(
                f"{salt}|{labels[role]}".encode()
            ).hexdigest(),
        )
    )


def _render(
    labels: dict[str, str],
    clean_p: int,
    p_value: int,
    p_first: bool,
    print_role: str,
) -> str:
    lines = {
        "root": f"{labels['root']} = ({labels['p']} + {labels['a0']}) % 10",
        "decoy": f"{labels['decoy']} = ({labels['x']} + {labels['ax0']}) % 10",
    }
    program = "\n".join(lines[r] for r in _ordered(("root", "decoy"), "program", labels))
    checkpoint_roles = list(_ordered(("a0", "ax0"), "checkpoint", labels))
    checkpoint_roles.extend(("p", "x") if p_first else ("x", "p"))
    values = {"a0": 0, "ax0": 0, "p": p_value, "x": clean_p}
    checkpoints = "\n".join(
        f"{labels[role]} = {values[role]}" for role in checkpoint_roles
    )
    return (
        "A straight-line Python-style program resumes from a checkpoint. "
        "All operations are modulo 10. Use the checkpoint values as the current "
        "state; do not recompute or replace them.\n\nDownstream program:\n"
        + program
        + f"\nprint({labels[print_role]})\n\nCheckpoint values:\n"
        + checkpoints
        + "\n\nWhat single digit is printed? Answer with one digit only."
    )


def generate_case(index: int, seed: int) -> RouteSwapCase:
    rng = random.Random(f"dependency-route-swap-p2|{seed}|{index}")
    roles = ("p", "x", "root", "decoy", "a0", "ax0")
    labels = dict(zip(roles, rng.sample(list("ABCDEFGHJKLMNPQRSTUVWXYZ"), len(roles))))
    clean_p = rng.randrange(10)
    corrupt_p = rng.choice([value for value in range(10) if value != clean_p])
    p_first = bool(rng.getrandbits(1))
    route_on = _render(labels, clean_p, clean_p, p_first, "root")
    route_off = _render(labels, clean_p, clean_p, p_first, "decoy")
    corrupt = _render(labels, clean_p, corrupt_p, p_first, "root")
    case_id = hashlib.sha256(
        f"dependency-route-swap-p2|{seed}|{index}".encode()
    ).hexdigest()[:16]
    return RouteSwapCase(
        id=case_id,
        labels=labels,
        clean_p=clean_p,
        corrupt_p=corrupt_p,
        route_on_user=route_on,
        route_off_user=route_off,
        corrupt_user=corrupt,
        clean_root=clean_p,
        corrupt_root=corrupt_p,
        p_first=p_first,
    )


def build_cases(n: int, seed: int):
    cases = [generate_case(i, seed) for i in range(n)]
    cases.sort(key=lambda case: case.id)
    return cases


def choose_donors(cases):
    buckets = {}
    for index, case in enumerate(cases):
        buckets.setdefault(case.clean_p, []).append(index)
    donors = []
    for index, case in enumerate(cases):
        candidates = [j for j in buckets[case.clean_p] if j != index]
        if not candidates:
            raise RuntimeError(f"no matched donor for {case.id}")
        donors.append(candidates[int(case.id, 16) % len(candidates)])
    return donors


class RouteSwapRunner(ModelRunner):
    def _encoded_pair(self, on_texts, off_texts, chunk):
        on_encoded = self.tokenize_with_offsets(on_texts)
        off_encoded = self.tokenize_with_offsets(off_texts)
        on_offsets = on_encoded.pop("offset_mapping").tolist()
        off_offsets = off_encoded.pop("offset_mapping").tolist()
        positions = []
        for row, case in enumerate(chunk):
            on_len = int(on_encoded["attention_mask"][row].sum())
            off_len = int(off_encoded["attention_mask"][row].sum())
            if on_len != off_len:
                raise RuntimeError(f"token length mismatch for {case.id}: {on_len}/{off_len}")
            on_pos = self.locate_checkpoint(
                on_texts[row], case.labels["p"], case.clean_p, on_offsets[row]
            )
            off_pos = self.locate_checkpoint(
                off_texts[row], case.labels["p"], case.clean_p, off_offsets[row]
            )
            if on_pos != off_pos:
                raise RuntimeError(f"checkpoint token position mismatch for {case.id}")
            on_ids = on_encoded["input_ids"][row, :on_len]
            off_ids = off_encoded["input_ids"][row, :off_len]
            diff = (on_ids != off_ids).nonzero().flatten().tolist()
            if len(diff) != 1 or diff[0] >= on_pos:
                raise RuntimeError(f"expected one earlier route token diff for {case.id}: {diff}")
            if int(on_ids[on_pos]) != int(off_ids[off_pos]):
                raise RuntimeError(f"checkpoint token ID mismatch for {case.id}")
            positions.append(on_pos)
        return on_encoded, off_encoded, positions

    def extract_donors(self, cases, layers, batch_size):
        torch = self.torch
        feats = {layer: {"p": [], "x": []} for layer in layers}
        metadata = {"p": [], "x": []}
        on_logp, on_pred, off_logp, off_pred = [], [], [], []
        integrity_pairs = 0
        for start in range(0, len(cases), batch_size):
            chunk = cases[start : start + batch_size]
            on_texts = [self.chat_text(case.route_on_user) for case in chunk]
            off_texts = [self.chat_text(case.route_off_user) for case in chunk]
            on_encoded, off_encoded, positions = self._encoded_pair(on_texts, off_texts, chunk)
            integrity_pairs += len(chunk)
            for row, case in enumerate(chunk):
                seq_len = int(on_encoded["attention_mask"][row].sum())
                checkpoint_start = on_texts[row].index("Checkpoint values:")
                mention = on_texts[row].index(f"({case.labels['p']} +", 0, checkpoint_start)
                item = [positions[row] / seq_len, mention / checkpoint_start, ord(case.labels["p"]) / 90.0]
                metadata["p"].append(item)
                metadata["x"].append(list(item))
            for condition, encoded in (("p", on_encoded), ("x", off_encoded)):
                inputs = {key: value.cuda() for key, value in encoded.items()}
                with torch.no_grad():
                    output = self.model(**inputs, output_hidden_states=True, use_cache=False)
                last = inputs["attention_mask"].sum(1) - 1
                logp, pred = self._digit_scores(
                    output.logits[torch.arange(len(chunk), device="cuda"), last]
                )
                if condition == "p":
                    on_logp.extend(logp.tolist())
                    on_pred.extend(pred.tolist())
                else:
                    off_logp.extend(logp.tolist())
                    off_pred.extend(pred.tolist())
                for layer in layers:
                    hidden = output.hidden_states[layer]
                    feats[layer][condition].extend(
                        hidden[row, positions[row]].detach().cpu().to(torch.float32)
                        for row in range(len(chunk))
                    )
                del output, inputs
            del on_encoded, off_encoded
        for layer in layers:
            for condition in ("p", "x"):
                feats[layer][condition] = torch.stack(feats[layer][condition])
        return feats, metadata, on_logp, on_pred, off_logp, off_pred, integrity_pairs

    def prepare_corrupt(self, cases):
        texts = [self.chat_text(case.corrupt_user) for case in cases]
        encoded = self.tokenize_with_offsets(texts)
        offsets = encoded.pop("offset_mapping").tolist()
        positions = [
            self.locate_checkpoint(text, case.labels["p"], case.corrupt_p, offsets[row])
            for row, (text, case) in enumerate(zip(texts, cases))
        ]
        return encoded, positions

    def score_corrupt_and_patches(self, cases, feats, layers, batch_size, donor_idx):
        torch = self.torch
        arms = ("correct_route_same_position", "wrong_route_same_position", "cross_problem")
        result = {
            "corrupt": {"logp": [], "pred": []},
            "layers": {str(layer): {arm: {"logp": [], "pred": []} for arm in arms} for layer in layers},
        }
        for start in range(0, len(cases), batch_size):
            chunk = cases[start : start + batch_size]
            encoded, positions = self.prepare_corrupt(chunk)
            inputs = {key: value.cuda() for key, value in encoded.items()}
            last = inputs["attention_mask"].sum(1) - 1
            with torch.no_grad():
                base = self.model(**inputs, use_cache=False)
            logp, pred = self._digit_scores(base.logits[torch.arange(len(chunk), device="cuda"), last])
            result["corrupt"]["logp"].extend(logp.tolist())
            result["corrupt"]["pred"].extend(pred.tolist())
            del base
            global_idx = list(range(start, start + len(chunk)))
            for layer in layers:
                vectors = {
                    "correct_route_same_position": feats[layer]["p"][global_idx],
                    "wrong_route_same_position": feats[layer]["x"][global_idx],
                    "cross_problem": feats[layer]["p"][[donor_idx[i] for i in global_idx]],
                }
                block = self.model.model.layers[layer - 1]
                for arm, cpu_vectors in vectors.items():
                    patch_vectors = cpu_vectors.to("cuda", dtype=torch.bfloat16)

                    def hook(_module, _args, block_output, pos=positions, patches=patch_vectors):
                        hidden = block_output[0] if isinstance(block_output, tuple) else block_output
                        changed = hidden.clone()
                        for row, position in enumerate(pos):
                            changed[row, int(position), :] = patches[row]
                        if isinstance(block_output, tuple):
                            return (changed,) + block_output[1:]
                        return changed

                    handle = block.register_forward_hook(hook)
                    try:
                        with torch.no_grad():
                            patched = self.model(**inputs, use_cache=False)
                    finally:
                        handle.remove()
                    logp, pred = self._digit_scores(
                        patched.logits[torch.arange(len(chunk), device="cuda"), last]
                    )
                    result["layers"][str(layer)][arm]["logp"].extend(logp.tolist())
                    result["layers"][str(layer)][arm]["pred"].extend(pred.tolist())
                    del patched
            del inputs, encoded
            torch.cuda.empty_cache()
        return result


def summarize(cases, calib_n, on_logp, on_pred, off_logp, off_pred, patch_output, probe, integrity_pairs):
    held = cases[calib_n:]
    route_on = arm_metrics(held, on_logp[calib_n:], on_pred[calib_n:])
    route_off = arm_metrics(held, off_logp[calib_n:], off_pred[calib_n:])
    corrupt_clean = arm_metrics(held, patch_output["corrupt"]["logp"][calib_n:], patch_output["corrupt"]["pred"][calib_n:])
    corrupt_own = arm_metrics(held, patch_output["corrupt"]["logp"][calib_n:], patch_output["corrupt"]["pred"][calib_n:], "corrupt_root")
    corruption = paired_delta(route_on["correct_logp"], corrupt_clean["correct_logp"], seed=1001)
    reports = {}
    for layer, arms in patch_output["layers"].items():
        summaries = {arm: arm_metrics(held, values["logp"][calib_n:], values["pred"][calib_n:]) for arm, values in arms.items()}
        correct = summaries["correct_route_same_position"]
        wrong = summaries["wrong_route_same_position"]
        correct_corrupt_lp = paired_delta(correct["correct_logp"], corrupt_clean["correct_logp"], seed=1100 + int(layer))
        correct_corrupt_acc = paired_delta(correct["correct"], corrupt_clean["correct"], seed=1200 + int(layer))
        correct_wrong_lp = paired_delta(correct["correct_logp"], wrong["correct_logp"], seed=1300 + int(layer))
        correct_wrong_acc = paired_delta(correct["correct"], wrong["correct"], seed=1400 + int(layer))
        recovery = correct_corrupt_lp["mean"] / max(1e-9, corruption["mean"])
        halves = {}
        for half in (0, 1):
            idx = [i for i, case in enumerate(held) if hash_half(case.id) == half]
            halves[str(half)] = mean(correct["correct_logp"][i] - wrong["correct_logp"][i] for i in idx)
        apparatus = (
            route_on["accuracy"] >= 0.90
            and route_off["accuracy"] >= 0.90
            and corrupt_own["accuracy"] >= 0.50
            and corruption["mean"] >= 0.20
            and corruption["ci95"][0] > 0.0
            and correct_corrupt_lp["mean"] >= 0.20
            and correct_corrupt_lp["ci95"][0] > 0.0
            and correct_corrupt_acc["mean"] >= 0.10
        )
        causal = (
            correct_wrong_lp["mean"] >= 0.10
            and correct_wrong_lp["ci95"][0] > 0.0
            and correct_wrong_acc["mean"] >= 0.03
            and min(halves.values()) >= 0.0
            and recovery >= 0.20
        )
        equivalent = (
            correct_wrong_lp["ci95"][0] >= -0.10
            and correct_wrong_lp["ci95"][1] <= 0.10
            and correct_wrong_acc["ci95"][0] >= -0.03
            and correct_wrong_acc["ci95"][1] <= 0.03
        )
        reports[layer] = {
            "arms": {arm: {"accuracy": values["accuracy"], "mean_correct_logp": mean(values["correct_logp"])} for arm, values in summaries.items()},
            "correct_minus_corrupt_logp": {k: v for k, v in correct_corrupt_lp.items() if k != "values"},
            "correct_minus_corrupt_accuracy": {k: v for k, v in correct_corrupt_acc.items() if k != "values"},
            "correct_minus_wrong_logp": {k: v for k, v in correct_wrong_lp.items() if k != "values"},
            "correct_minus_wrong_accuracy": {k: v for k, v in correct_wrong_acc.items() if k != "values"},
            "correct_recovery_fraction": recovery,
            "correct_minus_wrong_hash_halves": halves,
            "apparatus_gate_pass": bool(apparatus),
            "causal_gate_pass": bool(causal),
            "equivalence_pass": bool(equivalent),
        }
    selected = reports[str(LOCKED_LAYER)]
    if integrity_pairs != len(cases):
        verdict = "INVALID_CONTROL"
    elif not selected["apparatus_gate_pass"]:
        verdict = "APPARATUS_FAIL"
    elif selected["causal_gate_pass"]:
        verdict = "ROUTE_SWAP_CONFIRM"
    elif selected["equivalence_pass"]:
        verdict = "POSITION_CONFOUND_SUPPORTED"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "integrity_pairs": integrity_pairs,
        "route_on_clean": {"accuracy": route_on["accuracy"], "mean_correct_logp": mean(route_on["correct_logp"])},
        "route_off_clean": {"accuracy": route_off["accuracy"], "mean_correct_logp": mean(route_off["correct_logp"])},
        "corrupt_clean_answer": {"accuracy": corrupt_clean["accuracy"], "mean_correct_logp": mean(corrupt_clean["correct_logp"])},
        "corrupt_own_answer": {"accuracy": corrupt_own["accuracy"], "mean_correct_logp": mean(corrupt_own["correct_logp"])},
        "clean_minus_corrupt_clean_logp": {k: v for k, v in corruption.items() if k != "values"},
        "layers": reports,
        "representation_gate_pass": bool(probe["gate_pass"]),
        "verdict": verdict,
    }


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    if args.layers != [LOCKED_LAYER]:
        raise ValueError(f"P2 freezes --layers to [{LOCKED_LAYER}]")
    cases = build_cases(args.calib_n + args.test_n, args.seed)
    if args.dry_run:
        print(json.dumps({"n": len(cases), "first": asdict(cases[0])}, indent=2))
        return
    runner = RouteSwapRunner(args.model)
    extracted = runner.extract_donors(cases, args.layers, args.batch_size)
    feats, metadata, on_logp, on_pred, off_logp, off_pred, integrity_pairs = extracted
    probe = run_probe(feats, metadata, cases, args.calib_n, args.layers)
    donors = choose_donors(cases)
    patch_output = runner.score_corrupt_and_patches(cases, feats, args.layers, args.batch_size, donors)
    causal = summarize(cases, args.calib_n, on_logp, on_pred, off_logp, off_pred, patch_output, probe, integrity_pairs)
    report = {
        "protocol": "EXPERIMENT_PROTOCOL_DEPENDENCY_ROUTE_SWAP_P2.md",
        "model": args.model,
        "seed": args.seed,
        "calib_n": args.calib_n,
        "test_n": args.test_n,
        "locked_layer": LOCKED_LAYER,
        "case_id_sha256": hashlib.sha256("|".join(case.id for case in cases).encode()).hexdigest(),
        "p_first": sum(case.p_first for case in cases),
        "probe": probe,
        "causal": causal,
    }
    with open(os.path.join(args.out_dir, "dependency_route_swap_p2_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=json_scalar)
    with open(os.path.join(args.out_dir, "dependency_route_swap_p2_cases.jsonl"), "w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=json_scalar))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--calib-n", type=int, default=96)
    parser.add_argument("--test-n", type=int, default=192)
    parser.add_argument("--layers", type=int, nargs="+", default=[LOCKED_LAYER])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", default="dependency_route_swap_p2")
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
