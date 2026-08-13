"""Value-orthogonal route-subspace intervention for one-hop dependency use."""
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
    exact_sign_p,
    hash_half,
    json_scalar,
    mean,
    paired_delta,
)


PRIMARY_LAYER = 21
WINDOW_LAYERS = (19, 20, 21)
MODES = {"single21": (21,), "window19_21": WINDOW_LAYERS}
ARMS = ("correct_full", "wrong_full", "route_swap", "sham_plus", "sham_minus")


@dataclass
class RouteSubspaceCase:
    id: str
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


def _render(labels, p_value: int, x_value: int, p_first: bool, print_role: str):
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
    return (
        "A straight-line Python-style program resumes from a checkpoint. "
        "All operations are modulo 10. Use the checkpoint values as the current "
        "state; do not recompute or replace them.\n\nDownstream program:\n"
        + program
        + f"\nprint({labels[print_role]})\n\nCheckpoint values:\n"
        + checkpoints
        + "\n\nWhat single digit is printed? Answer with one digit only."
    )


def generate_case(index: int, seed: int) -> RouteSubspaceCase:
    rng = random.Random(f"dependency-route-subspace-p3|{seed}|{index}")
    roles = ("p", "x", "root", "decoy", "a0", "ax0")
    labels = dict(zip(roles, rng.sample(list("ABCDEFGHJKLMNPQRSTUVWXYZ"), len(roles))))
    clean_p, decoy_x, corrupt_p = rng.sample(range(10), 3)
    p_first = bool(rng.getrandbits(1))
    route_on = _render(labels, clean_p, decoy_x, p_first, "root")
    route_off = _render(labels, clean_p, decoy_x, p_first, "decoy")
    corrupt = _render(labels, corrupt_p, decoy_x, p_first, "root")
    case_id = hashlib.sha256(
        f"dependency-route-subspace-p3|{seed}|{index}".encode()
    ).hexdigest()[:16]
    return RouteSubspaceCase(
        id=case_id,
        labels=labels,
        clean_p=clean_p,
        decoy_x=decoy_x,
        corrupt_p=corrupt_p,
        route_on_user=route_on,
        route_off_user=route_off,
        corrupt_user=corrupt,
        clean_root=clean_p,
        corrupt_root=corrupt_p,
        decoy_root=decoy_x,
        p_first=p_first,
    )


def build_cases(n: int, seed: int):
    cases = [generate_case(index, seed) for index in range(n)]
    cases.sort(key=lambda case: case.id)
    return cases


def _project_out(x, basis):
    if basis.shape[1] == 0:
        return x
    return x - (x @ basis) @ basis.T


def fit_route_directions(feats, cases, calib_n, layers, seed):
    import torch

    directions, shams, reports = {}, {}, {}
    for layer in layers:
        on = feats[layer]["p"]
        off = feats[layer]["x"]
        midpoint = 0.5 * (on[:calib_n] + off[:calib_n])
        centroids = []
        for field, source in (("clean_p", midpoint), ("decoy_x", off[:calib_n])):
            source_mean = source.mean(0)
            for digit in range(10):
                idx = [i for i in range(calib_n) if getattr(cases[i], field) == digit]
                if not idx:
                    raise RuntimeError(f"calibration is missing {field} digit {digit}")
                centroids.append(source[idx].mean(0) - source_mean)
        centroid_matrix = torch.stack(centroids)
        _u, singular, vh = torch.linalg.svd(centroid_matrix, full_matrices=False)
        rank = int((singular > singular.max().clamp_min(1e-12) * 1e-6).sum())
        basis = vh[:rank].T.contiguous()

        deltas = _project_out(on[:calib_n] - off[:calib_n], basis)
        deltas = torch.nn.functional.normalize(deltas, dim=1)
        direction = _project_out(deltas.mean(0, keepdim=True), basis).squeeze(0)
        direction = torch.nn.functional.normalize(direction, dim=0)

        generator = torch.Generator().manual_seed(seed * 100 + layer)
        sham = torch.randn(direction.shape, generator=generator)
        sham = _project_out(sham.unsqueeze(0), basis).squeeze(0)
        sham = sham - torch.dot(sham, direction) * direction
        sham = torch.nn.functional.normalize(sham, dim=0)

        all_scores = ((on - off) @ direction).tolist()
        calibration_scores = all_scores[:calib_n]
        heldout_scores = all_scores[calib_n:]

        def score_report(scores, subset):
            wins = sum(score > 0 for score in scores)
            losses = sum(score < 0 for score in scores)
            ties = len(scores) - wins - losses
            return {
                "n": len(scores),
                "paired_accuracy": (wins + 0.5 * ties) / max(1, len(scores)),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "sign_p_one_sided": exact_sign_p(wins, losses),
                "mean_margin": mean(scores),
                "subset": subset,
            }

        halves = {}
        held_cases = cases[calib_n:]
        for half in (0, 1):
            idx = [i for i, case in enumerate(held_cases) if hash_half(case.id) == half]
            half_scores = [heldout_scores[i] for i in idx]
            halves[str(half)] = score_report(half_scores, f"hash_half_{half}")
        value_overlap = float(torch.linalg.vector_norm(basis.T @ direction)) if rank else 0.0
        sham_value_overlap = float(torch.linalg.vector_norm(basis.T @ sham)) if rank else 0.0
        reports[str(layer)] = {
            "value_basis_rank": rank,
            "value_overlap": value_overlap,
            "sham_value_overlap": sham_value_overlap,
            "route_sham_dot": float(torch.dot(direction, sham)),
            "calibration": score_report(calibration_scores, "calibration"),
            "heldout": score_report(heldout_scores, "heldout"),
            "hash_halves": halves,
        }
        directions[layer] = direction
        shams[layer] = sham
    primary = reports[str(PRIMARY_LAYER)]
    direction_gate = (
        primary["heldout"]["paired_accuracy"] >= 0.70
        and min(item["paired_accuracy"] for item in primary["hash_halves"].values()) >= 0.65
        and primary["value_overlap"] <= 1e-5
        and primary["sham_value_overlap"] <= 1e-5
        and abs(primary["route_sham_dot"]) <= 1e-5
    )
    return directions, shams, {"layers": reports, "gate_pass": bool(direction_gate)}


class P3Runner(ModelRunner):
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
                raise RuntimeError(f"token length mismatch for {case.id}")
            on_pos = self.locate_checkpoint(on_texts[row], case.labels["p"], case.clean_p, on_offsets[row])
            off_pos = self.locate_checkpoint(off_texts[row], case.labels["p"], case.clean_p, off_offsets[row])
            if on_pos != off_pos:
                raise RuntimeError(f"checkpoint token position mismatch for {case.id}")
            on_ids = on_encoded["input_ids"][row, :on_len]
            off_ids = off_encoded["input_ids"][row, :off_len]
            diff = (on_ids != off_ids).nonzero().flatten().tolist()
            if len(diff) != 1 or diff[0] >= on_pos:
                raise RuntimeError(f"expected one earlier route-token difference for {case.id}: {diff}")
            if int(on_ids[on_pos]) != int(off_ids[off_pos]):
                raise RuntimeError(f"checkpoint token ID mismatch for {case.id}")
            positions.append(on_pos)
        return on_encoded, off_encoded, positions

    def extract_sources(self, cases, layers, batch_size):
        torch = self.torch
        feats = {layer: {"p": [], "x": []} for layer in layers}
        outputs = {"route_on": {"logp": [], "pred": []}, "route_off": {"logp": [], "pred": []}}
        integrity_pairs = 0
        for start in range(0, len(cases), batch_size):
            chunk = cases[start : start + batch_size]
            on_texts = [self.chat_text(case.route_on_user) for case in chunk]
            off_texts = [self.chat_text(case.route_off_user) for case in chunk]
            on_encoded, off_encoded, positions = self._encoded_pair(on_texts, off_texts, chunk)
            integrity_pairs += len(chunk)
            for condition, encoded, feature_key in (
                ("route_on", on_encoded, "p"),
                ("route_off", off_encoded, "x"),
            ):
                inputs = {key: value.cuda() for key, value in encoded.items()}
                with torch.no_grad():
                    output = self.model(**inputs, output_hidden_states=True, use_cache=False)
                last = inputs["attention_mask"].sum(1) - 1
                logp, pred = self._digit_scores(output.logits[torch.arange(len(chunk), device="cuda"), last])
                outputs[condition]["logp"].extend(logp.tolist())
                outputs[condition]["pred"].extend(pred.tolist())
                for layer in layers:
                    hidden = output.hidden_states[layer]
                    feats[layer][feature_key].extend(
                        hidden[row, positions[row]].detach().cpu().to(torch.float32)
                        for row in range(len(chunk))
                    )
                del output, inputs
            del on_encoded, off_encoded
        for layer in layers:
            for key in ("p", "x"):
                feats[layer][key] = torch.stack(feats[layer][key])
        return feats, outputs, integrity_pairs

    def prepare_corrupt(self, cases):
        texts = [self.chat_text(case.corrupt_user) for case in cases]
        encoded = self.tokenize_with_offsets(texts)
        offsets = encoded.pop("offset_mapping").tolist()
        positions = [
            self.locate_checkpoint(text, case.labels["p"], case.corrupt_p, offsets[row])
            for row, (text, case) in enumerate(zip(texts, cases))
        ]
        return encoded, positions

    @staticmethod
    def arm_vectors(on, off, direction, sham):
        scalar = (off - on) @ direction
        magnitude = scalar.abs()
        return {
            "correct_full": on,
            "wrong_full": off,
            "route_swap": on + scalar[:, None] * direction[None, :],
            "sham_plus": on + magnitude[:, None] * sham[None, :],
            "sham_minus": on - magnitude[:, None] * sham[None, :],
        }

    def score_receiver(self, cases, feats, directions, shams, batch_size):
        torch = self.torch
        result = {
            "corrupt": {"logp": [], "pred": []},
            "modes": {mode: {arm: {"logp": [], "pred": []} for arm in ARMS} for mode in MODES},
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
            idx = list(range(start, start + len(chunk)))
            cached_vectors = {}
            for layer in WINDOW_LAYERS:
                cached_vectors[layer] = self.arm_vectors(
                    feats[layer]["p"][idx], feats[layer]["x"][idx], directions[layer], shams[layer]
                )
            for mode, active_layers in MODES.items():
                for arm in ARMS:
                    handles = []
                    for layer in active_layers:
                        patches = cached_vectors[layer][arm].to("cuda", dtype=torch.bfloat16)

                        def hook(_module, _args, block_output, pos=positions, vectors=patches):
                            hidden = block_output[0] if isinstance(block_output, tuple) else block_output
                            changed = hidden.clone()
                            for row, position in enumerate(pos):
                                changed[row, int(position), :] = vectors[row]
                            if isinstance(block_output, tuple):
                                return (changed,) + block_output[1:]
                            return changed

                        handles.append(self.model.model.layers[layer - 1].register_forward_hook(hook))
                    try:
                        with torch.no_grad():
                            patched = self.model(**inputs, use_cache=False)
                    finally:
                        for handle in handles:
                            handle.remove()
                    logp, pred = self._digit_scores(patched.logits[torch.arange(len(chunk), device="cuda"), last])
                    result["modes"][mode][arm]["logp"].extend(logp.tolist())
                    result["modes"][mode][arm]["pred"].extend(pred.tolist())
                    del patched
            del inputs, encoded
            torch.cuda.empty_cache()
        return result


def _equivalent(delta, low, high):
    return delta["ci95"][0] >= low and delta["ci95"][1] <= high


def summarize(cases, calib_n, sources, receiver, direction_report, integrity_pairs):
    held = cases[calib_n:]
    route_on = arm_metrics(held, sources["route_on"]["logp"][calib_n:], sources["route_on"]["pred"][calib_n:])
    route_off = arm_metrics(held, sources["route_off"]["logp"][calib_n:], sources["route_off"]["pred"][calib_n:], "decoy_root")
    corrupt_clean = arm_metrics(held, receiver["corrupt"]["logp"][calib_n:], receiver["corrupt"]["pred"][calib_n:])
    corrupt_own = arm_metrics(held, receiver["corrupt"]["logp"][calib_n:], receiver["corrupt"]["pred"][calib_n:], "corrupt_root")
    corruption = paired_delta(route_on["correct_logp"], corrupt_clean["correct_logp"], seed=2001)
    mode_reports = {}
    for mode, arms in receiver["modes"].items():
        clean = {arm: arm_metrics(held, values["logp"][calib_n:], values["pred"][calib_n:]) for arm, values in arms.items()}
        decoy = {arm: arm_metrics(held, values["logp"][calib_n:], values["pred"][calib_n:], "decoy_root") for arm, values in arms.items()}
        sham_clean_logp = [(a + b) / 2 for a, b in zip(clean["sham_plus"]["correct_logp"], clean["sham_minus"]["correct_logp"])]
        sham_clean_correct = [(a + b) / 2 for a, b in zip(clean["sham_plus"]["correct"], clean["sham_minus"]["correct"])]
        sham_decoy_logp = [(a + b) / 2 for a, b in zip(decoy["sham_plus"]["correct_logp"], decoy["sham_minus"]["correct_logp"])]
        sham_decoy_correct = [(a + b) / 2 for a, b in zip(decoy["sham_plus"]["correct"], decoy["sham_minus"]["correct"])]

        correct_corrupt_lp = paired_delta(clean["correct_full"]["correct_logp"], corrupt_clean["correct_logp"], seed=2100 + len(mode))
        correct_corrupt_acc = paired_delta(clean["correct_full"]["correct"], corrupt_clean["correct"], seed=2200 + len(mode))
        sham_route_clean_lp = paired_delta(sham_clean_logp, clean["route_swap"]["correct_logp"], seed=2300 + len(mode))
        sham_route_clean_acc = paired_delta(sham_clean_correct, clean["route_swap"]["correct"], seed=2400 + len(mode))
        route_sham_decoy_lp = paired_delta(decoy["route_swap"]["correct_logp"], sham_decoy_logp, seed=2500 + len(mode))
        route_sham_decoy_acc = paired_delta(decoy["route_swap"]["correct"], sham_decoy_correct, seed=2600 + len(mode))
        correct_sham_lp = paired_delta(clean["correct_full"]["correct_logp"], sham_clean_logp, seed=2700 + len(mode))
        correct_sham_acc = paired_delta(clean["correct_full"]["correct"], sham_clean_correct, seed=2800 + len(mode))
        correct_wrong_lp = paired_delta(clean["correct_full"]["correct_logp"], clean["wrong_full"]["correct_logp"], seed=2900 + len(mode))
        correct_wrong_acc = paired_delta(clean["correct_full"]["correct"], clean["wrong_full"]["correct"], seed=3000 + len(mode))
        half_deltas = {}
        for half in (0, 1):
            idx = [i for i, case in enumerate(held) if hash_half(case.id) == half]
            half_deltas[str(half)] = mean(sham_clean_logp[i] - clean["route_swap"]["correct_logp"][i] for i in idx)

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
        switch = (
            sham_route_clean_lp["mean"] >= 0.10
            and sham_route_clean_lp["ci95"][0] > 0.0
            and sham_route_clean_acc["mean"] >= 0.03
            and route_sham_decoy_lp["mean"] >= 0.10
            and route_sham_decoy_lp["ci95"][0] > 0.0
            and route_sham_decoy_acc["mean"] >= 0.03
            and _equivalent(correct_sham_lp, -0.10, 0.10)
            and _equivalent(correct_sham_acc, -0.03, 0.03)
            and min(half_deltas.values()) >= 0.0
        )
        null_equivalent = (
            _equivalent(sham_route_clean_lp, -0.10, 0.10)
            and _equivalent(sham_route_clean_acc, -0.03, 0.03)
            and _equivalent(route_sham_decoy_lp, -0.10, 0.10)
            and _equivalent(route_sham_decoy_acc, -0.03, 0.03)
        )
        mode_reports[mode] = {
            "arms": {
                arm: {
                    "clean_accuracy": clean[arm]["accuracy"],
                    "mean_clean_logp": mean(clean[arm]["correct_logp"]),
                    "decoy_accuracy": decoy[arm]["accuracy"],
                    "mean_decoy_logp": mean(decoy[arm]["correct_logp"]),
                }
                for arm in ARMS
            },
            "sham_average": {
                "clean_accuracy": mean(sham_clean_correct),
                "mean_clean_logp": mean(sham_clean_logp),
                "decoy_accuracy": mean(sham_decoy_correct),
                "mean_decoy_logp": mean(sham_decoy_logp),
            },
            "correct_minus_corrupt_logp": {k: v for k, v in correct_corrupt_lp.items() if k != "values"},
            "correct_minus_corrupt_accuracy": {k: v for k, v in correct_corrupt_acc.items() if k != "values"},
            "sham_minus_route_clean_logp": {k: v for k, v in sham_route_clean_lp.items() if k != "values"},
            "sham_minus_route_clean_accuracy": {k: v for k, v in sham_route_clean_acc.items() if k != "values"},
            "route_minus_sham_decoy_logp": {k: v for k, v in route_sham_decoy_lp.items() if k != "values"},
            "route_minus_sham_decoy_accuracy": {k: v for k, v in route_sham_decoy_acc.items() if k != "values"},
            "correct_minus_sham_clean_logp": {k: v for k, v in correct_sham_lp.items() if k != "values"},
            "correct_minus_sham_clean_accuracy": {k: v for k, v in correct_sham_acc.items() if k != "values"},
            "correct_minus_wrong_full_clean_logp": {k: v for k, v in correct_wrong_lp.items() if k != "values"},
            "correct_minus_wrong_full_clean_accuracy": {k: v for k, v in correct_wrong_acc.items() if k != "values"},
            "sham_minus_route_hash_halves": half_deltas,
            "apparatus_gate_pass": bool(apparatus),
            "route_switch_gate_pass": bool(switch),
            "route_null_equivalence_pass": bool(null_equivalent),
        }

    integrity_ok = integrity_pairs == len(cases)
    primary = mode_reports["single21"]
    window = mode_reports["window19_21"]
    if not integrity_ok:
        verdict = "INVALID_CONTROL"
    elif not direction_report["gate_pass"]:
        verdict = "DIRECTION_FAIL"
    elif not primary["apparatus_gate_pass"]:
        verdict = "APPARATUS_FAIL"
    elif primary["route_switch_gate_pass"]:
        verdict = "SUBSPACE_ROUTE_SWITCH"
    elif window["apparatus_gate_pass"] and window["route_switch_gate_pass"]:
        verdict = "WINDOW_ONLY"
    elif primary["route_null_equivalence_pass"] and window["route_null_equivalence_pass"]:
        verdict = "ROUTE_SUBSPACE_NULL"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "integrity_pairs": integrity_pairs,
        "route_on_clean": {"accuracy": route_on["accuracy"], "mean_correct_logp": mean(route_on["correct_logp"])},
        "route_off_clean_decoy_target": {"accuracy": route_off["accuracy"], "mean_correct_logp": mean(route_off["correct_logp"])},
        "corrupt_clean_target": {"accuracy": corrupt_clean["accuracy"], "mean_correct_logp": mean(corrupt_clean["correct_logp"])},
        "corrupt_own_target": {"accuracy": corrupt_own["accuracy"], "mean_correct_logp": mean(corrupt_own["correct_logp"])},
        "clean_minus_corrupt_clean_logp": {k: v for k, v in corruption.items() if k != "values"},
        "direction_gate_pass": bool(direction_report["gate_pass"]),
        "modes": mode_reports,
        "verdict": verdict,
    }


def main(args):
    expected = list(WINDOW_LAYERS)
    if args.layers != expected:
        raise ValueError(f"P3 freezes --layers to {expected}")
    os.makedirs(args.out_dir, exist_ok=True)
    cases = build_cases(args.calib_n + args.test_n, args.seed)
    if args.dry_run:
        print(json.dumps({"n": len(cases), "first": asdict(cases[0])}, indent=2))
        return
    runner = P3Runner(args.model)
    feats, sources, integrity_pairs = runner.extract_sources(cases, args.layers, args.batch_size)
    directions, shams, direction_report = fit_route_directions(feats, cases, args.calib_n, args.layers, args.seed)
    receiver = runner.score_receiver(cases, feats, directions, shams, args.batch_size)
    causal = summarize(cases, args.calib_n, sources, receiver, direction_report, integrity_pairs)
    report = {
        "protocol": "EXPERIMENT_PROTOCOL_DEPENDENCY_ROUTE_SUBSPACE_P3.md",
        "model": args.model,
        "seed": args.seed,
        "calib_n": args.calib_n,
        "test_n": args.test_n,
        "primary_layer": PRIMARY_LAYER,
        "window_layers": list(WINDOW_LAYERS),
        "case_id_sha256": hashlib.sha256("|".join(case.id for case in cases).encode()).hexdigest(),
        "p_first": sum(case.p_first for case in cases),
        "direction": direction_report,
        "causal": causal,
    }
    with open(os.path.join(args.out_dir, "dependency_route_subspace_p3_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=json_scalar)
    with open(os.path.join(args.out_dir, "dependency_route_subspace_p3_cases.jsonl"), "w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=json_scalar))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--calib-n", type=int, default=96)
    parser.add_argument("--test-n", type=int, default=192)
    parser.add_argument("--layers", type=int, nargs="+", default=list(WINDOW_LAYERS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", default="dependency_route_subspace_p3")
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
