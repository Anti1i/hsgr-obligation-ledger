"""One-hop dependency-role probe and activation-patching experiment P1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict

from dependency_apparatus_screen_p1 import build_cases as build_screen_cases
from dependency_patch_p0 import (
    ModelRunner,
    arm_metrics,
    hash_half,
    json_scalar,
    mean,
    paired_delta,
    run_probe,
)


FEATURE_LAYERS = (7, 14, 21)


def build_cases(n: int, seed: int):
    return build_screen_cases("chain1_copy", n, seed)


def choose_donors(cases):
    buckets = {}
    for i, case in enumerate(cases):
        buckets.setdefault(case.clean_p, []).append(i)
    donors = []
    for i, case in enumerate(cases):
        candidates = [j for j in buckets[case.clean_p] if j != i]
        if not candidates:
            raise RuntimeError(f"no matched donor for {case.id}")
        donors.append(candidates[int(case.id, 16) % len(candidates)])
    return donors


class P1Runner(ModelRunner):
    def extract_clean(self, cases, layers, batch_size):
        torch = self.torch
        feats = {layer: {"p": [], "x": []} for layer in layers}
        clean_logp, clean_pred = [], []
        metadata = {"p": [], "x": []}
        clean_texts = [self.chat_text(case.clean_user) for case in cases]
        for start in range(0, len(cases), batch_size):
            chunk = cases[start : start + batch_size]
            texts = clean_texts[start : start + len(chunk)]
            encoded = self.tokenize_with_offsets(texts)
            offsets = encoded.pop("offset_mapping").tolist()
            p_positions, x_positions = [], []
            for j, case in enumerate(chunk):
                p_positions.append(
                    self.locate_checkpoint(
                        texts[j], case.labels["p"], case.clean_p, offsets[j]
                    )
                )
                x_positions.append(
                    self.locate_checkpoint(
                        texts[j], case.labels["x"], case.clean_p, offsets[j]
                    )
                )
                seq_len = int(encoded["attention_mask"][j].sum())
                program_end = texts[j].index("Checkpoint values:")
                for role, position in (("p", p_positions[-1]), ("x", x_positions[-1])):
                    marker = f"({case.labels[role]} +"
                    mention = texts[j].index(marker, 0, program_end)
                    metadata[role].append(
                        [
                            position / seq_len,
                            mention / max(1, program_end),
                            ord(case.labels[role]) / 90.0,
                        ]
                    )
            inputs = {key: value.cuda() for key, value in encoded.items()}
            with torch.no_grad():
                output = self.model(
                    **inputs, output_hidden_states=True, use_cache=False
                )
            last = inputs["attention_mask"].sum(1) - 1
            logp, pred = self._digit_scores(
                output.logits[torch.arange(len(chunk), device="cuda"), last]
            )
            clean_logp.extend(logp.tolist())
            clean_pred.extend(pred.tolist())
            for layer in layers:
                hidden = output.hidden_states[layer]
                for j in range(len(chunk)):
                    feats[layer]["p"].append(
                        hidden[j, p_positions[j]].detach().cpu().to(torch.float32)
                    )
                    feats[layer]["x"].append(
                        hidden[j, x_positions[j]].detach().cpu().to(torch.float32)
                    )
            del output, inputs, encoded
        for layer in layers:
            for role in ("p", "x"):
                feats[layer][role] = torch.stack(feats[layer][role])
        return feats, metadata, clean_logp, clean_pred

    def score_corrupt_and_patches(
        self, cases, feats, layers, batch_size, donor_idx
    ):
        torch = self.torch
        result = {"corrupt": {"logp": [], "pred": []}, "layers": {}}
        for layer in layers:
            result["layers"][str(layer)] = {
                arm: {"logp": [], "pred": []}
                for arm in ("correct_role", "wrong_route", "cross_problem")
            }
        for start in range(0, len(cases), batch_size):
            chunk = cases[start : start + batch_size]
            encoded, p_positions = self.prepare_corrupt(chunk)
            inputs = {key: value.cuda() for key, value in encoded.items()}
            last = inputs["attention_mask"].sum(1) - 1
            with torch.no_grad():
                base = self.model(**inputs, use_cache=False)
            logp, pred = self._digit_scores(
                base.logits[torch.arange(len(chunk), device="cuda"), last]
            )
            result["corrupt"]["logp"].extend(logp.tolist())
            result["corrupt"]["pred"].extend(pred.tolist())
            del base

            global_idx = list(range(start, start + len(chunk)))
            for layer in layers:
                vectors = {
                    "correct_role": feats[layer]["p"][global_idx],
                    "wrong_route": feats[layer]["x"][global_idx],
                    "cross_problem": feats[layer]["p"][[donor_idx[i] for i in global_idx]],
                }
                block = self.model.model.layers[layer - 1]
                for arm, cpu_vectors in vectors.items():
                    patch_vectors = cpu_vectors.to("cuda", dtype=torch.bfloat16)

                    def hook(
                        _module,
                        _args,
                        block_output,
                        positions=p_positions,
                        patches=patch_vectors,
                    ):
                        hidden = (
                            block_output[0]
                            if isinstance(block_output, tuple)
                            else block_output
                        )
                        changed = hidden.clone()
                        for row, position in enumerate(positions):
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
                        patched.logits[
                            torch.arange(len(chunk), device="cuda"), last
                        ]
                    )
                    result["layers"][str(layer)][arm]["logp"].extend(
                        logp.tolist()
                    )
                    result["layers"][str(layer)][arm]["pred"].extend(
                        pred.tolist()
                    )
                    del patched
            del inputs, encoded
            torch.cuda.empty_cache()
        return result


def summarize(cases, calib_n, clean_logp, clean_pred, patch_output, probe):
    held = cases[calib_n:]
    clean = arm_metrics(held, clean_logp[calib_n:], clean_pred[calib_n:])
    corrupt_clean = arm_metrics(
        held,
        patch_output["corrupt"]["logp"][calib_n:],
        patch_output["corrupt"]["pred"][calib_n:],
    )
    corrupt_own = arm_metrics(
        held,
        patch_output["corrupt"]["logp"][calib_n:],
        patch_output["corrupt"]["pred"][calib_n:],
        "corrupt_root",
    )
    corruption = paired_delta(
        clean["correct_logp"], corrupt_clean["correct_logp"], seed=501
    )
    layer_reports = {}
    for layer, arms in patch_output["layers"].items():
        summaries = {
            arm: arm_metrics(
                held, values["logp"][calib_n:], values["pred"][calib_n:]
            )
            for arm, values in arms.items()
        }
        correct_corrupt_lp = paired_delta(
            summaries["correct_role"]["correct_logp"],
            corrupt_clean["correct_logp"],
            seed=600 + int(layer),
        )
        correct_corrupt_acc = paired_delta(
            summaries["correct_role"]["correct"],
            corrupt_clean["correct"],
            seed=700 + int(layer),
        )
        correct_wrong_lp = paired_delta(
            summaries["correct_role"]["correct_logp"],
            summaries["wrong_route"]["correct_logp"],
            seed=800 + int(layer),
        )
        correct_wrong_acc = paired_delta(
            summaries["correct_role"]["correct"],
            summaries["wrong_route"]["correct"],
            seed=900 + int(layer),
        )
        recovery = correct_corrupt_lp["mean"] / max(1e-9, corruption["mean"])
        half_deltas = {}
        for half in (0, 1):
            indices = [
                i for i, case in enumerate(held) if hash_half(case.id) == half
            ]
            half_deltas[str(half)] = mean(
                summaries["correct_role"]["correct_logp"][i]
                - summaries["wrong_route"]["correct_logp"][i]
                for i in indices
            )
        apparatus = (
            clean["accuracy"] >= 0.90
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
            and min(half_deltas.values()) >= 0.0
            and recovery >= 0.20
        )
        equivalent = (
            correct_wrong_lp["ci95"][0] >= -0.10
            and correct_wrong_lp["ci95"][1] <= 0.10
            and correct_wrong_acc["ci95"][0] >= -0.03
            and correct_wrong_acc["ci95"][1] <= 0.03
        )
        layer_reports[layer] = {
            "arms": {
                arm: {
                    "accuracy": values["accuracy"],
                    "mean_correct_logp": mean(values["correct_logp"]),
                }
                for arm, values in summaries.items()
            },
            "correct_minus_corrupt_logp": {
                key: value
                for key, value in correct_corrupt_lp.items()
                if key != "values"
            },
            "correct_minus_corrupt_accuracy": {
                key: value
                for key, value in correct_corrupt_acc.items()
                if key != "values"
            },
            "correct_minus_wrong_logp": {
                key: value
                for key, value in correct_wrong_lp.items()
                if key != "values"
            },
            "correct_minus_wrong_accuracy": {
                key: value
                for key, value in correct_wrong_acc.items()
                if key != "values"
            },
            "correct_recovery_fraction": recovery,
            "correct_minus_wrong_hash_halves": half_deltas,
            "apparatus_gate_pass": bool(apparatus),
            "causal_gate_pass": bool(causal),
            "equivalence_pass": bool(equivalent),
        }
    selected = layer_reports[str(probe["selected_layer"])]
    if not probe["gate_pass"]:
        verdict = "REPRESENTATION_FAIL"
    elif not selected["apparatus_gate_pass"]:
        verdict = "APPARATUS_FAIL"
    elif selected["causal_gate_pass"]:
        verdict = "CAUSAL_PASS"
    elif selected["equivalence_pass"]:
        verdict = "GAP_CANDIDATE"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "clean": {
            "accuracy": clean["accuracy"],
            "mean_correct_logp": mean(clean["correct_logp"]),
        },
        "corrupt_clean_answer": {
            "accuracy": corrupt_clean["accuracy"],
            "mean_correct_logp": mean(corrupt_clean["correct_logp"]),
        },
        "corrupt_own_answer": {
            "accuracy": corrupt_own["accuracy"],
            "mean_correct_logp": mean(corrupt_own["correct_logp"]),
        },
        "clean_minus_corrupt_clean_logp": {
            key: value for key, value in corruption.items() if key != "values"
        },
        "layers": layer_reports,
        "verdict": verdict,
    }


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    cases = build_cases(args.calib_n + args.test_n, args.seed)
    if args.dry_run:
        print(json.dumps({"n": len(cases), "first": asdict(cases[0])}, indent=2))
        return
    runner = P1Runner(args.model)
    feats, metadata, clean_logp, clean_pred = runner.extract_clean(
        cases, args.layers, args.batch_size
    )
    probe = run_probe(feats, metadata, cases, args.calib_n, args.layers)
    donors = choose_donors(cases)
    patch_output = runner.score_corrupt_and_patches(
        cases, feats, args.layers, args.batch_size, donors
    )
    causal = summarize(
        cases, args.calib_n, clean_logp, clean_pred, patch_output, probe
    )
    report = {
        "protocol": "EXPERIMENT_PROTOCOL_DEPENDENCY_PATCH_P1.md",
        "model": args.model,
        "family": "chain1_copy",
        "seed": args.seed,
        "calib_n": args.calib_n,
        "test_n": args.test_n,
        "layers": args.layers,
        "case_id_sha256": hashlib.sha256(
            "|".join(case.id for case in cases).encode()
        ).hexdigest(),
        "p_first": sum(case.p_first for case in cases),
        "probe": probe,
        "causal": causal,
    }
    report_path = os.path.join(args.out_dir, "dependency_patch_p1_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=json_scalar)
    cases_path = os.path.join(args.out_dir, "dependency_patch_p1_cases.jsonl")
    with open(cases_path, "w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=json_scalar))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--calib-n", type=int, default=96)
    parser.add_argument("--test-n", type=int, default=192)
    parser.add_argument("--layers", type=int, nargs="+", default=list(FEATURE_LAYERS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", default="dependency_patch_p1")
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())

