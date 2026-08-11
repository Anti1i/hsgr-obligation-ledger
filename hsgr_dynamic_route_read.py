"""Positive-only dynamic hidden RouteGuide action ceiling.

This is a final mechanism-level test after two failures:

* textual GUIDE-FOCUS markers were behaviorally ignored;
* a pooled selected-minus-other residual vector mainly damaged the wrong arm.

Every arm here receives the identical full prompt and all support evidence.  A
clean prefill caches token hidden states for one selected evidence span.  During
generation, the live decoder state queries only that cached span and a
norm-bounded route-local read is added at the active position.  No other block
is subtracted, masked, or pruned.

The oracle-correct span is compared with a length-matched predecessor span.
This tests action headroom, not a deployable selector.

Calibration stability gate
--------------------------
The 80-example calibration set is split into two fixed halves.  A configuration
may touch the held-out set only if, on both halves, correct-vs-base EM and
correct-vs-wrong EM are each >=2.5pp and correct-vs-base F1 is non-negative.

Held-out success gates
----------------------
1. correct beats base by >=3pp normalized EM and >=.03 official answer F1,
   with paired p<.05 for both;
2. correct beats wrong by >=3pp EM with exact McNemar p<.05;
3. correct-minus-base EM is non-decreasing from 2-hop to 3-hop to 4-hop.

Failure ends the hidden Guide control line; wrong-route damage is not success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict

from hsgr_focus_route_ceiling import (
    build_base_prompt,
    exact_mcnemar,
    make_unit,
    paired_signflip_p,
    score_text,
    unit_id,
)
from hsgr_hidden_route_guide import (
    HiddenRouteRunner,
    char_spans,
    mean,
    read_ids,
    select_units,
    token_indices,
)
from mh_e0 import load_rows


FEATURE_LAYERS = (14, 21, 28)
ALPHAS = (0.04, 0.08, 0.16)
VALUE_MODES = ("raw", "centered")
ATTENTION_TEMPERATURE = 0.05


def sha_ids(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def select_new_units(
    data: str,
    exclude_cases: list[str],
    original_limit: int,
    calib_n: int,
    test_n: int,
    seed: int,
):
    # Reconstruct all 320 IDs consumed by the immediately preceding hidden
    # RouteGuide experiment, including its calibration IDs (which were not
    # written to the cases artifact).
    old_calib, old_test, old_meta = select_units(
        data=data,
        exclude_cases=exclude_cases,
        original_limit=original_limit,
        calib_n=80,
        test_n=240,
        seed=20260813,
    )
    old_hidden_ids = {u["id"] for u in old_calib + old_test}
    if sha_ids({u["id"] for u in old_calib}) != old_meta["calib_id_sha256"]:
        raise RuntimeError("failed to reconstruct prior hidden calibration IDs")
    if sha_ids({u["id"] for u in old_test}) != old_meta["test_id_sha256"]:
        raise RuntimeError("failed to reconstruct prior hidden test IDs")

    original_ids = {unit_id(row) for row in load_rows(data, original_limit, seed=0)}
    source_sets = [(os.path.abspath(path), read_ids(path)) for path in exclude_cases]
    excluded = set(original_ids) | old_hidden_ids
    for _, ids in source_sets:
        excluded.update(ids)

    pool = []
    skipped = 0
    for row in load_rows(data, 0, seed=0):
        if unit_id(row) in excluded:
            continue
        unit = make_unit(row)
        if unit is None:
            skipped += 1
            continue
        pool.append(unit)
    need = calib_n + test_n
    if len(pool) < need:
        raise SystemExit(f"need {need} untouched units; found {len(pool)}")
    chosen = random.Random(seed).sample(pool, need)
    calib = chosen[:calib_n]
    test = chosen[calib_n:]
    meta = {
        "original_excluded": len(original_ids),
        "prior_case_sources": [
            {"path": path, "n": len(ids), "id_sha256": sha_ids(ids)}
            for path, ids in source_sets
        ],
        "prior_hidden_excluded": len(old_hidden_ids),
        "prior_hidden_id_sha256": sha_ids(old_hidden_ids),
        "excluded_union": len(excluded),
        "untouched_pool": len(pool),
        "skipped_incomplete_support": skipped,
        "selection_seed": seed,
        "calib_n": len(calib),
        "test_n": len(test),
        "calib_hops": dict(Counter(u["n_hops"] for u in calib)),
        "test_hops": dict(Counter(u["n_hops"] for u in test)),
        "calib_id_sha256": sha_ids({u["id"] for u in calib}),
        "test_id_sha256": sha_ids({u["id"] for u in test}),
    }
    return calib, test, meta


class DynamicRouteRunner(HiddenRouteRunner):
    def extract_memories(self, units, users, feature_layers, bs):
        torch = self.torch
        store = {
            layer: {"correct_route": [], "wrong_route": []}
            for layer in feature_layers
        }
        for start in range(0, len(units), bs):
            chunk_units = units[start : start + bs]
            chunk_users = users[start : start + bs]
            texts = [self.chat_text(user) for user in chunk_users]
            enc = self.encode_texts(texts)
            with torch.no_grad():
                out = self.model(
                    **enc, output_hidden_states=True, use_cache=False, return_dict=True
                )
            max_len = enc["input_ids"].shape[1]
            all_evidence_idxs = []
            for text, unit in zip(texts, chunk_units):
                single = self.tok(
                    text,
                    truncation=True,
                    max_length=self.max_context,
                    return_offsets_mapping=True,
                )
                pad = max_len - len(single["input_ids"])
                _, evidence_chars = char_spans(text, len(unit["blocks"]))
                all_evidence_idxs.append([
                    token_indices(single["offset_mapping"], span, pad)
                    for span in evidence_chars
                ])
            for layer in feature_layers:
                hidden = out.hidden_states[layer]
                for b, unit in enumerate(chunk_units):
                    evidence_idxs = all_evidence_idxs[b]
                    for arm, selected in (
                        ("correct_route", unit["correct_focus"]),
                        ("wrong_route", unit["wrong_focus"]),
                    ):
                        memory = hidden[b, evidence_idxs[selected]].detach()
                        store[layer][arm].append(memory.to("cpu", dtype=torch.bfloat16))
            del enc, out
            torch.cuda.empty_cache()
            print(
                f"[memory-prefill] {min(start + bs, len(units))}/{len(units)}",
                flush=True,
            )
        return store

    def _install_dynamic_hook(
        self,
        feature_layer: int,
        alpha: float,
        memories,
        value_mode: str,
        temperature: float,
    ):
        torch = self.torch
        if value_mode not in VALUE_MODES:
            raise ValueError(value_mode)
        lengths = [int(memory.shape[0]) for memory in memories]
        max_tokens = max(lengths)
        hidden_size = int(memories[0].shape[1])
        mem = torch.zeros(
            len(memories), max_tokens, hidden_size,
            device="cuda", dtype=torch.bfloat16,
        )
        mask = torch.zeros(len(memories), max_tokens, device="cuda", dtype=torch.bool)
        for i, memory in enumerate(memories):
            n = memory.shape[0]
            mem[i, :n] = memory.to("cuda", dtype=torch.bfloat16)
            mask[i, :n] = True
        values = mem.float()
        keys = values
        keys = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        span_mean = (values * mask.unsqueeze(-1)).sum(dim=1) / denom
        block = self.model.model.layers[feature_layer - 1]

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            active = hidden[:, -1, :].float()
            active_norm = active.norm(dim=-1, keepdim=True)
            query = active / (active_norm + 1e-8)
            scores = torch.einsum("bd,bsd->bs", query, keys) / float(temperature)
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            route = torch.einsum("bs,bsd->bd", weights, values)
            if value_mode == "centered":
                route = route - span_mean
            route = route / (route.norm(dim=-1, keepdim=True) + 1e-8)
            mixed = query + float(alpha) * route
            mixed = mixed / (mixed.norm(dim=-1, keepdim=True) + 1e-8)
            changed = hidden.clone()
            changed[:, -1, :] = (mixed * active_norm).to(hidden.dtype)
            if isinstance(output, tuple):
                return (changed,) + output[1:]
            return changed

        return block.register_forward_hook(hook)

    def generate_dynamic(
        self,
        users,
        arm,
        memories,
        feature_layer,
        alpha,
        value_mode,
        temperature,
        bs,
        max_new,
    ):
        torch = self.torch
        outputs = []
        for start in range(0, len(users), bs):
            chunk = users[start : start + bs]
            texts = [self.chat_text(user) for user in chunk]
            enc = self.encode_texts(texts)
            prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
            self.token_counts[f"{arm}_prompt"] += sum(int(x) for x in prompt_lens)
            handle = self._install_dynamic_hook(
                feature_layer=feature_layer,
                alpha=alpha,
                memories=memories[start : start + len(chunk)],
                value_mode=value_mode,
                temperature=temperature,
            )
            try:
                with torch.no_grad():
                    gen = self.model.generate(
                        **enc,
                        max_new_tokens=max_new,
                        do_sample=False,
                        pad_token_id=self.tok.pad_token_id,
                    )
            finally:
                handle.remove()
            plen = enc["input_ids"].shape[1]
            for j in range(len(chunk)):
                new = gen[j, plen:]
                self.token_counts[f"{arm}_generated"] += int(
                    (new != self.tok.pad_token_id).sum().item()
                )
                outputs.append(self.tok.decode(new, skip_special_tokens=True))
            del enc, gen
            torch.cuda.empty_cache()
        return outputs


def evaluate(units, texts):
    em, f1, answers = [], [], []
    for unit, text in zip(units, texts):
        hit, score, answer = score_text(unit, text)
        em.append(hit)
        f1.append(score)
        answers.append(answer)
    return em, f1, answers


def subset(values, idxs):
    return [values[i] for i in idxs]


def delta(a, b):
    return mean(a) - mean(b)


def calibration_row(base, correct, wrong, half_idxs):
    halves = {}
    for name, idxs in half_idxs.items():
        b_em = subset(base["em"], idxs)
        b_f1 = subset(base["f1"], idxs)
        c_em = subset(correct["em"], idxs)
        c_f1 = subset(correct["f1"], idxs)
        w_em = subset(wrong["em"], idxs)
        halves[name] = {
            "n": len(idxs),
            "base_em": mean(b_em),
            "correct_em": mean(c_em),
            "wrong_em": mean(w_em),
            "delta_correct_base_em": delta(c_em, b_em),
            "delta_correct_wrong_em": delta(c_em, w_em),
            "delta_correct_base_f1": delta(c_f1, b_f1),
        }
    em_floor = min(
        metric
        for h in halves.values()
        for metric in (
            h["delta_correct_base_em"], h["delta_correct_wrong_em"]
        )
    )
    f1_floor = min(h["delta_correct_base_f1"] for h in halves.values())
    pre_gate = em_floor >= 0.025 - 1e-12 and f1_floor >= -1e-12
    return halves, em_floor, f1_floor, pre_gate


def save_report(path, report):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, ensure_ascii=False)


def main(args):
    calib, test, data_meta = select_new_units(
        data=args.data,
        exclude_cases=args.exclude_cases,
        original_limit=args.original_exclude_limit,
        calib_n=args.calib,
        test_n=args.test,
        seed=args.seed,
    )
    print("[data] " + json.dumps(data_meta, ensure_ascii=False), flush=True)
    if args.dry_run:
        for unit in (calib[:1] + test[:1]):
            user = build_base_prompt(unit)
            goal_chars, evidence_chars = char_spans(user, len(unit["blocks"]))
            print(json.dumps({
                "id": unit["id"],
                "n_hops": unit["n_hops"],
                "goal": unit["fields"]["compiled_goal"],
                "correct_route": unit["correct_focus"],
                "wrong_route": unit["wrong_focus"],
                "goal_chars": goal_chars,
                "evidence_chars": evidence_chars,
            }, ensure_ascii=False))
        return

    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "hsgr_dynamic_route_report.json")
    runner = DynamicRouteRunner(args.model, args.max_context)
    cal_users = [build_base_prompt(u) for u in calib]
    cal_memories = runner.extract_memories(
        calib, cal_users, FEATURE_LAYERS, args.bs_hidden
    )
    base_texts = runner.generate(
        cal_users, "cal_base", args.bs_generate, args.max_new
    )
    b_em, b_f1, _ = evaluate(calib, base_texts)
    base = {"em": b_em, "f1": b_f1}
    half = len(calib) // 2
    half_idxs = {"A": list(range(0, half)), "B": list(range(half, len(calib)))}
    print(f"[cal-base] em={mean(b_em):.4f} f1={mean(b_f1):.4f}", flush=True)

    grid = []
    candidates = []
    for value_mode in VALUE_MODES:
        for layer in FEATURE_LAYERS:
            for alpha in ALPHAS:
                scores = {}
                for arm in ("correct_route", "wrong_route"):
                    name = f"cal_{value_mode}_l{layer}_a{alpha:.2f}_{arm}"
                    texts = runner.generate_dynamic(
                        cal_users,
                        name,
                        cal_memories[layer][arm],
                        feature_layer=layer,
                        alpha=alpha,
                        value_mode=value_mode,
                        temperature=ATTENTION_TEMPERATURE,
                        bs=args.bs_generate,
                        max_new=args.max_new,
                    )
                    em, f1, _ = evaluate(calib, texts)
                    scores[arm] = {"em": em, "f1": f1}
                halves, em_floor, f1_floor, pre_gate = calibration_row(
                    base, scores["correct_route"], scores["wrong_route"], half_idxs
                )
                row = {
                    "value_mode": value_mode,
                    "feature_layer": layer,
                    "alpha": alpha,
                    "temperature": ATTENTION_TEMPERATURE,
                    "halves": halves,
                    "em_floor": em_floor,
                    "f1_floor": f1_floor,
                    "calibration_pre_gate": pre_gate,
                }
                grid.append(row)
                if pre_gate:
                    candidates.append(row)
                print("[cal-grid] " + json.dumps(row, sort_keys=True), flush=True)

    def selection_key(row):
        avg_em = mean([
            h["delta_correct_base_em"] + h["delta_correct_wrong_em"]
            for h in row["halves"].values()
        ])
        return (
            row["em_floor"], row["f1_floor"], avg_em,
            -row["alpha"], -row["feature_layer"], row["value_mode"] == "centered",
        )

    selected = max(candidates, key=selection_key) if candidates else max(grid, key=selection_key)
    calibration_pass = bool(candidates)
    calibration_report = {
        "experiment": "HSGR positive-only dynamic route-local hidden read",
        "claim_boundary": (
            "Gold final-hop route indices measure oracle action headroom only; "
            "no deployable hidden route observer is established."
        ),
        "data": data_meta,
        "mechanism": {
            "positive_only": True,
            "other_evidence_subtracted_or_pruned": False,
            "temperature": ATTENTION_TEMPERATURE,
        },
        "calibration": {
            "base_em": mean(b_em),
            "base_f1": mean(b_f1),
            "half_sizes": {name: len(idxs) for name, idxs in half_idxs.items()},
            "pre_gate": (
                "both halves: correct-base EM >=2.5pp, correct-wrong EM >=2.5pp, "
                "correct-base F1 non-negative"
            ),
            "grid": grid,
            "any_candidate_pass": calibration_pass,
        },
        "selected": selected,
        "heldout_touched": False,
        "advance_to_heldout": calibration_pass,
        "token_counts": dict(runner.token_counts),
        "model": args.model,
    }
    print("[selected] " + json.dumps(selected, sort_keys=True), flush=True)
    if not calibration_pass:
        calibration_report["decision"] = (
            "CALIBRATION GATE FAIL: held-out set was not decoded; stop hidden Guide control line."
        )
        save_report(report_path, calibration_report)
        print(json.dumps({
            "advance_to_heldout": False,
            "selected_descriptive_only": selected,
            "report": report_path,
        }, indent=1), flush=True)
        return

    del cal_memories
    runner.torch.cuda.empty_cache()
    test_users = [build_base_prompt(u) for u in test]
    layer = int(selected["feature_layer"])
    test_memories = runner.extract_memories(test, test_users, [layer], args.bs_hidden)
    outputs = {
        "base": runner.generate(
            test_users, "test_base", args.bs_generate, args.max_new
        )
    }
    for arm in ("correct_route", "wrong_route"):
        outputs[arm] = runner.generate_dynamic(
            test_users,
            f"test_{arm}",
            test_memories[layer][arm],
            feature_layer=layer,
            alpha=float(selected["alpha"]),
            value_mode=selected["value_mode"],
            temperature=float(selected["temperature"]),
            bs=args.bs_generate,
            max_new=args.max_new,
        )

    results = {}
    for arm, texts in outputs.items():
        em, f1, answers = evaluate(test, texts)
        results[arm] = {"em": em, "f1": f1, "answers": answers}
        print(f"[test] {arm} em={mean(em):.4f} f1={mean(f1):.4f}", flush=True)
    paired = {
        "correct_vs_base_em": exact_mcnemar(
            results["correct_route"]["em"], results["base"]["em"]
        ),
        "correct_vs_wrong_em": exact_mcnemar(
            results["correct_route"]["em"], results["wrong_route"]["em"]
        ),
        "correct_vs_base_f1": paired_signflip_p(
            results["correct_route"]["f1"], results["base"]["f1"],
            seed=args.seed, samples=args.permutation_samples,
        ),
    }
    deltas = {
        "correct_vs_base_em": delta(
            results["correct_route"]["em"], results["base"]["em"]
        ),
        "correct_vs_wrong_em": delta(
            results["correct_route"]["em"], results["wrong_route"]["em"]
        ),
        "correct_vs_base_f1": delta(
            results["correct_route"]["f1"], results["base"]["f1"]
        ),
    }
    depth_delta = {}
    for hop in sorted({u["n_hops"] for u in test}):
        idxs = [i for i, u in enumerate(test) if u["n_hops"] == hop]
        depth_delta[str(hop)] = mean(
            [results["correct_route"]["em"][i] for i in idxs]
        ) - mean([results["base"]["em"][i] for i in idxs])
    depth_keys = [str(h) for h in (2, 3, 4) if str(h) in depth_delta]
    depth_non_decreasing = all(
        depth_delta[b] >= depth_delta[a] - 1e-12
        for a, b in zip(depth_keys, depth_keys[1:])
    )
    gates = {
        "gain_vs_base": (
            deltas["correct_vs_base_em"] >= 0.03
            and deltas["correct_vs_base_f1"] >= 0.03
            and paired["correct_vs_base_em"]["p"] < 0.05
            and paired["correct_vs_base_f1"]["p"] < 0.05
        ),
        "route_specificity": (
            deltas["correct_vs_wrong_em"] >= 0.03
            and paired["correct_vs_wrong_em"]["p"] < 0.05
        ),
        "depth_non_decreasing": depth_non_decreasing,
    }

    cases_path = os.path.join(args.out_dir, "hsgr_dynamic_route_cases.jsonl")
    with open(cases_path, "w", encoding="utf-8") as handle:
        for i, unit in enumerate(test):
            handle.write(json.dumps({
                "id": unit["id"],
                "n_hops": unit["n_hops"],
                "compiled_goal": unit["fields"]["compiled_goal"],
                "gold": unit["gold"],
                "correct_route": unit["correct_focus"],
                "wrong_route": unit["wrong_focus"],
                "arms": {
                    arm: {
                        "answer": results[arm]["answers"][i],
                        "em": results[arm]["em"][i],
                        "f1": results[arm]["f1"][i],
                        "text": outputs[arm][i][:1000],
                    }
                    for arm in outputs
                },
            }, ensure_ascii=False) + "\n")

    calibration_report.update({
        "heldout_touched": True,
        "test": {
            "official_em": {arm: mean(v["em"]) for arm, v in results.items()},
            "official_f1": {arm: mean(v["f1"]) for arm, v in results.items()},
            "deltas": deltas,
            "paired": paired,
            "depth_correct_vs_base_em": depth_delta,
            "gates": gates,
            "advance_to_hidden_route_observer": all(gates.values()),
        },
        "token_counts": dict(runner.token_counts),
        "decision": "PASS" if all(gates.values()) else "HELD-OUT GATE FAIL",
    })
    save_report(report_path, calibration_report)
    print("\n== HSGR dynamic route-local read held-out result ==")
    print(json.dumps({
        "selected": selected,
        "official_em": calibration_report["test"]["official_em"],
        "official_f1": calibration_report["test"]["official_f1"],
        "deltas": deltas,
        "paired": paired,
        "depth": depth_delta,
        "gates": gates,
        "advance": calibration_report["test"]["advance_to_hidden_route_observer"],
    }, indent=1, ensure_ascii=False))
    print(f"saved {report_path} and {cases_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/musique_ans_val.jsonl")
    ap.add_argument("--exclude-cases", action="append", default=[])
    ap.add_argument("--out-dir", default="hsgr_dynamic_route")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--original-exclude-limit", type=int, default=200)
    ap.add_argument("--calib", type=int, default=80)
    ap.add_argument("--test", type=int, default=320)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--bs-hidden", type=int, default=4)
    ap.add_argument("--bs-generate", type=int, default=8)
    ap.add_argument("--permutation-samples", type=int, default=50000)
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args())
