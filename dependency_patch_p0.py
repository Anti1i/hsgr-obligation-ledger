"""Controlled dependency-role probing and activation patching P0.

The experiment uses exact, synthetic straight-line programs solely to isolate
dependency role.  It does not train or evaluate a Guide/policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Iterable


FEATURE_LAYERS = (7, 14, 21)
OPS = ("+", "-", "*")


def json_scalar(value):
    """Convert NumPy scalar diagnostics without changing experiment values."""
    if value.__class__.__module__.startswith("numpy") and hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def apply_op(left: int, op: str, right: int) -> int:
    if op == "+":
        return (left + right) % 10
    if op == "-":
        return (left - right) % 10
    if op == "*":
        return (left * right) % 10
    raise ValueError(op)


@dataclass
class ProgramCase:
    id: str
    labels: dict[str, str]
    ops: dict[str, str]
    clean_values: dict[str, int]
    corrupt_p: int
    clean_root: int
    corrupt_root: int
    p_first: bool
    clean_user: str
    corrupt_user: str
    program_mentions: dict[str, int]


def execute(values: dict[str, int], ops: dict[str, str]) -> dict[str, int]:
    out = dict(values)
    out["r"] = apply_op(out["p"], ops["r"], out["q"])
    out["s"] = apply_op(out["p"], ops["s"], out["t"])
    out["u"] = apply_op(out["r"], ops["u"], out["s"])
    out["root"] = apply_op(out["u"], ops["root"], out["q"])
    out["rd"] = apply_op(out["x"], ops["r"], out["qx"])
    out["sd"] = apply_op(out["x"], ops["s"], out["tx"])
    out["ud"] = apply_op(out["rd"], ops["u"], out["sd"])
    out["decoy_root"] = apply_op(out["ud"], ops["root"], out["qx"])
    return out


def _line(lhs: str, left: str, op: str, right: str) -> str:
    return f"{lhs} = ({left} {op} {right}) % 10"


def build_user(
    labels: dict[str, str],
    ops: dict[str, str],
    values: dict[str, int],
    p_value: int,
    p_first: bool,
) -> tuple[str, dict[str, int]]:
    role_line = {
        "r": _line(labels["r"], labels["p"], ops["r"], labels["q"]),
        "s": _line(labels["s"], labels["p"], ops["s"], labels["t"]),
        "u": _line(labels["u"], labels["r"], ops["u"], labels["s"]),
        "root": _line(
            labels["root"], labels["u"], ops["root"], labels["q"]
        ),
        "rd": _line(labels["rd"], labels["x"], ops["r"], labels["qx"]),
        "sd": _line(labels["sd"], labels["x"], ops["s"], labels["tx"]),
        "ud": _line(labels["ud"], labels["rd"], ops["u"], labels["sd"]),
        "decoy_root": _line(
            labels["decoy_root"], labels["ud"], ops["root"], labels["qx"]
        ),
    }
    def order_level(items, salt):
        return tuple(
            sorted(
                items,
                key=lambda role: hashlib.sha256(
                    f"{salt}|{labels[role]}".encode()
                ).hexdigest(),
            )
        )

    level1 = order_level(("r", "s", "rd", "sd"), "level1")
    level2 = order_level(("u", "ud"), "level2")
    level3 = order_level(("root", "decoy_root"), "level3")
    code_roles = (*level1, *level2, *level3)
    code = [role_line[r] for r in code_roles]
    program_mentions = {
        role: next(i for i, line in enumerate(code) if labels[role] in line)
        for role in ("p", "x")
    }
    tail = ("p", "x") if p_first else ("x", "p")
    checkpoint_head = order_level(("q", "t", "qx", "tx"), "checkpoint")
    checkpoint_roles = (*checkpoint_head, *tail)
    checkpoint_lines = []
    for role in checkpoint_roles:
        value = p_value if role == "p" else values[role]
        checkpoint_lines.append(f"{labels[role]} = {value}")
    user = (
        "A straight-line Python-style program resumes from a checkpoint. "
        "All operations are modulo 10. Use the checkpoint values as the current "
        "state; do not recompute or replace them.\n\n"
        "Downstream program:\n"
        + "\n".join(code)
        + f"\nprint({labels['root']})\n\nCheckpoint values:\n"
        + "\n".join(checkpoint_lines)
        + "\n\nWhat single digit is printed? Answer with one digit only."
    )
    return user, program_mentions


def generate_case(index: int, seed: int) -> ProgramCase:
    rng = random.Random(f"dependency-patch-p0|{seed}|{index}")
    roles = [
        "p", "x", "q", "t", "qx", "tx",
        "r", "s", "u", "root", "rd", "sd", "ud", "decoy_root",
    ]
    alphabet = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
    chosen = rng.sample(alphabet, len(roles))
    labels = dict(zip(roles, chosen))
    for _ in range(1000):
        ops = {role: rng.choice(OPS) for role in ("r", "s", "u", "root")}
        p = rng.randrange(10)
        q, t = rng.randrange(10), rng.randrange(10)
        values = {"p": p, "x": p, "q": q, "t": t, "qx": q, "tx": t}
        clean = execute(values, ops)
        alternatives = list(range(10))
        rng.shuffle(alternatives)
        for corrupt_p in alternatives:
            if corrupt_p == p:
                continue
            corrupt_values = dict(values, p=corrupt_p)
            corrupt = execute(corrupt_values, ops)
            if corrupt["root"] != clean["root"]:
                p_first = bool(rng.getrandbits(1))
                clean_user, mentions = build_user(labels, ops, values, p, p_first)
                corrupt_user, _ = build_user(
                    labels, ops, values, corrupt_p, p_first
                )
                case_id = hashlib.sha256(
                    f"dependency-patch-p0|{seed}|{index}".encode()
                ).hexdigest()[:16]
                return ProgramCase(
                    id=case_id,
                    labels=labels,
                    ops=ops,
                    clean_values=clean,
                    corrupt_p=corrupt_p,
                    clean_root=clean["root"],
                    corrupt_root=corrupt["root"],
                    p_first=p_first,
                    clean_user=clean_user,
                    corrupt_user=corrupt_user,
                    program_mentions=mentions,
                )
    raise RuntimeError(f"could not construct sensitive case {index}")


def build_cases(n: int, seed: int) -> list[ProgramCase]:
    cases = [generate_case(i, seed) for i in range(n)]
    cases.sort(key=lambda c: c.id)
    return cases


def exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n)
    return min(1.0, tail)


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / max(1, len(xs))


def auc(labels, scores) -> float:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(int(y) for _, y in pairs)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return 0.5
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(int(y) for _, y in pairs[i:j])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def bootstrap_ci(values, samples=20000, seed=20260813):
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    if not len(arr):
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk = 1000
    for start in range(0, samples, chunk):
        take = min(chunk, samples - start)
        idx = rng.integers(0, len(arr), size=(take, len(arr)))
        means[start : start + take] = arr[idx].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def hash_half(case_id: str) -> int:
    return int(hashlib.sha256(("dep-patch-half|" + case_id).encode()).hexdigest(), 16) % 2


def fit_probe(x, y, seed=0, steps=500):
    import torch

    torch.manual_seed(seed)
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    mu = x.mean(0, keepdim=True)
    sd = x.std(0, keepdim=True).clamp_min(1e-4)
    z = (x - mu) / sd
    model = torch.nn.Linear(z.shape[1], 1)
    opt = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-3)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = model(z).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()
    return model.eval(), mu, sd


def probe_predict(fit, x):
    import torch

    model, mu, sd = fit
    with torch.no_grad():
        z = (torch.as_tensor(x, dtype=torch.float32) - mu) / sd
        return model(z).squeeze(-1).numpy()


def random_project(features, layer: int, out_dim: int = 128):
    import torch

    x = torch.as_tensor(features, dtype=torch.float32)
    x = torch.nn.functional.normalize(x, dim=1)
    gen = torch.Generator().manual_seed(91000 + int(layer))
    proj = torch.randn(x.shape[1], out_dim, generator=gen) / math.sqrt(out_dim)
    return (x @ proj).numpy()


def paired_probe_metrics(scores, n_cases: int):
    p = scores[:n_cases]
    x = scores[n_cases:]
    wins = int((p > x).sum())
    losses = int((p < x).sum())
    ties = n_cases - wins - losses
    paired_acc = (wins + 0.5 * ties) / n_cases
    labels = [1] * n_cases + [0] * n_cases
    return {
        "paired_accuracy": float(paired_acc),
        "pooled_auroc": float(auc(labels, scores.tolist())),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "sign_p_one_sided": exact_sign_p(wins, losses),
    }


class ModelRunner:
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(0)
        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
        except Exception:
            pass
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).cuda().eval()
        self.digit_ids = []
        for digit in range(10):
            ids = self.tok.encode(str(digit), add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"digit {digit} is not one token: {ids}")
            self.digit_ids.append(ids[0])
        print(
            f"[model] {model_id} layers={len(self.model.model.layers)} "
            f"hidden={self.model.config.hidden_size} digit_ids={self.digit_ids}",
            flush=True,
        )

    def chat_text(self, user: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def tokenize_with_offsets(self, texts):
        return self.tok(
            texts,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

    @staticmethod
    def _span_token(offsets, start: int, end: int) -> int:
        hits = [i for i, (l, r) in enumerate(offsets) if r > start and l < end]
        if len(hits) != 1:
            raise RuntimeError(f"expected one token for span {start}:{end}; got {hits}")
        return hits[0]

    def locate_checkpoint(self, text: str, label: str, value: int, offsets) -> int:
        section = text.index("Checkpoint values:")
        marker = f"{label} = {value}"
        start = text.index(marker, section) + len(marker) - len(str(value))
        return self._span_token(offsets, start, start + len(str(value)))

    def _digit_scores(self, logits):
        torch = self.torch
        digits = logits[:, self.digit_ids].float()
        logp = torch.log_softmax(digits, dim=-1)
        pred = digits.argmax(-1)
        return logp.cpu(), pred.cpu()

    def extract_clean(self, cases, layers, batch_size):
        torch = self.torch
        feats = {layer: {"p": [], "x": [], "root": []} for layer in layers}
        clean_logp, clean_pred = [], []
        metadata = {"p": [], "x": []}
        clean_texts = [self.chat_text(c.clean_user) for c in cases]
        root_texts = [text + str(c.clean_root) for text, c in zip(clean_texts, cases)]
        for start in range(0, len(cases), batch_size):
            chunk = cases[start : start + batch_size]
            texts = clean_texts[start : start + len(chunk)]
            encoded = self.tokenize_with_offsets(texts)
            offsets = encoded.pop("offset_mapping").tolist()
            p_pos, x_pos = [], []
            for j, case in enumerate(chunk):
                p_pos.append(self.locate_checkpoint(texts[j], case.labels["p"], case.clean_values["p"], offsets[j]))
                x_pos.append(self.locate_checkpoint(texts[j], case.labels["x"], case.clean_values["x"], offsets[j]))
                seq_len = int(encoded["attention_mask"][j].sum())
                metadata["p"].append([
                    p_pos[-1] / seq_len,
                    case.program_mentions["p"] / 4.0,
                    ord(case.labels["p"]) / 90.0,
                ])
                metadata["x"].append([
                    x_pos[-1] / seq_len,
                    case.program_mentions["x"] / 4.0,
                    ord(case.labels["x"]) / 90.0,
                ])
            model_inputs = {k: v.cuda() for k, v in encoded.items()}
            with torch.no_grad():
                out = self.model(**model_inputs, output_hidden_states=True, use_cache=False)
            last = model_inputs["attention_mask"].sum(1) - 1
            scores, preds = self._digit_scores(out.logits[torch.arange(len(chunk), device="cuda"), last])
            clean_logp.extend(scores.tolist())
            clean_pred.extend(preds.tolist())
            for layer in layers:
                h = out.hidden_states[layer]
                for j in range(len(chunk)):
                    feats[layer]["p"].append(h[j, p_pos[j]].detach().cpu().to(torch.float32))
                    feats[layer]["x"].append(h[j, x_pos[j]].detach().cpu().to(torch.float32))
            del out, model_inputs, encoded

            rtexts = root_texts[start : start + len(chunk)]
            renc = self.tokenize_with_offsets(rtexts)
            renc.pop("offset_mapping")
            rinputs = {k: v.cuda() for k, v in renc.items()}
            with torch.no_grad():
                rout = self.model(**rinputs, output_hidden_states=True, use_cache=False)
            rlast = rinputs["attention_mask"].sum(1) - 1
            for layer in layers:
                h = rout.hidden_states[layer]
                for j in range(len(chunk)):
                    feats[layer]["root"].append(h[j, rlast[j]].detach().cpu().to(torch.float32))
            del rout, rinputs, renc
        for layer in layers:
            for role in feats[layer]:
                feats[layer][role] = torch.stack(feats[layer][role])
        return feats, metadata, clean_logp, clean_pred

    def prepare_corrupt(self, cases):
        texts = [self.chat_text(c.corrupt_user) for c in cases]
        encoded = self.tokenize_with_offsets(texts)
        offsets = encoded.pop("offset_mapping").tolist()
        p_positions = []
        for j, case in enumerate(cases):
            p_positions.append(
                self.locate_checkpoint(texts[j], case.labels["p"], case.corrupt_p, offsets[j])
            )
            clean_ids = self.tok(self.chat_text(case.clean_user), add_special_tokens=False)["input_ids"]
            corrupt_ids = encoded["input_ids"][j, : int(encoded["attention_mask"][j].sum())].tolist()
            if len(clean_ids) != len(corrupt_ids):
                raise RuntimeError(f"token length mismatch for {case.id}")
            diffs = [i for i, (a, b) in enumerate(zip(clean_ids, corrupt_ids)) if a != b]
            if diffs != [p_positions[-1]]:
                raise RuntimeError(f"expected one-token corruption for {case.id}; diffs={diffs}, p={p_positions[-1]}")
        return encoded, p_positions

    def score_corrupt_and_patches(self, cases, feats, layers, batch_size, donor_idx):
        torch = self.torch
        output = {"corrupt": {"logp": [], "pred": []}, "layers": {}}
        for layer in layers:
            output["layers"][str(layer)] = {
                arm: {"logp": [], "pred": []}
                for arm in ("correct_role", "wrong_route", "cross_problem", "root_positive")
            }
        for start in range(0, len(cases), batch_size):
            chunk = cases[start : start + batch_size]
            encoded, p_positions = self.prepare_corrupt(chunk)
            inputs = {k: v.cuda() for k, v in encoded.items()}
            last = inputs["attention_mask"].sum(1) - 1
            with torch.no_grad():
                base = self.model(**inputs, use_cache=False)
            lp, pred = self._digit_scores(base.logits[torch.arange(len(chunk), device="cuda"), last])
            output["corrupt"]["logp"].extend(lp.tolist())
            output["corrupt"]["pred"].extend(pred.tolist())
            del base

            global_idx = list(range(start, start + len(chunk)))
            for layer in layers:
                vectors = {
                    "correct_role": feats[layer]["p"][global_idx],
                    "wrong_route": feats[layer]["x"][global_idx],
                    "cross_problem": feats[layer]["p"][[donor_idx[i] for i in global_idx]],
                    "root_positive": feats[layer]["root"][global_idx],
                }
                for arm, cpu_vec in vectors.items():
                    patch_pos = last.tolist() if arm == "root_positive" else p_positions
                    block = self.model.model.layers[layer - 1]
                    vec = cpu_vec.to("cuda", dtype=torch.bfloat16)

                    def hook(_module, _args, block_output, positions=patch_pos, patch=vec):
                        hidden = block_output[0] if isinstance(block_output, tuple) else block_output
                        changed = hidden.clone()
                        for row, pos in enumerate(positions):
                            changed[row, int(pos), :] = patch[row]
                        if isinstance(block_output, tuple):
                            return (changed,) + block_output[1:]
                        return changed

                    handle = block.register_forward_hook(hook)
                    try:
                        with torch.no_grad():
                            patched = self.model(**inputs, use_cache=False)
                    finally:
                        handle.remove()
                    lp, pred = self._digit_scores(
                        patched.logits[torch.arange(len(chunk), device="cuda"), last]
                    )
                    output["layers"][str(layer)][arm]["logp"].extend(lp.tolist())
                    output["layers"][str(layer)][arm]["pred"].extend(pred.tolist())
                    del patched
            del inputs, encoded
            torch.cuda.empty_cache()
        return output


def choose_donors(cases):
    buckets = {}
    for i, case in enumerate(cases):
        key = (case.clean_values["p"], case.clean_root)
        buckets.setdefault(key, []).append(i)
    value_buckets = {}
    for i, case in enumerate(cases):
        value_buckets.setdefault(case.clean_values["p"], []).append(i)
    donors = []
    for i, case in enumerate(cases):
        candidates = [j for j in buckets[(case.clean_values["p"], case.clean_root)] if j != i]
        if not candidates:
            candidates = [j for j in value_buckets[case.clean_values["p"]] if j != i]
        if not candidates:
            raise RuntimeError(f"no donor for {case.id}")
        donors.append(candidates[int(case.id, 16) % len(candidates)])
    return donors


def run_probe(feats, metadata, cases, calib_n, layers):
    import numpy as np

    calibration = {}
    selected = None
    for layer in layers:
        p = random_project(feats[layer]["p"][:calib_n], layer)
        x = random_project(feats[layer]["x"][:calib_n], layer)
        all_x = np.concatenate([p, x], axis=0)
        all_y = np.asarray([1] * calib_n + [0] * calib_n, dtype=np.float32)
        oof = np.zeros(2 * calib_n, dtype=np.float32)
        fold_by_case = [int(cases[i].id, 16) % 5 for i in range(calib_n)]
        for fold in range(5):
            train_cases = [i for i in range(calib_n) if fold_by_case[i] != fold]
            val_cases = [i for i in range(calib_n) if fold_by_case[i] == fold]
            tr = train_cases + [i + calib_n for i in train_cases]
            va = val_cases + [i + calib_n for i in val_cases]
            fit = fit_probe(all_x[tr], all_y[tr], seed=layer * 10 + fold)
            oof[va] = probe_predict(fit, all_x[va])
        metrics = paired_probe_metrics(oof, calib_n)
        calibration[str(layer)] = metrics
        key = (metrics["paired_accuracy"], metrics["pooled_auroc"], -layer)
        if selected is None or key > selected[0]:
            selected = (key, layer)

    selected_layer = selected[1]
    p_cal = random_project(feats[selected_layer]["p"][:calib_n], selected_layer)
    x_cal = random_project(feats[selected_layer]["x"][:calib_n], selected_layer)
    p_test = random_project(feats[selected_layer]["p"][calib_n:], selected_layer)
    x_test = random_project(feats[selected_layer]["x"][calib_n:], selected_layer)
    train_x = np.concatenate([p_cal, x_cal])
    train_y = np.asarray([1] * calib_n + [0] * calib_n, dtype=np.float32)
    test_n = len(cases) - calib_n
    test_x = np.concatenate([p_test, x_test])
    fit = fit_probe(train_x, train_y, seed=selected_layer * 100)
    test_scores = probe_predict(fit, test_x)
    heldout = paired_probe_metrics(test_scores, test_n)
    p_scores, x_scores = test_scores[:test_n], test_scores[test_n:]
    halves = {}
    for half in (0, 1):
        idx = [i for i, c in enumerate(cases[calib_n:]) if hash_half(c.id) == half]
        wins = sum(p_scores[i] > x_scores[i] for i in idx)
        losses = sum(p_scores[i] < x_scores[i] for i in idx)
        ties = len(idx) - wins - losses
        halves[str(half)] = {
            "n": len(idx),
            "paired_accuracy": (wins + 0.5 * ties) / max(1, len(idx)),
            "wins": wins,
            "losses": losses,
            "ties": ties,
        }

    mp = np.asarray(metadata["p"], dtype=np.float32)
    mx = np.asarray(metadata["x"], dtype=np.float32)
    mtrain = np.concatenate([mp[:calib_n], mx[:calib_n]])
    mtest = np.concatenate([mp[calib_n:], mx[calib_n:]])
    mfit = fit_probe(mtrain, train_y, seed=404)
    metadata_scores = probe_predict(mfit, mtest)
    metadata_metrics = paired_probe_metrics(metadata_scores, test_n)

    rep_gate = (
        heldout["paired_accuracy"] >= 0.70
        and heldout["sign_p_one_sided"] < 0.01
        and min(v["paired_accuracy"] for v in halves.values()) >= 0.65
        and heldout["paired_accuracy"] - metadata_metrics["paired_accuracy"] >= 0.10
    )
    return {
        "calibration_oof": calibration,
        "selected_layer": selected_layer,
        "heldout": heldout,
        "hash_halves": halves,
        "metadata_control": metadata_metrics,
        "gate_pass": bool(rep_gate),
    }


def arm_metrics(cases, logp, pred, answer_field="clean_root"):
    answers = [getattr(c, answer_field) for c in cases]
    correct_logp = [float(row[a]) for row, a in zip(logp, answers)]
    accuracy = [int(int(p) == a) for p, a in zip(pred, answers)]
    return {"accuracy": mean(accuracy), "correct_logp": correct_logp, "correct": accuracy}


def paired_delta(left, right, seed=20260813):
    vals = [a - b for a, b in zip(left, right)]
    return {"mean": mean(vals), "ci95": bootstrap_ci(vals, seed=seed), "values": vals}


def summarize(cases, calib_n, clean_logp, clean_pred, patch_output, probe):
    held = cases[calib_n:]
    clean = arm_metrics(held, clean_logp[calib_n:], clean_pred[calib_n:])
    corrupt_clean = arm_metrics(held, patch_output["corrupt"]["logp"][calib_n:], patch_output["corrupt"]["pred"][calib_n:])
    corrupt_own = arm_metrics(held, patch_output["corrupt"]["logp"][calib_n:], patch_output["corrupt"]["pred"][calib_n:], "corrupt_root")
    corruption = paired_delta(clean["correct_logp"], corrupt_clean["correct_logp"], seed=101)
    layer_reports = {}
    for layer, arms in patch_output["layers"].items():
        arm_summary = {
            arm: arm_metrics(held, values["logp"][calib_n:], values["pred"][calib_n:])
            for arm, values in arms.items()
        }
        correct_wrong_lp = paired_delta(
            arm_summary["correct_role"]["correct_logp"],
            arm_summary["wrong_route"]["correct_logp"],
            seed=200 + int(layer),
        )
        correct_wrong_acc = paired_delta(
            arm_summary["correct_role"]["correct"],
            arm_summary["wrong_route"]["correct"],
            seed=300 + int(layer),
        )
        root_lp = paired_delta(
            arm_summary["root_positive"]["correct_logp"],
            corrupt_clean["correct_logp"],
            seed=400 + int(layer),
        )
        root_acc = arm_summary["root_positive"]["accuracy"] - corrupt_clean["accuracy"]
        recover = (
            mean([
                a - b
                for a, b in zip(
                    arm_summary["correct_role"]["correct_logp"],
                    corrupt_clean["correct_logp"],
                )
            ])
            / max(1e-9, corruption["mean"])
        )
        half_deltas = {}
        for half in (0, 1):
            idx = [i for i, c in enumerate(held) if hash_half(c.id) == half]
            half_deltas[str(half)] = mean(
                arm_summary["correct_role"]["correct_logp"][i]
                - arm_summary["wrong_route"]["correct_logp"][i]
                for i in idx
            )
        apparatus = (
            clean["accuracy"] >= 0.50
            and corrupt_own["accuracy"] >= 0.40
            and corruption["mean"] >= 0.20
            and corruption["ci95"][0] > 0
            and root_acc >= 0.10
            and root_lp["mean"] >= 0.20
            and root_lp["ci95"][0] > 0
        )
        causal = (
            correct_wrong_lp["mean"] >= 0.10
            and correct_wrong_lp["ci95"][0] > 0
            and correct_wrong_acc["mean"] >= 0.03
            and min(half_deltas.values()) >= 0
            and recover >= 0.20
        )
        equivalent = (
            correct_wrong_lp["ci95"][0] >= -0.10
            and correct_wrong_lp["ci95"][1] <= 0.10
            and correct_wrong_acc["ci95"][0] >= -0.03
            and correct_wrong_acc["ci95"][1] <= 0.03
        )
        layer_reports[layer] = {
            "arms": {k: {"accuracy": v["accuracy"], "mean_correct_logp": mean(v["correct_logp"])} for k, v in arm_summary.items()},
            "correct_minus_wrong_logp": {k: v for k, v in correct_wrong_lp.items() if k != "values"},
            "correct_minus_wrong_accuracy": {k: v for k, v in correct_wrong_acc.items() if k != "values"},
            "root_positive_minus_corrupt_logp": {k: v for k, v in root_lp.items() if k != "values"},
            "root_positive_accuracy_gain": root_acc,
            "correct_recovery_fraction": recover,
            "correct_minus_wrong_hash_halves": half_deltas,
            "apparatus_gate_pass": bool(apparatus),
            "causal_gate_pass": bool(causal),
            "equivalence_pass": bool(equivalent),
        }
    selected = layer_reports[str(probe["selected_layer"])]
    if probe["gate_pass"] and selected["apparatus_gate_pass"] and selected["causal_gate_pass"]:
        verdict = "CAUSAL_PASS"
    elif probe["gate_pass"] and selected["apparatus_gate_pass"] and selected["equivalence_pass"]:
        verdict = "GAP_CANDIDATE"
    elif not probe["gate_pass"]:
        verdict = "REPRESENTATION_FAIL"
    elif not selected["apparatus_gate_pass"]:
        verdict = "APPARATUS_FAIL"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "clean": {"accuracy": clean["accuracy"], "mean_correct_logp": mean(clean["correct_logp"])},
        "corrupt_clean_answer": {"accuracy": corrupt_clean["accuracy"], "mean_correct_logp": mean(corrupt_clean["correct_logp"])},
        "corrupt_own_answer": {"accuracy": corrupt_own["accuracy"], "mean_correct_logp": mean(corrupt_own["correct_logp"])},
        "clean_minus_corrupt_clean_logp": {k: v for k, v in corruption.items() if k != "values"},
        "layers": layer_reports,
        "verdict": verdict,
    }


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    total = args.calib_n + args.test_n
    cases = build_cases(total, args.seed)
    if args.dry_run:
        print(json.dumps({"n": len(cases), "first": asdict(cases[0])}, indent=2))
        return

    runner = ModelRunner(args.model)
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
        "protocol": "EXPERIMENT_PROTOCOL_DEPENDENCY_PATCH_P0.md",
        "model": args.model,
        "seed": args.seed,
        "calib_n": args.calib_n,
        "test_n": args.test_n,
        "layers": list(args.layers),
        "case_id_sha256": hashlib.sha256(
            "\n".join(c.id for c in cases).encode()
        ).hexdigest(),
        "p_first": sum(c.p_first for c in cases),
        "probe": probe,
        "causal": causal,
    }
    report_path = os.path.join(args.out_dir, "dependency_patch_p0_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=json_scalar)
    cases_path = os.path.join(args.out_dir, "dependency_patch_p0_cases.jsonl")
    with open(cases_path, "w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
    print(
        json.dumps(report, ensure_ascii=False, indent=2, default=json_scalar),
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--calib-n", type=int, default=96)
    ap.add_argument("--test-n", type=int, default=192)
    ap.add_argument("--layers", type=int, nargs="+", default=list(FEATURE_LAYERS))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-dir", default="dependency_patch_p0")
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args())
