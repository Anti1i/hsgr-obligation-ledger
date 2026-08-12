"""Nested-OOF readers for the Hidden Graph-Energy Guide V0 experiment."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict

from hidden_graph_energy_guide import (
    LAYERS,
    aeo_asymmetric_loss,
    pairwise_energy_loss,
    stable_bucket,
)
from hsgr_focus_route_ceiling import exact_mcnemar
from mh_latent_rerank import auroc, within_problem_auroc


METHODS = (
    "nonhidden_bce",
    "root_dual_bce",
    "flat_dual_bce",
    "graph_dual_bce",
    "graph_dual_aeo",
    "graph_dual_energy",
    "root_dual_energy",
    "graph_last_energy",
    "graph_mean_energy",
    "graph_mismatch_energy",
)


def args_device(torch):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def simple_numeric(value: object) -> float:
    try:
        number = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return math.copysign(math.log1p(abs(number)), number)


def nonhidden_row(meta: dict) -> list[float]:
    bindings = [str(value) for value in meta.get("bindings", ("", ""))]
    while len(bindings) < 2:
        bindings.append("")
    frequencies = [float(value) for value in meta["frequencies"]]
    counts = [float(value) for value in meta["counts"]]
    return [
        frequencies[0], frequencies[1], frequencies[0] * frequencies[1],
        math.log1p(counts[0]), math.log1p(counts[1]),
        math.log1p(len(bindings[0])), math.log1p(len(bindings[1])),
        simple_numeric(bindings[0]), simple_numeric(bindings[1]),
        float(bindings[0] == bindings[1]),
        float(meta.get("is_modal", False)),
    ]


def equal_graph_weights(torch, labels, pids):
    """Equal graph mass; mixed graphs split mass equally across label classes."""
    by_problem = defaultdict(lambda: {0: [], 1: []})
    for index, (label, pid) in enumerate(zip(labels.tolist(), pids)):
        by_problem[pid][int(label > 0.5)].append(index)
    weights = torch.zeros_like(labels)
    graph_mass = 1.0 / max(1, len(by_problem))
    for groups in by_problem.values():
        present = [indices for indices in groups.values() if indices]
        class_mass = graph_mass / len(present)
        for indices in present:
            for index in indices:
                weights[index] = class_mass / len(indices)
    return weights / weights.sum()


def selection_outcomes(scores, labels, pids, modal_flags):
    by_problem = defaultdict(list)
    order = []
    for index, pid in enumerate(pids):
        if pid not in by_problem:
            order.append(pid)
        by_problem[pid].append(index)
    selected, modal = [], []
    for pid in order:
        indices = by_problem[pid]
        best = max(indices, key=lambda index: (scores[index], -index))
        modal_index = next(
            (index for index in indices if modal_flags[index]), indices[0]
        )
        selected.append(bool(labels[best]))
        modal.append(bool(labels[modal_index]))
    return order, selected, modal


def expected_calibration_error(probabilities, labels, bins: int = 10) -> float:
    total = len(labels)
    if not total:
        return float("nan")
    error = 0.0
    for bin_index in range(bins):
        low, high = bin_index / bins, (bin_index + 1) / bins
        indices = [
            index for index, value in enumerate(probabilities)
            if low <= value < high or (bin_index == bins - 1 and value == 1.0)
        ]
        if indices:
            confidence = sum(probabilities[index] for index in indices) / len(indices)
            accuracy = sum(labels[index] for index in indices) / len(indices)
            error += len(indices) / total * abs(confidence - accuracy)
    return error


def high_confidence_negative_rate(probabilities, labels, threshold: float = 0.9):
    negatives = [
        probability for probability, label in zip(probabilities, labels) if not label
    ]
    return (
        sum(probability >= threshold for probability in negatives) / len(negatives)
        if negatives else float("nan")
    )


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda name: pvalues[name])
    adjusted, running = {}, 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def bootstrap_delta(a, b, seed: int, samples: int = 10000):
    if len(a) != len(b) or not a:
        raise ValueError("paired bootstrap inputs must be aligned and non-empty")
    rng = random.Random(seed)
    deltas = []
    n = len(a)
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(a[i] - b[i] for i in indices) / n)
    deltas.sort()
    return [deltas[int(0.025 * samples)], deltas[int(0.975 * samples) - 1]]


def make_readers(torch, dimension: int, target_parameters: int = 90000):
    nn = torch.nn
    edge_input = 4 * dimension + 1
    edge_width = max(8, int((target_parameters - dimension - 1) / (edge_input + 2)))

    class SymmetricGraphReader(nn.Module):
        def __init__(self):
            super().__init__()
            self.edge = nn.Sequential(
                nn.Linear(edge_input, edge_width), nn.Tanh(), nn.Linear(edge_width, 1)
            )
            self.local = nn.Linear(dimension, 1)

        def forward(self, p0, p1, root, frequencies):
            def edge(parent, freq):
                values = torch.cat([
                    parent, root, parent * root, torch.abs(parent - root), freq,
                ], dim=1)
                return self.edge(values).squeeze(1)
            return (
                self.local(root).squeeze(1)
                + edge(p0, frequencies[:, 0:1])
                + edge(p1, frequencies[:, 1:2])
            )

    class RootReader(nn.Module):
        def __init__(self):
            super().__init__()
            width = max(8, int(target_parameters / max(1, dimension + 2)))
            self.net = nn.Sequential(
                nn.Linear(dimension, width), nn.Tanh(), nn.Linear(width, 1)
            )

        def forward(self, p0, p1, root, frequencies):
            return self.net(root).squeeze(1)

    flat_input = 3 * dimension + 2

    class FlatReader(nn.Module):
        def __init__(self):
            super().__init__()
            width = max(8, int(target_parameters / max(1, flat_input + 2)))
            self.net = nn.Sequential(
                nn.Linear(flat_input, width), nn.Tanh(), nn.Linear(width, 1)
            )

        def forward(self, p0, p1, root, frequencies):
            return self.net(torch.cat([p0, p1, root, frequencies], dim=1)).squeeze(1)

    return SymmetricGraphReader, RootReader, FlatReader


def tensor_dataset(torch, payload: dict, layer: int, view: str) -> dict:
    parent = payload["parent"]
    root = payload["root"]
    parent_map = {tuple(key): index for index, key in enumerate(parent["keys"])}
    root_map = {tuple(key): index for index, key in enumerate(root["keys"])}

    def selected(section, index):
        last = section["features"][layer]["last"][index].float()
        mean = section["features"][layer]["mean"][index].float()
        if view == "last":
            return last
        if view == "mean":
            return mean
        return torch.cat([last, mean])

    p0_rows, p1_rows, root_rows = [], [], []
    for meta in payload["metas"]:
        pid = int(meta["id"])
        parent_rows = []
        for slot in (0, 1):
            members = [
                selected(parent, parent_map[("parent", pid, slot, int(index))])
                for index in meta["member_indices"][slot]
            ]
            parent_rows.append(torch.stack(members).mean(dim=0))
        p0_rows.append(parent_rows[0])
        p1_rows.append(parent_rows[1])
        root_index = root_map[("root", pid, *meta["norms"])]
        root_rows.append(selected(root, root_index))
    pids = [int(meta["id"]) for meta in payload["metas"]]
    return {
        "p0": torch.stack(p0_rows),
        "p1": torch.stack(p1_rows),
        "root": torch.stack(root_rows),
        "frequencies": torch.tensor(
            [meta["frequencies"] for meta in payload["metas"]], dtype=torch.float32
        ),
        "nonhidden": torch.tensor(
            [nonhidden_row(meta) for meta in payload["metas"]], dtype=torch.float32
        ),
        "labels": torch.tensor(
            [meta["label"] for meta in payload["metas"]], dtype=torch.float32
        ),
        "pids": pids,
        "modal": [bool(meta["is_modal"]) for meta in payload["metas"]],
    }


def mismatch_roots(torch, data: dict):
    pids = sorted(set(data["pids"]))
    donors = {pid: pids[(index + 1) % len(pids)] for index, pid in enumerate(pids)}
    modal_root = {}
    for index, (pid, modal) in enumerate(zip(data["pids"], data["modal"])):
        if modal:
            modal_root[pid] = data["root"][index]
    return torch.stack([modal_root[donors[pid]] for pid in data["pids"]])


def subset(data: dict, indices):
    return {
        key: value[indices]
        for key, value in data.items()
        if key in (
            "p0", "p1", "root", "mismatch_root", "frequencies",
            "nonhidden", "labels",
        )
    } | {
        "pids": [data["pids"][index] for index in indices],
        "modal": [data["modal"][index] for index in indices],
    }


def method_model(torch, method: str, data: dict):
    if method == "nonhidden_bce":
        dimension = data["nonhidden"].shape[1]
        width = 32
        model = torch.nn.Sequential(
            torch.nn.Linear(dimension, width), torch.nn.Tanh(), torch.nn.Linear(width, 1)
        )
        return model.to(data["nonhidden"].device), "nonhidden"
    dimension = data["root"].shape[1]
    Graph, Root, Flat = make_readers(torch, dimension)
    if method in ("root_dual_bce", "root_dual_energy"):
        return Root().to(data["root"].device), "structured"
    if method == "flat_dual_bce":
        return Flat().to(data["root"].device), "structured"
    return Graph().to(data["root"].device), "structured"


def forward(torch, model, kind: str, data: dict, method: str):
    if kind == "nonhidden":
        return model(data["nonhidden"]).squeeze(1)
    root = data["root"]
    if method == "graph_mismatch_energy":
        root = data["mismatch_root"]
    return model(data["p0"], data["p1"], root, data["frequencies"])


def objective(torch, method, logits, data, hyper):
    weights = equal_graph_weights(torch, data["labels"], data["pids"])
    if method == "graph_dual_aeo":
        return aeo_asymmetric_loss(
            torch, logits, data["labels"], hyper, weights=weights
        )
    if method.endswith("energy"):
        return pairwise_energy_loss(
            torch, logits, data["labels"], data["pids"], hyper
        )
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, data["labels"], reduction="none"
    )
    return (losses * weights).sum()


def fit_once(torch, method, train, validation, hyper, seed, epochs=160):
    torch.manual_seed(seed)
    model, kind = method_model(torch, method, train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-3)
    best = (-1.0, -1.0, 0, None)
    patience = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = forward(torch, model, kind, train, method)
        loss = objective(torch, method, logits, train, hyper)
        loss.backward()
        optimizer.step()
        if epoch % 10:
            continue
        model.eval()
        with torch.no_grad():
            scores = forward(torch, model, kind, validation, method).tolist()
        _, outcomes, _ = selection_outcomes(
            scores, validation["labels"].tolist(), validation["pids"],
            validation["modal"],
        )
        accuracy = sum(outcomes) / max(1, len(outcomes))
        ranking, _ = within_problem_auroc(
            scores, validation["labels"].tolist(), validation["pids"]
        )
        ranking = ranking if ranking == ranking else -1.0
        if (accuracy, ranking) > best[:2]:
            best = (
                accuracy, ranking, epoch,
                {key: value.detach().clone() for key, value in model.state_dict().items()},
            )
            patience = 0
        else:
            patience += 10
        if patience >= 80:
            break
    if best[3] is None:
        raise RuntimeError(f"no valid checkpoint for {method}")
    model.load_state_dict(best[3])
    return model, kind, {"accuracy": best[0], "ranking": best[1], "epoch": best[2]}


def refit(torch, method, train, hyper, seed, epochs):
    torch.manual_seed(seed)
    model, kind = method_model(torch, method, train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-3)
    for _ in range(max(10, epochs)):
        model.train()
        optimizer.zero_grad()
        logits = forward(torch, model, kind, train, method)
        loss = objective(torch, method, logits, train, hyper)
        loss.backward()
        optimizer.step()
    return model.eval(), kind


def hyper_grid(method: str):
    if method == "graph_dual_aeo":
        return (0.1, 0.3, 1.0)
    if method.endswith("energy"):
        return (0.0, 0.5, 1.0)
    return (0.0,)


def view_for_method(method: str) -> str:
    if method == "graph_last_energy":
        return "last"
    if method == "graph_mean_energy":
        return "mean"
    return "dual"


def nested_oof(torch, payload: dict, method: str, seed: int):
    all_scores = [None] * len(payload["metas"])
    fold_records = []
    invariance_max = 0.0
    for outer in range(5):
        hold_pids = {
            int(meta["id"]) for meta in payload["metas"]
            if stable_bucket(int(meta["id"]), 5, "outer") == outer
        }
        outer_test = [
            index for index, meta in enumerate(payload["metas"])
            if int(meta["id"]) in hold_pids
        ]
        outer_train = [
            index for index, meta in enumerate(payload["metas"])
            if int(meta["id"]) not in hold_pids
        ]
        train_pids = sorted({int(payload["metas"][index]["id"]) for index in outer_train})
        validation_pids = {
            pid for pid in train_pids
            if stable_bucket(pid, 5, f"inner-{outer}") == 0
        }
        inner_train_idx = [
            index for index in outer_train
            if int(payload["metas"][index]["id"]) not in validation_pids
        ]
        inner_val_idx = [
            index for index in outer_train
            if int(payload["metas"][index]["id"]) in validation_pids
        ]
        choices = []
        candidate_layers = (LAYERS[0],) if method == "nonhidden_bce" else LAYERS
        for layer in candidate_layers:
            data = tensor_dataset(torch, payload, layer, view_for_method(method))
            data = {
                key: value.to(args_device(torch)) if hasattr(value, "to") else value
                for key, value in data.items()
            }
            data["mismatch_root"] = mismatch_roots(torch, data)
            for hyper in hyper_grid(method):
                model, kind, validation = fit_once(
                    torch, method, subset(data, inner_train_idx),
                    subset(data, inner_val_idx), hyper,
                    seed + 1000 * outer + 17 * layer + int(100 * hyper),
                )
                choices.append((
                    validation["accuracy"], validation["ranking"], -layer, -hyper,
                    layer, hyper, validation["epoch"],
                ))
        best = max(choices)
        layer, hyper, epochs = best[4], best[5], best[6]
        data = tensor_dataset(torch, payload, layer, view_for_method(method))
        data = {
            key: value.to(args_device(torch)) if hasattr(value, "to") else value
            for key, value in data.items()
        }
        data["mismatch_root"] = mismatch_roots(torch, data)
        train = subset(data, outer_train)
        test = subset(data, outer_test)
        model, kind = refit(
            torch, method, train, hyper, seed + 7000 + outer, epochs
        )
        with torch.no_grad():
            scores = forward(torch, model, kind, test, method)
            if method.startswith("graph_") and method != "graph_mismatch_energy":
                swapped = dict(test)
                swapped["p0"], swapped["p1"] = test["p1"], test["p0"]
                swapped["frequencies"] = test["frequencies"][:, [1, 0]]
                difference = torch.max(torch.abs(
                    scores - forward(torch, model, kind, swapped, method)
                )).item()
                invariance_max = max(invariance_max, difference)
        for index, score in zip(outer_test, scores.tolist()):
            all_scores[index] = score
        _, outcomes, modal = selection_outcomes(
            scores.tolist(), test["labels"].tolist(), test["pids"], test["modal"]
        )
        fold_records.append({
            "fold": outer,
            "n_graphs": len(outcomes),
            "layer": layer,
            "hyper": hyper,
            "epochs": epochs,
            "accuracy": sum(outcomes) / len(outcomes),
            "modal_accuracy": sum(modal) / len(modal),
            "delta": sum(outcomes) / len(outcomes) - sum(modal) / len(modal),
        })
    if any(score is None for score in all_scores):
        raise RuntimeError(f"incomplete OOF scores for {method}")
    return all_scores, fold_records, invariance_max


def report(args) -> None:
    import torch

    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    labels = [int(meta["label"]) for meta in payload["metas"]]
    pids = [int(meta["id"]) for meta in payload["metas"]]
    modal_flags = [bool(meta["is_modal"]) for meta in payload["metas"]]
    method_reports = {}
    pid_order = None
    modal_outcomes = None
    for method in METHODS:
        print(f"[OOF] method={method}", flush=True)
        scores, folds, invariance = nested_oof(torch, payload, method, args.seed)
        order, outcomes, modal = selection_outcomes(scores, labels, pids, modal_flags)
        if pid_order is None:
            pid_order, modal_outcomes = order, modal
        if order != pid_order or modal != modal_outcomes:
            raise RuntimeError("method outcome order mismatch")
        probabilities = [1.0 / (1.0 + math.exp(-max(-40, min(40, s)))) for s in scores]
        within, n_within = within_problem_auroc(scores, labels, pids)
        method_reports[method] = {
            "accuracy": sum(outcomes) / len(outcomes),
            "delta_vs_modal": sum(outcomes) / len(outcomes) - sum(modal) / len(modal),
            "outcomes": outcomes,
            "folds": folds,
            "assignment_auroc": auroc(scores, labels),
            "within_graph_auroc": within,
            "n_within_graphs": n_within,
            "ece": expected_calibration_error(probabilities, labels),
            "high_confidence_negative_rate": high_confidence_negative_rate(
                probabilities, labels
            ),
            "sibling_swap_max_abs": invariance,
        }
    modal_accuracy = sum(modal_outcomes) / len(modal_outcomes)
    energy = method_reports["graph_dual_energy"]
    primary_names = ("modal", "flat_dual_bce", "graph_dual_bce")
    primary_outcomes = {
        "modal": modal_outcomes,
        "flat_dual_bce": method_reports["flat_dual_bce"]["outcomes"],
        "graph_dual_bce": method_reports["graph_dual_bce"]["outcomes"],
    }
    paired = {
        name: exact_mcnemar(energy["outcomes"], outcomes)
        for name, outcomes in primary_outcomes.items()
    }
    adjusted = holm_adjust({name: result["p"] for name, result in paired.items()})
    for name in primary_names:
        paired[name]["holm_p"] = adjusted[name]
        paired[name]["bootstrap_95"] = bootstrap_delta(
            energy["outcomes"], primary_outcomes[name], args.seed + len(name)
        )
    hash_halves = []
    for half in (0, 1):
        indices = [
            index for index, pid in enumerate(pid_order)
            if stable_bucket(pid, 2, "half") == half
        ]
        delta = (
            sum(energy["outcomes"][index] - modal_outcomes[index] for index in indices)
            / len(indices)
        )
        hash_halves.append(delta)
    aeo = method_reports["graph_dual_aeo"]
    bce = method_reports["graph_dual_bce"]
    graph_gates = {
        "energy_plus_3pp_modal": energy["accuracy"] - modal_accuracy >= 0.03,
        "energy_modal_holm_p_lt_005": paired["modal"]["holm_p"] < 0.05,
        "energy_plus_1pp_nonhidden": (
            energy["accuracy"] - method_reports["nonhidden_bce"]["accuracy"] >= 0.01
        ),
        "energy_plus_1pp_flat": (
            energy["accuracy"] - method_reports["flat_dual_bce"]["accuracy"] >= 0.01
        ),
        "energy_plus_1pp_graph_bce": energy["accuracy"] - bce["accuracy"] >= 0.01,
        "energy_plus_1pp_root_only": (
            energy["accuracy"] - method_reports["root_dual_energy"]["accuracy"] >= 0.01
        ),
        "energy_plus_2pp_mismatch": (
            energy["accuracy"]
            - method_reports["graph_mismatch_energy"]["accuracy"] >= 0.02
        ),
        "positive_four_of_five_folds": sum(
            fold["delta"] > 0 for fold in energy["folds"]
        ) >= 4,
        "positive_both_hash_halves": all(delta > 0 for delta in hash_halves),
        "sibling_invariance": energy["sibling_swap_max_abs"] < 1e-6,
    }
    bce_hcn = bce["high_confidence_negative_rate"]
    aeo_gates = {
        "accuracy_plus_1pp_bce": aeo["accuracy"] - bce["accuracy"] >= 0.01,
        "ece_not_worse": aeo["ece"] <= bce["ece"],
        "high_conf_negative_relative_minus_20pct": (
            aeo["high_confidence_negative_rate"] <= 0.8 * bce_hcn
            if bce_hcn > 0 else aeo["high_confidence_negative_rate"] == 0
        ),
    }
    dual_gates = {
        "plus_1pp_last": (
            energy["accuracy"] - method_reports["graph_last_energy"]["accuracy"] >= 0.01
        ),
        "plus_1pp_mean": (
            energy["accuracy"] - method_reports["graph_mean_energy"]["accuracy"] >= 0.01
        ),
    }
    output = {
        "experiment": "hidden_graph_energy_guide_v0_development",
        "features": os.path.abspath(args.features),
        "n_graphs": len(pid_order),
        "n_assignments": len(labels),
        "modal_accuracy": modal_accuracy,
        "methods": method_reports,
        "primary_paired": paired,
        "hash_half_deltas": hash_halves,
        "energy_gates": graph_gates,
        "energy_pass": all(graph_gates.values()),
        "aeo_gates": aeo_gates,
        "aeo_pass": all(aeo_gates.values()),
        "dual_view_gates": dual_gates,
        "dual_view_pass": all(dual_gates.values()),
    }
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=1)
    print(json.dumps({
        "modal_accuracy": modal_accuracy,
        "method_accuracy": {
            name: values["accuracy"] for name, values in method_reports.items()
        },
        "energy_gates": graph_gates,
        "aeo_gates": aeo_gates,
        "dual_view_gates": dual_gates,
    }, indent=1), flush=True)
    print(f"ENERGY_OVERALL={'PASS' if output['energy_pass'] else 'FAIL'}")


def self_test() -> None:
    assert stable_bucket(10, 5, "outer") == stable_bucket(10, 5, "outer")
    assert simple_numeric("1,000") > simple_numeric("10") > 0
    assert nonhidden_row({
        "bindings": ["10", "20"], "frequencies": [0.5, 0.25],
        "counts": [2, 1], "is_modal": False,
    })[-1] == 0.0
    _, chosen, modal = selection_outcomes(
        [0.1, 0.9, 0.5], [0, 1, 0], [1, 1, 2], [True, False, True]
    )
    assert chosen == [True, False] and modal == [False, False]
    assert expected_calibration_error([0.1, 0.9], [0, 1]) < 0.11
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.2})
    assert adjusted["a"] == 0.03 and adjusted["b"] == 0.08
    print("SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features")
    parser.add_argument("--report", default="hidden_graph_energy_oof_report.json")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.features:
        raise SystemExit("--features is required")
    report(args)


if __name__ == "__main__":
    main()
