"""Build the S1 datasets: harder / genuinely compositional problem sets.

V8 showed that on MATH-500 and GSM8K the gold answer is reachable via the
decomposition path but almost never ONLY via that path (3.6% / 0.0%), so the
hierarchy has no room to demonstrate value. S1 needs sets where composition is
forced.

Builders
  math_l5    : MATH-500 restricted to Level 5 (larger oracle gap).
  gsm_deep   : GSM8K problems whose annotated solution has >= MIN_STEPS
               calculator steps. The <<expr=value>> annotations give GOLD
               INTERMEDIATE VALUES, which the current pilot lacks entirely, so
               node-level oracle metrics become measurable.
  gsm_chain  : compositional GSM. One number in problem B is replaced by the
               answer of problem A, and B's gold is recomputed by re-evaluating
               B's annotated arithmetic chain under that substitution. Every
               composed item is validated by an identity-substitution check:
               re-evaluating with the ORIGINAL number must reproduce B's
               original gold, otherwise the item is discarded.

Usage:
  python data_prep.py --which all --data-dir data
"""
import argparse
import itertools
import json
import os
import random
import re
from fractions import Fraction

MIN_STEPS = 3
STEP_RE = re.compile(r"<<([^<>=]+)=([^<>]+)>>")
NUM_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {path}  ({len(rows)} rows)")


# ------------------------------------------------------------------ helpers
def num_str(x):
    """Render a Fraction the way GSM8K annotations do (int if integral)."""
    if x.denominator == 1:
        return str(x.numerator)
    f = float(x)
    s = f"{f:.10g}"
    return s


def safe_eval(expr):
    """Evaluate an arithmetic expression from a GSM8K annotation."""
    e = expr.replace(",", "").replace("x", "*").replace("X", "*")
    e = e.replace("%", "/100").replace("$", "").strip()
    if not re.fullmatch(r"[0-9+\-*/(). ]+", e):
        return None
    try:
        return Fraction(eval(e, {"__builtins__": {}}, {}))  # noqa: S307
    except (ZeroDivisionError, SyntaxError, TypeError, ValueError):
        return None


def parse_steps(answer):
    """[(expr, value)] from annotations, plus the final #### gold."""
    steps = []
    for expr, val in STEP_RE.findall(answer):
        v = safe_eval(val)
        if v is None:
            return None, None
        steps.append((expr.strip(), v))
    m = re.search(r"####\s*(.+)", answer)
    if not m:
        return None, None
    gold = safe_eval(m.group(1).strip())
    return steps, gold


def re_evaluate(steps, subst):
    """Re-run the annotated chain, mapping old literal values to new ones.

    `subst` maps an original literal string to its replacement Fraction. Values
    produced by earlier steps are added to the map as they change, so the whole
    downstream chain follows the substitution.
    """
    mapping = dict(subst)
    last = None
    for expr, old_val in steps:
        e = expr
        # longest first so "100" is not clobbered by "10"
        for old in sorted(mapping, key=len, reverse=True):
            e = e.replace(old, num_str(mapping[old]))
        new_val = safe_eval(e)
        if new_val is None:
            return None
        if new_val != old_val:
            mapping[num_str(old_val)] = new_val
        last = new_val
    return last


# ------------------------------------------------------------------ builders
def build_math_l5(data_dir):
    rows = read_jsonl(os.path.join(data_dir, "math500_test.jsonl"))
    out = [{"problem": r["problem"], "answer": r["answer"], "level": r.get("level"),
            "subject": r.get("subject"), "source": "math_l5"}
           for r in rows if str(r.get("level", "")).endswith("5")]
    write_jsonl(os.path.join(data_dir, "math_l5.jsonl"), out)
    return out


def build_gsm_deep(data_dir, split="test"):
    rows = read_jsonl(os.path.join(data_dir, f"gsm8k_{split}.jsonl"))
    out = []
    for r in rows:
        steps, gold = parse_steps(r["answer"])
        if not steps or gold is None or len(steps) < MIN_STEPS:
            continue
        if re_evaluate(steps, {}) != gold:  # annotations must be self-consistent
            continue
        out.append({"problem": r["question"], "answer": num_str(gold),
                    "n_steps": len(steps),
                    "steps": [{"expr": e, "value": num_str(v)} for e, v in steps],
                    "source": f"gsm_deep_{split}"})
    write_jsonl(os.path.join(data_dir, f"gsm_deep_{split}.jsonl"), out)
    return out


UNITLESS_RE = re.compile(r"percent|percentage|%|what fraction|what ratio", re.I)


def substitutable_numbers(B):
    """Numbers in B's question that drive its arithmetic and can be swapped
    without making the wording absurd.

    Requirements: nonzero, not the literal 1, appears in an annotated
    expression, not followed by '%', and occurs EXACTLY ONCE in the question
    text (otherwise replacing the first occurrence contradicts the rest).
    """
    out = []
    for m in NUM_RE.finditer(B["q"]):
        n = m.group(0)
        try:
            val = Fraction(n.replace(",", ""))
        except ValueError:
            continue
        if val == 0 or val == 1:
            continue
        if not any(n in e for e, _ in B["steps"]):
            continue
        if B["q"][m.end(): m.end() + 1] == "%":
            continue
        if B["q"].count(n) != 1:
            continue
        out.append((n, val))
    return out


def build_gsm_chain(data_dir, split="test", limit=400, seed=0, max_ratio=3.0):
    """Compose two problems: a number in B is replaced by A's answer.

    Filters keep the composed item both arithmetically valid and semantically
    plausible: the substituted value must stay within `max_ratio` of the original
    magnitude, and the recomputed gold must be a positive integer.
    """
    rows = read_jsonl(os.path.join(data_dir, f"gsm8k_{split}.jsonl"))
    parsed = []
    for r in rows:
        steps, gold = parse_steps(r["answer"])
        if steps and gold is not None and re_evaluate(steps, {}) == gold:
            parsed.append({"q": r["question"], "steps": steps, "gold": gold})
    rng = random.Random(seed)
    order = list(range(len(parsed)))
    rng.shuffle(order)
    out, used_b = [], set()
    for ai in order:
        A = parsed[ai]
        if A["gold"] <= 1 or A["gold"].denominator != 1:
            continue
        # A's answer must be a plain quantity, else it reads absurdly inside B
        if UNITLESS_RE.search(A["q"]):
            continue
        for _ in range(30):
            bi = rng.randrange(len(parsed))
            if bi == ai or bi in used_b:
                continue
            B = parsed[bi]
            cands = [n for n, val in substitutable_numbers(B)
                     if 1.0 / max_ratio <= float(A["gold"]) / float(val) <= max_ratio]
            if not cands:
                continue
            target = max(cands, key=len)
            new_gold = re_evaluate(B["steps"], {target: A["gold"]})
            if (new_gold is None or new_gold == B["gold"] or new_gold <= 0
                    or new_gold.denominator != 1):
                continue
            used_b.add(bi)
            out.append({
                "problem": ("Question 1: " + A["q"].strip() + "\n\n"
                            "Question 2: " + B["q"].strip().replace(
                                target, "(the answer to Question 1)", 1) + "\n\n"
                            "Give the answer to Question 2."),
                "answer": num_str(new_gold),
                "hop1_answer": num_str(A["gold"]),
                "substituted": target,
                "n_steps": len(A["steps"]) + len(B["steps"]),
                "source": f"gsm_chain_{split}"})
            break
        if len(out) >= limit:
            break
    write_jsonl(os.path.join(data_dir, f"gsm_chain_{split}.jsonl"), out)
    return out


def build_gsm_join(data_dir, split="test", limit=400, seed=17, max_ratio=3.0):
    """Compose a three-node join: two independent parents feed one root.

    Two distinct numeric literals in problem B are replaced by the answers to
    independent problems A and C.  Each replacement must change B's symbolic
    result on its own, so both incoming edges have a verified causal effect.
    The A/C order and their assignment to B's two literals are randomized.

    This set is for dependency-source diagnostics: a wrong root can originate
    on parent edge 1, parent edge 2, both edges, or the local root computation.
    """
    rows = read_jsonl(os.path.join(data_dir, f"gsm8k_{split}.jsonl"))
    parsed = []
    for r in rows:
        steps, gold = parse_steps(r["answer"])
        if steps and gold is not None and re_evaluate(steps, {}) == gold:
            parsed.append({"q": r["question"], "steps": steps, "gold": gold})

    parent_ids = [
        i for i, row in enumerate(parsed)
        if row["gold"] > 1
        and row["gold"].denominator == 1
        and not UNITLESS_RE.search(row["q"])
    ]
    rng = random.Random(seed)
    root_order = list(range(len(parsed)))
    rng.shuffle(root_order)
    out = []

    for bi in root_order:
        B = parsed[bi]
        literals = []
        for text, value in substitutable_numbers(B):
            if all(text != old for old, _ in literals):
                literals.append((text, value))
        pairs = [
            pair for pair in itertools.combinations(literals, 2)
            # Avoid ambiguous textual replacement such as 10 inside 100.
            if pair[0][0] not in pair[1][0] and pair[1][0] not in pair[0][0]
        ]
        rng.shuffle(pairs)
        if not pairs:
            continue

        built = None
        for target_pair in pairs:
            for _ in range(80):
                ai, ci = rng.sample(parent_ids, 2)
                if bi in (ai, ci):
                    continue
                A, C = parsed[ai], parsed[ci]
                parents = [A, C]
                rng.shuffle(parents)  # break source label vs textual position
                replacements = [parents[0]["gold"], parents[1]["gold"]]
                ratios = [
                    float(new) / float(old)
                    for new, (_, old) in zip(replacements, target_pair)
                ]
                if any(not (1.0 / max_ratio <= ratio <= max_ratio) for ratio in ratios):
                    continue
                mapping = {
                    target_pair[0][0]: replacements[0],
                    target_pair[1][0]: replacements[1],
                }
                new_gold = re_evaluate(B["steps"], mapping)
                first_only = re_evaluate(
                    B["steps"], {target_pair[0][0]: replacements[0]}
                )
                second_only = re_evaluate(
                    B["steps"], {target_pair[1][0]: replacements[1]}
                )
                if (
                    new_gold is None
                    or new_gold <= 0
                    or new_gold.denominator != 1
                    or new_gold == B["gold"]
                    or first_only in (None, B["gold"])
                    or second_only in (None, B["gold"])
                ):
                    continue

                q3 = B["q"].strip()
                q3 = q3.replace(
                    target_pair[0][0], "(the answer to Question 1)", 1
                )
                q3 = q3.replace(
                    target_pair[1][0], "(the answer to Question 2)", 1
                )
                if target_pair[0][0] in q3 or target_pair[1][0] in q3:
                    continue
                built = {
                    "problem": (
                        "Question 1: " + parents[0]["q"].strip() + "\n\n"
                        "Question 2: " + parents[1]["q"].strip() + "\n\n"
                        "Question 3: " + q3 + "\n\n"
                        "Give the answer to Question 3."
                    ),
                    "answer": num_str(new_gold),
                    "parent_answers": [
                        num_str(parents[0]["gold"]), num_str(parents[1]["gold"])
                    ],
                    "substituted": [target_pair[0][0], target_pair[1][0]],
                    "single_edge_answers": [
                        num_str(first_only), num_str(second_only)
                    ],
                    "original_root_answer": num_str(B["gold"]),
                    "parent_step_counts": [
                        len(parents[0]["steps"]), len(parents[1]["steps"])
                    ],
                    "root_step_count": len(B["steps"]),
                    "graph": {
                        "nodes": ["parent_0", "parent_1", "root"],
                        "edges": [
                            ["parent_0", "root"], ["parent_1", "root"]
                        ],
                    },
                    "n_steps": (
                        len(parents[0]["steps"])
                        + len(parents[1]["steps"])
                        + len(B["steps"])
                    ),
                    "source": f"gsm_join_{split}",
                }
                break
            if built is not None:
                break
        if built is not None:
            out.append(built)
        if len(out) >= limit:
            break

    write_jsonl(os.path.join(data_dir, f"gsm_join_{split}.jsonl"), out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all",
                    choices=["all", "math_l5", "gsm_deep", "gsm_chain", "gsm_join",
                             "gsm_join_train"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--join-train-limit", type=int, default=400)
    a = ap.parse_args()
    if a.which in ("all", "math_l5"):
        build_math_l5(a.data_dir)
    if a.which in ("all", "gsm_deep"):
        build_gsm_deep(a.data_dir, "test")
        build_gsm_deep(a.data_dir, "train")
    if a.which in ("all", "gsm_chain"):
        build_gsm_chain(a.data_dir, "test")
    if a.which in ("all", "gsm_join"):
        build_gsm_join(a.data_dir, "test")
    if a.which == "gsm_join_train":
        build_gsm_join(
            a.data_dir, "train", limit=a.join_train_limit, seed=29
        )
