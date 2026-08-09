"""Construct label-guaranteed hard negatives for scorer training by
programmatically corrupting verified-correct assignments.

Negative types (mirroring the proposal's hard-negative taxonomy):
  wrong_value    : one sub answer numerically perturbed, final kept
                   -> final no longer follows (incompatible edge)
  wrong_final    : subs kept, final answer perturbed
  sibling_swap   : answers of two subquestions exchanged (wrong slot)

Writes outputs_gsm_train/synthetic_negatives.jsonl with the same fields
train_scorer.build_examples needs.
"""
import argparse
import json
import os
import random
import re
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_scorer import jread_glob  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_ap = argparse.ArgumentParser()
_ap.add_argument("--out-dir", default="outputs_gsm_train")
_args = _ap.parse_args()
OUT = os.path.join(HERE, _args.out_dir)


def perturb_number(s, rng):
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        val = Fraction(m.group(0))
    except ValueError:
        return None
    ops = [
        lambda v: v + 1, lambda v: v - 1, lambda v: v * 2,
        lambda v: v / 2, lambda v: v * 10, lambda v: v + 10,
    ]
    for _ in range(6):
        nv = rng.choice(ops)(val)
        if nv != val:
            break
    nv_str = str(nv) if nv.denominator != 1 else str(nv.numerator)
    return s[: m.start()] + nv_str + s[m.end():]


def main():
    dec = {r["id"]: r for r in jread_glob(os.path.join(OUT, "decompose.s*.jsonl"))}
    rows = jread_glob(os.path.join(OUT, "aggregate.s*.jsonl"))
    positives = [r for r in rows if r["label"] == 1 and r["final_ans"]]
    out_rows = []
    for r in positives:
        rng = random.Random(r["id"] * 1000 + r["assign_idx"])
        subs = dec[r["id"]]["subquestions"]

        # wrong_value: corrupt one sub answer, keep final
        si = rng.randrange(len(r["sub_answers"]))
        bad = perturb_number(r["sub_answers"][si], rng)
        if bad is not None:
            sa = list(r["sub_answers"])
            sa[si] = bad
            out_rows.append({"id": r["id"], "sub_answers": sa,
                             "final_ans": r["final_ans"], "kind": "wrong_value"})

        # wrong_final: keep subs, corrupt final
        badf = perturb_number(r["final_ans"], rng)
        if badf is not None:
            out_rows.append({"id": r["id"], "sub_answers": list(r["sub_answers"]),
                             "final_ans": badf, "kind": "wrong_final"})

        # sibling_swap: exchange two sub answers (only if distinct and >=2 subs)
        if len(r["sub_answers"]) >= 2:
            i, j = rng.sample(range(len(r["sub_answers"])), 2)
            if r["sub_answers"][i] != r["sub_answers"][j]:
                sa = list(r["sub_answers"])
                sa[i], sa[j] = sa[j], sa[i]
                out_rows.append({"id": r["id"], "sub_answers": sa,
                                 "final_ans": r["final_ans"], "kind": "sibling_swap"})
        _ = subs
    path = os.path.join(OUT, "synthetic_negatives.jsonl")
    with open(path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"positives base: {len(positives)}")
    print(f"synthetic negatives: {len(out_rows)}  {Counter(r['kind'] for r in out_rows)}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
