"""Print raw decomposition outputs, to tell prompt failures apart from model
capability failures when `usable` is unexpectedly low."""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", required=True)
ap.add_argument("--n", type=int, default=3)
a = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
rows = []
for fn in sorted(os.listdir(os.path.join(here, a.out_dir))):
    if fn.startswith("decompose.") and fn.endswith(".jsonl"):
        with open(os.path.join(here, a.out_dir, fn), encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]

ok = sum(1 for r in rows if r.get("subquestions"))
print(f"{len(rows)} rows, {ok} parsed with subquestions")
for r in rows[: a.n]:
    print(f"\n===== id={r['id']} parsed={bool(r.get('subquestions'))} =====")
    print("PROBLEM:", (r.get("problem") or "")[:300].replace("\n", " "))
    print("RAW    :", (r.get("raw") or "")[:600])
