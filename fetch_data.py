"""Download small evaluation datasets into ./data (cluster-friendly).

Prefer raw HTTP / huggingface_hub over `datasets` so the scratch venv does
not need extra packages. Safe to re-run; skips files that already exist
unless --force.

Usage:
  python fetch_data.py --which socratic,musique
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# Official OpenAI grade-school-math socratic split (same questions as main).
SOCRATIC_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test_socratic.jsonl"
)

# MuSiQue answerable validation via HF (voidful mirror; includes decomposition).
MUSIQUE_HF = "voidful/MuSiQue"


def _download(url, dest):
    print(f"  GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "dch-hsgr/fetch"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    print(f"  wrote {dest} ({os.path.getsize(dest)} bytes)")


def fetch_socratic(force=False):
    os.makedirs(DATA, exist_ok=True)
    dest = os.path.join(DATA, "gsm8k_socratic_test.jsonl")
    if os.path.exists(dest) and not force:
        print(f"  skip {dest} (exists)")
        return dest
    tmp = dest + ".tmp"
    _download(SOCRATIC_URL, tmp)
    # Normalize field names to match our gsm8k_* files (question/answer).
    n = 0
    with open(tmp, encoding="utf-8") as fin, open(dest, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            # upstream uses {"question","answer"} already
            if "question" not in r and "problem" in r:
                r["question"] = r["problem"]
            fout.write(json.dumps({
                "question": r["question"],
                "answer": r["answer"],
            }, ensure_ascii=False) + "\n")
            n += 1
    os.remove(tmp)
    print(f"  normalized {n} socratic rows -> {dest}")
    return dest


def fetch_musique(force=False, limit=0):
    """Write a compact jsonl with gold decomposition for screening."""
    os.makedirs(DATA, exist_ok=True)
    dest = os.path.join(DATA, "musique_ans_val.jsonl")
    if os.path.exists(dest) and not force:
        print(f"  skip {dest} (exists)")
        return dest
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub required for MuSiQue; install in venv"
        ) from e
    # Prefer parquet via datasets if present; else try json files in the repo.
    path = None
    for cand in (
        "data/validation-00000-of-00001.parquet",
        "musique_ans_v1.0_dev.jsonl",
        "data/musique_ans_v1.0_dev.jsonl",
    ):
        try:
            path = hf_hub_download(MUSIQUE_HF, cand, repo_type="dataset")
            print(f"  hub file: {cand} -> {path}")
            break
        except Exception as ex:
            print(f"  no {cand}: {ex}")
    if path is None:
        # Fall back to datasets.load_dataset
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise SystemExit(
                "Could not download MuSiQue; install `datasets` or check hub"
            ) from e
        print(f"  load_dataset({MUSIQUE_HF})")
        ds = load_dataset(MUSIQUE_HF, split="validation")
        rows = []
        for i, r in enumerate(ds):
            if not r.get("answerable", True):
                continue
            rows.append({
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "answer_aliases": list(r.get("answer_aliases") or []),
                "question_decomposition": list(r["question_decomposition"]),
                "paragraphs": list(r.get("paragraphs") or []),
                "n_paragraphs": len(r.get("paragraphs") or []),
            })
            if limit and len(rows) >= limit:
                break
        with open(dest, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {dest} ({len(rows)} rows)")
        return dest

    if path.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError:
            try:
                from datasets import Dataset
                table = Dataset.from_parquet(path)
                records = list(table)
            except Exception as e:
                raise SystemExit(f"need pyarrow or datasets to read parquet: {e}") from e
        else:
            records = pq.read_table(path).to_pylist()
    else:
        with open(path, encoding="utf-8") as f:
            records = [json.loads(l) for l in f if l.strip()]

    rows = []
    for r in records:
        if r.get("answerable") is False:
            continue
        decomp = r.get("question_decomposition") or []
        # Keep supporting paragraphs only (full corpus is large / unused for
        # the open-book ceiling which uses gold supports).
        paras = r.get("paragraphs") or []
        support_idxs = {
            step.get("paragraph_support_idx")
            for step in decomp
            if step.get("paragraph_support_idx") is not None
        }
        slim_paras = []
        for i, p in enumerate(paras):
            idx = p.get("idx", i) if isinstance(p, dict) else i
            if idx in support_idxs or (isinstance(p, dict) and p.get("is_supporting")):
                slim_paras.append(p if isinstance(p, dict) else {
                    "idx": i, "paragraph_text": str(p), "title": ""
                })
        rows.append({
            "id": r.get("id"),
            "question": r["question"],
            "answer": r.get("answer") or (r.get("golden_answers") or [None])[0],
            "answer_aliases": list(r.get("answer_aliases") or r.get("golden_answers") or []),
            "question_decomposition": decomp,
            "paragraphs": slim_paras,
            "n_paragraphs": len(paras),
        })
        if limit and len(rows) >= limit:
            break
    with open(dest, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {dest} ({len(rows)} rows)")
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="socratic",
                    help="comma list: socratic,musique")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    for w in [x.strip() for x in a.which.split(",") if x.strip()]:
        print(f"=== fetch {w} ===")
        if w == "socratic":
            fetch_socratic(force=a.force)
        elif w == "musique":
            fetch_musique(force=a.force, limit=a.limit)
        else:
            raise SystemExit(f"unknown dataset: {w}")


if __name__ == "__main__":
    main()
