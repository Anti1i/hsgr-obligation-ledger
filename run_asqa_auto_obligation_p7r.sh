#!/usr/bin/env bash
set -euo pipefail

SCR="${SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
ALCE="${ASQA_ALCE:-$SCR/data/asqa_eval_gtr_top100_reranked_oracle.json}"
ORIGINAL="${ASQA_ORIGINAL:-$SCR/data/ASQA.json}"
P1X="${ASQA_P1X_GENERATIONS:-$SCR/results/asqa_clean_fixed_support_p1x_733287/asqa_clean_fixed_support_p1x_generations.jsonl}"
P7BASE="${ASQA_P7X_BASE:-$SCR/results/asqa_auto_obligation_p7x_749083}"
OUT_DIR="${ASQA_P7R_OUT:-$SCR/results/asqa_auto_obligation_p7r_${SLURM_JOB_ID:-manual}}"

for path in "$ALCE" "$ORIGINAL" "$P1X" "$P7BASE/asqa_auto_obligation_p7x_inductions.jsonl" "$P7BASE/asqa_auto_obligation_p7x_candidates.jsonl" "$P7BASE/asqa_auto_obligation_p7x_selections.jsonl"; do
  [ -s "$path" ] || { echo "FATAL: missing $path" >&2; exit 1; }
done

"${PY:-python}" -X utf8 -m unittest -v test_asqa_auto_obligation_p7r.py
"${PY:-python}" -X utf8 asqa_auto_obligation_p7r.py \
  --alce "$ALCE" \
  --original "$ORIGINAL" \
  --p1x-generations "$P1X" \
  --p7x-inductions "$P7BASE/asqa_auto_obligation_p7x_inductions.jsonl" \
  --p7x-candidates "$P7BASE/asqa_auto_obligation_p7x_candidates.jsonl" \
  --p7x-selections "$P7BASE/asqa_auto_obligation_p7x_selections.jsonl" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --selector-batch-size 16 \
  --generation-batch-size 8 \
  --out-dir "$OUT_DIR"

sha256sum "$OUT_DIR/asqa_auto_obligation_p7r_report.json" "$OUT_DIR/asqa_auto_obligation_p7r_new_candidates.jsonl" "$OUT_DIR/asqa_auto_obligation_p7r_selections.jsonl"
echo "result_dir=$OUT_DIR"
