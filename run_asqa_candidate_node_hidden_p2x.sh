#!/usr/bin/env bash
set -euo pipefail

SCR="${SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
ALCE="${ASQA_ALCE:-$SCR/data/asqa_eval_gtr_top100_reranked_oracle.json}"
ORIGINAL="${ASQA_ORIGINAL:-$SCR/data/ASQA.json}"
P1X_DIR="${ASQA_P1X_DIR:-$SCR/results/asqa_clean_fixed_support_p1x_733287}"
P1X_REPORT="$P1X_DIR/asqa_clean_fixed_support_p1x_report.json"
GENERATIONS="$P1X_DIR/asqa_clean_fixed_support_p1x_generations.jsonl"
OUT_DIR="${ASQA_P2X_OUT:-$SCR/results/asqa_candidate_node_hidden_p2x_${SLURM_JOB_ID:-manual}}"

for path in "$ALCE" "$ORIGINAL" "$P1X_REPORT" "$GENERATIONS"; do
  [ -s "$path" ] || { echo "FATAL: missing $path" >&2; exit 1; }
done

"${PY:-python}" -X utf8 -m unittest -v \
  test_asqa_fixed_support_audit.py \
  test_asqa_clean_fixed_support_p1x.py \
  test_asqa_candidate_node_hidden_p2x.py

"${PY:-python}" -X utf8 asqa_candidate_node_hidden_p2x.py \
  --alce "$ALCE" \
  --original "$ORIGINAL" \
  --p1x-report "$P1X_REPORT" \
  --generations "$GENERATIONS" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --projection-seed 20260815 \
  --node-batch-size 16 \
  --prefix-batch-size 4 \
  --out-dir "$OUT_DIR"

sha256sum \
  "$OUT_DIR/asqa_candidate_node_hidden_p2x_report.json" \
  "$OUT_DIR/asqa_candidate_node_hidden_p2x_scores.jsonl"
echo "result_dir=$OUT_DIR"
