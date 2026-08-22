#!/usr/bin/env bash
set -euo pipefail

SCR="${SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
ALCE="${ASQA_ALCE:-$SCR/data/asqa_eval_gtr_top100_reranked_oracle.json}"
ORIGINAL="${ASQA_ORIGINAL:-$SCR/data/ASQA.json}"
P1X="${ASQA_P1X_GENERATIONS:-$SCR/results/asqa_clean_fixed_support_p1x_733287/asqa_clean_fixed_support_p1x_generations.jsonl}"
P3X="${ASQA_P3X_GENERATIONS:-$SCR/results/asqa_single_node_intervention_p3x_733401/asqa_single_node_intervention_p3x_generations.jsonl}"
OUT_DIR="${ASQA_P6X_OUT:-$SCR/results/asqa_missing_selector_p6x_${SLURM_JOB_ID:-manual}}"

for path in "$ALCE" "$ORIGINAL" "$P1X" "$P3X"; do
  [ -s "$path" ] || { echo "FATAL: missing $path" >&2; exit 1; }
done

"${PY:-python}" -X utf8 -m unittest -v \
  test_asqa_fixed_support_audit.py \
  test_asqa_clean_fixed_support_p1x.py \
  test_asqa_single_node_intervention_p3x.py \
  test_asqa_obligation_repair_p5x.py \
  test_asqa_missing_selector_p6x.py

"${PY:-python}" -X utf8 asqa_missing_selector_p6x.py \
  --alce "$ALCE" \
  --original "$ORIGINAL" \
  --p1x-generations "$P1X" \
  --p3x-generations "$P3X" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --selector-batch-size 16 \
  --generation-batch-size 8 \
  --out-dir "$OUT_DIR"

sha256sum \
  "$OUT_DIR/asqa_missing_selector_p6x_report.json" \
  "$OUT_DIR/asqa_missing_selector_p6x_candidates.jsonl" \
  "$OUT_DIR/asqa_missing_selector_p6x_generations.jsonl"
echo "result_dir=$OUT_DIR"
