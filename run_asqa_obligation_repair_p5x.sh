#!/usr/bin/env bash
set -euo pipefail

SCR="${SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
ALCE="${ASQA_ALCE:-$SCR/data/asqa_eval_gtr_top100_reranked_oracle.json}"
ORIGINAL="${ASQA_ORIGINAL:-$SCR/data/ASQA.json}"
P3X="${ASQA_P3X_GENERATIONS:-$SCR/results/asqa_single_node_intervention_p3x_733401/asqa_single_node_intervention_p3x_generations.jsonl}"
OUT_DIR="${ASQA_P5X_OUT:-$SCR/results/asqa_obligation_repair_p5x_${SLURM_JOB_ID:-manual}}"

[ -s "$ALCE" ] || { echo "FATAL: missing $ALCE" >&2; exit 1; }
[ -s "$ORIGINAL" ] || { echo "FATAL: missing $ORIGINAL" >&2; exit 1; }
[ -s "$P3X" ] || { echo "FATAL: missing $P3X" >&2; exit 1; }

"${PY:-python}" -X utf8 -m unittest -v \
  test_asqa_fixed_support_audit.py \
  test_asqa_clean_fixed_support_p1x.py \
  test_asqa_single_node_intervention_p3x.py \
  test_asqa_obligation_repair_p5x.py

"${PY:-python}" -X utf8 asqa_obligation_repair_p5x.py \
  --alce "$ALCE" \
  --original "$ORIGINAL" \
  --p3x-generations "$P3X" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --batch-size 8 \
  --out-dir "$OUT_DIR"

sha256sum \
  "$OUT_DIR/asqa_obligation_repair_p5x_report.json" \
  "$OUT_DIR/asqa_obligation_repair_p5x_generations.jsonl" \
  "$OUT_DIR/repair_ids.txt"
echo "result_dir=$OUT_DIR"
