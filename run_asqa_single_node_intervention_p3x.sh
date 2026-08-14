#!/usr/bin/env bash
set -euo pipefail

SCR="${SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
ALCE="${ASQA_ALCE:-$SCR/data/asqa_eval_gtr_top100_reranked_oracle.json}"
ORIGINAL="${ASQA_ORIGINAL:-$SCR/data/ASQA.json}"
OUT_DIR="${ASQA_P3X_OUT:-$SCR/results/asqa_single_node_intervention_p3x_${SLURM_JOB_ID:-manual}}"

[ -s "$ALCE" ] || { echo "FATAL: missing $ALCE" >&2; exit 1; }
[ -s "$ORIGINAL" ] || { echo "FATAL: missing $ORIGINAL" >&2; exit 1; }

"${PY:-python}" -X utf8 -m unittest -v \
  test_asqa_fixed_support_audit.py \
  test_asqa_clean_fixed_support_p1x.py \
  test_asqa_single_node_intervention_p3x.py

"${PY:-python}" -X utf8 asqa_single_node_intervention_p3x.py \
  --alce "$ALCE" \
  --original "$ORIGINAL" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n 192 \
  --batch-size 8 \
  --max-new-tokens 192 \
  --out-dir "$OUT_DIR"

sha256sum \
  "$OUT_DIR/asqa_single_node_intervention_p3x_report.json" \
  "$OUT_DIR/asqa_single_node_intervention_p3x_generations.jsonl" \
  "$OUT_DIR/selected_ids.txt" \
  "$OUT_DIR/decoy_mapping.json"
echo "result_dir=$OUT_DIR"
