#!/usr/bin/env bash
set -euo pipefail

SCR="${SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
ALCE="${ASQA_ALCE:-$SCR/data/asqa_eval_gtr_top100_reranked_oracle.json}"
ORIGINAL="${ASQA_ORIGINAL:-$SCR/data/ASQA.json}"
P1X_GENERATIONS="${ASQA_P1X_GENERATIONS:-$SCR/results/asqa_clean_fixed_support_p1x_733287/asqa_clean_fixed_support_p1x_generations.jsonl}"
OUT_DIR="${ASQA_P4X_OUT:-$SCR/results/asqa_set_guide_patch_p4x_${SLURM_JOB_ID:-manual}}"

[ -s "$ALCE" ] || { echo "FATAL: missing $ALCE" >&2; exit 1; }
[ -s "$ORIGINAL" ] || { echo "FATAL: missing $ORIGINAL" >&2; exit 1; }
[ -s "$P1X_GENERATIONS" ] || { echo "FATAL: missing $P1X_GENERATIONS" >&2; exit 1; }

"${PY:-python}" -X utf8 -m unittest -v \
  test_asqa_fixed_support_audit.py \
  test_asqa_clean_fixed_support_p1x.py \
  test_asqa_set_guide_patch_p4x.py

"${PY:-python}" -X utf8 asqa_set_guide_patch_p4x.py \
  --alce "$ALCE" \
  --original "$ORIGINAL" \
  --p1x-generations "$P1X_GENERATIONS" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --batch-size 4 \
  --source-batch-size 4 \
  --max-new-tokens 192 \
  --out-dir "$OUT_DIR"

sha256sum \
  "$OUT_DIR/asqa_set_guide_patch_p4x_report.json" \
  "$OUT_DIR/asqa_set_guide_patch_p4x_calibration.jsonl" \
  "$OUT_DIR/asqa_set_guide_patch_p4x_heldout.jsonl" \
  "$OUT_DIR/calibration_ids.txt" \
  "$OUT_DIR/heldout_ids.txt"
echo "result_dir=$OUT_DIR"
