#!/usr/bin/env bash
set -euo pipefail

SCR="${SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
ALCE="${ASQA_ALCE:-$SCR/data/asqa_eval_gtr_top100_reranked_oracle.json}"
ORIGINAL="${ASQA_ORIGINAL:-$SCR/data/ASQA.json}"
P1X="${ASQA_P1X_GENERATIONS:-$SCR/results/asqa_clean_fixed_support_p1x_733287/asqa_clean_fixed_support_p1x_generations.jsonl}"
P7R="${ASQA_P7R_SELECTIONS:-$SCR/results/asqa_auto_obligation_p7r_749095/asqa_auto_obligation_p7r_selections.jsonl}"
OUT_DIR="${ASQA_P8X_OUT:-$SCR/results/asqa_answer_bearing_obligation_p8x_${SLURM_JOB_ID:-manual}}"
for path in "$ALCE" "$ORIGINAL" "$P1X" "$P7R"; do [ -s "$path" ] || { echo "FATAL: missing $path" >&2; exit 1; }; done
"${PY:-python}" -X utf8 -m unittest -v test_asqa_answer_bearing_obligation_p8x.py
"${PY:-python}" -X utf8 asqa_answer_bearing_obligation_p8x.py \
  --alce "$ALCE" --original "$ORIGINAL" --p1x-generations "$P1X" \
  --p7r-selections "$P7R" --model Qwen/Qwen2.5-7B-Instruct \
  --induction-batch-size 4 --selector-batch-size 16 --out-dir "$OUT_DIR"
sha256sum "$OUT_DIR/asqa_answer_bearing_obligation_p8x_report.json" "$OUT_DIR/asqa_answer_bearing_obligation_p8x_inductions.jsonl" "$OUT_DIR/asqa_answer_bearing_obligation_p8x_candidates.jsonl" "$OUT_DIR/asqa_answer_bearing_obligation_p8x_selections.jsonl"
echo "result_dir=$OUT_DIR"
