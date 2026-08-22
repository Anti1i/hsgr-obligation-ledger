#!/usr/bin/env bash
set -euo pipefail

SCR="${SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
ALCE="${ASQA_ALCE:-$SCR/data/asqa_eval_gtr_top100_reranked_oracle.json}"
ORIGINAL="${ASQA_ORIGINAL:-$SCR/data/ASQA.json}"
P1X="${ASQA_P1X_GENERATIONS:-$SCR/results/asqa_clean_fixed_support_p1x_733287/asqa_clean_fixed_support_p1x_generations.jsonl}"
P3X="${ASQA_P3X_GENERATIONS:-$SCR/results/asqa_single_node_intervention_p3x_733401/asqa_single_node_intervention_p3x_generations.jsonl}"
P6X="${ASQA_P6X_CANDIDATES:-$SCR/results/asqa_missing_selector_p6x_749008/asqa_missing_selector_p6x_candidates.jsonl}"
OUT_DIR="${ASQA_P6R_OUT:-$SCR/results/asqa_hidden_probe_convergence_p6r_${SLURM_JOB_ID:-manual}}"

for path in "$ALCE" "$ORIGINAL" "$P1X" "$P3X" "$P6X"; do
  [ -s "$path" ] || { echo "FATAL: missing $path" >&2; exit 1; }
done

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "${PY:-python}" -X utf8 -m unittest -v test_asqa_hidden_probe_convergence_p6r.py
OPENBLAS_NUM_THREADS=12 OMP_NUM_THREADS=12 "${PY:-python}" -X utf8 asqa_hidden_probe_convergence_p6r.py \
  --alce "$ALCE" \
  --original "$ORIGINAL" \
  --p1x-generations "$P1X" \
  --p3x-generations "$P3X" \
  --p6x-candidates "$P6X" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --batch-size 16 \
  --out-dir "$OUT_DIR"

sha256sum "$OUT_DIR/asqa_hidden_probe_convergence_p6r_report.json" "$OUT_DIR/asqa_hidden_probe_convergence_p6r_scores.jsonl"
echo "result_dir=$OUT_DIR"
