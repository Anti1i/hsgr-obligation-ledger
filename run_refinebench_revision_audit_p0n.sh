#!/usr/bin/env bash
set -euo pipefail

PROJECT_SCR="${HSGR_PROJECT_SCR:-/mnt/scratch/z/${USER}/hsgr-obligation-ledger}"
RUNTIME_SCR="${HSGR_RUNTIME_SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
PY="${HSGR_PYTHON:-$RUNTIME_SCR/venv/bin/python}"
OUT_DIR="${REFINEBENCH_P0N_OUT:-$PROJECT_SCR/results/refinebench_revision_audit_p0n_${SLURM_JOB_ID:-manual}}"

[ -x "$PY" ] || { echo "FATAL: no project Python at $PY" >&2; exit 1; }
for model in Qwen3-8B Qwen2.5-14B-Instruct; do
  [ -d "$RUNTIME_SCR/hf/hub/models--Qwen--$model" ] || {
    echo "FATAL: cached model missing: $model" >&2; exit 2;
  }
done
mkdir -p "$OUT_DIR" "$PROJECT_SCR/tmp" "$PROJECT_SCR/hf_datasets"
export HF_HOME="${HSGR_HF_HOME:-$RUNTIME_SCR/hf}"
export HF_DATASETS_CACHE="${HSGR_DATASET_CACHE:-$PROJECT_SCR/hf_datasets}"
export TORCH_HOME="${HSGR_TORCH_HOME:-$RUNTIME_SCR/torch}"
export XDG_CACHE_HOME="${HSGR_XDG_CACHE_HOME:-$RUNTIME_SCR/xdg}"
export TMPDIR="$PROJECT_SCR/tmp"
# RefineBench data must be fetched at its frozen revision on the first run.
# Model weights are already present in the project cache and are checked above.
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 TOKENIZERS_PARALLELISM=false

"$PY" -X utf8 -m unittest -v test_qwen3_non_thinking.py test_refinebench_revision_audit_p0n.py
"$PY" -X utf8 refinebench_revision_audit_p0n.py \
  --generator-models Qwen/Qwen3-8B \
  --judge-model Qwen/Qwen2.5-14B-Instruct \
  --n-per-stratum 8 --generation-batch-size 1 --generation-cap 2048 \
  --judge-batch-size 4 --judge-cap 256 \
  --dataset-cache "$HF_DATASETS_CACHE" --out-dir "$OUT_DIR"
sha256sum "$OUT_DIR/p0n_report.json" "$OUT_DIR/p0n_transition_rows.jsonl" \
  "$OUT_DIR/p0n_manual_review.jsonl"
