#!/usr/bin/env bash
set -euo pipefail

PROJECT_SCR="${HSGR_PROJECT_SCR:-/mnt/scratch/z/${USER}/hsgr-obligation-ledger}"
RUNTIME_SCR="${HSGR_RUNTIME_SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
PY="${HSGR_PYTHON:-$RUNTIME_SCR/venv/bin/python}"
OUT_DIR="${WITNESS_P0G_OUT:-$PROJECT_SCR/results/witness_interference_p0g_${SLURM_JOB_ID:-manual}}"

[ -x "$PY" ] || { echo "FATAL: no project Python at $PY" >&2; exit 1; }
for model_dir in \
  "$RUNTIME_SCR/hf/hub/models--Qwen--Qwen2.5-7B-Instruct" \
  "$RUNTIME_SCR/hf/hub/models--Qwen--Qwen2.5-14B-Instruct"; do
  [ -d "$model_dir" ] || { echo "FATAL: cached model missing at $model_dir" >&2; exit 2; }
done
mkdir -p "$OUT_DIR" "$PROJECT_SCR/tmp"
export HF_HOME="${HSGR_HF_HOME:-$RUNTIME_SCR/hf}"
export TORCH_HOME="${HSGR_TORCH_HOME:-$RUNTIME_SCR/torch}"
export XDG_CACHE_HOME="${HSGR_XDG_CACHE_HOME:-$RUNTIME_SCR/xdg}"
export TMPDIR="$PROJECT_SCR/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

"$PY" -X utf8 -m unittest -v test_witness_interference_p0g.py
"$PY" -X utf8 witness_interference_p0g.py \
  --generator-model Qwen/Qwen2.5-7B-Instruct \
  --judge-model Qwen/Qwen2.5-14B-Instruct \
  --generation-batch-size 4 --judge-batch-size 4 \
  --generation-cap 512 --judge-cap 384 --out-dir "$OUT_DIR"
sha256sum "$OUT_DIR/witness_interference_p0g_report.json" \
  "$OUT_DIR/witness_interference_p0g_candidates.jsonl" \
  "$OUT_DIR/witness_interference_p0g_controls.jsonl" \
  "$OUT_DIR/witness_interference_p0g_review.jsonl"

