#!/usr/bin/env bash
set -euo pipefail

PROJECT_SCR="${HSGR_PROJECT_SCR:-/mnt/scratch/z/${USER}/hsgr-obligation-ledger}"
RUNTIME_SCR="${HSGR_RUNTIME_SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
PY="${HSGR_PYTHON:-$RUNTIME_SCR/venv/bin/python}"
OUT_DIR="${SEMANTIC_P0K_OUT:-$PROJECT_SCR/results/semantic_staleness_p0k_${SLURM_JOB_ID:-manual}}"

[ -x "$PY" ] || { echo "FATAL: no project Python at $PY" >&2; exit 1; }
for model in Qwen3-8B Qwen2.5-14B-Instruct; do
  [ -d "$RUNTIME_SCR/hf/hub/models--Qwen--$model" ] || {
    echo "FATAL: cached model missing: $model" >&2; exit 2;
  }
done
mkdir -p "$OUT_DIR" "$PROJECT_SCR/tmp"
export HF_HOME="${HSGR_HF_HOME:-$RUNTIME_SCR/hf}"
export TORCH_HOME="${HSGR_TORCH_HOME:-$RUNTIME_SCR/torch}"
export XDG_CACHE_HOME="${HSGR_XDG_CACHE_HOME:-$RUNTIME_SCR/xdg}"
export TMPDIR="$PROJECT_SCR/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

"$PY" -X utf8 -m unittest -v test_qwen3_non_thinking.py test_semantic_staleness_p0k.py
"$PY" -X utf8 semantic_staleness_p0k.py \
  --judge-models Qwen/Qwen3-8B Qwen/Qwen2.5-14B-Instruct \
  --judge-batch-size 4 --judge-cap 192 \
  --out-dir "$OUT_DIR"
sha256sum "$OUT_DIR/semantic_staleness_p0k_report.json" \
  "$OUT_DIR/semantic_staleness_p0k_cases.jsonl" \
  "$OUT_DIR/semantic_staleness_p0k_judgments.jsonl" \
  "$OUT_DIR/semantic_staleness_p0k_rows.jsonl" \
  "$OUT_DIR/semantic_staleness_p0k_review.jsonl"
