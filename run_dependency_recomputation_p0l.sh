#!/usr/bin/env bash
set -euo pipefail

PROJECT_SCR="${HSGR_PROJECT_SCR:-/mnt/scratch/z/${USER}/hsgr-obligation-ledger}"
RUNTIME_SCR="${HSGR_RUNTIME_SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
PY="${HSGR_PYTHON:-$RUNTIME_SCR/venv/bin/python}"
OUT_DIR="${RECOMPUTE_P0L_OUT:-$PROJECT_SCR/results/dependency_recomputation_p0l_${SLURM_JOB_ID:-manual}}"

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

"$PY" -X utf8 -m unittest -v test_qwen3_non_thinking.py test_semantic_staleness_p0k.py test_dependency_recomputation_p0l.py
"$PY" -X utf8 dependency_recomputation_p0l.py \
  --models Qwen/Qwen3-8B Qwen/Qwen2.5-14B-Instruct \
  --batch-size 8 --cap 64 --out-dir "$OUT_DIR"
sha256sum "$OUT_DIR/dependency_recomputation_p0l_report.json" \
  "$OUT_DIR/dependency_recomputation_p0l_rows.jsonl" \
  "$OUT_DIR/dependency_recomputation_p0l_review.jsonl"
