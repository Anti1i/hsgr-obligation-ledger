#!/usr/bin/env bash
set -euo pipefail

PROJECT_SCR="${HSGR_PROJECT_SCR:-/mnt/scratch/z/${USER}/hsgr-obligation-ledger}"
RUNTIME_SCR="${HSGR_RUNTIME_SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
PY="${HSGR_PYTHON:-$RUNTIME_SCR/venv/bin/python}"
OUT_DIR="${STALE_P0J_OUT:-$PROJECT_SCR/results/stale_verdict_p0j_${SLURM_JOB_ID:-manual}}"

[ -x "$PY" ] || { echo "FATAL: no project Python at $PY" >&2; exit 1; }
for model in Qwen2.5-7B-Instruct Qwen3-8B Qwen2.5-14B-Instruct; do
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

"$PY" -X utf8 -m unittest -v test_qwen3_non_thinking.py test_stale_verdict_p0j.py
"$PY" -X utf8 stale_verdict_p0j.py \
  --generator-models Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-8B \
  --generation-batch-size 4 --generation-cap 640 \
  --judge-model Qwen/Qwen2.5-14B-Instruct --judge-batch-size 2 --judge-cap 512 \
  --out-dir "$OUT_DIR"
sha256sum "$OUT_DIR/stale_verdict_p0j_report.json" \
  "$OUT_DIR/stale_verdict_p0j_candidates.jsonl" \
  "$OUT_DIR/stale_verdict_p0j_controls.jsonl" \
  "$OUT_DIR/stale_verdict_p0j_transitions.jsonl" \
  "$OUT_DIR/stale_verdict_p0j_review.jsonl"
