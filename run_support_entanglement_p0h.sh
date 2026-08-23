#!/usr/bin/env bash
set -euo pipefail

PROJECT_SCR="${HSGR_PROJECT_SCR:-/mnt/scratch/z/${USER}/hsgr-obligation-ledger}"
RUNTIME_SCR="${HSGR_RUNTIME_SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
PY="${HSGR_PYTHON:-$RUNTIME_SCR/venv/bin/python}"
OUT_DIR="${SUPPORT_P0H_OUT:-$PROJECT_SCR/results/support_entanglement_p0h_${SLURM_JOB_ID:-manual}}"

[ -x "$PY" ] || { echo "FATAL: no project Python at $PY" >&2; exit 1; }
[ -d "$RUNTIME_SCR/hf/hub/models--Qwen--Qwen2.5-7B-Instruct" ] || {
  echo "FATAL: cached generator missing" >&2; exit 2;
}
[ -d "$RUNTIME_SCR/hf/hub/models--Qwen--Qwen2.5-14B-Instruct" ] || {
  echo "FATAL: cached judge missing" >&2; exit 3;
}
mkdir -p "$OUT_DIR" "$PROJECT_SCR/tmp"
export HF_HOME="${HSGR_HF_HOME:-$RUNTIME_SCR/hf}"
export TORCH_HOME="${HSGR_TORCH_HOME:-$RUNTIME_SCR/torch}"
export XDG_CACHE_HOME="${HSGR_XDG_CACHE_HOME:-$RUNTIME_SCR/xdg}"
export TMPDIR="$PROJECT_SCR/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

"$PY" -X utf8 -m unittest -v test_support_entanglement_p0h.py
"$PY" -X utf8 support_entanglement_p0h.py \
  --generator-model Qwen/Qwen2.5-7B-Instruct --generation-batch-size 4 --generation-cap 384 \
  --judge-model Qwen/Qwen2.5-14B-Instruct --judge-batch-size 4 --judge-cap 320 \
  --out-dir "$OUT_DIR"
sha256sum "$OUT_DIR/support_entanglement_p0h_report.json" \
  "$OUT_DIR/support_entanglement_p0h_candidates.jsonl" \
  "$OUT_DIR/support_entanglement_p0h_controls.jsonl" \
  "$OUT_DIR/support_entanglement_p0h_review.jsonl"
