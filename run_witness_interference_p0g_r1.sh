#!/usr/bin/env bash
set -euo pipefail

PROJECT_SCR="${HSGR_PROJECT_SCR:-/mnt/scratch/z/${USER}/hsgr-obligation-ledger}"
RUNTIME_SCR="${HSGR_RUNTIME_SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
PY="${HSGR_PYTHON:-$RUNTIME_SCR/venv/bin/python}"
SOURCE_DIR="${WITNESS_P0G_SOURCE:-$PROJECT_SCR/results/witness_interference_p0g_751560}"
OUT_DIR="${WITNESS_P0G_R1_OUT:-$PROJECT_SCR/results/witness_interference_p0g_r1_${SLURM_JOB_ID:-manual}}"

[ -x "$PY" ] || { echo "FATAL: no project Python at $PY" >&2; exit 1; }
[ -s "$SOURCE_DIR/witness_interference_p0g_candidates.jsonl" ] || {
  echo "FATAL: frozen P0g candidates missing" >&2; exit 2;
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

"$PY" -X utf8 -m unittest -v test_witness_interference_p0g.py
"$PY" -X utf8 witness_interference_p0g.py \
  --generator-model Qwen/Qwen2.5-7B-Instruct \
  --judge-model Qwen/Qwen2.5-14B-Instruct --judge-batch-size 4 --judge-cap 384 \
  --reuse-candidates "$SOURCE_DIR/witness_interference_p0g_candidates.jsonl" \
  --out-dir "$OUT_DIR"
sha256sum "$SOURCE_DIR/witness_interference_p0g_candidates.jsonl" \
  "$OUT_DIR/witness_interference_p0g_report.json" \
  "$OUT_DIR/witness_interference_p0g_candidates.jsonl" \
  "$OUT_DIR/witness_interference_p0g_controls.jsonl"

