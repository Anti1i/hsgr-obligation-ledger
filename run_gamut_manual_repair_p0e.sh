#!/usr/bin/env bash
set -euo pipefail

PROJECT_SCR="${HSGR_PROJECT_SCR:-/mnt/scratch/z/${USER}/hsgr-obligation-ledger}"
RUNTIME_SCR="${HSGR_RUNTIME_SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
PY="${HSGR_PYTHON:-$RUNTIME_SCR/venv/bin/python}"
DATA="$PROJECT_SCR/data/gamut_test_text_only.parquet"
P0D_DIR="${GAMUT_P0D_DIR:-$PROJECT_SCR/results/gamut_long_baseline_p0d_751249}"
OUT_DIR="${GAMUT_P0E_OUT:-$PROJECT_SCR/results/gamut_manual_repair_p0e_${SLURM_JOB_ID:-manual}}"

[ -x "$PY" ] || { echo "FATAL: no project Python at $PY" >&2; exit 1; }
[ -s "$DATA" ] || { echo "FATAL: frozen GAMUT parquet missing" >&2; exit 2; }
[ -s "$P0D_DIR/gamut_long_baseline_p0d_rows.jsonl" ] || { echo "FATAL: P0d rows missing" >&2; exit 3; }
for model_dir in \
  "$RUNTIME_SCR/hf/hub/models--Qwen--Qwen2.5-7B-Instruct" \
  "$RUNTIME_SCR/hf/hub/models--Qwen--Qwen2.5-14B-Instruct"; do
  [ -d "$model_dir" ] || { echo "FATAL: cached model missing at $model_dir" >&2; exit 4; }
done
mkdir -p "$OUT_DIR" "$PROJECT_SCR/tmp"
export HF_HOME="${HSGR_HF_HOME:-$RUNTIME_SCR/hf}"
export TORCH_HOME="${HSGR_TORCH_HOME:-$RUNTIME_SCR/torch}"
export XDG_CACHE_HOME="${HSGR_XDG_CACHE_HOME:-$RUNTIME_SCR/xdg}"
export TMPDIR="$PROJECT_SCR/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

"$PY" -X utf8 -m unittest -v \
  test_gamut_process_repair_p0.py test_gamut_process_repair_p0b.py \
  test_gamut_relation_judge_p0c.py test_gamut_manual_repair_p0e.py
"$PY" -X utf8 gamut_manual_repair_p0e.py \
  --dataset "$DATA" --p0d-rows "$P0D_DIR/gamut_long_baseline_p0d_rows.jsonl" \
  --split test_text_only --generator-model Qwen/Qwen2.5-7B-Instruct \
  --judge-model Qwen/Qwen2.5-14B-Instruct --skip 48 --n 192 \
  --generation-batch-size 2 --judge-batch-size 4 --out-dir "$OUT_DIR"
sha256sum "$OUT_DIR/gamut_manual_repair_p0e_report.json" \
  "$OUT_DIR/gamut_manual_repair_p0e_candidates.jsonl"
echo "result_dir=$OUT_DIR"

