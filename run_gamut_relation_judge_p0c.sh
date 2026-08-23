#!/usr/bin/env bash
set -euo pipefail

PROJECT_SCR="${HSGR_PROJECT_SCR:-/mnt/scratch/z/${USER}/hsgr-obligation-ledger}"
RUNTIME_SCR="${HSGR_RUNTIME_SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
PY="${HSGR_PYTHON:-$RUNTIME_SCR/venv/bin/python}"
DATA="$PROJECT_SCR/data/gamut_test_text_only.parquet"
BASELINES="${GAMUT_P0B_BASELINES:-$PROJECT_SCR/results/gamut_process_repair_p0b_751095/gamut_process_repair_p0b_baselines.jsonl}"
OUT_DIR="${GAMUT_P0C_OUT:-$PROJECT_SCR/results/gamut_relation_judge_p0c_${SLURM_JOB_ID:-manual}}"

[ -x "$PY" ] || { echo "FATAL: no project Python at $PY" >&2; exit 1; }
[ -s "$DATA" ] || { echo "FATAL: frozen GAMUT parquet missing at $DATA" >&2; exit 2; }
[ -s "$BASELINES" ] || { echo "FATAL: frozen P0b baselines missing at $BASELINES" >&2; exit 3; }
[ -d "$RUNTIME_SCR/hf/hub/models--Qwen--Qwen2.5-14B-Instruct" ] || {
  echo "FATAL: cached judge model missing" >&2; exit 4;
}
mkdir -p "$OUT_DIR" "$PROJECT_SCR/tmp"
export HF_HOME="${HSGR_HF_HOME:-$RUNTIME_SCR/hf}"
export TORCH_HOME="${HSGR_TORCH_HOME:-$RUNTIME_SCR/torch}"
export XDG_CACHE_HOME="${HSGR_XDG_CACHE_HOME:-$RUNTIME_SCR/xdg}"
export TMPDIR="$PROJECT_SCR/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

"$PY" -X utf8 -m unittest -v test_gamut_process_repair_p0.py test_gamut_process_repair_p0b.py test_gamut_relation_judge_p0c.py
"$PY" -X utf8 gamut_relation_judge_p0c.py \
  --dataset "$DATA" --baseline-jsonl "$BASELINES" --split test_text_only \
  --judge-model Qwen/Qwen2.5-14B-Instruct --skip 48 --n 192 \
  --judge-batch-size 8 --generation-batch-size 4 --out-dir "$OUT_DIR"
sha256sum "$OUT_DIR/gamut_relation_judge_p0c_report.json" \
  "$OUT_DIR/gamut_relation_judge_p0c_rows.jsonl"
echo "result_dir=$OUT_DIR"

