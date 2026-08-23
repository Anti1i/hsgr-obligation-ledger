#!/usr/bin/env bash
set -euo pipefail

PROJECT_SCR="${HSGR_PROJECT_SCR:-/mnt/scratch/z/${USER}/hsgr-obligation-ledger}"
RUNTIME_SCR="${HSGR_RUNTIME_SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
PY="${HSGR_PYTHON:-$RUNTIME_SCR/venv/bin/python}"
DATA_DIR="$PROJECT_SCR/data"
DATA="$DATA_DIR/gamut_test_text_only.parquet"
OUT_DIR="${GAMUT_P0_OUT:-$PROJECT_SCR/results/gamut_process_repair_p0_${SLURM_JOB_ID:-manual}}"

mkdir -p "$DATA_DIR" "$OUT_DIR" "$PROJECT_SCR/tmp"
[ -x "$PY" ] || { echo "FATAL: no project Python at $PY" >&2; exit 1; }
if [ ! -s "$DATA" ]; then
  curl -fL --retry 3 --connect-timeout 20 \
    'https://huggingface.co/datasets/facebook/GAMUT/resolve/main/data/test_text_only-00000-of-00001.parquet?download=true' \
    -o "$DATA.part"
  mv -- "$DATA.part" "$DATA"
fi

export HF_HOME="${HSGR_HF_HOME:-$RUNTIME_SCR/hf}"
export TORCH_HOME="${HSGR_TORCH_HOME:-$RUNTIME_SCR/torch}"
export XDG_CACHE_HOME="${HSGR_XDG_CACHE_HOME:-$RUNTIME_SCR/xdg}"
export TMPDIR="$PROJECT_SCR/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

"$PY" -c 'import datasets, torch, transformers; print("datasets", datasets.__version__, "torch", torch.__version__, "transformers", transformers.__version__)'
"$PY" -X utf8 -m unittest -v test_gamut_process_repair_p0.py
"$PY" -X utf8 gamut_process_repair_p0.py \
  --dataset "$DATA" --split test_text_only \
  --model Qwen/Qwen2.5-7B-Instruct --n 48 \
  --generation-batch-size 4 --judge-batch-size 12 --out-dir "$OUT_DIR"
sha256sum "$OUT_DIR/gamut_process_repair_p0_report.json" \
  "$OUT_DIR/gamut_process_repair_p0_baselines.jsonl" \
  "$OUT_DIR/gamut_process_repair_p0_candidates.jsonl"
echo "result_dir=$OUT_DIR"
