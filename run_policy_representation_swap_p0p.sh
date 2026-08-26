#!/usr/bin/env bash
set -euo pipefail

PROJECT_SCR="${HSGR_PROJECT_SCR:-/mnt/scratch/z/${USER}/hsgr-obligation-ledger}"
RUNTIME_SCR="${HSGR_RUNTIME_SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
PY="${HSGR_PYTHON:-$RUNTIME_SCR/venv/bin/python}"
OUT_DIR="${POLICY_SWAP_P0P_OUT:-$PROJECT_SCR/results/policy_representation_swap_p0p_${SLURM_JOB_ID:-manual}}"
DATASET_CACHE="${REFINEBENCH_CACHE:-$PROJECT_SCR/data/refinebench_cache}"

[ -x "$PY" ] || { echo "FATAL: no project Python at $PY" >&2; exit 1; }
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "FATAL: no Slurm GPU allocation" >&2; exit 2; }
mkdir -p "$OUT_DIR" "$PROJECT_SCR/tmp" "$DATASET_CACHE"
export HF_HOME="${HSGR_HF_HOME:-$RUNTIME_SCR/hf}"
export TORCH_HOME="${HSGR_TORCH_HOME:-$RUNTIME_SCR/torch}"
export XDG_CACHE_HOME="${HSGR_XDG_CACHE_HOME:-$RUNTIME_SCR/xdg}"
export TMPDIR="$PROJECT_SCR/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

for model_dir in \
  models--Qwen--Qwen3-8B \
  models--mistralai--Mistral-7B-Instruct-v0.3 \
  models--allenai--OLMo-2-1124-7B-Instruct \
  models--Qwen--Qwen2.5-14B-Instruct; do
  [ -d "$HF_HOME/hub/$model_dir" ] || {
    echo "FATAL: cached model missing: $model_dir" >&2
    exit 3
  }
done

"$PY" -X utf8 -m unittest -v test_qwen3_non_thinking.py \
  test_policy_representation_swap_p0p.py
"$PY" -X utf8 policy_representation_swap_p0p.py \
  --dataset-cache "$DATASET_CACHE" \
  --out-dir "$OUT_DIR" \
  --judge-model Qwen/Qwen2.5-14B-Instruct \
  --per-stratum 2 \
  --query-char-limit 8000 \
  --representation-batch-size 1 \
  --answer-batch-size 2 \
  --judge-batch-size 3 \
  --checklist-cap 768 \
  --plan-cap 384 \
  --answer-cap 1024 \
  --judge-cap 384 \
  --permutations 10000

sha256sum "$OUT_DIR/p0p_report.json"
for artifact in p0p_selected_tasks.jsonl p0p_representations.jsonl \
  p0p_answers.jsonl p0p_judgments.jsonl p0p_native_coverage.jsonl \
  p0p_manual_review.jsonl; do
  if [ -f "$OUT_DIR/$artifact" ]; then
    sha256sum "$OUT_DIR/$artifact"
  fi
done
