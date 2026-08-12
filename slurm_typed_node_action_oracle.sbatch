#!/usr/bin/env bash
#SBATCH --job-name=dch-action0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/scratch/z/zitong/dch-hsgr/logs/dch-action0-%j.out

set -euo pipefail
REPO="$HOME/dch-hsgr"
SCR="/mnt/scratch/z/$USER/dch-hsgr"
WORK="$SCR/work-dch-action0-${SLURM_JOB_ID}"
PY="$SCR/venv/bin/python"
BASELINE="$SCR/work-dch-shard0-728006/structural_hardness_screen/calibration/parents.jsonl"

export HF_HOME="$SCR/hf"
export TORCH_HOME="$SCR/torch"
export XDG_CACHE_HOME="$SCR/xdg"
export TMPDIR="$SCR/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset LD_LIBRARY_PATH || true

test "$(uname -m)" = x86_64
test -x "$PY"
test -s "$BASELINE"
test "$(wc -l < "$BASELINE")" -eq 800
test -n "${CUDA_VISIBLE_DEVICES:-}"
mkdir -p "$WORK" "$TMPDIR" "$SCR/logs"

echo "host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID} commit=$(git -C "$REPO" rev-parse --short HEAD)"
echo "repo=$REPO work=$WORK baseline=$BASELINE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "soft_vmem=$(ulimit -Sv) hard_vmem=$(ulimit -Hv)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv

cp -r "$REPO/." "$WORK/"
cd "$WORK"
"$PY" -c 'import torch, transformers; print("IMPORT_OK", torch.__version__, transformers.__version__, torch.cuda.is_available())'
"$PY" -m unittest -v test_typed_node_action_oracle.py
"$PY" typed_node_action_oracle.py --self-test --baseline-parents "$BASELINE"
"$PY" typed_node_action_oracle.py \
  --data data/gsm_join_train.jsonl \
  --baseline-parents "$BASELINE" \
  --out-dir typed_node_action_oracle_v0 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --batch-size 8 2>&1 | tee "$SCR/logs/typed_node_action_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
cp -f typed_node_action_oracle_v0/report.json \
  "$SCR/logs/typed_node_action_report_${SLURM_JOB_ID}.json" 2>/dev/null || true
echo "JOB_DONE rc=$RC"
exit "$RC"
