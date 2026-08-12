#!/usr/bin/env bash
#SBATCH --job-name=dch-joinv3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/scratch/z/zitong/dch-hsgr/logs/dch-joinv3-%j.out

set -euo pipefail
REPO="$HOME/dch-hsgr"
SCR="/mnt/scratch/z/$USER/dch-hsgr"
WORK="$SCR/work-dch-joinv3-${SLURM_JOB_ID}"
PY="$SCR/venv/bin/python"
REV="cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
SNAPSHOT="$SCR/hf/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/$REV"

export HF_HOME="$SCR/hf"
export TORCH_HOME="$SCR/torch"
export XDG_CACHE_HOME="$SCR/xdg"
export TMPDIR="$SCR/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset LD_LIBRARY_PATH || true

test "$(uname -m)" = x86_64
test -x "$PY"
test -n "${CUDA_VISIBLE_DEVICES:-}"
test -d "$SNAPSHOT"
mkdir -p "$WORK" "$TMPDIR" "$SCR/logs"

echo "host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID} commit=$(git -C "$REPO" rev-parse --short HEAD)"
echo "repo=$REPO work=$WORK CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "model_snapshot=$SNAPSHOT"
echo "soft_vmem=$(ulimit -Sv) hard_vmem=$(ulimit -Hv)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv

cp -r "$REPO/." "$WORK/"
cd "$WORK"
test "$(sha256sum data/gsm_join_easy_train.jsonl | awk '{print $1}')" = \
  "576fcbf7d6cee0d0d3c9e4a1cf059c0d474b17f675da183c1f47e41df30ae129"
test "$(sha256sum data/gsm_join_easy_test.jsonl | awk '{print $1}')" = \
  "5d274c136e149481e47b7ebe534599f7f899efc8e9ae073cce757b3c95c4e1ad"
"$PY" -c 'import torch, transformers; print("IMPORT_OK", torch.__version__, transformers.__version__, torch.cuda.is_available())'
"$PY" -m unittest -v \
  test_gsm_join_easy_data.py \
  test_join_viability_easy_v2.py \
  test_join_viability_easy_v2_1.py \
  test_join_viability_easy_v3.py
"$PY" join_viability_easy_v3.py --self-test
"$PY" join_viability_easy_v3.py \
  --calibration-data data/gsm_join_easy_train.jsonl \
  --confirmation-data data/gsm_join_easy_test.jsonl \
  --out-dir join_viability_easy_v3 \
  --hf-home "$HF_HOME" \
  --batch-size 8 2>&1 | tee "$SCR/logs/join_viability_easy_v3_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
cp -f join_viability_easy_v3/report.json \
  "$SCR/logs/join_viability_easy_v3_report_${SLURM_JOB_ID}.json" 2>/dev/null || true
echo "JOB_DONE rc=$RC"
exit "$RC"
