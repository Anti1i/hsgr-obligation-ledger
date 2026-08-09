#!/usr/bin/env bash
#SBATCH --job-name=dch-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/scratch/z/%u/dch-hsgr/logs/%x-%j.out
#
# One-GPU smoke test: confirm SDPA works after disabling cuDNN SDP.
# Override GPU: sbatch --gres=gpu:h200-141:1 slurm_smoke.sbatch

set -uo pipefail

SCR="/mnt/scratch/z/${USER}/dch-hsgr"
REPO="$HOME/dch-hsgr"
PY="$SCR/venv/bin/python"
export HF_HOME="$SCR/hf"
export TORCH_HOME="$SCR/torch"
export XDG_CACHE_HOME="$SCR/xdg"
export TMPDIR="$SCR/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset LD_LIBRARY_PATH || true

echo "host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
cp "$REPO/gpu_probe.py" "$SCR/work/gpu_probe.py"
cd "$SCR/work" || exit 1
"$PY" gpu_probe.py
echo "SMOKE_DONE"
