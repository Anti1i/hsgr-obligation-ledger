#!/usr/bin/env bash
#SBATCH --job-name=dch-dd
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/scratch/z/%u/dch-hsgr/logs/%x-%j.out
#
# Decomposition-level candidate domain (replaces the per-node candidate domain
# that node_oracle.py showed to be degenerate at 1.16 distinct answers).
# Requires slurm_setup.sbatch to have completed once.
#
# Override the GPU type at submit time, e.g.:
#   sbatch --gres=gpu:h200-141:2 slurm_dd.sbatch
# Per-user QOS caps: h200-141 max 2; h100-47 / h100-96 / a100-80 max 8.

set -uo pipefail

REPO="$HOME/dch-hsgr"
SCR="/mnt/scratch/z/${USER}/dch-hsgr"
WORK="$SCR/work-${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
PY="$SCR/venv/bin/python"

export HF_HOME="$SCR/hf"
export TORCH_HOME="$SCR/torch"
export XDG_CACHE_HOME="$SCR/xdg"
export TMPDIR="$SCR/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset LD_LIBRARY_PATH || true

[ -x "$PY" ] || { echo "FATAL: no venv at $PY -- run slurm_setup.sbatch first"; exit 1; }
mkdir -p "$WORK" "$TMPDIR"

echo "=== provenance ==="
echo "host       : $(hostname)"
echo "job        : ${SLURM_JOB_ID}  (${SLURM_JOB_NAME})"
echo "gpus       : ${CUDA_VISIBLE_DEVICES:-unset}  [${SLURM_JOB_GPUS:-?}]"
echo "repo commit: $(git -C "$REPO" rev-parse --short HEAD)"
echo "scratch    : $SCR  ($(df -h --output=avail /mnt/scratch | tail -1 | tr -d ' ') free)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

echo "=== sync code -> $WORK ==="
cp -r "$REPO/." "$WORK/"
cd "$WORK" || exit 1

"$PY" - <<'EOF'
import torch
torch.backends.cuda.enable_cudnn_sdp(False)
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "gpus", torch.cuda.device_count(),
      "name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
EOF

echo "=== data prep ==="
"$PY" data_prep.py --which all --data-dir data

NGPU=$(nvidia-smi --list-gpus | wc -l)
echo "=== decomposition-domain run across $NGPU GPU(s) ==="
PY="$PY" NS="$NGPU" bash run_dd.sh
RC=$?

echo "=== footprint ==="
du -sh "$WORK" "$SCR" 2>/dev/null
echo "results under: $WORK  (dd_deep, dd_mathl5, logs/)"
echo "JOB_DONE rc=$RC"
exit "$RC"
