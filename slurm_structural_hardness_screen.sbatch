#!/usr/bin/env bash
#SBATCH --job-name=dch-shard0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/scratch/z/zitong/dch-hsgr/logs/dch-shard0-%j.out

set -euo pipefail
REPO="$HOME/dch-hsgr"
SCR="/mnt/scratch/z/$USER/dch-hsgr"
WORK="$SCR/work-dch-shard0-${SLURM_JOB_ID}"
PY="$SCR/venv/bin/python"

export HF_HOME="$SCR/hf"
export TORCH_HOME="$SCR/torch"
export XDG_CACHE_HOME="$SCR/xdg"
export TMPDIR="$SCR/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset LD_LIBRARY_PATH || true

[ -x "$PY" ] || { echo "FATAL: no project venv at $PY"; exit 1; }
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || {
  echo "FATAL: Slurm did not set CUDA_VISIBLE_DEVICES"; exit 1;
}
export CUDA_VISIBLE_DEVICES
mkdir -p "$WORK" "$TMPDIR" "$SCR/logs"

echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git -C "$REPO" rev-parse --short HEAD)"
echo "repo=$REPO work=$WORK CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || exit 2

cp -r "$REPO/." "$WORK/"
cd "$WORK"

"$PY" -m unittest -v test_structural_hardness_screen.py
"$PY" - <<'PYEOF'
import torch
torch.backends.cuda.enable_cudnn_sdp(False)
print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0))
PYEOF

export PY
bash run_structural_hardness_screen.sh 2>&1 | tee "$SCR/logs/structural_hardness_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
cp -f structural_hardness_screen/report.json \
  "$SCR/logs/structural_hardness_report_${SLURM_JOB_ID}.json" 2>/dev/null || true
echo "JOB_DONE rc=$RC"
exit "$RC"
