#!/usr/bin/env bash
#SBATCH --job-name=dch-deppatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=72G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/scratch/z/%u/dch-hsgr/logs/%x-%j.out

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

[ -x "$PY" ] || { echo "FATAL: no project venv at $PY"; exit 1; }
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || {
  echo "FATAL: Slurm did not assign CUDA_VISIBLE_DEVICES"; exit 1;
}
mkdir -p "$WORK" "$TMPDIR" "$SCR/logs"
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git -C "$REPO" rev-parse --short HEAD)"
echo "repo=$REPO work=$WORK python=$PY CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || exit 2

cp -r "$REPO/." "$WORK/"
cd "$WORK" || exit 1

"$PY" - <<'PYEOF' || exit 3
import torch
torch.backends.cuda.enable_cudnn_sdp(False)
print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0))
PYEOF

export PY
bash run_dependency_patch_p0.sh 2>&1 | tee "$SCR/logs/dependency_patch_p0_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
cp -f dependency_patch_p0/dependency_patch_p0_report.json "$SCR/logs/dependency_patch_p0_report_${SLURM_JOB_ID}.json" 2>/dev/null || true
cp -f dependency_patch_p0/dependency_patch_p0_cases.jsonl "$SCR/logs/dependency_patch_p0_cases_${SLURM_JOB_ID}.jsonl" 2>/dev/null || true
echo "JOB_DONE rc=$RC"
exit "$RC"
