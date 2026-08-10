#!/usr/bin/env bash
#SBATCH --job-name=dch-mhe0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
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
export HF_HUB_OFFLINE_KEEP=1
export TRANSFORMERS_OFFLINE_KEEP=1
unset LD_LIBRARY_PATH || true

[ -x "$PY" ] || { echo "FATAL: no venv at $PY"; exit 1; }
mkdir -p "$WORK" "$TMPDIR" "$SCR/logs"
echo "host $(hostname) job ${SLURM_JOB_ID} commit $(git -C "$REPO" rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
cp -r "$REPO/." "$WORK/"
cd "$WORK" || exit 1
"$PY" - <<'EOF'
import torch
torch.backends.cuda.enable_cudnn_sdp(False)
print("ok", torch.cuda.get_device_name(0))
EOF

REUSE=""
# Prefer completed mh_ceil workdir with matching Random(0) limit=200 sample.
for d in "$SCR"/work-dch-mhceil-*/mh_ceil; do
  if [ -d "$d" ] && ls "$d"/sc.s*.jsonl >/dev/null 2>&1; then
    REUSE="$d"
  fi
done
echo "REUSE_SC=$REUSE"
export PY REUSE_SC="$REUSE" LIMIT=200
bash run_mh_e0.sh
RC=$?
cp -f mh_e0_report.json "$SCR/logs/" 2>/dev/null || true
echo "JOB_DONE rc=$RC"
exit "$RC"
