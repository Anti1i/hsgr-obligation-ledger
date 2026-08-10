#!/usr/bin/env bash
#SBATCH --job-name=dch-e0soc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/scratch/z/%u/dch-hsgr/logs/%x-%j.out
#
# E0 socratic rerun: measure decomposition tax with clean goals.

set -uo pipefail

REPO="$HOME/dch-hsgr"
SCR="/mnt/scratch/z/${USER}/dch-hsgr"
WORK="$SCR/work-${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
PY="$SCR/venv/bin/python"

export HF_HOME="$SCR/hf"
export TORCH_HOME="$SCR/torch"
export XDG_CACHE_HOME="$SCR/xdg"
export TMPDIR="$SCR/tmp"
# Model weights are cached; socratic jsonl is fetched with network briefly
# inside run_e0.sh (it unsets these). Keep offline default for generate().
export HF_HUB_OFFLINE_KEEP=1
export TRANSFORMERS_OFFLINE_KEEP=1
unset LD_LIBRARY_PATH || true

[ -x "$PY" ] || { echo "FATAL: no venv at $PY"; exit 1; }
mkdir -p "$WORK" "$TMPDIR" "$SCR/logs"

echo "=== provenance ==="
echo "host       : $(hostname)"
echo "job        : ${SLURM_JOB_ID}  (${SLURM_JOB_NAME})"
echo "gpus       : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "repo commit: $(git -C "$REPO" rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

echo "=== sync code + small data -> $WORK ==="
cp -r "$REPO/." "$WORK/"
cd "$WORK" || exit 1

"$PY" - <<'EOF'
import torch
torch.backends.cuda.enable_cudnn_sdp(False)
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "gpus", torch.cuda.device_count(),
      "name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
EOF

DD_ROOT="$SCR/work-dch-dd-g200-723813/dd_deep"
REUSE=""
if [ -d "$DD_ROOT" ] && ls "$DD_ROOT"/rootext.s*.jsonl >/dev/null 2>&1; then
  REUSE="$DD_ROOT"
  echo "reusing rootext from $REUSE"
else
  echo "no reusable rootext; will sample SC baseline"
fi

NGPU=$(nvidia-smi --list-gpus | wc -l)
echo "=== E0 socratic across $NGPU GPU(s) ==="
export PY REUSE_ROOTEXT="$REUSE" NS="$NGPU" LIMIT=300 OUTDIR=e0_soc
bash run_e0.sh
RC=$?

echo "=== footprint ==="
du -sh "$WORK" 2>/dev/null
echo "results under: $WORK  (e0_soc, logs/)"
echo "JOB_DONE rc=$RC"
exit "$RC"
