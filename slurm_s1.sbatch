#!/usr/bin/env bash
#SBATCH --job-name=dch-s1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#
# S1 acceptance pilot. Everything heavy (venv, model weights, HF cache) lives on
# the compute node's local disk, because the shared home is at its quota. Only
# the small JSONL results are copied back at the end.
#
# Override the GPU type at submit time, e.g.:
#   sbatch --gres=gpu:h200-141:2 slurm_s1.sbatch
# Note the per-user QOS cap: h200-141 max 2, h100-47/h100-96/a100-80 max 8.

set -uo pipefail

REPO="$HOME/dch-hsgr"
SCRATCH="/tmp/${USER}-${SLURM_JOB_ID}"
mkdir -p "$SCRATCH" || { echo "FATAL: cannot create $SCRATCH"; exit 1; }
trap 'rm -rf "$SCRATCH"' EXIT

# Keep every cache off the shared home and out of public cache dirs.
export HF_HOME="$SCRATCH/hf"
export TORCH_HOME="$SCRATCH/torch"
export XDG_CACHE_HOME="$SCRATCH/xdg"
export TMPDIR="$SCRATCH/tmp"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR"

echo "=== provenance ==="
echo "host       : $(hostname)"
echo "job        : $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "gpus       : ${CUDA_VISIBLE_DEVICES:-unset}  [${SLURM_JOB_GPUS:-?}]"
echo "scratch    : $SCRATCH  ($(df -h --output=avail "$SCRATCH" | tail -1 | tr -d ' ') free)"
echo "repo commit: $(git -C "$REPO" rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

echo "=== venv on node-local disk ==="
PY="$SCRATCH/venv/bin/python"
python3 -m venv "$SCRATCH/venv" || { echo "FATAL: venv creation failed"; exit 1; }
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet torch transformers accelerate sympy huggingface_hub \
  || { echo "FATAL: pip install failed"; exit 1; }
"$PY" - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "devices", torch.cuda.device_count())
print("transformers", transformers.__version__)
EOF

echo "=== model weights -> $HF_HOME ==="
# The `hf` entry point only exists in newer huggingface_hub; fall back to the API.
"$SCRATCH/venv/bin/hf" download Qwen/Qwen2.5-7B-Instruct \
  || "$PY" -c "from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct')" \
  || { echo "FATAL: model download failed"; exit 1; }
du -sh "$HF_HOME"

echo "=== run S1 from node-local working copy ==="
# Work on a node-local copy so the pipeline never writes into the shared home.
cp -r "$REPO" "$SCRATCH/work"
cd "$SCRATCH/work" || exit 1
"$PY" data_prep.py --which all --data-dir data

NGPU=$(nvidia-smi --list-gpus | wc -l)
echo "sharding across $NGPU GPU(s)"
PY="$PY" NS="$NGPU" bash run_s1.sh
S1_RC=$?
echo "run_s1.sh exit: $S1_RC"

echo "=== copy small results back ==="
# Home is near its quota; report clearly instead of failing silently.
DEST="$REPO/results/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$DEST"
for d in outputs_deep2 outputs_mathl5 logs; do
  [ -d "$d" ] || continue
  if cp -r "$d" "$DEST/" 2>"$SCRATCH/cp.err"; then
    echo "copied $d -> $DEST"
  else
    echo "WARNING: could not copy $d (likely home quota): $(tail -1 "$SCRATCH/cp.err")"
  fi
done
du -sh "$DEST" 2>/dev/null
echo "JOB_DONE rc=$S1_RC"
exit "$S1_RC"
