#!/usr/bin/env bash
#SBATCH --job-name=dch-setup
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/scratch/z/%u/dch-hsgr/logs/%x-%j.out
#
# One-time environment build on the scratch volume. No GPU needed: this only
# creates the venv and downloads model weights. The shared home is at its quota,
# so nothing heavy may live there -- /mnt/scratch is a separate 157T volume.
#
# Submit once:  sbatch slurm_setup.sbatch
# Safe to re-run; pip and hf both skip work that is already done.

set -uo pipefail

SCR="/mnt/scratch/z/${USER}/dch-hsgr"
export HF_HOME="$SCR/hf"
export TORCH_HOME="$SCR/torch"
export XDG_CACHE_HOME="$SCR/xdg"
export TMPDIR="$SCR/tmp"
mkdir -p "$SCR" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$SCR/logs"

echo "=== provenance ==="
echo "host    : $(hostname)"
echo "job     : ${SLURM_JOB_ID:-none}"
echo "scratch : $SCR"
df -h /mnt/scratch | tail -1

echo "=== venv ==="
PY="$SCR/venv/bin/python"
[ -x "$PY" ] || python3 -m venv "$SCR/venv" || { echo "FATAL: venv failed"; exit 1; }
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet torch transformers accelerate sympy huggingface_hub \
  || { echo "FATAL: pip install failed"; exit 1; }
"$PY" - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
EOF

echo "=== model weights -> $HF_HOME ==="
for m in Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-1.5B-Instruct; do
  echo "--- $m ---"
  "$SCR/venv/bin/hf" download "$m" || "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('$m')" || { echo "FATAL: download failed for $m"; exit 1; }
done

echo "=== footprint ==="
du -sh "$SCR" "$HF_HOME" "$SCR/venv"
echo "SETUP_DONE"
