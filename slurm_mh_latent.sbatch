#!/usr/bin/env bash
#SBATCH --job-name=dch-mhlat
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
export SCR
unset LD_LIBRARY_PATH || true

[ -x "$PY" ] || { echo "FATAL: no venv at $PY"; exit 1; }
mkdir -p "$WORK" "$TMPDIR" "$SCR/logs"
echo "host $(hostname) job ${SLURM_JOB_ID} commit $(git -C "$REPO" rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
cp -r "$REPO/." "$WORK/"
cd "$WORK" || exit 1
if [ -f "$SCR/data/musique_ans_val.jsonl" ]; then
  mkdir -p data && cp -f "$SCR/data/musique_ans_val.jsonl" data/
fi
"$PY" - <<'EOF'
import torch
torch.backends.cuda.enable_cudnn_sdp(False)
print("ok", torch.cuda.get_device_name(0))
EOF

# Prefer the richest completed SC cache (mh_ceil or mh_e0 / mh_controls).
SC_DIR=""
BEST_N=0
for d in "$SCR"/work-dch-mhceil-*/mh_ceil \
         "$SCR"/work-dch-mhe0-*/mh_e0 \
         "$SCR"/work-dch-mhctl-*/mh_e0; do
  [ -d "$d" ] || continue
  n=0
  for f in "$d"/sc.s*.jsonl; do
    [ -f "$f" ] || continue
    n=$((n + $(wc -l < "$f")))
  done
  if [ "$n" -gt "$BEST_N" ]; then
    BEST_N=$n
    SC_DIR="$d"
  fi
done
echo "SC_DIR=$SC_DIR (n=$BEST_N)"
[ -n "$SC_DIR" ] || { echo "FATAL: no SC cache"; exit 1; }

export PY SC_DIR LIMIT=200
bash run_mh_latent.sh
RC=$?
cp -f mh_latent_report.json mh_latent/mh_latent_report.json "$SCR/logs/" 2>/dev/null || true
cp -f mh_latent/selections.jsonl "$SCR/logs/mh_latent_selections.jsonl" 2>/dev/null || true
echo "JOB_DONE rc=$RC"
exit "$RC"
