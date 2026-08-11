#!/usr/bin/env bash
#SBATCH --job-name=dch-hsgrdg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:15:00
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

[ -x "$PY" ] || { echo "FATAL: no venv at $PY"; exit 1; }
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || {
  echo "FATAL: Slurm did not set CUDA_VISIBLE_DEVICES"; exit 1;
}
mkdir -p "$WORK" "$TMPDIR" "$SCR/logs"
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git -C "$REPO" rev-parse --short HEAD)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || exit 2

cp -r "$REPO/." "$WORK/"
cd "$WORK" || exit 1
mkdir -p data
cp -f "$SCR/data/musique_ans_val.jsonl" data/
for FEATURE in structured_hidden_features_726228.pt heldout_hidden_features_726310.pt dual_route_hidden_features_726354.pt; do
  [ -f "$SCR/logs/$FEATURE" ] || { echo "FATAL: missing $SCR/logs/$FEATURE"; exit 1; }
done

"$PY" - <<'EOF' || exit 3
import torch
torch.backends.cuda.enable_cudnn_sdp(False)
print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0))
EOF

bash run_hsgr_dual_route_diagnostic.sh 2>&1 | tee "$SCR/logs/hsgr_dual_route_diagnostic_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
for NAME in dual_route_guide_report.json dual_route_guide_cases.jsonl dual_route_hidden_features.pt; do
  cp -f "hsgr_dual_route_diagnostic/$NAME" "$SCR/logs/${NAME%.*}_${SLURM_JOB_ID}.${NAME##*.}" 2>/dev/null || true
done
echo "JOB_DONE rc=$RC"
exit "$RC"
