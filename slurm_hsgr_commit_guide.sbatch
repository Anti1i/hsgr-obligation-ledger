#!/usr/bin/env bash
#SBATCH --job-name=dch-hsgrcg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
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
export CUDA_VISIBLE_DEVICES
mkdir -p "$WORK" "$TMPDIR" "$SCR/logs"
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git -C "$REPO" rev-parse --short HEAD)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || exit 2

cp -r "$REPO/." "$WORK/"
cd "$WORK" || exit 1
mkdir -p data
[ -f "$SCR/data/musique_ans_val.jsonl" ] || { echo "FATAL: missing data"; exit 1; }
cp -f "$SCR/data/musique_ans_val.jsonl" data/
for NAME in hsgr_edge_repair_cases_724849.jsonl hsgr_focus_route_cases_725772.jsonl; do
  [ -f "$SCR/logs/$NAME" ] || { echo "FATAL: missing $SCR/logs/$NAME"; exit 1; }
  cp -f "$SCR/logs/$NAME" "data/$NAME"
done

"$PY" - <<'EOF' || exit 3
import sklearn, torch
torch.backends.cuda.enable_cudnn_sdp(False)
print("torch", torch.__version__, "sklearn", sklearn.__version__, "gpu", torch.cuda.get_device_name(0))
EOF

export PY
bash run_hsgr_commit_guide.sh 2>&1 | tee "$SCR/logs/hsgr_commit_guide_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
cp -f hsgr_commit_guide/hsgr_commit_guide_report.json "$SCR/logs/hsgr_commit_guide_report_${SLURM_JOB_ID}.json" 2>/dev/null || true
cp -f hsgr_commit_guide/hsgr_commit_guide_cases.jsonl "$SCR/logs/hsgr_commit_guide_cases_${SLURM_JOB_ID}.jsonl" 2>/dev/null || true
echo "JOB_DONE rc=$RC"
exit "$RC"
