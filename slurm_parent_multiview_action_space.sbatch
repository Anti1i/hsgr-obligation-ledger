#!/usr/bin/env bash
#SBATCH --job-name=dch-mview0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/scratch/z/zitong/dch-hsgr/logs/dch-mview0-%j.out

set -euo pipefail
REPO="$HOME/dch-hsgr"
SCR="/mnt/scratch/z/$USER/dch-hsgr"
WORK="$SCR/work-dch-mview0-${SLURM_JOB_ID}"
PY="$SCR/venv/bin/python"
BASELINE=${BASELINE:-$SCR/work-dch-shard0-728006/structural_hardness_screen/calibration/parents.jsonl}

export HF_HOME="$SCR/hf"
export TORCH_HOME="$SCR/torch"
export XDG_CACHE_HOME="$SCR/xdg"
export TMPDIR="$SCR/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset LD_LIBRARY_PATH || true

test -x "$PY"
test -s "$BASELINE"
test "$(wc -l < "$BASELINE")" -eq 800
test -n "${CUDA_VISIBLE_DEVICES:-}"
mkdir -p "$WORK" "$TMPDIR" "$SCR/logs"

echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git -C "$REPO" rev-parse --short HEAD)"
echo "repo=$REPO work=$WORK baseline=$BASELINE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

cp -r "$REPO/." "$WORK/"
cd "$WORK"
"$PY" -m unittest -v test_parent_multiview_action_space.py
"$PY" parent_multiview_action_space.py --self-test --baseline-parents "$BASELINE"
"$PY" parent_multiview_action_space.py \
  --data data/gsm_join_train.jsonl \
  --baseline-parents "$BASELINE" \
  --out-dir parent_multiview_action_space \
  --model Qwen/Qwen2.5-7B-Instruct \
  --batch-size 16 2>&1 | tee "$SCR/logs/parent_multiview_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
cp -f parent_multiview_action_space/report.json \
  "$SCR/logs/parent_multiview_report_${SLURM_JOB_ID}.json" 2>/dev/null || true
echo "JOB_DONE rc=$RC"
exit "$RC"
