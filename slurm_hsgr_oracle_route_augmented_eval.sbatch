#!/usr/bin/env bash
#SBATCH --job-name=dch-hsgroa
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:20:00
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
export PYTHONHASHSEED=0

[ -x "$PY" ] || { echo "FATAL: no venv at $PY"; exit 1; }
mkdir -p "$WORK" "$TMPDIR" "$SCR/logs"
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git -C "$REPO" rev-parse --short HEAD)"
echo "pwd=$WORK threads=${SLURM_CPUS_PER_TASK:-16}"
cp -r "$REPO/." "$WORK/"
cd "$WORK" || exit 1
mkdir -p data
[ -f "$SCR/data/musique_ans_val.jsonl" ] || { echo "FATAL: missing data"; exit 1; }
cp -f "$SCR/data/musique_ans_val.jsonl" data/

FEATURE_A="$SCR/logs/structured_hidden_features_726228.pt"
FEATURE_B="$SCR/logs/dual_route_hidden_features_726354.pt"
FEATURE_C="$SCR/logs/transition_fresh_features_726389.pt"
for FEATURE in "$FEATURE_A" "$FEATURE_B" "$FEATURE_C"; do
  [ -f "$FEATURE" ] || { echo "FATAL: missing $FEATURE"; exit 1; }
done

"$PY" -m unittest test_route_counterfactual.py || exit 3
"$PY" hsgr_oracle_route_augmented_eval.py \
  --data data/musique_ans_val.jsonl \
  --features "$FEATURE_A" "$FEATURE_B" "$FEATURE_C" \
  --out-dir hsgr_oracle_route_augmented_eval \
  --threads "${SLURM_CPUS_PER_TASK:-16}"
RC=$?
cp -f hsgr_oracle_route_augmented_eval/oracle_route_augmented_report.json \
  "$SCR/logs/oracle_route_augmented_report_${SLURM_JOB_ID}.json" 2>/dev/null || true
echo "JOB_DONE rc=$RC"
exit "$RC"

