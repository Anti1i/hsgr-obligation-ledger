#!/usr/bin/env bash
#SBATCH --job-name=dch-hsgrra
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/scratch/z/%u/dch-hsgr/logs/%x-%j.out

set -uo pipefail
REPO="$HOME/dch-hsgr"
SCR="/mnt/scratch/z/${USER}/dch-hsgr"
WORK="$SCR/work-${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
PY="$SCR/venv/bin/python"
FEATURES="$SCR/cache-route-counterfactual-v1-1b074fc/route_counterfactual_features.pt"

export TORCH_HOME="$SCR/torch"
export XDG_CACHE_HOME="$SCR/xdg"
export TMPDIR="$SCR/tmp"
unset LD_LIBRARY_PATH || true

[ -x "$PY" ] || { echo "FATAL: no venv at $PY"; exit 1; }
[ -f "$FEATURES" ] || { echo "FATAL: missing $FEATURES"; exit 1; }
mkdir -p "$WORK" "$TMPDIR" "$SCR/logs"
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git -C "$REPO" rev-parse --short HEAD)"
echo "features=$FEATURES work=$WORK"
cp -r "$REPO/." "$WORK/"
cd "$WORK" || exit 1

"$PY" -m unittest -v test_route_counterfactual.py || exit 3

"$PY" hsgr_route_augmented_eval.py \
  --features "$FEATURES" \
  --out-dir hsgr_route_augmented_eval \
  --threads 16 \
  2>&1 | tee "$SCR/logs/hsgr_route_augmented_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
cp -f hsgr_route_augmented_eval/route_augmented_report.json \
  "$SCR/logs/route_augmented_report_${SLURM_JOB_ID}.json" 2>/dev/null || true
echo "JOB_DONE rc=$RC"
exit "$RC"
