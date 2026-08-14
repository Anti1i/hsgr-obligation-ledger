#!/usr/bin/env bash
#SBATCH --job-name=dch-asqa-fetch
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/scratch/z/%u/dch-hsgr/logs/%x-%j.out

set -uo pipefail
REPO="$HOME/dch-hsgr"
SCR="/mnt/scratch/z/${USER}/dch-hsgr"
PY="$SCR/venv/bin/python"

[ -x "$PY" ] || { echo "FATAL: no project venv at $PY"; exit 1; }
mkdir -p "$SCR/data" "$SCR/tmp" "$SCR/logs"
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git -C "$REPO" rev-parse --short HEAD)"
echo "repo=$REPO python=$PY scratch=$SCR"
df -h "$HOME" "$SCR"

cd "$REPO" || exit 1
export SCR PY
bash fetch_asqa_oracle.sh 2>&1 | tee "$SCR/logs/asqa_fetch_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
echo "JOB_DONE rc=$RC"
exit "$RC"

