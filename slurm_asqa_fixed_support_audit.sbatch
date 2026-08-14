#!/usr/bin/env bash
#SBATCH --job-name=dch-asqa-p0
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/scratch/z/%u/dch-hsgr/logs/%x-%j.out

set -uo pipefail
REPO="$HOME/dch-hsgr"
SCR="/mnt/scratch/z/${USER}/dch-hsgr"
PY="$SCR/venv/bin/python"

[ -x "$PY" ] || { echo "FATAL: no project venv at $PY"; exit 1; }
mkdir -p "$SCR/data" "$SCR/results" "$SCR/logs"
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git -C "$REPO" rev-parse --short HEAD)"
echo "repo=$REPO python=$PY scratch=$SCR"
echo "cpus=${SLURM_CPUS_PER_TASK:-unknown} mem_per_node=${SLURM_MEM_PER_NODE:-unknown}"
df -h "$HOME" "$SCR"

cd "$REPO" || exit 1
export SCR PY
bash run_asqa_fixed_support_audit.sh 2>&1 | tee "$SCR/logs/asqa_p0_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
echo "JOB_DONE rc=$RC"
exit "$RC"
