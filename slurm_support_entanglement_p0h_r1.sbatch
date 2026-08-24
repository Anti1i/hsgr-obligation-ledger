#!/usr/bin/env bash
#SBATCH --job-name=hsgr-support-p0hr1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=72G
#SBATCH --time=00:35:00
#SBATCH --output=/mnt/scratch/z/%u/hsgr-obligation-ledger/logs/%x-%j.out

set -uo pipefail
REPO="${HSGR_REPO:-$HOME/hsgr-obligation-ledger}"
PROJECT_SCR="${HSGR_PROJECT_SCR:-/mnt/scratch/z/${USER}/hsgr-obligation-ledger}"
RUNTIME_SCR="${HSGR_RUNTIME_SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
WORK="$PROJECT_SCR/work-${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
PY="${HSGR_PYTHON:-$RUNTIME_SCR/venv/bin/python}"

[ -x "$PY" ] || { echo "FATAL: no project Python at $PY"; exit 1; }
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "FATAL: no Slurm GPU allocation"; exit 1; }
mkdir -p "$WORK" "$PROJECT_SCR/tmp" "$PROJECT_SCR/logs" "$PROJECT_SCR/results"
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git -C "$REPO" rev-parse --short HEAD)"
echo "repo=$REPO work=$WORK python=$PY CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || exit 2
cp -r "$REPO/." "$WORK/"
cd "$WORK" || exit 1
export HSGR_PROJECT_SCR="$PROJECT_SCR" HSGR_RUNTIME_SCR="$RUNTIME_SCR" HSGR_PYTHON="$PY"
bash run_support_entanglement_p0h_r1.sh 2>&1 | tee "$PROJECT_SCR/logs/support_entanglement_p0h_r1_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
echo "JOB_DONE rc=$RC"
exit "$RC"
