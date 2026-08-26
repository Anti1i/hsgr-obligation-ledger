#!/usr/bin/env bash
#SBATCH --job-name=hsgr-policy-p0p-cache
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/scratch/z/%u/hsgr-obligation-ledger/logs/%x-%j.out

set -uo pipefail
REPO="${HSGR_REPO:-$HOME/hsgr-obligation-ledger}"
PROJECT_SCR="${HSGR_PROJECT_SCR:-/mnt/scratch/z/${USER}/hsgr-obligation-ledger}"
RUNTIME_SCR="${HSGR_RUNTIME_SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
WORK="$PROJECT_SCR/work-${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
PY="${HSGR_PYTHON:-$RUNTIME_SCR/venv/bin/python}"

[ -x "$PY" ] || { echo "FATAL: no project Python at $PY"; exit 1; }
mkdir -p "$WORK" "$PROJECT_SCR/tmp" "$PROJECT_SCR/logs" \
  "$PROJECT_SCR/data/refinebench_cache"
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git -C "$REPO" rev-parse --short HEAD)"
echo "repo=$REPO work=$WORK python=$PY"
df -h "$HOME" "$PROJECT_SCR"
cp -r "$REPO/." "$WORK/"
cd "$WORK" || exit 1
export HF_HOME="${HSGR_HF_HOME:-$RUNTIME_SCR/hf}"
export TORCH_HOME="${HSGR_TORCH_HOME:-$RUNTIME_SCR/torch}"
export XDG_CACHE_HOME="${HSGR_XDG_CACHE_HOME:-$RUNTIME_SCR/xdg}"
export TMPDIR="$PROJECT_SCR/tmp"
export TOKENIZERS_PARALLELISM=false
"$PY" -X utf8 cache_policy_representation_models_p0p.py \
  --hf-home "$HF_HOME" \
  --dataset-cache "$PROJECT_SCR/data/refinebench_cache" 2>&1 | \
  tee "$PROJECT_SCR/logs/policy_representation_p0p_cache_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
echo "JOB_DONE rc=$RC"
exit "$RC"
