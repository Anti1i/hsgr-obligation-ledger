#!/usr/bin/env bash
#SBATCH --job-name=dch-mhscreen
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/scratch/z/%u/dch-hsgr/logs/%x-%j.out
#
# CPU-only MuSiQue download + structural screen.

set -uo pipefail

REPO="$HOME/dch-hsgr"
SCR="/mnt/scratch/z/${USER}/dch-hsgr"
WORK="$SCR/work-${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
PY="$SCR/venv/bin/python"

export HF_HOME="$SCR/hf"
export TORCH_HOME="$SCR/torch"
export XDG_CACHE_HOME="$SCR/xdg"
export TMPDIR="$SCR/tmp"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true

[ -x "$PY" ] || { echo "FATAL: no venv at $PY"; exit 1; }
mkdir -p "$WORK" "$TMPDIR" "$SCR/logs"

echo "=== provenance ==="
hostname; echo "job ${SLURM_JOB_ID}"
echo "repo $(git -C "$REPO" rev-parse --short HEAD)"

cp -r "$REPO/." "$WORK/"
cd "$WORK" || exit 1
export PY
bash run_mh_screen.sh
RC=$?
# Keep the compact jsonl in the repo data/ via scratch copy for later GPU ceiling
mkdir -p "$REPO/data"
cp -f data/musique_ans_val.jsonl "$REPO/data/" 2>/dev/null || true
cp -f data/gsm8k_socratic_test.jsonl "$REPO/data/" 2>/dev/null || true
cp -f mh_screen_*.json "$SCR/logs/" 2>/dev/null || true
echo "JOB_DONE rc=$RC"
exit "$RC"
