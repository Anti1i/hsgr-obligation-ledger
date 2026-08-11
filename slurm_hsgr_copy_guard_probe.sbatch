#!/usr/bin/env bash
#SBATCH --job-name=dch-hsgrcp
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/scratch/z/%u/dch-hsgr/logs/%x-%j.out

set -uo pipefail
REPO="$HOME/dch-hsgr"
SCR="/mnt/scratch/z/${USER}/dch-hsgr"
WORK="$SCR/work-${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
PY="$SCR/venv/bin/python"
FEATURES="$SCR/logs/structured_hidden_features_726228.pt"

export TORCH_HOME="$SCR/torch"
export XDG_CACHE_HOME="$SCR/xdg"
export TMPDIR="$SCR/tmp"
unset LD_LIBRARY_PATH || true

[ -x "$PY" ] || { echo "FATAL: no venv at $PY"; exit 1; }
[ -f "$FEATURES" ] || { echo "FATAL: missing $FEATURES"; exit 1; }
mkdir -p "$WORK" "$TMPDIR" "$SCR/logs"
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git -C "$REPO" rev-parse --short HEAD)"
cp -r "$REPO/." "$WORK/"
cd "$WORK" || exit 1
mkdir -p data
cp -f "$SCR/data/musique_ans_val.jsonl" data/

"$PY" hsgr_copy_guard_probe.py \
  --data data/musique_ans_val.jsonl \
  --features "$FEATURES" \
  --out-dir hsgr_copy_guard_probe \
  2>&1 | tee "$SCR/logs/hsgr_copy_guard_probe_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
cp -f hsgr_copy_guard_probe/hsgr_copy_guard_probe_report.json "$SCR/logs/hsgr_copy_guard_probe_report_${SLURM_JOB_ID}.json" 2>/dev/null || true
echo "JOB_DONE rc=$RC"
exit "$RC"
