#!/usr/bin/env bash
#SBATCH --job-name=dch-finqa-a0
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/scratch/z/zitong/dch-hsgr/logs/dch-finqa-a0-%j.out

set -euo pipefail
REPO="$HOME/dch-hsgr"
SCR="/mnt/scratch/z/$USER/dch-hsgr"
WORK="$SCR/work-dch-finqa-a0-${SLURM_JOB_ID}"
PY="$SCR/venv/bin/python"
SOURCE="$SCR/data/FinQA-source"
SOURCE_URL="https://github.com/czyssrs/FinQA.git"

test -x "$PY"
mkdir -p "$WORK" "$SCR/data" "$SCR/logs" "$SCR/tmp"

echo "host=$(hostname) job=${SLURM_JOB_ID} repo=$REPO work=$WORK"
echo "commit=$(git -C "$REPO" rev-parse --short HEAD)"
echo "source=$SOURCE_URL data_dir=$SOURCE/dataset"

if [ ! -d "$SOURCE/.git" ]; then
  git clone --depth 1 "$SOURCE_URL" "$SOURCE"
else
  git -C "$SOURCE" fetch --depth 1 origin main
  git -C "$SOURCE" checkout --detach FETCH_HEAD
fi
SOURCE_COMMIT="$(git -C "$SOURCE" rev-parse HEAD)"
echo "finqa_source_commit=$SOURCE_COMMIT"

cp -r "$REPO/." "$WORK/"
cd "$WORK"
"$PY" -m unittest -v test_finqa_program.py
"$PY" finqa_structure_audit.py --self-test
"$PY" finqa_structure_audit.py \
  --data-dir "$SOURCE/dataset" \
  --source-commit "$SOURCE_COMMIT" \
  --out finqa_structure_audit_report.json \
  2>&1 | tee "$SCR/logs/finqa_structure_audit_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
cp -f finqa_structure_audit_report.json \
  "$SCR/logs/finqa_structure_audit_report_${SLURM_JOB_ID}.json"
echo "JOB_DONE rc=$RC report=$SCR/logs/finqa_structure_audit_report_${SLURM_JOB_ID}.json"
exit "$RC"
