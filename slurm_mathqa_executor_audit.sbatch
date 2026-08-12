#!/usr/bin/env bash
#SBATCH --job-name=dch-mqexec
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/scratch/z/zitong/dch-hsgr/logs/dch-mqexec-%j.out

set -euo pipefail
REPO="$HOME/dch-hsgr"
SCR="/mnt/scratch/z/$USER/dch-hsgr"
WORK="$SCR/work-dch-mqexec-${SLURM_JOB_ID}"
PY="$SCR/venv/bin/python"

test -x "$PY"
mkdir -p "$WORK" "$SCR/logs"
echo "host=$(hostname) job=${SLURM_JOB_ID} repo=$REPO work=$WORK"
echo "commit=$(git -C "$REPO" rev-parse --short HEAD)"
echo "resources=partition:normal cpus:2 mem:8G time:00:30:00"

cp -r "$REPO/." "$WORK/code/"
cd "$WORK/code"
"$PY" -m unittest -v test_executable_dag_audit.py test_mathqa_executor_audit.py

curl -L --fail --retry 3 --connect-timeout 30 \
  'https://math-qa.github.io/math-QA/data/MathQA.zip' \
  -o "$WORK/MathQA.zip"
unzip -q "$WORK/MathQA.zip" -d "$WORK/MathQA"
echo "mathqa_zip_sha256=$(sha256sum "$WORK/MathQA.zip" | awk '{print $1}')"

"$PY" mathqa_executor_audit.py \
  --root "$WORK/MathQA" \
  --out mathqa_executor_audit_report.json \
  2>&1 | tee "$SCR/logs/mathqa_executor_audit_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
cp -f mathqa_executor_audit_report.json \
  "$SCR/logs/mathqa_executor_audit_report_${SLURM_JOB_ID}.json"
echo "JOB_DONE rc=$RC report=$SCR/logs/mathqa_executor_audit_report_${SLURM_JOB_ID}.json"
exit "$RC"
