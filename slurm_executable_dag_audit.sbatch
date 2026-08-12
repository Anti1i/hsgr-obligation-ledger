#!/usr/bin/env bash
#SBATCH --job-name=dch-dagaudit
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/scratch/z/zitong/dch-hsgr/logs/dch-dagaudit-%j.out

set -euo pipefail
REPO="$HOME/dch-hsgr"
SCR="/mnt/scratch/z/$USER/dch-hsgr"
WORK="$SCR/work-dch-dagaudit-${SLURM_JOB_ID}"
PY="$SCR/venv/bin/python"
SRC="$WORK/sources"
VENDOR="$WORK/vendor"

test -x "$PY"
mkdir -p "$WORK" "$SRC" "$VENDOR" "$SCR/logs" "$SCR/tmp"
echo "host=$(hostname) job=${SLURM_JOB_ID} repo=$REPO work=$WORK"
echo "commit=$(git -C "$REPO" rev-parse --short HEAD)"
echo "resources=partition:normal cpus:2 mem:12G time:01:00:00"

cp -r "$REPO/." "$WORK/code/"
cd "$WORK/code"
"$PY" -m unittest -v test_executable_dag_audit.py
"$PY" executable_dag_audit.py --self-test

git clone --depth 1 https://github.com/psunlpgroup/MultiHiertt.git "$SRC/MultiHiertt-code"

HITAB_COMMIT="$(git ls-remote https://github.com/microsoft/HiTab.git refs/heads/main | awk '{print $1}')"
test -n "$HITAB_COMMIT"
mkdir -p "$SRC/HiTab/data"
for split in train dev test; do
  curl -L --fail --retry 3 --connect-timeout 30 \
    "https://raw.githubusercontent.com/microsoft/HiTab/$HITAB_COMMIT/data/${split}_samples.jsonl" \
    -o "$SRC/HiTab/data/${split}_samples.jsonl"
done

"$PY" -m pip install --quiet --target "$VENDOR" 'gdown==5.2.0'
PYTHONPATH="$VENDOR" "$PY" -m gdown --folder \
  'https://drive.google.com/drive/folders/1ituEWZ5F7G9T9AZ0kzZZLrHNhRigHCZJ?usp=sharing' \
  -O "$SRC/MultiHiertt-dataset"

curl -L --fail --retry 3 --connect-timeout 30 \
  'https://math-qa.github.io/math-QA/data/MathQA.zip' \
  -o "$SRC/MathQA.zip"
unzip -q "$SRC/MathQA.zip" -d "$SRC/MathQA"

MULTI_COMMIT="$(git -C "$SRC/MultiHiertt-code" rev-parse HEAD)"
MATHQA_SHA="$(sha256sum "$SRC/MathQA.zip" | awk '{print $1}')"
"$PY" -c 'import json,sys; json.dump({"multihiertt_repo_commit":sys.argv[1],"multihiertt_dataset_url":sys.argv[2],"mathqa_url":sys.argv[3],"mathqa_zip_sha256":sys.argv[4],"hitab_repo_commit":sys.argv[5]},open("source_meta.json","w"),indent=1)' \
  "$MULTI_COMMIT" \
  'https://drive.google.com/drive/folders/1ituEWZ5F7G9T9AZ0kzZZLrHNhRigHCZJ' \
  'https://math-qa.github.io/math-QA/data/MathQA.zip' \
  "$MATHQA_SHA" "$HITAB_COMMIT"

"$PY" executable_dag_audit.py \
  --multihiertt-root "$SRC/MultiHiertt-dataset" \
  --mathqa-root "$SRC/MathQA" \
  --hitab-root "$SRC/HiTab/data" \
  --source-meta source_meta.json \
  --out executable_dag_audit_report.json \
  2>&1 | tee "$SCR/logs/executable_dag_audit_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
cp -f executable_dag_audit_report.json \
  "$SCR/logs/executable_dag_audit_report_${SLURM_JOB_ID}.json"
echo "JOB_DONE rc=$RC report=$SCR/logs/executable_dag_audit_report_${SLURM_JOB_ID}.json"
exit "$RC"
