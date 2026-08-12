#!/usr/bin/env bash
#SBATCH --job-name=dch-cache14b
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/scratch/z/zitong/dch-hsgr/logs/dch-cache14b-%j.out

set -euo pipefail
SCR="/mnt/scratch/z/$USER/dch-hsgr"
PY="$SCR/venv/bin/python"
export HF_HOME="$SCR/hf"
export XDG_CACHE_HOME="$SCR/xdg"
export TMPDIR="$SCR/tmp"
unset HF_HUB_OFFLINE || true
unset TRANSFORMERS_OFFLINE || true

test -x "$PY"
mkdir -p "$SCR/logs" "$TMPDIR"
echo "host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID}"
echo "HF_HOME=$HF_HOME"
df -h "$SCR"
"$PY" -c 'from huggingface_hub import snapshot_download; print("SNAPSHOT=" + snapshot_download(repo_id="Qwen/Qwen2.5-14B-Instruct"))'
ref="$HF_HOME/hub/models--Qwen--Qwen2.5-14B-Instruct/refs/main"
test -s "$ref"
echo "MODEL_REVISION=$(tr -d '\r\n' < "$ref")"
du -sh "$HF_HOME/hub/models--Qwen--Qwen2.5-14B-Instruct"
echo 'CACHE_DONE'
