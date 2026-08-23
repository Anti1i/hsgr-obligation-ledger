#!/usr/bin/env bash
#SBATCH --job-name=hsgr-gamut-p0c
#SBATCH --partition=ai
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/scratch/z/zitong/hsgr-obligation-ledger/logs/hsgr-gamut-p0c-%j.out

set -euo pipefail
PROJECT_HOME="$HOME/hsgr-obligation-ledger"
PROJECT_SCR="/mnt/scratch/z/${USER}/hsgr-obligation-ledger"
mkdir -p "$PROJECT_SCR/logs"
echo "host=$(hostname) pwd=$PROJECT_HOME commit=$(git -C "$PROJECT_HOME" rev-parse HEAD)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
cd "$PROJECT_HOME"
bash run_gamut_relation_judge_p0c.sh

