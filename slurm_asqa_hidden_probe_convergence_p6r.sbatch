#!/usr/bin/env bash
#SBATCH --job-name=dch-asqa-p6r
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=72G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/scratch/z/%u/dch-hsgr/logs/%x-%j.out

set -uo pipefail
REPO="${HSGR_REPO:-$HOME/hsgr-obligation-ledger}"
SCR="/mnt/scratch/z/${USER}/dch-hsgr"
WORK="$SCR/work-${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
PY="$SCR/venv/bin/python"

export HF_HOME="$SCR/hf"
export TORCH_HOME="$SCR/torch"
export XDG_CACHE_HOME="$SCR/xdg"
export TMPDIR="$SCR/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
unset LD_LIBRARY_PATH || true

[ -x "$PY" ] || { echo "FATAL: no project venv at $PY"; exit 1; }
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "FATAL: no Slurm GPU allocation"; exit 1; }
mkdir -p "$WORK" "$TMPDIR" "$SCR/logs" "$SCR/results"
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git -C "$REPO" rev-parse --short HEAD)"
echo "repo=$REPO work=$WORK python=$PY CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || exit 2
cp -r "$REPO/." "$WORK/"
cd "$WORK" || exit 1
export SCR PY
bash run_asqa_hidden_probe_convergence_p6r.sh 2>&1 | tee "$SCR/logs/asqa_hidden_probe_convergence_p6r_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
echo "JOB_DONE rc=$RC"
exit "$RC"
