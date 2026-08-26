#!/usr/bin/env bash
#SBATCH --job-name=hsgr-policy-p0p-env
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/scratch/z/%u/hsgr-obligation-ledger/logs/%x-%j.out

set -euo pipefail
PROJECT_SCR="${HSGR_PROJECT_SCR:-/mnt/scratch/z/${USER}/hsgr-obligation-ledger}"
ENV_DIR="$PROJECT_SCR/venv-p0p-v1"
TMP_DIR="$PROJECT_SCR/tmp"

mkdir -p "$PROJECT_SCR/logs" "$TMP_DIR"
echo "host=$(hostname) job=${SLURM_JOB_ID} env=$ENV_DIR"
df -h "$PROJECT_SCR"
if [ ! -x "$ENV_DIR/bin/python" ]; then
  python3 -m venv "$ENV_DIR"
fi
export TMPDIR="$TMP_DIR" PIP_DISABLE_PIP_VERSION_CHECK=1
"$ENV_DIR/bin/python" -m pip install --no-cache-dir \
  torch==2.8.0 \
  numpy==2.2.6 \
  pandas==2.3.2 \
  pyarrow==21.0.0 \
  transformers==4.55.4 \
  tokenizers==0.21.4 \
  accelerate==1.10.1 \
  datasets==4.0.0 \
  huggingface_hub==0.34.4 \
  safetensors==0.6.2 \
  sentencepiece==0.2.1
"$ENV_DIR/bin/python" - <<'PY'
import accelerate
import datasets
import numpy
import torch
import transformers

print("torch", torch.__version__, "cuda_build", torch.version.cuda)
print("numpy", numpy.__version__)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("accelerate", accelerate.__version__)
PY
du -sh "$ENV_DIR"
echo "ENV_SETUP_DONE"
