#!/usr/bin/env bash
#SBATCH --job-name=hsgr-policy-p0p-env
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/scratch/z/%u/hsgr-obligation-ledger/logs/%x-%j.out

set -euo pipefail
PROJECT_SCR="${HSGR_PROJECT_SCR:-/mnt/scratch/z/${USER}/hsgr-obligation-ledger}"
ARCHIVE_DIR="$PROJECT_SCR/env"
ARCHIVE="$ARCHIVE_DIR/venv-p0p-v1.tar"
ARCHIVE_PART="$ARCHIVE.part-${SLURM_JOB_ID}"
TMP_DIR="$PROJECT_SCR/tmp"
LOCAL_ROOT="$(mktemp -d "/tmp/hsgr-p0p-build-${SLURM_JOB_ID}-XXXXXX")"
ENV_DIR="$LOCAL_ROOT/venv"

cleanup() {
  rm -rf -- "$LOCAL_ROOT"
  rm -f -- "$ARCHIVE_PART"
}
trap cleanup EXIT

mkdir -p "$PROJECT_SCR/logs" "$TMP_DIR" "$ARCHIVE_DIR"
echo "host=$(hostname) job=${SLURM_JOB_ID} local_env=$ENV_DIR archive=$ARCHIVE"
df -h /tmp "$PROJECT_SCR"
python3 -m venv "$ENV_DIR"
export TMPDIR="$LOCAL_ROOT/tmp" PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$TMPDIR"
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
tar -C "$LOCAL_ROOT" -cf "$ARCHIVE_PART" venv
mv -f "$ARCHIVE_PART" "$ARCHIVE"
sha256sum "$ARCHIVE"
echo "ENV_SETUP_DONE"
