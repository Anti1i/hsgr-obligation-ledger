#!/usr/bin/env bash
#SBATCH --job-name=hsgr-policy-p0p-audit
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/scratch/z/%u/hsgr-obligation-ledger/logs/%x-%j.out

set -euo pipefail
REPO="${HSGR_REPO:-$HOME/hsgr-obligation-ledger}"
PROJECT_SCR="${HSGR_PROJECT_SCR:-/mnt/scratch/z/${USER}/hsgr-obligation-ledger}"
RUNTIME_SCR="${HSGR_RUNTIME_SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
source "$REPO/prepare_policy_representation_env_p0p.sh"
p0p_prepare_env "$PROJECT_SCR" || exit 1
trap p0p_cleanup_env EXIT
PY="$P0P_ENV_PYTHON"
OUT_DIR="$PROJECT_SCR/results/policy_representation_selection_audit_p0p_${SLURM_JOB_ID}"

[ -x "$PY" ] || { echo "FATAL: no project Python at $PY"; exit 1; }
mkdir -p "$OUT_DIR" "$PROJECT_SCR/tmp" "$PROJECT_SCR/logs"
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git -C "$REPO" rev-parse --short HEAD)"
echo "repo=$REPO python=$PY out=$OUT_DIR"
cd "$REPO"
export HF_HOME="$RUNTIME_SCR/hf"
export TORCH_HOME="$RUNTIME_SCR/torch"
export XDG_CACHE_HOME="$RUNTIME_SCR/xdg"
export TMPDIR="$PROJECT_SCR/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$PY" -X utf8 policy_representation_swap_p0p.py \
  --dataset-cache "$PROJECT_SCR/data/refinebench_cache" \
  --out-dir "$OUT_DIR" \
  --query-char-limit 8000 \
  --selection-audit-only
sha256sum "$OUT_DIR/p0p_selection_audit.json"
