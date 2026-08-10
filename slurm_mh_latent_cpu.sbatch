#!/usr/bin/env bash
#SBATCH --job-name=dch-mhlat2
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/scratch/z/%u/dch-hsgr/logs/%x-%j.out

# Probe stage only: reuses the cached hidden features, so no GPU is needed.
set -uo pipefail
REPO="$HOME/dch-hsgr"
SCR="/mnt/scratch/z/${USER}/dch-hsgr"
PY="$SCR/venv/bin/python"

export HF_HOME="$SCR/hf"
export XDG_CACHE_HOME="$SCR/xdg"
export TMPDIR="$SCR/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset LD_LIBRARY_PATH || true

# Reuse the GPU run's workdir so hidden_feats.pt and data are already there.
W=""
for d in "$SCR"/work-dch-mhlat-*; do
  [ -f "$d/mh_latent/hidden_feats.pt" ] && W="$d"
done
[ -n "$W" ] || { echo "FATAL: no cached hidden_feats.pt"; exit 1; }
echo "reusing $W"
cd "$W" || exit 1
cp -f "$REPO/mh_latent_rerank.py" . || exit 1

"$PY" mh_latent_rerank.py --data data/musique_ans_val.jsonl --out-dir mh_latent \
  --limit 200 --folds 5 --probe-only 2>&1 | tee "$SCR/logs/mh_latent2.txt"
RC=$?
cp -f mh_latent_report.json "$SCR/logs/mh_latent_report2.json" 2>/dev/null || true
echo "JOB_DONE rc=$RC"
exit "$RC"
