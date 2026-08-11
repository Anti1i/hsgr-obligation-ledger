#!/usr/bin/env bash
#SBATCH --job-name=dch-hsgrrc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/scratch/z/%u/dch-hsgr/logs/%x-%j.out

set -uo pipefail
REPO="$HOME/dch-hsgr"
SCR="/mnt/scratch/z/${USER}/dch-hsgr"
WORK="$SCR/work-${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
PY="$SCR/venv/bin/python"

export HF_HOME="$SCR/hf"
export TORCH_HOME="$SCR/torch"
export XDG_CACHE_HOME="$SCR/xdg"
export TMPDIR="$SCR/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset LD_LIBRARY_PATH || true

[ -x "$PY" ] || { echo "FATAL: no venv at $PY"; exit 1; }
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || {
  echo "FATAL: Slurm did not set CUDA_VISIBLE_DEVICES"; exit 1;
}
[ -f "$SCR/data/musique_ans_val.jsonl" ] || {
  echo "FATAL: missing $SCR/data/musique_ans_val.jsonl"; exit 1;
}
for FEATURE in \
  structured_hidden_features_726228.pt \
  dual_route_hidden_features_726354.pt \
  transition_fresh_features_726389.pt; do
  [ -f "$SCR/logs/$FEATURE" ] || {
    echo "FATAL: missing $SCR/logs/$FEATURE"; exit 1;
  }
done

mkdir -p "$WORK" "$TMPDIR" "$SCR/logs"
COMMIT="$(git -C "$REPO" rev-parse --short HEAD)"
export HSGR_RC_OUT="$SCR/cache-route-counterfactual-v1-$COMMIT"
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$COMMIT"
echo "work=$WORK cache=$HSGR_RC_OUT CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || exit 2

cp -r "$REPO/." "$WORK/"
cd "$WORK" || exit 1
mkdir -p data
cp -f "$SCR/data/musique_ans_val.jsonl" data/

"$PY" -m unittest -v test_route_counterfactual.py || exit 3

bash run_hsgr_route_counterfactual_guide.sh \
  2>&1 | tee "$SCR/logs/hsgr_route_counterfactual_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
for SOURCE in \
  "$HSGR_RC_OUT/route_counterfactual_accounting.json" \
  "$HSGR_RC_OUT/eval/route_counterfactual_report.json" \
  "$HSGR_RC_OUT/eval/route_counterfactual_cases.jsonl"; do
  if [ -f "$SOURCE" ]; then
    NAME="$(basename "$SOURCE")"
    STEM="${NAME%.*}"
    EXT="${NAME##*.}"
    cp -f "$SOURCE" "$SCR/logs/${STEM}_${SLURM_JOB_ID}.${EXT}"
  fi
done
echo "feature_cache=$HSGR_RC_OUT/route_counterfactual_features.pt"
echo "JOB_DONE rc=$RC"
exit "$RC"
