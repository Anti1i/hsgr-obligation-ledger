#!/usr/bin/env bash
#SBATCH --job-name=dch-hsgrhb
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/scratch/z/%u/dch-hsgr/logs/%x-%j.out

set -uo pipefail
REPO="$HOME/dch-hsgr"
SCR="/mnt/scratch/z/${USER}/dch-hsgr"
WORK="$SCR/work-${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
PY="$SCR/venv/bin/python"

export TORCH_HOME="$SCR/torch"
export XDG_CACHE_HOME="$SCR/xdg"
export TMPDIR="$SCR/tmp"
unset LD_LIBRARY_PATH || true

[ -x "$PY" ] || { echo "FATAL: no venv at $PY"; exit 1; }
for FEATURE in \
  structured_hidden_features_726228.pt \
  dual_route_hidden_features_726354.pt \
  transition_fresh_features_726389.pt; do
  [ -f "$SCR/logs/$FEATURE" ] || {
    echo "FATAL: missing $SCR/logs/$FEATURE"; exit 1;
  }
done
mkdir -p "$WORK" "$TMPDIR" "$SCR/logs"
echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git -C "$REPO" rev-parse --short HEAD)"
cp -r "$REPO/." "$WORK/"
cd "$WORK" || exit 1
mkdir -p data
cp -f "$SCR/data/musique_ans_val.jsonl" data/

"$PY" - <<'PY'
import torch
from hsgr_listwise_guide_verifier import (
    hop_class_balanced_bce,
    mean_loss_by_hop,
    pair_accuracy,
)

losses = torch.tensor([1.0, 3.0, 5.0])
hops = torch.tensor([2, 2, 4])
assert abs(float(mean_loss_by_hop(torch, losses, hops)) - 3.5) < 1e-6
logits = torch.tensor([0.0, 0.0, 0.0, 0.0])
labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
balanced = hop_class_balanced_bce(
    torch, logits, labels, torch.tensor([2, 2, 4, 4])
)
assert abs(float(balanced) - 0.69314718) < 1e-6
data = {
    "labels": labels,
    "pids": ["a", "a", "b", "b"],
    "hops": torch.tensor([2, 2, 4, 4]),
}
assert pair_accuracy([2.0, 1.0, 0.0, 1.0], data, range(4), True) == 0.5
print("SELFTEST_OK", flush=True)
PY
if [ "$?" -ne 0 ]; then
  echo "FATAL: compute-node self-test failed"
  exit 1
fi

bash run_hsgr_hop_balanced_guide_verifier.sh \
  2>&1 | tee "$SCR/logs/hsgr_hop_balanced_guide_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
for NAME in listwise_guide_report.json listwise_guide_cases.jsonl; do
  STEM="${NAME%.*}"
  EXT="${NAME##*.}"
  cp -f "hsgr_hop_balanced_guide_verifier/$NAME" \
    "$SCR/logs/hop_balanced_${STEM}_${SLURM_JOB_ID}.${EXT}" \
    2>/dev/null || true
done
echo "JOB_DONE rc=$RC"
exit "$RC"
