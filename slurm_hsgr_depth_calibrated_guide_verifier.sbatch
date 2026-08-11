#!/usr/bin/env bash
#SBATCH --job-name=dch-hsgrdc
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
from hsgr_listwise_guide_verifier import select_policy_with_weights

metas = [
    {
        "id": "p2", "norm": "a", "gold": "a", "aliases": [],
        "mentions": {"correct": 0},
    },
    {
        "id": "p2", "norm": "b", "gold": "a", "aliases": [],
        "mentions": {"correct": 0},
    },
    {
        "id": "p4", "norm": "c", "gold": "d", "aliases": [],
        "mentions": {"correct": 0},
    },
    {
        "id": "p4", "norm": "d", "gold": "d", "aliases": [],
        "mentions": {"correct": 0},
    },
]
state = {"p2": {"values": set()}, "p4": {"values": set()}}
outcomes, _ = select_policy_with_weights(
    metas, state, [1.0, 0.0, 0.0, 1.0], {2: 0.0, 4: 1.0},
    {"p2": 2, "p4": 4}, routed=False, explicit=False,
)
assert outcomes == {"p2": True, "p4": True}
print("SELFTEST_OK", flush=True)
PY
if [ "$?" -ne 0 ]; then
  echo "FATAL: compute-node self-test failed"
  exit 1
fi

bash run_hsgr_depth_calibrated_guide_verifier.sh \
  2>&1 | tee "$SCR/logs/hsgr_depth_calibrated_guide_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
for NAME in listwise_guide_report.json listwise_guide_cases.jsonl; do
  STEM="${NAME%.*}"
  EXT="${NAME##*.}"
  cp -f "hsgr_depth_calibrated_guide_verifier/$NAME" \
    "$SCR/logs/depth_calibrated_${STEM}_${SLURM_JOB_ID}.${EXT}" \
    2>/dev/null || true
done
echo "JOB_DONE rc=$RC"
exit "$RC"
