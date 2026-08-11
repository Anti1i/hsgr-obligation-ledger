#!/usr/bin/env bash
#SBATCH --job-name=dch-hsgrfi
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
from hsgr_listwise_guide_verifier import LAYERS, make_model

n = 4
data = {
    "scalar": torch.randn(n, 5),
    "features": {
        route: {layer: torch.randn(n, 256) for layer in LAYERS}
        for route in ("correct", "wrong")
    },
    "rotation": torch.tensor([1, 0, 3, 2]),
    "hops": torch.tensor([2, 2, 4, 4]),
    "permuted_hops": torch.tensor([4, 4, 2, 2]),
}
model = make_model(torch, hidden=True, depth_conditioned=True)
for mode in ("normal", "swap", "mismatch", "depth_permute"):
    assert tuple(model(data, route_mode=mode).shape) == (n,)
assert model.depth_regularization() is not None
print("SELFTEST_OK", flush=True)
PY
if [ "$?" -ne 0 ]; then
  echo "FATAL: compute-node self-test failed"
  exit 1
fi

bash run_hsgr_depth_conditioned_guide_verifier.sh \
  2>&1 | tee "$SCR/logs/hsgr_depth_conditioned_guide_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
for NAME in listwise_guide_report.json listwise_guide_cases.jsonl; do
  STEM="${NAME%.*}"
  EXT="${NAME##*.}"
  cp -f "hsgr_depth_conditioned_guide_verifier/$NAME" \
    "$SCR/logs/depth_conditioned_${STEM}_${SLURM_JOB_ID}.${EXT}" \
    2>/dev/null || true
done
echo "JOB_DONE rc=$RC"
exit "$RC"
