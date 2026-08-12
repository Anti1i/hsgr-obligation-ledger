#!/usr/bin/env bash
#SBATCH --job-name=dch-henergy0a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/scratch/z/zitong/dch-hsgr/logs/dch-henergy0a-%j.out

set -euo pipefail
REPO="$HOME/dch-hsgr"
SCR="/mnt/scratch/z/$USER/dch-hsgr"
WORK="$SCR/work-dch-henergy0a-${SLURM_JOB_ID}"
PY="$SCR/venv/bin/python"
SCREEN_DIR=${SCREEN_DIR:-$SCR/work-dch-shard0-728006/structural_hardness_screen}

export HF_HOME="$SCR/hf"
export TORCH_HOME="$SCR/torch"
export XDG_CACHE_HOME="$SCR/xdg"
export TMPDIR="$SCR/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset LD_LIBRARY_PATH || true

[ -x "$PY" ] || { echo "FATAL: no project venv at $PY"; exit 1; }
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || {
  echo "FATAL: Slurm did not set CUDA_VISIBLE_DEVICES"; exit 1;
}
[ -s "$SCREEN_DIR/report.json" ] || {
  echo "FATAL: missing completed screen report under $SCREEN_DIR"; exit 1;
}
export CUDA_VISIBLE_DEVICES
mkdir -p "$WORK" "$TMPDIR" "$SCR/logs"

echo "host=$(hostname) job=${SLURM_JOB_ID} commit=$(git -C "$REPO" rev-parse --short HEAD)"
echo "repo=$REPO work=$WORK screen=$SCREEN_DIR CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || exit 2

cp -r "$REPO/." "$WORK/"
cd "$WORK"

"$PY" -m unittest -v test_hidden_graph_energy_guide.py test_hidden_graph_energy_train.py
"$PY" hidden_graph_energy_guide.py --stage self-test
"$PY" hidden_graph_energy_train.py --self-test
bash -n run_hidden_graph_energy_assignment_audit.sh

export PY SCREEN_DIR
bash run_hidden_graph_energy_assignment_audit.sh 2>&1 \
  | tee "$SCR/logs/hidden_graph_energy_assignment_${SLURM_JOB_ID}.txt"
RC=${PIPESTATUS[0]}
cp -f hidden_graph_energy_guide/assignment_audit_report.json \
  "$SCR/logs/hidden_graph_energy_assignment_report_${SLURM_JOB_ID}.json" 2>/dev/null || true
GATE_PASS="$($PY - <<'PYEOF'
import json
with open("hidden_graph_energy_guide/assignment_audit_report.json", encoding="utf-8") as handle:
    print("1" if json.load(handle).get("gate_pass") else "0")
PYEOF
)"
if [ "$GATE_PASS" = 1 ]; then
  echo "ASSIGNMENT_GATE_PASS: extracting frozen dual-view hidden states"
  "$PY" hidden_graph_energy_guide.py \
    --stage hidden-extract \
    --data data/gsm_join_train.jsonl \
    --screen-dir "$SCREEN_DIR" \
    --out-dir hidden_graph_energy_guide \
    --model Qwen/Qwen2.5-7B-Instruct \
    --batch-size 8 \
    --max-context 2048
  "$PY" hidden_graph_energy_train.py \
    --features hidden_graph_energy_guide/calibration_dual_features.pt \
    --report hidden_graph_energy_guide/oof_report.json
  cp -f hidden_graph_energy_guide/oof_report.json \
    "$SCR/logs/hidden_graph_energy_oof_report_${SLURM_JOB_ID}.json"
else
  echo "ASSIGNMENT_GATE_FAIL: hidden extraction and reader training stopped"
fi
echo "JOB_DONE rc=$RC"
exit "$RC"
