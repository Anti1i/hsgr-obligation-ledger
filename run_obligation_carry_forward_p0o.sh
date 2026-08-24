#!/usr/bin/env bash
set -euo pipefail

PROJECT_SCR="${HSGR_PROJECT_SCR:-/mnt/scratch/z/${USER}/hsgr-obligation-ledger}"
RUNTIME_SCR="${HSGR_RUNTIME_SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
PY="${HSGR_PYTHON:-$RUNTIME_SCR/venv/bin/python}"
P0N_DIR="${REFINEBENCH_P0N_SOURCE:-$PROJECT_SCR/results/refinebench_revision_audit_p0n_753693}"
OUT_DIR="${OBLIGATION_P0O_OUT:-$PROJECT_SCR/results/obligation_carry_forward_p0o_${SLURM_JOB_ID:-manual}}"

[ -x "$PY" ] || { echo "FATAL: no project Python at $PY" >&2; exit 1; }
[ -f "$P0N_DIR/p0n_initial_judgments.jsonl" ] || {
  echo "FATAL: missing frozen P0n source: $P0N_DIR" >&2
  exit 2
}
for model in Qwen3-8B Qwen2.5-14B-Instruct; do
  [ -d "$RUNTIME_SCR/hf/hub/models--Qwen--$model" ] || {
    echo "FATAL: cached model missing: $model" >&2
    exit 3
  }
done
mkdir -p "$OUT_DIR" "$PROJECT_SCR/tmp"
export HF_HOME="${HSGR_HF_HOME:-$RUNTIME_SCR/hf}"
export TORCH_HOME="${HSGR_TORCH_HOME:-$RUNTIME_SCR/torch}"
export XDG_CACHE_HOME="${HSGR_XDG_CACHE_HOME:-$RUNTIME_SCR/xdg}"
export TMPDIR="$PROJECT_SCR/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

"$PY" -X utf8 -m unittest -v test_qwen3_non_thinking.py \
  test_refinebench_revision_audit_p0n.py test_obligation_carry_forward_p0o.py
"$PY" -X utf8 obligation_carry_forward_p0o.py \
  --p0n-result-dir "$P0N_DIR" --out-dir "$OUT_DIR" \
  --generator-model Qwen/Qwen3-8B --judge-model Qwen/Qwen2.5-14B-Instruct \
  --replicates 5 --generation-cap 2048 --judge-cap 256 \
  --temperature 0.7 --top-p 0.9
sha256sum "$OUT_DIR/p0o_report.json"
for artifact in p0o_transition_rows.jsonl p0o_manual_review.jsonl; do
  if [ -f "$OUT_DIR/$artifact" ]; then
    sha256sum "$OUT_DIR/$artifact"
  fi
done
