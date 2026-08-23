#!/usr/bin/env bash
set -euo pipefail

PROJECT_SCR="${HSGR_PROJECT_SCR:-/mnt/scratch/z/${USER}/hsgr-obligation-ledger}"
RUNTIME_SCR="${HSGR_RUNTIME_SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
PY="${HSGR_PYTHON:-$RUNTIME_SCR/venv/bin/python}"
P0D_DIR="${GAMUT_P0D_DIR:-$PROJECT_SCR/results/gamut_long_baseline_p0d_751249}"
P0E_DIR="${GAMUT_P0E_DIR:-$PROJECT_SCR/results/gamut_manual_repair_p0e_751359}"
OUT_DIR="${GAMUT_P0F_OUT:-$PROJECT_SCR/results/gamut_repair_dynamics_p0f_751359}"

[ -x "$PY" ] || { echo "FATAL: project Python missing at $PY" >&2; exit 1; }
[ -s "$P0D_DIR/gamut_long_baseline_p0d_rows.jsonl" ] || { echo "FATAL: P0d rows missing" >&2; exit 2; }
[ -s "$P0E_DIR/gamut_manual_repair_p0e_candidates.jsonl" ] || { echo "FATAL: P0e candidates missing" >&2; exit 3; }
mkdir -p "$OUT_DIR"

"$PY" -X utf8 -m unittest -v test_gamut_repair_dynamics_p0f.py
"$PY" -X utf8 gamut_repair_dynamics_p0f.py \
  --p0e-candidates "$P0E_DIR/gamut_manual_repair_p0e_candidates.jsonl" \
  --p0d-rows "$P0D_DIR/gamut_long_baseline_p0d_rows.jsonl" \
  --out-dir "$OUT_DIR"
sha256sum "$OUT_DIR/gamut_repair_dynamics_p0f_report.json" \
  "$OUT_DIR/gamut_repair_dynamics_p0f_rows.jsonl" \
  "$OUT_DIR/gamut_repair_dynamics_p0f_review.md"

