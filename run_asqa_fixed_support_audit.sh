#!/usr/bin/env bash
set -euo pipefail

SCR="${SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
ALCE="${ASQA_ALCE:-$SCR/data/asqa_eval_gtr_top100_reranked_oracle.json}"
ORIGINAL="${ASQA_ORIGINAL:-$SCR/data/ASQA.json}"
OUT_DIR="${ASQA_AUDIT_OUT:-$SCR/results/asqa_fixed_support_p0}"
ORIGINAL_URL="https://storage.googleapis.com/gresearch/ASQA/data/ASQA.json"

mkdir -p "$SCR/data" "$OUT_DIR"

if [ ! -s "$ALCE" ]; then
  echo "FATAL: missing frozen ALCE oracle data at $ALCE" >&2
  echo "Run slurm_fetch_asqa_oracle.sbatch first." >&2
  exit 1
fi

if [ ! -s "$ORIGINAL" ]; then
  tmp="$SCR/data/ASQA.${SLURM_JOB_ID:-manual}.json"
  curl --fail --location --retry 4 --retry-delay 3 --output "$tmp" "$ORIGINAL_URL"
  "${PY:-python}" -X utf8 - "$tmp" <<'PYEOF'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(data, dict) or not isinstance(data.get("dev"), dict) or len(data["dev"]) != 948:
    raise SystemExit("invalid official ASQA JSON")
print("validated", path, "dev_records", len(data["dev"]))
PYEOF
  mv "$tmp" "$ORIGINAL"
fi

"${PY:-python}" -X utf8 -m unittest -v test_asqa_fixed_support_audit.py
"${PY:-python}" -X utf8 asqa_fixed_support_audit.py \
  --alce "$ALCE" \
  --original "$ORIGINAL" \
  --output "$OUT_DIR/report.json" \
  --selected-output "$OUT_DIR/selected_p1_ids.txt"

sha256sum "$ALCE" "$ORIGINAL" "$OUT_DIR/report.json" "$OUT_DIR/selected_p1_ids.txt"
echo "report=$OUT_DIR/report.json"
echo "selected_ids=$OUT_DIR/selected_p1_ids.txt"
