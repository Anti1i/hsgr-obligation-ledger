#!/usr/bin/env bash
set -euo pipefail

SCR="${SCR:-/mnt/scratch/z/${USER}/dch-hsgr}"
OUT="$SCR/data/asqa_eval_gtr_top100_reranked_oracle.json"
TMP="$SCR/tmp/asqa_eval_gtr_top100_reranked_oracle.${SLURM_JOB_ID:-manual}.json"
URL="https://huggingface.co/datasets/princeton-nlp/ALCE-data/resolve/main/ALCE-data.tar"
MEMBER="ALCE-data/asqa_eval_gtr_top100_reranked_oracle.json"

mkdir -p "$SCR/data" "$SCR/tmp" "$SCR/logs"

validate() {
  "${PY:-python}" -X utf8 - "$1" <<'PYEOF'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(data, list) or len(data) < 900:
    raise SystemExit(f"invalid ALCE ASQA object: type={type(data).__name__} len={len(data) if hasattr(data, '__len__') else 'NA'}")
first = data[0]
if not isinstance(first, dict):
    raise SystemExit("first ALCE ASQA record is not an object")
print("validated", path, "bytes", path.stat().st_size, "records", len(data), "keys", sorted(first))
PYEOF
}

if [ -s "$OUT" ]; then
  validate "$OUT"
  sha256sum "$OUT"
  exit 0
fi

echo "streaming $MEMBER from official ALCE archive"
curl --fail --location --retry 4 --retry-delay 3 "$URL" \
  | tar -xOf - "$MEMBER" > "$TMP"
validate "$TMP"
sha256sum "$TMP"
mv "$TMP" "$OUT"
echo "saved=$OUT"

