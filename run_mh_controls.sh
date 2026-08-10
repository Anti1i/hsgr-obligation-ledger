#!/usr/bin/env bash
# MuSiQue E0 controls: isolate structure from retrieval supervision.
#   A) evidence=all  -> hop executor sees the SAME evidence as SC baseline
#   B) token accounting (prompt+gen) for a budget-fair comparison
#   C) paired McNemar significance on the original evidence=hop run
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs data

PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
LIMIT=${LIMIT:-200}
REUSE=${REUSE_SC:-}
PRIOR=${PRIOR_E0_DIR:-}

echo "=== ensure musique jsonl ==="
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
$PY -m pip install --quiet datasets pyarrow 2>/dev/null || true
$PY fetch_data.py --which musique
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE_KEEP:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_KEEP:-1}

# Seed the original evidence=hop outputs so both arms share SC samples and ids.
if [ -n "$PRIOR" ] && [ -d "$PRIOR" ]; then
  mkdir -p mh_e0
  cp -n "$PRIOR"/*.jsonl mh_e0/ 2>/dev/null || true
  echo "seeded mh_e0 from $PRIOR"
fi

echo "=== control A/B: evidence=all ==="
$PY mh_e0.py --data data/musique_ans_val.jsonl --out-dir mh_e0 \
  --limit "$LIMIT" --model "$MODEL" --evidence-mode all \
  ${REUSE:+--reuse-sc "$REUSE"} \
  | tee logs/mh_e0_evall.txt

echo "=== control B': token accounting for evidence=hop ==="
$PY mh_e0.py --data data/musique_ans_val.jsonl --out-dir mh_e0 \
  --limit "$LIMIT" --model "$MODEL" --evidence-mode hop \
  ${REUSE:+--reuse-sc "$REUSE"} \
  | tee logs/mh_e0_evhop.txt

echo "=== control C: paired significance + SC@k budget curve ==="
$PY mh_e0_stats.py --dir mh_e0 --data data/musique_ans_val.jsonl \
  | tee logs/mh_e0_stats_evhop.txt
$PY mh_e0_stats.py --dir mh_e0 --data data/musique_ans_val.jsonl --suffix _evall \
  | tee logs/mh_e0_stats_evall.txt
