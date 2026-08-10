#!/usr/bin/env bash
# E2': latent verifier re-ranks the cached MuSiQue 8-sample pool.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs data mh_latent

PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
LIMIT=${LIMIT:-200}
SC_DIR=${SC_DIR:-}

echo "=== ensure musique jsonl ==="
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
$PY -m pip install --quiet datasets pyarrow 2>/dev/null || true
$PY fetch_data.py --which musique
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE_KEEP:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_KEEP:-1}

echo "=== E2' latent verifier rerank ==="
ARGS=(--data data/musique_ans_val.jsonl --out-dir mh_latent
      --limit "$LIMIT" --model "$MODEL" --bs 8 --folds 5)
if [ -n "$SC_DIR" ]; then
  ARGS+=(--sc-dir "$SC_DIR")
fi
$PY mh_latent_rerank.py "${ARGS[@]}" | tee logs/mh_latent.txt
echo "DONE"
