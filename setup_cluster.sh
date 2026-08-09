#!/bin/bash
# One-time environment setup: project-local venv + model weights.
# Run from the repo root. Safe to re-run; pip and hf both skip existing work.
set -u

VENV="${VENV:-$HOME/venvs/dch-hsgr}"
PY="$VENV/bin/python"

echo "=== venv: $VENV ==="
[ -x "$PY" ] || python3 -m venv "$VENV"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet torch transformers accelerate sympy huggingface_hub

echo "=== versions ==="
"$PY" - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
EOF

echo "=== model weights -> ${HF_HOME:-$HOME/.cache/huggingface} ==="
for m in Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-1.5B-Instruct; do
  echo "--- $m ---"
  "$VENV/bin/hf" download "$m" || "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download('$m')
"
done

echo "=== disk used by HF cache ==="
du -sh "${HF_HOME:-$HOME/.cache/huggingface}"
echo "SETUP_DONE"
