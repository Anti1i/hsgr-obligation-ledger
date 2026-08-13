#!/usr/bin/env bash
set -euo pipefail

"${PY:-python}" -X utf8 -m unittest -v test_dependency_route_apparatus_p3a.py
"${PY:-python}" -X utf8 dependency_route_apparatus_p3a.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --seed 20260819 \
  --n 96 \
  --batch-size 32 \
  --output dependency_route_apparatus_p3a_report.json
