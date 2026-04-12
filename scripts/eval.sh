#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from repro_shiptraj_r1.metrics import ade, fde

pred = [(120.10, 30.20), (120.15, 30.22), (120.20, 30.25)]
gt   = [(120.11, 30.19), (120.14, 30.23), (120.22, 30.26)]

print("ADE(m):", round(ade(pred, gt), 3))
print("FDE(m):", round(fde(pred, gt), 3))
PY
