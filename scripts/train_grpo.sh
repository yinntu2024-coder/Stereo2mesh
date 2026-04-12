#!/usr/bin/env bash
set -euo pipefail

# Reward-composition candidate selection demo
PYTHONPATH=src python tools/train_grpo.py \
  --config configs/shiptraj_r1_grpo.yaml \
  --input data/sample_shiptraj.jsonl

# Tiny scalar policy-gradient GRPO demo
PYTHONPATH=src python tools/train_grpo_policy.py \
  --input data/sample_shiptraj.jsonl \
  --epochs 15 \
  --num-samples 8 \
  --lr 1e-2 \
  --seed 42

# Linear-feature policy-gradient GRPO demo
PYTHONPATH=src python tools/train_grpo_linear.py \
  --input data/sample_shiptraj.jsonl \
  --epochs 15 \
  --num-samples 8 \
  --lr 1e-3 \
  --seed 42

# MLP ES policy-gradient GRPO demo (optional numpy)
if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec('numpy') else 1)
PY
then
  PYTHONPATH=src python tools/train_grpo_mlp.py \
    --input data/sample_shiptraj.jsonl \
    --epochs 10 \
    --num-samples 16 \
    --lr 1e-2 \
    --seed 42
else
  echo "[WARN] numpy not installed, skip train_grpo_mlp.py"
fi
