#!/usr/bin/env bash
set -euo pipefail

# Reward-composition candidate selection demo
PYTHONPATH=src python tools/train_grpo.py \
  --config configs/shiptraj_r1_grpo.yaml \
  --input data/sample_shiptraj.jsonl

# Tiny policy-gradient GRPO demo (actual parameter update)
PYTHONPATH=src python tools/train_grpo_policy.py \
  --input data/sample_shiptraj.jsonl \
  --epochs 15 \
  --num-samples 8 \
  --lr 1e-2 \
  --seed 42
