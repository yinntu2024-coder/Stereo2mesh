#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python tools/train_grpo.py \
  --config configs/shiptraj_r1_grpo.yaml \
  --input data/sample_shiptraj.jsonl
