#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python tools/train_sft.py \
  --config configs/shiptraj_r1_sft.yaml \
  --input data/sample_shiptraj.jsonl \
  --output artifacts/sft_ready.jsonl
