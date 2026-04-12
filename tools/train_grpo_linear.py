#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def iter_jsonl(path: Path):
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train linear-feature policy with GRPO-style update.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from repro_shiptraj_r1.policy_linear import LinearFeaturePolicy, grpo_linear_step

    random.seed(args.seed)
    data = list(iter_jsonl(Path(args.input)))
    policy = LinearFeaturePolicy()

    log = []
    for epoch in range(args.epochs):
        rs = []
        for rec in data:
            hist = [tuple(x) for x in rec["target_history"]]
            fut = [tuple(x) for x in rec["future"]]
            stats = grpo_linear_step(policy, hist, fut, num_samples=args.num_samples, lr=args.lr)
            rs.append(stats["mean_reward"])
        log.append({"epoch": epoch, "mean_reward": sum(rs) / len(rs), "w": policy.w})

    print(json.dumps({"epochs": args.epochs, "final_w": policy.w, "tail": log[-5:]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
