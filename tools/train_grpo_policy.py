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
    parser = argparse.ArgumentParser(description="Train tiny Gaussian policy with GRPO-style updates.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from repro_shiptraj_r1.policy import GaussianVelocityPolicy, grpo_step

    random.seed(args.seed)
    data = list(iter_jsonl(Path(args.input)))
    policy = GaussianVelocityPolicy(theta=1.0, sigma=0.001)

    history_log = []
    for epoch in range(args.epochs):
        epoch_rewards = []
        for rec in data:
            hist = [tuple(x) for x in rec["target_history"]]
            fut = [tuple(x) for x in rec["future"]]
            stats = grpo_step(policy, hist, fut, num_samples=args.num_samples, lr=args.lr)
            epoch_rewards.append(stats["mean_reward"])

        history_log.append({
            "epoch": epoch,
            "theta": policy.theta,
            "mean_reward": sum(epoch_rewards) / len(epoch_rewards),
        })

    print(json.dumps({
        "final_theta": policy.theta,
        "epochs": args.epochs,
        "history_tail": history_log[-5:],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
