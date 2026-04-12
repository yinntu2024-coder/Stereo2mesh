#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def iter_jsonl(path: Path):
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLP ES policy with GRPO-style updates.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from repro_shiptraj_r1.policy_mlp_es import MLPESPolicy, grpo_es_step

    data = list(iter_jsonl(Path(args.input)))
    policy = MLPESPolicy(hidden=8)

    history = []
    for epoch in range(args.epochs):
        rs = []
        for i, rec in enumerate(data):
            hist = [tuple(x) for x in rec["target_history"]]
            fut = [tuple(x) for x in rec["future"]]
            stats = grpo_es_step(policy, hist, fut, num_samples=args.num_samples, lr=args.lr, seed=args.seed + epoch * 101 + i)
            rs.append(stats["mean_reward"])
        history.append({"epoch": epoch, "mean_reward": sum(rs) / len(rs)})

    print(json.dumps({"epochs": args.epochs, "history_tail": history[-5:]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
