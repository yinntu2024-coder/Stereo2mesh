#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency: pyyaml. Install with `pip install pyyaml`.") from exc
    return yaml.safe_load(path.read_text())


def iter_jsonl(path: Path):
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def extrapolate(history: list[list[float]], horizon: int, noise: float) -> list[tuple[float, float]]:
    if len(history) < 2:
        lon, lat = history[-1]
        return [(lon, lat) for _ in range(horizon)]
    lon1, lat1 = history[-2]
    lon2, lat2 = history[-1]
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    out = []
    for t in range(1, horizon + 1):
        out.append((lon2 + t * dlon + random.uniform(-noise, noise), lat2 + t * dlat + random.uniform(-noise, noise)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GRPO-style reward selection baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="JSONL with target_history and future")
    args = parser.parse_args()

    from repro_shiptraj_r1.metrics import ade
    from repro_shiptraj_r1.rewards import accuracy_reward

    cfg = load_yaml(Path(args.config))
    records = list(iter_jsonl(Path(args.input)))

    num_generations = int(cfg.get("num_generations", 4))
    rewards = []
    ades = []

    for rec in records:
        history = rec["target_history"]
        future = [tuple(x) for x in rec["future"]]

        cands = [extrapolate(history, horizon=len(future), noise=0.002) for _ in range(num_generations)]
        cand_rewards = [accuracy_reward(c, future) for c in cands]
        best_idx = max(range(len(cands)), key=lambda i: cand_rewards[i])
        best = cands[best_idx]

        rewards.append(cand_rewards[best_idx])
        ades.append(ade(best, future))

    summary = {
        "stage": "grpo_reward_select",
        "model_name": cfg.get("model_name"),
        "samples": len(records),
        "num_generations": num_generations,
        "avg_best_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "avg_best_ade_m": sum(ades) / len(ades) if ades else 0.0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
