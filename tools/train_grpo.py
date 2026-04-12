#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _parse_scalar(v: str):
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        out = []
        for item in inner.split(","):
            item = item.strip()
            try:
                out.append(int(item))
                continue
            except ValueError:
                pass
            try:
                out.append(float(item))
                continue
            except ValueError:
                pass
            out.append(item)
        return out
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _minimal_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current_map = root
    stack: list[tuple[int, dict[str, Any]]] = [(0, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        while len(stack) > 1 and indent < stack[-1][0]:
            stack.pop()
        current_map = stack[-1][1]
        if line.endswith(":"):
            key = line[:-1].strip()
            new_map: dict[str, Any] = {}
            current_map[key] = new_map
            stack.append((indent + 2, new_map))
            continue
        key, val = line.split(":", 1)
        current_map[key.strip()] = _parse_scalar(val)
    return root


def load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text()
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except Exception:
        return _minimal_yaml(text)


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


def as_answer_text(traj: list[tuple[float, float]]) -> str:
    return f'<think>candidate rollout</think><answer>{{"trajectory": {json.dumps(traj)} }}</answer>'


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GRPO-style reward selection baseline with optional COLREGs/CVaR reward.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="JSONL with target_history, neighbor_histories and future")
    args = parser.parse_args()

    from repro_shiptraj_r1.metrics import ade
    from repro_shiptraj_r1.rewards import combined_reward

    cfg = load_yaml(Path(args.config))
    records = list(iter_jsonl(Path(args.input)))

    num_generations = int(cfg.get("num_generations", 4))
    reward_cfg = cfg.get("reward", {})
    w_format = float(reward_cfg.get("w_format", 0.3))
    w_acc = float(reward_cfg.get("w_acc", 0.5))
    w_colregs = float(reward_cfg.get("w_colregs", 0.2))
    w_cvar = float(reward_cfg.get("w_cvar", 0.0))
    cvar_alpha = float(reward_cfg.get("cvar_alpha", 0.5))
    safety_dcpa_m = float(reward_cfg.get("safety_dcpa_m", 500.0))
    colregs_mode = str(reward_cfg.get("colregs_mode", "softmin"))

    rewards = []
    ades = []

    for rec in records:
        history = rec["target_history"]
        future = [tuple(x) for x in rec["future"]]

        neighbors = rec.get("neighbor_histories", [])
        neighbor_futures = [extrapolate(n, horizon=len(future), noise=0.0) for n in neighbors if n]

        cands = [extrapolate(history, horizon=len(future), noise=0.002) for _ in range(num_generations)]
        cand_rewards = [
            combined_reward(
                as_answer_text(c),
                future,
                neighbor_trajs=neighbor_futures,
                w_format=w_format,
                w_acc=w_acc,
                w_colregs=w_colregs,
                w_cvar=w_cvar,
                cvar_alpha=cvar_alpha,
                safety_dcpa_m=safety_dcpa_m,
                colregs_mode=colregs_mode,
            )
            for c in cands
        ]

        best_idx = max(range(len(cands)), key=lambda i: cand_rewards[i])
        best = cands[best_idx]

        rewards.append(cand_rewards[best_idx])
        ades.append(ade(best, future))

    summary = {
        "stage": "grpo_reward_select",
        "model_name": cfg.get("model_name"),
        "samples": len(records),
        "num_generations": num_generations,
        "w_format": w_format,
        "w_acc": w_acc,
        "w_colregs": w_colregs,
        "w_cvar": w_cvar,
        "colregs_mode": colregs_mode,
        "avg_best_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "avg_best_ade_m": sum(ades) / len(ades) if ades else 0.0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
