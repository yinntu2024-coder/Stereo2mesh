#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency: pyyaml. Install with `pip install pyyaml`.") from exc
    return yaml.safe_load(path.read_text())


def iter_jsonl(path: Path):
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {i} of {path}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT prompts for ShipTraj-R1 style training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="JSONL with target_history, neighbor_histories, future")
    parser.add_argument("--output", required=True, help="Output JSONL with prompt/response")
    args = parser.parse_args()

    from repro_shiptraj_r1.prompting import build_prompt

    cfg = load_yaml(Path(args.config))
    records = list(iter_jsonl(Path(args.input)))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            target_history = rec["target_history"]
            neighbors = rec.get("neighbor_histories", [])
            future = rec["future"]
            prompt = build_prompt(target_history, neighbors, tpred=len(future))
            response = f'<think>trajectory reasoning</think><answer>{{"trajectory": {json.dumps(future)} }}</answer>'
            f.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")
            num_written += 1

    summary = {
        "stage": "sft_prepare",
        "model_name": cfg.get("model_name"),
        "global_batch_size": cfg.get("global_batch_size"),
        "samples": num_written,
        "output": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
