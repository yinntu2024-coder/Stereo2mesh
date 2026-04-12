#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
