from __future__ import annotations

import json
import re
from typing import Iterable, Sequence, Tuple

Point = Tuple[float, float]


def build_prompt(target_history: Sequence[Point], neighbor_histories: Iterable[Sequence[Point]], tpred: int) -> str:
    neighbors = "\n".join([f"- ship{i+1}: {list(hist)}" for i, hist in enumerate(neighbor_histories)])
    return (
        f"Context: target ship past trajectory: {list(target_history)}\n"
        f"Adjacent conflict ships:\n{neighbors}\n"
        f"Question: predict next {tpred} coordinates.\n"
        "Return in format:\n"
        "<think>...</think>\n"
        "<answer>{\"trajectory\": [[lon,lat], ...]}</answer>"
    )


def parse_answer_trajectory(text: str) -> list[Point]:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if not match:
        raise ValueError("missing <answer> block")
    payload = match.group(1).strip()
    obj = json.loads(payload)
    coords = obj.get("trajectory")
    if not isinstance(coords, list):
        raise ValueError("trajectory must be list")
    out: list[Point] = []
    for item in coords:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise ValueError("invalid coordinate format")
        lon, lat = float(item[0]), float(item[1])
        out.append((lon, lat))
    return out
