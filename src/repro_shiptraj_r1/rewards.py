from __future__ import annotations

from typing import Sequence, Tuple

from .metrics import ade
from .prompting import parse_answer_trajectory

Point = Tuple[float, float]


def format_reward(text: str, expected_len: int) -> float:
    """Return 1.0 if output follows <think>/<answer> and trajectory length, else 0.0."""
    if "<think>" not in text or "</think>" not in text:
        return 0.0
    if "<answer>" not in text or "</answer>" not in text:
        return 0.0
    try:
        traj = parse_answer_trajectory(text)
    except Exception:
        return 0.0
    return 1.0 if len(traj) == expected_len else 0.0


def accuracy_reward(pred_traj: Sequence[Point], gt_traj: Sequence[Point], scale_m: float = 1000.0) -> float:
    """Convert ADE to bounded reward in (0, 1]."""
    err = ade(pred_traj, gt_traj)
    return 1.0 / (1.0 + err / max(scale_m, 1e-6))


def combined_reward(text: str, gt_traj: Sequence[Point], w_format: float = 0.3, w_acc: float = 0.7) -> float:
    traj = parse_answer_trajectory(text)
    r_f = format_reward(text, expected_len=len(gt_traj))
    r_a = accuracy_reward(traj, gt_traj)
    return w_format * r_f + w_acc * r_a
