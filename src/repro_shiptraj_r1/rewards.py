from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

from .colregs import colregs_risk_reward
from .metrics import ade, geodesic_distance_m
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


def cvar_accuracy_reward(pred_traj: Sequence[Point], gt_traj: Sequence[Point], alpha: float = 0.5, scale_m: float = 1000.0) -> float:
    """Tail-risk aware reward: reward on worst-alpha trajectory errors (CVaR-style)."""
    if len(pred_traj) != len(gt_traj) or not pred_traj:
        raise ValueError("pred_traj and gt_traj must have same non-zero length")
    errors = sorted(geodesic_distance_m(p, g) for p, g in zip(pred_traj, gt_traj))
    k = max(1, math.ceil(len(errors) * alpha))
    tail_mean = sum(errors[-k:]) / k
    return 1.0 / (1.0 + tail_mean / max(scale_m, 1e-6))


def colregs_reward(pred_traj: Sequence[Point], nei_traj: Sequence[Point], safety_dcpa_m: float = 500.0) -> float:
    """Rule-based COLREGs risk reward in [0,1]."""
    return colregs_risk_reward(pred_traj, nei_traj, safety_dcpa_m=safety_dcpa_m)


def multi_colregs_reward(
    pred_traj: Sequence[Point],
    neighbor_trajs: Iterable[Sequence[Point]],
    *,
    safety_dcpa_m: float = 500.0,
    mode: str = "softmin",
    temperature: float = 8.0,
) -> float:
    vals = [colregs_reward(pred_traj, n, safety_dcpa_m=safety_dcpa_m) for n in neighbor_trajs]
    if not vals:
        return 1.0
    if mode == "min":
        return min(vals)
    if mode == "mean":
        return sum(vals) / len(vals)
    if mode != "softmin":
        raise ValueError("mode must be one of: min, mean, softmin")

    weights = [math.exp(-temperature * v) for v in vals]
    z = sum(weights)
    return sum(v * w for v, w in zip(vals, weights)) / max(z, 1e-9)


def combined_reward(
    text: str,
    gt_traj: Sequence[Point],
    *,
    neighbor_traj: Sequence[Point] | None = None,
    neighbor_trajs: Iterable[Sequence[Point]] | None = None,
    w_format: float = 0.3,
    w_acc: float = 0.5,
    w_colregs: float = 0.2,
    w_cvar: float = 0.0,
    cvar_alpha: float = 0.5,
    safety_dcpa_m: float = 500.0,
    colregs_mode: str = "softmin",
) -> float:
    traj = parse_answer_trajectory(text)
    r_f = format_reward(text, expected_len=len(gt_traj))
    r_a = accuracy_reward(traj, gt_traj)
    r_v = cvar_accuracy_reward(traj, gt_traj, alpha=cvar_alpha)

    if neighbor_trajs is not None:
        r_c = multi_colregs_reward(traj, neighbor_trajs, safety_dcpa_m=safety_dcpa_m, mode=colregs_mode)
    elif neighbor_traj is not None:
        r_c = colregs_reward(traj, neighbor_traj, safety_dcpa_m=safety_dcpa_m)
    else:
        r_c = 1.0

    weight_sum = max(w_format + w_acc + w_colregs + w_cvar, 1e-6)
    return (w_format * r_f + w_acc * r_a + w_colregs * r_c + w_cvar * r_v) / weight_sum
