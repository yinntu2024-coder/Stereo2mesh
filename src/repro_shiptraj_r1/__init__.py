"""Utilities for reproducing ShipTraj-R1 style trajectory prediction."""

from .colregs import colregs_risk_reward, tcpa_dcpa
from .metrics import ade, fde, geodesic_distance_m
from .policy import GaussianVelocityPolicy, grpo_step, trajectory_reward
from .rewards import (
    accuracy_reward,
    colregs_reward,
    combined_reward,
    cvar_accuracy_reward,
    format_reward,
    multi_colregs_reward,
)
from .prompting import build_prompt, parse_answer_trajectory

__all__ = [
    "ade",
    "fde",
    "geodesic_distance_m",
    "tcpa_dcpa",
    "colregs_risk_reward",
    "format_reward",
    "accuracy_reward",
    "cvar_accuracy_reward",
    "colregs_reward",
    "multi_colregs_reward",
    "combined_reward",
    "GaussianVelocityPolicy",
    "trajectory_reward",
    "grpo_step",
    "build_prompt",
    "parse_answer_trajectory",
]
