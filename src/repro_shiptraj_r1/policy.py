from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence, Tuple

from .metrics import geodesic_distance_m

Point = Tuple[float, float]


@dataclass
class GaussianVelocityPolicy:
    """Tiny policy for trajectory rollout with REINFORCE-style updates.

    mean step = theta * last_velocity
    action = mean + Normal(0, sigma)
    """

    theta: float = 1.0
    sigma: float = 0.001

    def rollout(self, history: Sequence[Point], horizon: int) -> list[Point]:
        if len(history) < 2:
            raise ValueError("history length must be >= 2")
        (lon1, lat1), (lon2, lat2) = history[-2], history[-1]
        vlon, vlat = lon2 - lon1, lat2 - lat1
        cur_lon, cur_lat = lon2, lat2
        out: list[Point] = []
        for _ in range(horizon):
            dlon = self.theta * vlon + random.gauss(0.0, self.sigma)
            dlat = self.theta * vlat + random.gauss(0.0, self.sigma)
            cur_lon += dlon
            cur_lat += dlat
            out.append((cur_lon, cur_lat))
        return out

    def logprob_grad_theta(self, history: Sequence[Point], traj: Sequence[Point]) -> float:
        (lon1, lat1), (lon2, lat2) = history[-2], history[-1]
        vlon, vlat = lon2 - lon1, lat2 - lat1
        prev_lon, prev_lat = lon2, lat2
        grad = 0.0
        var = max(self.sigma * self.sigma, 1e-12)
        for lon, lat in traj:
            dlon = lon - prev_lon
            dlat = lat - prev_lat
            grad += ((dlon - self.theta * vlon) * vlon + (dlat - self.theta * vlat) * vlat) / var
            prev_lon, prev_lat = lon, lat
        return grad


def trajectory_reward(pred: Sequence[Point], gt: Sequence[Point], scale_m: float = 1000.0) -> float:
    if len(pred) != len(gt) or not pred:
        return 0.0
    err = sum(geodesic_distance_m(p, g) for p, g in zip(pred, gt)) / len(pred)
    return 1.0 / (1.0 + err / scale_m)


def grpo_step(
    policy: GaussianVelocityPolicy,
    history: Sequence[Point],
    gt_future: Sequence[Point],
    *,
    num_samples: int = 8,
    lr: float = 1e-2,
) -> dict:
    samples = [policy.rollout(history, horizon=len(gt_future)) for _ in range(num_samples)]
    rewards = [trajectory_reward(s, gt_future) for s in samples]
    mu = sum(rewards) / len(rewards)
    var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var + 1e-8)
    advantages = [(r - mu) / std for r in rewards]

    grad = 0.0
    for a, traj in zip(advantages, samples):
        grad += a * policy.logprob_grad_theta(history, traj)
    grad /= len(samples)

    policy.theta += lr * grad
    return {
        "mean_reward": mu,
        "std_reward": std,
        "grad": grad,
        "theta": policy.theta,
    }
