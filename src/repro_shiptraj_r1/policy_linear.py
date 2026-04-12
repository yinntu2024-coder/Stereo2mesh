from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence, Tuple

from .metrics import geodesic_distance_m

Point = Tuple[float, float]


@dataclass
class LinearFeaturePolicy:
    """Slightly stronger policy than scalar-theta model.

    Predict per-step delta via linear map:
    [dlon, dlat] = W @ [1, vlon, vlat]
    """

    w: list[list[float]] = field(default_factory=lambda: [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    sigma: float = 0.001

    def _step(self, vlon: float, vlat: float) -> tuple[float, float]:
        x = [1.0, vlon, vlat]
        dlon = sum(self.w[0][i] * x[i] for i in range(3))
        dlat = sum(self.w[1][i] * x[i] for i in range(3))
        return dlon, dlat

    def rollout(self, history: Sequence[Point], horizon: int) -> list[Point]:
        if len(history) < 2:
            raise ValueError("history length must be >= 2")
        (lon1, lat1), (lon2, lat2) = history[-2], history[-1]
        vlon, vlat = lon2 - lon1, lat2 - lat1

        cur_lon, cur_lat = lon2, lat2
        out: list[Point] = []
        for _ in range(horizon):
            md_lon, md_lat = self._step(vlon, vlat)
            cur_lon += md_lon + random.gauss(0.0, self.sigma)
            cur_lat += md_lat + random.gauss(0.0, self.sigma)
            out.append((cur_lon, cur_lat))
        return out

    def grad_logprob(self, history: Sequence[Point], traj: Sequence[Point]) -> list[list[float]]:
        (lon1, lat1), (lon2, lat2) = history[-2], history[-1]
        vlon, vlat = lon2 - lon1, lat2 - lat1
        x = [1.0, vlon, vlat]
        var = max(self.sigma * self.sigma, 1e-12)

        g = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        p_lon, p_lat = lon2, lat2
        for lon, lat in traj:
            dlon = lon - p_lon
            dlat = lat - p_lat
            md_lon, md_lat = self._step(vlon, vlat)
            for i in range(3):
                g[0][i] += (dlon - md_lon) * x[i] / var
                g[1][i] += (dlat - md_lat) * x[i] / var
            p_lon, p_lat = lon, lat
        return g

    def apply_grad(self, grad: list[list[float]], lr: float) -> None:
        for r in range(2):
            for c in range(3):
                self.w[r][c] += lr * grad[r][c]


def reward(pred: Sequence[Point], gt: Sequence[Point], scale_m: float = 1000.0) -> float:
    if len(pred) != len(gt) or not pred:
        return 0.0
    err = sum(geodesic_distance_m(p, g) for p, g in zip(pred, gt)) / len(pred)
    return 1.0 / (1.0 + err / scale_m)


def grpo_linear_step(policy: LinearFeaturePolicy, history: Sequence[Point], gt_future: Sequence[Point], num_samples: int = 8, lr: float = 1e-3) -> dict:
    samples = [policy.rollout(history, horizon=len(gt_future)) for _ in range(num_samples)]
    rewards = [reward(s, gt_future) for s in samples]
    mu = sum(rewards) / len(rewards)
    std = (sum((r - mu) ** 2 for r in rewards) / len(rewards) + 1e-8) ** 0.5
    adv = [(r - mu) / std for r in rewards]

    g = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for a, traj in zip(adv, samples):
        gi = policy.grad_logprob(history, traj)
        for r in range(2):
            for c in range(3):
                g[r][c] += a * gi[r][c]
    for r in range(2):
        for c in range(3):
            g[r][c] /= len(samples)

    policy.apply_grad(g, lr=lr)
    return {"mean_reward": mu, "std_reward": std, "weights": policy.w}
