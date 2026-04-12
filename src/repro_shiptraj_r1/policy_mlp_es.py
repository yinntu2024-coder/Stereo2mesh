from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .metrics import geodesic_distance_m

Point = Tuple[float, float]


@dataclass
class MLPESPolicy:
    """Small MLP policy optimized by evolution-strategy GRPO-like updates.

    Input: [1, vlon, vlat]
    Output: [dlon, dlat]
    """

    hidden: int = 8
    sigma_action: float = 1e-3
    sigma_es: float = 5e-3

    def __post_init__(self):
        rng = np.random.default_rng(0)
        self.w1 = rng.normal(0, 0.1, size=(3, self.hidden))
        self.b1 = np.zeros((self.hidden,))
        self.w2 = rng.normal(0, 0.1, size=(self.hidden, 2))
        self.b2 = np.zeros((2,))

    def _forward(self, x: np.ndarray, params: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        if params is None:
            w1, b1, w2, b2 = self.w1, self.b1, self.w2, self.b2
        else:
            w1, b1, w2, b2 = params
        h = np.tanh(x @ w1 + b1)
        y = h @ w2 + b2
        return y

    def _params(self):
        return (self.w1, self.b1, self.w2, self.b2)

    def _set_params(self, p):
        self.w1, self.b1, self.w2, self.b2 = p

    def rollout(self, history: Sequence[Point], horizon: int, rng: np.random.Generator, params=None) -> list[Point]:
        (lon1, lat1), (lon2, lat2) = history[-2], history[-1]
        vlon, vlat = lon2 - lon1, lat2 - lat1
        x = np.array([1.0, vlon, vlat], dtype=float)
        mean = self._forward(x, params=params)

        cur_lon, cur_lat = lon2, lat2
        out: list[Point] = []
        for _ in range(horizon):
            dlon = float(mean[0] + rng.normal(0, self.sigma_action))
            dlat = float(mean[1] + rng.normal(0, self.sigma_action))
            cur_lon += dlon
            cur_lat += dlat
            out.append((cur_lon, cur_lat))
        return out


def reward(pred: Sequence[Point], gt: Sequence[Point], scale_m: float = 1000.0) -> float:
    if len(pred) != len(gt) or not pred:
        return 0.0
    err = sum(geodesic_distance_m(p, g) for p, g in zip(pred, gt)) / len(pred)
    return 1.0 / (1.0 + err / scale_m)


def _add_noise(params, eps, scale):
    return tuple(p + scale * e for p, e in zip(params, eps))


def grpo_es_step(policy: MLPESPolicy, history: Sequence[Point], gt_future: Sequence[Point], *, num_samples: int = 16, lr: float = 1e-2, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = policy._params()

    eps_list = []
    rewards = []
    for _ in range(num_samples):
        eps = tuple(rng.normal(0, 1, size=p.shape) for p in base)
        perturbed = _add_noise(base, eps, policy.sigma_es)
        traj = policy.rollout(history, horizon=len(gt_future), rng=rng, params=perturbed)
        r = reward(traj, gt_future)
        eps_list.append(eps)
        rewards.append(r)

    rs = np.array(rewards)
    adv = (rs - rs.mean()) / (rs.std() + 1e-8)

    grad = []
    for i in range(len(base)):
        g = np.zeros_like(base[i])
        for a, eps in zip(adv, eps_list):
            g += a * eps[i]
        g /= (len(eps_list) * policy.sigma_es)
        grad.append(g)

    new_params = tuple(p + lr * g for p, g in zip(base, grad))
    policy._set_params(new_params)

    return {"mean_reward": float(rs.mean()), "std_reward": float(rs.std())}
