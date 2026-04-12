from __future__ import annotations

import math
from typing import Sequence, Tuple

Point = Tuple[float, float]


def _to_local_xy_m(p: Point, ref_lat_deg: float) -> tuple[float, float]:
    """Approximate lon/lat to local tangent-plane meters."""
    lon, lat = p
    x = math.radians(lon) * 6_371_000.0 * math.cos(math.radians(ref_lat_deg))
    y = math.radians(lat) * 6_371_000.0
    return x, y


def _velocity(p_prev: Point, p_last: Point, dt_s: float) -> tuple[float, float]:
    ref_lat = (p_prev[1] + p_last[1]) / 2.0
    x1, y1 = _to_local_xy_m(p_prev, ref_lat)
    x2, y2 = _to_local_xy_m(p_last, ref_lat)
    return (x2 - x1) / dt_s, (y2 - y1) / dt_s


def tcpa_dcpa(own_hist: Sequence[Point], nei_hist: Sequence[Point], dt_s: float = 5.0) -> tuple[float, float]:
    """Compute Time/Distance at Closest Point of Approach.

    Returns:
        (tcpa_seconds, dcpa_meters)
    """
    if len(own_hist) < 2 or len(nei_hist) < 2:
        raise ValueError("Need at least 2 points for each vessel")

    ref_lat = (own_hist[-1][1] + nei_hist[-1][1]) / 2.0
    ox, oy = _to_local_xy_m(own_hist[-1], ref_lat)
    nx, ny = _to_local_xy_m(nei_hist[-1], ref_lat)

    ovx, ovy = _velocity(own_hist[-2], own_hist[-1], dt_s)
    nvx, nvy = _velocity(nei_hist[-2], nei_hist[-1], dt_s)

    rx, ry = ox - nx, oy - ny
    rvx, rvy = ovx - nvx, ovy - nvy
    rv2 = rvx * rvx + rvy * rvy

    if rv2 < 1e-9:
        return 0.0, math.hypot(rx, ry)

    tcpa = - (rx * rvx + ry * rvy) / rv2
    cx = rx + rvx * tcpa
    cy = ry + rvy * tcpa
    dcpa = math.hypot(cx, cy)
    return tcpa, dcpa


def colregs_risk_reward(
    own_hist: Sequence[Point],
    nei_hist: Sequence[Point],
    safety_dcpa_m: float = 500.0,
    dt_s: float = 5.0,
    horizon_tcpa_s: float = 300.0,
) -> float:
    """Reward in [0,1], high when collision risk is low.

    If TCPA is in a near-future horizon and DCPA is below threshold, reward decreases.
    """
    tcpa_s, dcpa_m = tcpa_dcpa(own_hist, nei_hist, dt_s=dt_s)

    if tcpa_s < 0 or tcpa_s > horizon_tcpa_s:
        return 1.0

    risk = max(0.0, (safety_dcpa_m - dcpa_m) / max(safety_dcpa_m, 1e-6))
    return max(0.0, min(1.0, 1.0 - risk))
