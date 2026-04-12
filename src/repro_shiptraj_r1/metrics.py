from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

Point = Tuple[float, float]


def geodesic_distance_m(p1: Point, p2: Point) -> float:
    """Compute great-circle distance in meters using haversine formula.

    Args:
        p1: (lon, lat)
        p2: (lon, lat)
    """
    lon1, lat1 = p1
    lon2, lat2 = p2

    r = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def _validate(pred: Sequence[Point], gt: Sequence[Point]) -> None:
    if len(pred) != len(gt):
        raise ValueError(f"pred and gt lengths differ: {len(pred)} vs {len(gt)}")
    if not pred:
        raise ValueError("pred and gt cannot be empty")


def ade(pred: Sequence[Point], gt: Sequence[Point]) -> float:
    """Average Displacement Error in meters."""
    _validate(pred, gt)
    return sum(geodesic_distance_m(p, g) for p, g in zip(pred, gt)) / len(pred)


def fde(pred: Sequence[Point], gt: Sequence[Point]) -> float:
    """Final Displacement Error in meters."""
    _validate(pred, gt)
    return geodesic_distance_m(pred[-1], gt[-1])


def batch_ade_fde(batch_pred: Iterable[Sequence[Point]], batch_gt: Iterable[Sequence[Point]]) -> tuple[float, float]:
    ades = []
    fdes = []
    for pred, gt in zip(batch_pred, batch_gt):
        ades.append(ade(pred, gt))
        fdes.append(fde(pred, gt))
    if not ades:
        raise ValueError("empty batch")
    return sum(ades) / len(ades), sum(fdes) / len(fdes)
