"""Small statistics helpers for success rates."""

from __future__ import annotations

import math


def wilson_interval(successes: int, trials: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95 %).

    Unlike the normal approximation it stays inside [0, 1] and behaves at 0/n and n/n,
    which matters for 50-trial evals of policies that are either very good or very bad.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be in [0, trials]")
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    # At 0/n and n/n the bound is exactly 0 or 1; avoid 0.9999999999999999.
    lo = 0.0 if successes == 0 else max(0.0, centre - half)
    hi = 1.0 if successes == trials else min(1.0, centre + half)
    return lo, hi
