"""Split a campaign's daily budget across its ad sets.

Each ad set's curve f_i(s) = sum_j w_j a_j (1 - exp(-s / k_j)) is concave, so the optimal split
equalises marginal conversions per euro (f_i'(s_i) = lambda) across ad sets that are not at a
bound — "water-filling". An outer bisection finds lambda so that the spends add up to the budget,
an inner bisection solves f_i'(s) = lambda per ad set. Exact to a cent in well under a millisecond
for a typical campaign.

Guardrails are business rules, not tuning knobs:
- every ad set keeps a minimum spend (the platform's minimum, and enough to keep learning);
- no ad set moves more than `max_change` relative to yesterday, because large jumps reset the ad
  platforms' delivery learning and make results unreadable for the advertiser.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Curve = tuple[np.ndarray, np.ndarray, np.ndarray]  # (weights, a, k)


def marginal(curve: Curve, s: float) -> float:
    w, a, k = curve
    return float((w * a / k * np.exp(-s / k)).sum())


def value(curve: Curve, s: float) -> float:
    w, a, k = curve
    return float((w * a * (1 - np.exp(-s / k))).sum())


def _spend_at(curve: Curve, lam: float, lo: float, hi: float) -> float:
    if marginal(curve, lo) <= lam:
        return lo
    if marginal(curve, hi) >= lam:
        return hi
    a, b = lo, hi
    for _ in range(50):
        m = 0.5 * (a + b)
        if marginal(curve, m) > lam:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


@dataclass(frozen=True)
class Guardrails:
    min_spend: float = 10.0
    min_share: float = 0.03  # each ad set gets at least 3% of the budget, to keep learning
    max_change: float = 0.30  # +-30% vs yesterday


def bounds(budget: float, n: int, previous: np.ndarray | None, g: Guardrails) -> tuple[np.ndarray, np.ndarray, bool]:
    """Per-ad-set spend corridor for today, and whether it had to be relaxed to be feasible."""
    floor = min(max(g.min_spend, g.min_share * budget), budget / n)
    if previous is None or len(previous) != n or np.sum(previous) <= 0:
        return np.full(n, floor), np.full(n, budget), False
    lo = np.maximum(floor, previous * (1 - g.max_change))
    hi = np.maximum(previous * (1 + g.max_change), floor)
    relaxed = False
    # the budget itself may have changed: widen the corridor proportionally until it is feasible
    if lo.sum() > budget:
        lo = np.maximum(floor, lo * (budget / lo.sum()))
        relaxed = True
    if hi.sum() < budget:
        hi = hi * (budget / hi.sum())
        relaxed = True
    return np.minimum(lo, budget), np.minimum(hi, budget), relaxed


def allocate(curves: list[Curve], budget: float, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    if lo.sum() >= budget:
        return lo * budget / lo.sum()
    if hi.sum() <= budget:
        return hi.copy()
    lam_hi = max(marginal(c, lo_i) for c, lo_i in zip(curves, lo, strict=True)) * 2 + 1e-12
    lam_lo = 0.0
    for _ in range(60):
        lam = 0.5 * (lam_lo + lam_hi)
        total = sum(_spend_at(c, lam, a, b) for c, a, b in zip(curves, lo, hi, strict=True))
        if total > budget:
            lam_lo = lam
        else:
            lam_hi = lam
    s = np.array([_spend_at(c, lam_hi, a, b) for c, a, b in zip(curves, lo, hi, strict=True)])
    # distribute the bisection's rounding remainder to ad sets with room, so the sum is exact
    gap = budget - s.sum()
    room = np.where(gap > 0, hi - s, s - lo)
    if room.sum() > 0:
        s += gap * room / room.sum()
    return s
