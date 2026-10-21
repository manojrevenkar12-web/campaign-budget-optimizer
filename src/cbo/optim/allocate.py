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


def _stack(curves: list[Curve]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Curves as padded (ad sets x components) arrays; padding has zero weight."""
    m = max(len(c[0]) for c in curves)
    w = np.zeros((len(curves), m))
    a = np.zeros((len(curves), m))
    k = np.ones((len(curves), m))
    for i, (wi, ai, ki) in enumerate(curves):
        w[i, : len(wi)], a[i, : len(ai)], k[i, : len(ki)] = wi, ai, ki
    return w, a, k


GRID = 2049


def allocate(curves: list[Curve], budget: float, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Water-filling. Each ad set's marginal return f_i'(s) is tabulated once on a fine grid over
    its corridor [lo_i, hi_i] (it is monotone decreasing), so finding the spend for a given lambda
    is an interpolation, and the outer bisection on lambda costs microseconds per step."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    if lo.sum() >= budget:
        return lo * budget / lo.sum()
    if hi.sum() <= budget:
        return hi.copy()
    w, a, k = _stack(curves)
    wa_k = w * a / k
    t = np.linspace(0.0, 1.0, GRID)
    grid = lo[:, None] + (hi - lo)[:, None] * t[None, :]  # ad sets x grid
    marg = np.einsum("nm,ngm->ng", wa_k, np.exp(-grid[:, :, None] / k[:, None, :]))  # decreasing along the grid
    rev_m, rev_s = marg[:, ::-1], grid[:, ::-1]  # increasing marginal, for np.interp

    def spend(lam: float) -> np.ndarray:
        return np.array([np.interp(lam, rev_m[i], rev_s[i]) for i in range(len(curves))])

    lam_lo, lam_hi = 0.0, float(marg[:, 0].max()) * 2 + 1e-12
    for _ in range(60):
        lam = 0.5 * (lam_lo + lam_hi)
        if spend(lam).sum() > budget:
            lam_lo = lam
        else:
            lam_hi = lam
    x = spend(lam_hi)
    # distribute the remaining cents to ad sets with room, so the sum is exact
    gap = budget - x.sum()
    room = np.where(gap > 0, hi - x, x - lo)
    if room.sum() > 0:
        x += gap * room / room.sum()
    return x
