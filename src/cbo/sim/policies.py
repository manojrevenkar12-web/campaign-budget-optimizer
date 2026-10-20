"""Budget policies, played day by day on simulated campaigns."""

from __future__ import annotations

import numpy as np

from cbo.model.response import fit, pooled_cpa, weekday_multipliers
from cbo.optim.allocate import Guardrails, allocate, bounds
from cbo.sim.market import Campaign, oracle_curve


def history_arrays(c: Campaign) -> dict[str, np.ndarray]:
    h = c.history
    return {k: np.array([r[k] for r in h]) for k in ("ad_set_id", "platform", "day", "weekday", "spend", "conversions")}


def even(c: Campaign, prev: np.ndarray | None, rng, g: Guardrails) -> np.ndarray:
    n = len(c.ad_sets)
    return np.full(n, c.budget / n)


def rule_of_thumb(c: Campaign, prev: np.ndarray | None, rng, g: Guardrails) -> np.ndarray:
    """What many teams do by hand: once a week, move 20% of the worst ad set's budget (by cost per
    conversion over the last 7 days) to the best one."""
    n = len(c.ad_sets)
    if prev is None:
        return np.full(n, c.budget / n)
    if c.day % 7 != 0 or len(c.history) < 7 * n:
        return prev
    h = history_arrays(c)
    recent = h["day"] >= c.day - 7
    cpa = []
    for s in c.ad_sets:
        m = recent & (h["ad_set_id"] == s.ad_set_id)
        cpa.append(h["spend"][m].sum() / max(h["conversions"][m].sum(), 0.5))
    cpa = np.array(cpa)
    out = prev.copy()
    worst, best = int(np.argmax(cpa)), int(np.argmin(cpa))
    move = 0.2 * out[worst]
    out[worst] -= move
    out[best] += move
    return out


def fit_all(c: Campaign) -> list:
    h = history_arrays(c)
    wd_mult = {}
    for p in np.unique(h["platform"]):
        m = h["platform"] == p
        wd_mult[p] = weekday_multipliers(h["weekday"][m], h["conversions"][m], h["spend"][m])
    posts = []
    for s in c.ad_sets:
        m = h["ad_set_id"] == s.ad_set_id
        pm = h["platform"] == s.platform
        posts.append(
            fit(
                s.ad_set_id,
                h["spend"][m],
                h["conversions"][m],
                c.day - 1 - h["day"][m],
                wd_mult[s.platform][h["weekday"][m]],
                prior_cpa=pooled_cpa(h["spend"][pm], h["conversions"][pm]),
            )
        )
    return posts


def greedy(c: Campaign, prev: np.ndarray | None, rng, g: Guardrails) -> np.ndarray:
    posts = fit_all(c)
    lo, hi, _ = bounds(c.budget, len(posts), prev, g)
    return allocate([p.mean_curve() for p in posts], c.budget, lo, hi)


def thompson(c: Campaign, prev: np.ndarray | None, rng, g: Guardrails) -> np.ndarray:
    posts = fit_all(c)
    curves = []
    for p in posts:
        a, k = p.sample(rng)
        curves.append((np.array([1.0]), np.array([a]), np.array([k])))
    lo, hi, _ = bounds(c.budget, len(posts), prev, g)
    return allocate(curves, c.budget, lo, hi)


def damped_thompson(c: Campaign, prev: np.ndarray | None, rng, g: Guardrails, share: float = 0.3) -> np.ndarray:
    """Production default: the posterior-mean allocation, moved `share` of the way towards one
    Thompson draw. Keeps exploring the curves without pure Thompson's day-to-day churn."""
    posts = fit_all(c)
    lo, hi, _ = bounds(c.budget, len(posts), prev, g)
    mean = allocate([p.mean_curve() for p in posts], c.budget, lo, hi)
    draws = []
    for p in posts:
        a, k = p.sample(rng)
        draws.append((np.array([1.0]), np.array([a]), np.array([k])))
    ts = allocate(draws, c.budget, lo, hi)
    return (1 - share) * mean + share * ts


def oracle(c: Campaign, prev: np.ndarray | None, rng, g: Guardrails) -> np.ndarray:
    """Knows today's true curves; held to the same guardrails, so it is the best any policy can do."""
    weekday = c.day % 7
    curves = []
    for s in c.ad_sets:
        a, k = oracle_curve(s, weekday)
        curves.append((np.array([1.0]), np.array([a]), np.array([k])))
    lo, hi, _ = bounds(c.budget, len(curves), prev, g)
    return allocate(curves, c.budget, lo, hi)


POLICIES = {
    "even_split": even,
    "rule_of_thumb": rule_of_thumb,
    "greedy_bayes": greedy,
    "thompson": thompson,
    "damped_thompson": damped_thompson,
    "oracle": oracle,
}
