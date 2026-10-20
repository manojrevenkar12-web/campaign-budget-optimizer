import itertools

import numpy as np
import pytest

from cbo.optim.allocate import Guardrails, allocate, bounds, marginal, value


def curve(a, k):
    return (np.array([1.0]), np.array([a]), np.array([k]))


def test_matches_brute_force_and_equalises_marginal_returns():
    curves = [curve(60, 800), curve(30, 300), curve(90, 2500)]
    budget = 3000.0
    lo, hi = np.full(3, 50.0), np.full(3, budget)
    s = allocate(curves, budget, lo, hi)
    assert abs(s.sum() - budget) < 1e-6
    best = max(
        (sum(value(c, x) for c, x in zip(curves, (x1, x2, budget - x1 - x2), strict=True)), (x1, x2))
        for x1, x2 in itertools.product(np.arange(50, 3000, 25), repeat=2)
        if budget - x1 - x2 >= 50
    )[0]
    assert sum(value(c, x) for c, x in zip(curves, s, strict=True)) >= best - 1e-6
    m = [marginal(c, x) for c, x in zip(curves, s, strict=True)]
    assert max(m) / min(m) < 1.001


def test_guardrails_limit_daily_moves_and_keep_a_floor():
    g = Guardrails(min_spend=10, min_share=0.05, max_change=0.3)
    prev = np.array([1000.0, 1000.0, 1000.0])
    lo, hi, relaxed = bounds(3000, 3, prev, g)
    assert not relaxed and np.allclose(lo, 700) and np.allclose(hi, 1300)
    s = allocate([curve(500, 100), curve(5, 1000), curve(5, 1000)], 3000, lo, hi)
    assert np.all(s >= lo - 1e-6) and np.all(s <= hi + 1e-6) and abs(s.sum() - 3000) < 1e-6


@pytest.mark.parametrize("budget", [1000.0, 9000.0])
def test_corridor_is_relaxed_when_the_budget_changes_a_lot(budget):
    lo, hi, relaxed = bounds(budget, 3, np.array([1000.0, 1000.0, 1000.0]), Guardrails())
    assert relaxed and lo.sum() <= budget + 1e-6 and hi.sum() >= budget - 1e-6


def test_first_day_has_only_the_floor():
    lo, hi, relaxed = bounds(1000, 4, None, Guardrails(min_spend=10, min_share=0.03))
    assert np.allclose(lo, 30) and np.allclose(hi, 1000) and not relaxed
