import numpy as np

from cbo.model.response import fit, pooled_cpa


def simulate(a, k, spend, rng):
    return rng.poisson(a * (1 - np.exp(-spend / k)))


def test_posterior_recovers_the_curve_when_spend_varies():
    rng = np.random.default_rng(0)
    spend = rng.uniform(200, 3000, 120)
    conv = simulate(80.0, 1500.0, spend, rng)
    p = fit("x", spend, conv, np.arange(120)[::-1], prior_cpa=25.0, forgetting=1.0)
    for s in (500, 1500, 3000):
        truth = 80 * (1 - np.exp(-s / 1500))
        assert abs(p.expected(s)[0] - truth) / truth < 0.08
    lo, hi = p.interval(1500.0, level=0.95)
    assert lo < 80 * (1 - np.exp(-1)) < hi


def test_forgetting_follows_a_shift():
    rng = np.random.default_rng(1)
    spend = rng.uniform(500, 1500, 100)
    a = np.where(np.arange(100) < 70, 40.0, 80.0)  # the ad set doubled its potential 30 days ago
    conv = rng.poisson(a * (1 - np.exp(-spend / 1000)))
    age = np.arange(100)[::-1]
    fresh = fit("x", spend, conv, age, forgetting=0.95).expected(1000)[0]
    stale = fit("x", spend, conv, age, forgetting=1.0).expected(1000)[0]
    truth = 80 * (1 - np.exp(-1))
    assert abs(fresh - truth) < abs(stale - truth)


def test_cold_start_uses_the_pooled_cost_per_conversion():
    p = fit("new", np.array([]), np.array([]), np.array([]), prior_cpa=40.0, typical_spend=1000.0)
    # with no data, expected cost per conversion at low spend is the prior's
    assert abs(10.0 / p.expected(10.0)[0] - 40.0) / 40.0 < 0.15
    assert p.n_days == 0
    assert 35 < pooled_cpa(np.array([4000.0]), np.array([100.0])) < 40


def test_thompson_draws_are_spread_like_the_posterior():
    rng = np.random.default_rng(2)
    spend = rng.uniform(500, 1500, 10)
    p = fit("x", spend, simulate(50, 1000, spend, rng), np.arange(10)[::-1])
    draws = np.array([p.sample(np.random.default_rng(i)) for i in range(400)])
    values = draws[:, 0] * (1 - np.exp(-1000 / draws[:, 1]))
    lo, hi = p.interval(1000.0)
    inside = ((values >= lo) & (values <= hi)).mean()
    assert 0.7 < inside < 0.9
