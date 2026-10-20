"""Bayesian spend-response curves per ad set.

    conversions_t ~ Poisson( a * (1 - exp(-spend_t / k)) * weekday_t )

For each value of k on a grid, the Gamma prior on `a` is conjugate to the Poisson likelihood, so
the posterior of `a` and the marginal likelihood of k are exact; the posterior over k is a
discrete distribution on the grid. Two production details:

- forgetting: day t's observation is weighted gamma**age, so the curve follows drift and creative
  refreshes (half-life ~23 days at gamma=0.97);
- pooled prior: the prior on `a` is centred on the cost per conversion that ad sets on the same
  platform achieve (empirical Bayes), so a new ad set starts from a sensible guess, not from zero.

The posterior is a small, JSON-serialisable object stored in PostgreSQL by the worker and read by
the API on every request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import special

GRID_SPAN = 1.6  # k from exp(-1.6) to exp(1.6) times the typical daily spend
GRID_SIZE = 25


@dataclass
class Posterior:
    ad_set_id: str
    k: list[float]  # grid
    weight: list[float]  # posterior probability of each k
    alpha: list[float]  # Gamma posterior of a, per k
    beta: list[float]
    n_days: int
    typical_spend: float
    prior_cpa: float

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> Posterior:
        return cls(**d)

    # -------------------------------------------------------------------- curve functions
    def mean_curve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(weights, a, k) of the posterior-mean curve: a mixture of saturating curves."""
        w, al, be, k = (np.asarray(x, float) for x in (self.weight, self.alpha, self.beta, self.k))
        return w, al / be, k

    def sample(self, rng: np.random.Generator) -> tuple[float, float]:
        """One Thompson draw: k from its grid posterior, then a from its Gamma posterior."""
        i = int(rng.choice(len(self.k), p=np.asarray(self.weight) / np.sum(self.weight)))
        a = float(rng.gamma(self.alpha[i], 1.0 / self.beta[i]))
        return a, float(self.k[i])

    def expected(self, spend: float | np.ndarray) -> np.ndarray:
        w, a, k = self.mean_curve()
        s = np.atleast_1d(np.asarray(spend, float))
        return (w[None, :] * a[None, :] * (1 - np.exp(-s[:, None] / k[None, :]))).sum(1)

    def interval(self, spend: float, level: float = 0.8, draws: int = 2000, seed: int = 0) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.k), size=draws, p=np.asarray(self.weight) / np.sum(self.weight))
        a = rng.gamma(np.asarray(self.alpha)[idx], 1.0 / np.asarray(self.beta)[idx])
        y = a * (1 - np.exp(-spend / np.asarray(self.k)[idx]))
        lo, hi = np.quantile(y, [(1 - level) / 2, (1 + level) / 2])
        return float(lo), float(hi)


def fit(
    ad_set_id: str,
    spend: np.ndarray,
    conversions: np.ndarray,
    age_days: np.ndarray,
    weekday_mult: np.ndarray | None = None,
    prior_cpa: float = 30.0,
    prior_strength: float = 1.0,
    forgetting: float = 0.97,
    typical_spend: float | None = None,
) -> Posterior:
    spend = np.asarray(spend, float)
    conv = np.asarray(conversions, float)
    wd = np.ones_like(spend) if weekday_mult is None else np.asarray(weekday_mult, float)
    w = forgetting ** np.asarray(age_days, float)
    active = spend > 0
    typical = float(np.average(spend[active], weights=w[active])) if active.any() else (typical_spend or 100.0)
    k = typical * np.exp(np.linspace(-GRID_SPAN, GRID_SPAN, GRID_SIZE))
    g = (1 - np.exp(-spend[:, None] / k[None, :])) * wd[:, None]  # days x grid
    # prior on a given k: mean k / prior_cpa (cost per conversion at low spend), shape prior_strength
    a0 = np.full_like(k, prior_strength)
    b0 = prior_strength * prior_cpa / k
    an = a0 + (w * conv).sum()
    bn = b0 + (w[:, None] * g).sum(0)
    with np.errstate(divide="ignore"):
        log_g = np.where(g > 0, np.log(np.maximum(g, 1e-300)), 0.0)
    loglik = (
        special.gammaln(an)
        - special.gammaln(a0)
        + a0 * np.log(b0)
        - an * np.log(bn)
        + (w[:, None] * conv[:, None] * log_g).sum(0)
    )
    log_prior_k = -0.5 * (np.log(k / typical) / 1.0) ** 2
    lp = loglik + log_prior_k
    post = np.exp(lp - lp.max())
    post /= post.sum()
    return Posterior(
        ad_set_id,
        k.tolist(),
        post.tolist(),
        np.broadcast_to(an, k.shape).tolist(),
        bn.tolist(),
        int(active.sum()),
        typical,
        prior_cpa,
    )


def pooled_cpa(spend: np.ndarray, conversions: np.ndarray, default: float = 30.0) -> float:
    """Platform-level cost per conversion, shrunk towards a default when data is thin."""
    s, c = float(np.sum(spend)), float(np.sum(conversions))
    return (s + default * 20) / (c + 20)


def weekday_multipliers(weekday: np.ndarray, conversions: np.ndarray, spend: np.ndarray) -> np.ndarray:
    """Conversions per euro by weekday relative to the average, shrunk to 1 when data is thin."""
    out = np.ones(7)
    total = conversions.sum() / max(spend.sum(), 1e-9)
    for d in range(7):
        m = weekday == d
        if m.any():
            r = conversions[m].sum() / max(spend[m].sum(), 1e-9) / max(total, 1e-12)
            n = conversions[m].sum()
            out[d] = (r * n + 1.0 * 50) / (n + 50)
    return out / out.mean()
