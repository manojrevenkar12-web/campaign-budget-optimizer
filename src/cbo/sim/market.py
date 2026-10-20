"""A simulated ad market with known ground truth.

Each campaign has a daily budget and several ad sets (an audience on a platform). An ad set's
expected conversions follow a saturating spend-response curve

    E[conversions | spend s] = a * (1 - exp(-s / k)) * weekday(t) * creative(t)

where `a` is the most conversions the audience can produce in a day, `k` the spend at which
returns start to diminish, `weekday` a platform-specific day-of-week effect and `creative` the
current creative's freshness, which decays as the audience sees it more often and recovers when the
advertiser swaps in a new creative. `a` drifts slowly (seasonality, competition).

Advertisers never see a, k or the fatigue: they see spend, impressions, clicks and conversions,
which is what the optimizer has to learn from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PLATFORMS = {
    # cpm (EUR), base ctr, weekday multipliers Mon..Sun
    "meta": (9.0, 0.012, (0.95, 0.96, 0.98, 1.0, 1.02, 1.06, 1.05)),
    "tiktok": (6.5, 0.010, (0.92, 0.94, 0.96, 1.0, 1.05, 1.10, 1.08)),
    "pinterest": (5.5, 0.008, (1.0, 1.0, 1.0, 1.0, 0.98, 1.04, 1.06)),
    "google": (12.0, 0.020, (1.05, 1.05, 1.03, 1.02, 0.98, 0.90, 0.92)),
}
FORMATS = {"static": 9.0, "carousel": 13.0, "video": 18.0}  # fatigue scale (frequency at which CTR falls to 1/e)


@dataclass
class AdSet:
    ad_set_id: str
    platform: str
    a: float  # max daily conversions
    k: float  # spend scale (EUR)
    audience: float  # people reachable
    creative_format: str
    creative_id: str = ""
    creative_age: int = 0
    exposures: float = 0.0  # impressions of the current creative
    refresh_every: int = 30
    log_a_drift: float = 0.0
    creative_seq: int = 0

    def freshness(self) -> float:
        freq = self.exposures / self.audience
        return 0.35 + 0.65 * float(np.exp(-freq / FORMATS[self.creative_format]))


@dataclass
class Campaign:
    campaign_id: str
    budget: float
    ad_sets: list[AdSet]
    day: int = 0
    history: list[dict] = field(default_factory=list)


def make_campaign(rng: np.random.Generator, idx: int, n_min: int = 3, n_max: int = 8) -> Campaign:
    n = int(rng.integers(n_min, n_max + 1))
    budget = float(np.round(rng.uniform(1500, 15000), -1))
    sets = []
    for j in range(n):
        platform = str(rng.choice(list(PLATFORMS), p=[0.45, 0.2, 0.1, 0.25]))
        k = budget / n * float(np.exp(rng.normal(0.0, 0.9)))
        cpa_low = float(np.exp(rng.normal(np.log(28), 0.6)))  # cost per conversion at low spend
        a = k / cpa_low
        fmt = str(rng.choice(list(FORMATS), p=[0.45, 0.2, 0.35]))
        s = AdSet(
            ad_set_id=f"c{idx:04d}-as{j}",
            platform=platform,
            a=a,
            k=k,
            audience=float(np.exp(rng.normal(np.log(400_000), 0.6))),
            creative_format=fmt,
            refresh_every=int(rng.integers(18, 50)),
        )
        s.creative_id = f"{s.ad_set_id}-cr0"
        sets.append(s)
    return Campaign(f"c{idx:04d}", budget, sets)


def expected_conversions(s: AdSet, spend: float, weekday: int) -> float:
    w = PLATFORMS[s.platform][2][weekday]
    return float(s.a * np.exp(s.log_a_drift) * (1 - np.exp(-spend / s.k)) * w * s.freshness() ** 0.8)


def step(
    c: Campaign,
    spends: dict[str, float],
    rng: np.random.Generator,
    refresh: dict[str, bool] | None = None,
    world_rng: np.random.Generator | None = None,
) -> list[dict]:
    """Run one day: spend, observe, age creatives, drift. Returns one row per ad set.

    `rng` draws what is observed; `world_rng` (defaults to `rng`) drives drift and creative
    changes, so that policies compared on the same world see the same market."""
    world_rng = world_rng or rng
    weekday = c.day % 7
    rows = []
    for s in c.ad_sets:
        spend = float(max(spends.get(s.ad_set_id, 0.0), 0.0))
        cpm_base, ctr0, _ = PLATFORMS[s.platform]
        cpm = cpm_base * float(np.exp(rng.normal(0, 0.08)))
        imps = int(spend / cpm * 1000)
        fresh = s.freshness()
        day_ctr = float(np.exp(rng.normal(0, 0.12)))  # auction mix and placements move CTR day to day
        clicks = int(rng.poisson(imps * ctr0 * fresh * day_ctr)) if imps else 0
        lam = expected_conversions(s, spend, weekday)
        conv = int(rng.poisson(lam))
        rows.append(
            {
                "campaign_id": c.campaign_id,
                "ad_set_id": s.ad_set_id,
                "platform": s.platform,
                "day": c.day,
                "weekday": weekday,
                "spend": round(spend, 2),
                "impressions": imps,
                "clicks": clicks,
                "conversions": conv,
                "creative_id": s.creative_id,
                "creative_format": s.creative_format,
                "creative_age": s.creative_age,
                "frequency": round(s.exposures / s.audience, 4),
                "true_expected_conversions": lam,
                "true_freshness": fresh,
            }
        )
        s.exposures += imps
        s.creative_age += 1
        if (refresh or {}).get(s.ad_set_id) or s.creative_age >= s.refresh_every:
            s.creative_seq += 1
            s.creative_id = f"{s.ad_set_id}-cr{s.creative_seq}"
            s.creative_age, s.exposures = 0, 0.0
            s.creative_format = str(world_rng.choice(list(FORMATS), p=[0.45, 0.2, 0.35]))
        s.log_a_drift += float(world_rng.normal(0, 0.02))
    c.day += 1
    c.history += rows
    return rows


def oracle_curve(s: AdSet, weekday: int) -> tuple[float, float]:
    """True (a, k) of today's curve, for the oracle policy and regret."""
    w = PLATFORMS[s.platform][2][weekday]
    return float(s.a * np.exp(s.log_a_drift) * w * s.freshness() ** 0.8), s.k
