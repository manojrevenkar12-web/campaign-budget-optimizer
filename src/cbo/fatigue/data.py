"""Creative-fatigue training data: one row per creative per day, with features that use only what
an advertiser can see up to that day, and the label "CTR over the next 7 days falls below 70% of
the creative's best 3-day CTR so far" (the creative is wearing out and should be replaced soon).

Only creatives that still look healthy are scored (current CTR at least 80% of their peak, at least
three days old): flagging a creative that has already collapsed is easy and useless; the value is in
warning a week ahead.

Features are computed the same way in training (from simulated histories) and in serving (from the
`creative_daily` table), by `features_from_history`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cbo.sim.market import FORMATS, PLATFORMS, make_campaign, step

FEATURES = [
    "age_days",
    "log_frequency",
    "ctr_ratio_to_peak",
    "ctr_slope_7d",
    "log_impressions_3d",
    "days_since_peak",
    *[f"platform_{p}" for p in PLATFORMS],
    *[f"format_{f}" for f in FORMATS],
]
HORIZON = 7
THRESHOLD = 0.7
HEALTHY = 0.8


def features_from_history(h: pd.DataFrame) -> pd.DataFrame:
    """h: rows of one or more creatives with columns creative_id, platform, creative_format,
    creative_age, frequency, impressions, clicks, sorted by day. Returns one feature row per input row."""
    out = []
    for cid, g in h.sort_values(["creative_id", "creative_age"]).groupby("creative_id", sort=False):
        imps = g["impressions"].to_numpy(float)
        clicks = g["clicks"].to_numpy(float)
        r_imp = pd.Series(imps).rolling(3, min_periods=1).sum().to_numpy()
        r_clk = pd.Series(clicks).rolling(3, min_periods=1).sum().to_numpy()
        ctr3 = (r_clk + 0.5) / (r_imp + 50)  # smoothed 3-day CTR
        peak = np.maximum.accumulate(ctr3)
        peak_day = np.array([int(np.argmax(ctr3[: i + 1])) for i in range(len(ctr3))])
        log_ctr = np.log(ctr3)
        slope = np.zeros(len(g))
        for i in range(len(g)):
            lo = max(0, i - 6)
            if i - lo >= 2:
                x = np.arange(lo, i + 1)
                slope[i] = np.polyfit(x, log_ctr[lo : i + 1], 1)[0]
        f = pd.DataFrame(
            {
                "creative_id": cid,
                "day": g["day"].to_numpy() if "day" in g else np.arange(len(g)),
                "age_days": g["creative_age"].to_numpy(float),
                "log_frequency": np.log1p(g["frequency"].to_numpy(float)),
                "ctr_ratio_to_peak": ctr3 / peak,
                "ctr_slope_7d": slope,
                "log_impressions_3d": np.log1p(r_imp),
                "days_since_peak": np.arange(len(g)) - peak_day,
                "ctr3": ctr3,
                "peak_ctr3": peak,
            }
        )
        for p in PLATFORMS:
            f[f"platform_{p}"] = float(g["platform"].iloc[0] == p)
        for fm in FORMATS:
            f[f"format_{fm}"] = float(g["creative_format"].iloc[0] == fm)
        out.append(f)
    return pd.concat(out, ignore_index=True)


def label(f: pd.DataFrame, h: pd.DataFrame) -> pd.Series:
    """1 if the next 7 days' CTR is below 70% of the peak 3-day CTR seen so far; NaN when the
    creative was replaced before 7 more days were observed."""
    y = np.full(len(f), np.nan)
    hh = h.sort_values(["creative_id", "creative_age"])
    pos = 0
    for _cid, g in hh.groupby("creative_id", sort=False):
        imps = g["impressions"].to_numpy(float)
        clicks = g["clicks"].to_numpy(float)
        n = len(g)
        for i in range(n):
            if i + HORIZON < n:
                future = (clicks[i + 1 : i + 1 + HORIZON].sum() + 0.5) / (imps[i + 1 : i + 1 + HORIZON].sum() + 50)
                y[pos + i] = float(future < THRESHOLD * f["peak_ctr3"].iloc[pos + i])
        pos += n
    return pd.Series(y, index=f.index)


def generate(n_campaigns: int = 120, days: int = 150, seed: int = 21) -> pd.DataFrame:
    """Creative histories from campaigns run on an even split (creatives are refreshed on the
    advertiser's own schedule)."""
    rows = []
    for i in range(n_campaigns):
        rng = np.random.default_rng([seed, i])
        c = make_campaign(rng, 10_000 + i)
        for _ in range(days):
            rows += step(
                c, {s.ad_set_id: c.budget / len(c.ad_sets) * float(np.exp(rng.normal(0, 0.15))) for s in c.ad_sets}, rng
            )
    h = pd.DataFrame(rows)
    f = features_from_history(h)
    f["label"] = label(f, h)
    f["campaign"] = f["creative_id"].str.split("-").str[0]
    f = f.dropna(subset=["label"])
    return f[(f["ctr_ratio_to_peak"] >= HEALTHY) & (f["age_days"] >= 3)].reset_index(drop=True)
