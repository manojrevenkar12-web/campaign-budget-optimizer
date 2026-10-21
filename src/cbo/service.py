"""Domain operations shared by the API and the worker."""

from __future__ import annotations

import hashlib
from datetime import date, timedelta

import numpy as np
from psycopg import Connection
from psycopg.types.json import Jsonb

from cbo.fatigue.data import FEATURES, HEALTHY, features_from_history
from cbo.fatigue.model import FatigueModel
from cbo.model.response import Posterior, fit, intervals, pooled_cpa, weekday_multipliers
from cbo.optim.allocate import Guardrails, allocate, bounds, marginal

HISTORY_DAYS = 120


class NotFound(Exception):
    pass


def campaign(conn: Connection, tenant: str, campaign_id: str) -> dict:
    c = conn.execute("select * from campaigns where id = %s and tenant_id = %s", (campaign_id, tenant)).fetchone()
    if not c:
        raise NotFound(f"campaign {campaign_id}")  # also for other tenants' campaigns: no existence leak
    c["ad_sets"] = conn.execute(
        "select * from ad_sets where campaign_id = %s and active order by id", (campaign_id,)
    ).fetchall()
    return c


# ------------------------------------------------------------------------------------- model fitting
def refresh_campaign(conn: Connection, campaign_id: str, fatigue: FatigueModel | None) -> dict:
    """Fit the response curve of every ad set and score its creatives; called by the worker."""
    sets = conn.execute("select id, platform from ad_sets where campaign_id = %s and active", (campaign_id,)).fetchall()
    rows = conn.execute(
        """select s.ad_set_id, a.platform, s.day, s.spend::float as spend, s.conversions
           from daily_stats s join ad_sets a on a.id = s.ad_set_id
           where a.campaign_id = %s and s.day > (select max(day) from daily_stats s2 join ad_sets a2 on a2.id = s2.ad_set_id where a2.campaign_id = %s) - %s
           order by s.day""",
        (campaign_id, campaign_id, HISTORY_DAYS),
    ).fetchall()
    if not rows:
        return {"ad_sets": 0, "creatives": 0}
    ids = np.array([r["ad_set_id"] for r in rows])
    plat = np.array([r["platform"] for r in rows])
    days = np.array([r["day"] for r in rows])
    last = days.max()
    age = np.array([(last - d).days for d in days])
    wday = np.array([d.weekday() for d in days])
    spend = np.array([r["spend"] for r in rows], float)
    conv = np.array([r["conversions"] for r in rows], float)
    fitted = 0
    for s in sets:
        pm = plat == s["platform"]
        mult = weekday_multipliers(wday[pm], conv[pm], spend[pm])
        m = ids == s["id"]
        post = fit(s["id"], spend[m], conv[m], age[m], mult[wday[m]], prior_cpa=pooled_cpa(spend[pm], conv[pm]))
        state = {**post.to_json(), "weekday_mult": mult.tolist()}
        conn.execute(
            """insert into posteriors (ad_set_id, state, fitted_through, updated_at) values (%s, %s, %s, now())
               on conflict (ad_set_id) do update set state = excluded.state, fitted_through = excluded.fitted_through, updated_at = now()""",
            (s["id"], Jsonb(state), last),
        )
        fitted += 1
    scored = score_creatives(conn, campaign_id, fatigue) if fatigue else 0
    return {"ad_sets": fitted, "creatives": scored, "fitted_through": last.isoformat()}


def score_creatives(conn: Connection, campaign_id: str, model: FatigueModel) -> int:
    import pandas as pd

    rows = conn.execute(
        """select c.creative_id, c.ad_set_id, c.day, a.platform, c.creative_format, c.creative_age, c.frequency, c.impressions, c.clicks
           from creative_daily c join ad_sets a on a.id = c.ad_set_id
           where a.campaign_id = %s and c.creative_id in (
               select distinct on (c2.ad_set_id) c2.creative_id from creative_daily c2 join ad_sets a2 on a2.id = c2.ad_set_id
               where a2.campaign_id = %s order by c2.ad_set_id, c2.day desc)
           order by c.creative_id, c.day""",
        (campaign_id, campaign_id),
    ).fetchall()
    if not rows:
        return 0
    h = pd.DataFrame(rows)
    h["day"] = pd.to_datetime(h["day"])
    f = features_from_history(h)
    latest = f.groupby("creative_id").tail(1)
    p = model.predict(latest[FEATURES].to_numpy(float))
    # the model is trained on creatives that still look healthy; one already below 80% of its peak
    # CTR is reported as fatigued, not scored
    worn = latest["ctr_ratio_to_peak"].to_numpy() < HEALTHY
    meta = h.groupby("creative_id").agg(ad_set_id=("ad_set_id", "first"), day=("day", "max"))
    for cid, prob, w in zip(latest["creative_id"], p, worn, strict=True):
        conn.execute(
            """insert into fatigue_scores (creative_id, ad_set_id, day, probability, state, model_version, scored_at) values (%s, %s, %s, %s, %s, %s, now())
               on conflict (creative_id) do update set day = excluded.day, probability = excluded.probability, state = excluded.state,
                 model_version = excluded.model_version, scored_at = now()""",
            (
                cid,
                meta.loc[cid, "ad_set_id"],
                meta.loc[cid, "day"].date(),
                1.0 if w else float(prob),
                "already_fatigued" if w else "at_risk_scored",
                model.version,
            ),
        )
    return len(latest)


# ------------------------------------------------------------------------------------ recommendation
def _seed(campaign_id: str, day: date) -> int:
    return int.from_bytes(hashlib.sha256(f"{campaign_id}:{day.isoformat()}".encode()).digest()[:8], "big")


def recommend(
    conn: Connection,
    camp: dict,
    day: date,
    budget: float | None,
    policy: str,
    guardrails: Guardrails,
    exploration: float = 0.3,
) -> dict:
    """Allocation for `day`. Deterministic per campaign and day: the Thompson draw is seeded by
    both, so asking twice gives the same answer."""
    sets = camp["ad_sets"]
    if not sets:
        raise ValueError("campaign has no active ad sets")
    budget = float(budget if budget is not None else camp["daily_budget"])
    ids = [s["id"] for s in sets]
    states = {
        r["ad_set_id"]: r
        for r in conn.execute(
            "select ad_set_id, state, updated_at from posteriors where ad_set_id = any(%s)", (ids,)
        ).fetchall()
    }
    prev_rows = conn.execute(
        """select ad_set_id, spend::float as spend from daily_stats where ad_set_id = any(%s)
           and day = (select max(day) from daily_stats where ad_set_id = any(%s))""",
        (ids, ids),
    ).fetchall()
    prev_map = {r["ad_set_id"]: r["spend"] for r in prev_rows}
    previous = np.array([prev_map.get(i, 0.0) for i in ids]) if prev_map else None
    wd = day.weekday()
    posts, mults, cold = [], [], []
    for s in sets:
        st = states.get(s["id"])
        if st:
            state = dict(st["state"])
            mult = state.pop("weekday_mult", [1.0] * 7)[wd]
            posts.append(Posterior.from_json(state))
        else:  # new ad set: prior only, centred on the platform's typical cost per conversion
            posts.append(
                fit(s["id"], np.array([]), np.array([]), np.array([]), prior_cpa=30.0, typical_spend=budget / len(sets))
            )
            mult = 1.0
            cold.append(s["id"])
        mults.append(mult)
    lo, hi, relaxed = bounds(budget, len(sets), previous, guardrails)

    def scaled(curve, m):
        w, a, k = curve
        return (w, a * m, k)

    mean_curves = [scaled(p.mean_curve(), m) for p, m in zip(posts, mults, strict=True)]
    mean_alloc = allocate(mean_curves, budget, lo, hi)
    if policy == "posterior_mean":
        alloc = mean_alloc
    elif policy == "damped_thompson":
        rng = np.random.default_rng(_seed(camp["id"], day))
        draws = []
        for p, m in zip(posts, mults, strict=True):
            a, k = p.sample(rng)
            draws.append((np.array([1.0]), np.array([a * m]), np.array([k])))
        alloc = (1 - exploration) * mean_alloc + exploration * allocate(draws, budget, lo, hi)
    else:
        raise ValueError(f"unknown policy {policy}")
    lines = []
    ilos, ihis = intervals(posts, alloc)
    for s, p, m, x, mc, lo_i, hi_i, ilo, ihi in zip(
        sets, posts, mults, alloc, mean_curves, lo, hi, ilos, ihis, strict=True
    ):
        exp_conv = float(p.expected(x)[0] * m)
        mg = marginal(mc, float(x))
        lines.append(
            {
                "ad_set_id": s["id"],
                "platform": s["platform"],
                "previous_spend": None if previous is None else round(float(previous[ids.index(s["id"])]), 2),
                "recommended_spend": round(float(x), 2),
                "expected_conversions": round(exp_conv, 2),
                "conversions_80pct": [round(ilo * m, 2), round(ihi * m, 2)],
                "marginal_cost_per_conversion": round(1 / mg, 2) if mg > 0 else None,
                "at_lower_guardrail": bool(x <= lo_i + 0.01),
                "at_upper_guardrail": bool(x >= hi_i - 0.01),
                "cold_start": s["id"] in cold,
                "days_of_data": p.n_days,
            }
        )
    expected_total = sum(line["expected_conversions"] for line in lines)
    model_as_of = max((st["updated_at"] for st in states.values()), default=None)
    return {
        "campaign_id": camp["id"],
        "day": day.isoformat(),
        "policy": policy,
        "budget": round(budget, 2),
        "expected_conversions": round(expected_total, 2),
        "expected_cost_per_conversion": round(budget / expected_total, 2) if expected_total else None,
        "guardrails": {
            "min_spend": guardrails.min_spend,
            "min_share": guardrails.min_share,
            "max_change": guardrails.max_change,
            "relaxed": relaxed,
        },
        "model_state_as_of": model_as_of.isoformat() if model_as_of else None,
        "ad_sets": lines,
    }


def curve_points(conn: Connection, camp: dict, points: int = 25) -> list[dict]:
    """Posterior-mean response curve per ad set, for the dashboard."""
    ids = [s["id"] for s in camp["ad_sets"]]
    states = {
        r["ad_set_id"]: r["state"]
        for r in conn.execute("select ad_set_id, state from posteriors where ad_set_id = any(%s)", (ids,)).fetchall()
    }
    budget = float(camp["daily_budget"])
    xs = np.linspace(0, budget * 0.8, points)
    out = []
    for s in camp["ad_sets"]:
        st = states.get(s["id"])
        if not st:
            continue
        st = dict(st)
        st.pop("weekday_mult", None)
        p = Posterior.from_json(st)
        out.append(
            {
                "ad_set_id": s["id"],
                "platform": s["platform"],
                "spend": xs.round(2).tolist(),
                "conversions": p.expected(xs).round(3).tolist(),
            }
        )
    return out


def yesterday() -> date:
    return date.today() - timedelta(days=1)
