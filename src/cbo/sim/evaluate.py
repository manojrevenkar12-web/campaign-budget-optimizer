"""Compare budget policies on simulated campaigns, on the same market (common random numbers).

Every campaign starts with three weeks of an even split (the history an advertiser brings), then
each policy runs it for `days` days. Reported: conversions and cost per conversion per policy, the
uplift over the rule of thumb with a 95% bootstrap interval over campaigns, the share of the
oracle's gain each policy captures, and how much budgets move day to day.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from cbo.optim.allocate import Guardrails
from cbo.sim.market import Campaign, make_campaign, step
from cbo.sim.policies import POLICIES, even

WARMUP = 21


def run_campaign(base: Campaign, idx: int, policy: str, days: int, seed: int, g: Guardrails) -> dict:
    c = copy.deepcopy(base)
    fn = POLICIES[policy]
    prev = None
    conv = spend = expected = 0.0
    moves = []
    for d in range(WARMUP + days):
        obs = np.random.default_rng([seed, idx, d, 1])
        world = np.random.default_rng([seed, idx, d, 2])
        decide = np.random.default_rng([seed, idx, d, 3, hash(policy) % 1000])
        alloc = even(c, prev, decide, g) if d < WARMUP else fn(c, prev, decide, g)
        rows = step(c, {s.ad_set_id: float(x) for s, x in zip(c.ad_sets, alloc, strict=True)}, obs, world_rng=world)
        if d >= WARMUP:
            conv += sum(r["conversions"] for r in rows)
            expected += sum(r["true_expected_conversions"] for r in rows)
            spend += sum(r["spend"] for r in rows)
            if prev is not None:
                moves.append(float(np.abs(alloc - prev).sum() / 2 / c.budget))
        prev = np.asarray(alloc, float)
    return {
        "campaign": base.campaign_id,
        "policy": policy,
        "conversions": conv,
        "expected_conversions": expected,
        "spend": spend,
        "daily_reallocation": float(np.mean(moves)) if moves else 0.0,
    }


def evaluate(
    n_campaigns: int = 60,
    days: int = 60,
    seed: int = 11,
    policies: list[str] | None = None,
    g: Guardrails = Guardrails(),
) -> tuple[pd.DataFrame, dict]:
    policies = policies or list(POLICIES)
    rows = []
    for i in range(n_campaigns):
        base = make_campaign(np.random.default_rng([seed, i, 0]), i)
        for p in policies:
            rows.append(run_campaign(base, i, p, days, seed, g))
    res = pd.DataFrame(rows)
    return res, summarise(res)


def summarise(res: pd.DataFrame, reference: str = "rule_of_thumb", n_boot: int = 4000, seed: int = 0) -> dict:
    # expected conversions (the true mean behind each day's draw) remove Poisson noise from the comparison
    piv = res.pivot(index="campaign", columns="policy", values="expected_conversions")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(piv), size=(n_boot, len(piv)))
    out = {}
    for p in piv.columns:
        g = res[res["policy"] == p]
        row = {
            "conversions": float(g["conversions"].sum()),
            "expected_conversions": float(g["expected_conversions"].sum()),
            "cost_per_conversion": float(g["spend"].sum() / g["expected_conversions"].sum()),
            "daily_reallocation": float(g["daily_reallocation"].mean()),
        }
        if p != reference:
            d = piv[p].to_numpy() - piv[reference].to_numpy()
            base = piv[reference].to_numpy()
            boot = d[idx].sum(1) / base[idx].sum(1)
            row["uplift_vs_" + reference] = float(d.sum() / base.sum())
            row["uplift_ci95"] = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
            row["campaigns_better"] = float((d > 0).mean())
        out[p] = row
    if "oracle" in out:
        gap = out["oracle"]["expected_conversions"] - out[reference]["expected_conversions"]
        for p in out:
            if p not in ("oracle", reference):
                out[p]["share_of_oracle_gain"] = (
                    float((out[p]["expected_conversions"] - out[reference]["expected_conversions"]) / gap)
                    if gap > 0
                    else None
                )
    return out
