"""Load simulated advertisers into a running API, the way a platform connector would."""

from __future__ import annotations

import hashlib
import secrets
from datetime import date, timedelta

import numpy as np

from cbo.sim.market import make_campaign, step


def create_tenant(pool, tenant_id: str, name: str = "") -> str:
    key = "cbo_" + secrets.token_urlsafe(24)
    with pool.connection() as conn:
        conn.execute(
            "insert into tenants (id, name) values (%s, %s) on conflict (id) do nothing", (tenant_id, name or tenant_id)
        )
        conn.execute(
            "insert into api_keys (key_hash, tenant_id, label) values (%s, %s, 'seed')",
            (hashlib.sha256(key.encode()).hexdigest(), tenant_id),
        )
        conn.commit()
    return key


def seed(client, campaigns: int = 10, days: int = 60, seed_: int = 5, end: date | None = None) -> list[str]:
    """`client` is an httpx.Client (or FastAPI TestClient) already carrying the API key."""
    end = end or date.today() - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    ids = []
    for i in range(campaigns):
        rng = np.random.default_rng([seed_, i])
        c = make_campaign(rng, i)
        ids.append(c.campaign_id)
        r = client.put(
            f"/v1/campaigns/{c.campaign_id}",
            json={
                "name": f"Campaign {i}",
                "daily_budget": c.budget,
                "ad_sets": [{"id": s.ad_set_id, "platform": s.platform} for s in c.ad_sets],
            },
        )
        r.raise_for_status()
        rows = []
        for d in range(days):
            spends = {s.ad_set_id: c.budget / len(c.ad_sets) * float(np.exp(rng.normal(0, 0.2))) for s in c.ad_sets}
            for r_ in step(c, spends, rng):
                rows.append(
                    {
                        "ad_set_id": r_["ad_set_id"],
                        "day": (start + timedelta(days=d)).isoformat(),
                        "spend": r_["spend"],
                        "impressions": r_["impressions"],
                        "clicks": r_["clicks"],
                        "conversions": r_["conversions"],
                        "creative": {
                            "id": r_["creative_id"],
                            "format": r_["creative_format"],
                            "age_days": r_["creative_age"],
                            "frequency": r_["frequency"],
                        },
                    }
                )
        for k in range(0, len(rows), 2000):
            client.post(f"/v1/campaigns/{c.campaign_id}/stats", json={"rows": rows[k : k + 2000]}).raise_for_status()
    return ids
