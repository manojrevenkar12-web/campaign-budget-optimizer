"""Morning traffic: mostly recommendations, some campaign reads and stats deliveries.

    locust -f loadtest/locustfile.py --headless -u 20 -r 5 -t 60s --host http://localhost:8000 --csv out
with CBO_API_KEY and CBO_CAMPAIGNS (comma-separated ids) in the environment.
"""

import os
import random
from datetime import date, timedelta

from locust import HttpUser, between, task

CAMPAIGNS = os.environ.get("CBO_CAMPAIGNS", "c0000").split(",")
KEY = os.environ.get("CBO_API_KEY", "")


class Planner(HttpUser):
    wait_time = between(0.05, 0.2)

    def on_start(self):
        self.client.headers["Authorization"] = f"Bearer {KEY}"
        self.ad_sets = {}
        for c in CAMPAIGNS:
            r = self.client.get(f"/v1/campaigns/{c}", name="/v1/campaigns/{id}")
            self.ad_sets[c] = [a["id"] for a in r.json()["ad_sets"]]

    @task(7)
    def recommend(self):
        c = random.choice(CAMPAIGNS)
        day = date.today() + timedelta(days=random.randint(0, 6))
        self.client.post(
            f"/v1/campaigns/{c}/recommendations",
            json={"day": day.isoformat()},
            name="/v1/campaigns/{id}/recommendations",
        )

    @task(2)
    def read(self):
        self.client.get(f"/v1/campaigns/{random.choice(CAMPAIGNS)}", name="/v1/campaigns/{id}")

    @task(1)
    def deliver_stats(self):
        c = random.choice(CAMPAIGNS)
        day = (date.today() - timedelta(days=1)).isoformat()
        rows = [
            {
                "ad_set_id": a,
                "day": day,
                "spend": round(random.uniform(100, 2000), 2),
                "impressions": 100000,
                "clicks": 1200,
                "conversions": random.randint(5, 60),
            }
            for a in self.ad_sets[c]
        ]
        self.client.post(f"/v1/campaigns/{c}/stats", json={"rows": rows}, name="/v1/campaigns/{id}/stats")
