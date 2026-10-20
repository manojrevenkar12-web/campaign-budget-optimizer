"""The services against a real PostgreSQL: tenancy, validation, idempotency, recommendations,
the job queue and its failure handling."""

import threading
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from cbo.api.app import create_app
from cbo.db.db import make_pool, migrate
from cbo.seed import create_tenant, seed
from cbo.worker import worker

pytestmark = pytest.mark.db
TODAY = date.today()


@pytest.fixture(scope="module")
def env(database_url, fatigue_model):
    pool = make_pool(database_url, max_size=8)
    migrate(pool)
    app = create_app(database_url)
    with TestClient(app) as tc:
        key_a, key_b = create_tenant(pool, "acme"), create_tenant(pool, "globex")
        a = TestClient(app, headers={"Authorization": f"Bearer {key_a}"})
        b = TestClient(app, headers={"Authorization": f"Bearer {key_b}"})
        ids = seed(a, campaigns=3, days=45)
        worker.drain(pool, fatigue_model[0])
        yield {"pool": pool, "a": a, "b": b, "ids": ids, "anon": tc}
    pool.close()


def test_auth_and_tenant_isolation(env):
    cid = env["ids"][0]
    assert env["anon"].get("/v1/campaigns").status_code == 401
    assert env["anon"].get("/v1/campaigns", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert env["b"].get(f"/v1/campaigns/{cid}").status_code == 404  # not 403: no existence leak
    assert (
        env["b"]
        .put(f"/v1/campaigns/{cid}", json={"daily_budget": 1, "ad_sets": [{"id": "x1", "platform": "meta"}]})
        .status_code
        == 404
    )
    assert env["b"].get("/v1/campaigns").json()["campaigns"] == []


def test_ingest_validates_and_is_idempotent(env):
    cid = env["ids"][0]
    camp = env["a"].get(f"/v1/campaigns/{cid}").json()
    ad = camp["ad_sets"][0]["id"]
    bad = env["a"].post(
        f"/v1/campaigns/{cid}/stats",
        json={
            "rows": [
                {"ad_set_id": ad, "day": "2026-01-01", "spend": 10, "impressions": 5, "clicks": 9, "conversions": 0}
            ]
        },
    )
    assert bad.status_code == 422
    unknown = env["a"].post(
        f"/v1/campaigns/{cid}/stats",
        json={
            "rows": [
                {"ad_set_id": "nope", "day": "2026-01-01", "spend": 10, "impressions": 5, "clicks": 1, "conversions": 0}
            ]
        },
    )
    assert unknown.status_code == 422 and unknown.json()["detail"]["ad_sets"] == ["nope"]
    row = {
        "ad_set_id": ad,
        "day": (TODAY - timedelta(days=3)).isoformat(),
        "spend": 123.45,
        "impressions": 1000,
        "clicks": 10,
        "conversions": 2,
    }
    for _ in range(2):
        assert env["a"].post(f"/v1/campaigns/{cid}/stats", json={"rows": [row]}).status_code == 200
    with env["pool"].connection() as conn:
        n = conn.execute(
            "select count(*) as n from daily_stats where ad_set_id = %s and day = %s", (ad, row["day"])
        ).fetchone()["n"]
        queued = conn.execute(
            "select count(*) as n from jobs where campaign_id = %s and status = 'queued'", (cid,)
        ).fetchone()["n"]
    assert n == 1 and queued == 1  # one row per day, one pending refresh however many deliveries
    worker.drain(env["pool"])


def test_recommendation_respects_budget_and_guardrails(env):
    cid = env["ids"][0]
    r = env["a"].post(
        f"/v1/campaigns/{cid}/recommendations",
        json={"day": TODAY.isoformat(), "budget": 5000, "guardrails": {"max_change": 0.2}},
    )
    assert r.status_code == 200
    rec = r.json()
    spends = [a["recommended_spend"] for a in rec["ad_sets"]]
    assert abs(sum(spends) - 5000) < 0.05 * len(spends)
    for a in rec["ad_sets"]:
        assert a["conversions_80pct"][0] <= a["expected_conversions"] <= a["conversions_80pct"][1]
        if not rec["guardrails"]["relaxed"]:
            assert a["recommended_spend"] <= a["previous_spend"] * 1.2 + 0.05
    assert rec["model_state_as_of"] is not None


def test_same_day_same_answer_and_idempotency_key(env):
    cid = env["ids"][1]
    body = {"day": TODAY.isoformat()}
    r1 = env["a"].post(f"/v1/campaigns/{cid}/recommendations", json=body).json()
    r2 = env["a"].post(f"/v1/campaigns/{cid}/recommendations", json=body).json()
    assert [a["recommended_spend"] for a in r1["ad_sets"]] == [a["recommended_spend"] for a in r2["ad_sets"]]
    k1 = env["a"].post(f"/v1/campaigns/{cid}/recommendations", json=body, headers={"Idempotency-Key": "abc"}).json()
    k2 = env["a"].post(f"/v1/campaigns/{cid}/recommendations", json=body, headers={"Idempotency-Key": "abc"}).json()
    assert k1["recommendation_id"] == k2["recommendation_id"]
    assert (
        env["a"].get(f"/v1/campaigns/{cid}/recommendations/latest").json()["recommendation_id"]
        == k1["recommendation_id"]
    )


def test_new_ad_set_starts_from_the_prior(env):
    cid = env["ids"][2]
    camp = env["a"].get(f"/v1/campaigns/{cid}").json()
    sets = [{"id": s["id"], "platform": s["platform"]} for s in camp["ad_sets"]] + [
        {"id": f"{cid}-new", "platform": "tiktok"}
    ]
    assert (
        env["a"].put(f"/v1/campaigns/{cid}", json={"daily_budget": camp["daily_budget"], "ad_sets": sets}).status_code
        == 200
    )
    rec = env["a"].post(f"/v1/campaigns/{cid}/recommendations", json={"day": TODAY.isoformat()}).json()
    new = [a for a in rec["ad_sets"] if a["ad_set_id"] == f"{cid}-new"][0]
    assert new["cold_start"] and new["recommended_spend"] > 0


def test_creatives_are_scored(env):
    r = env["a"].get(f"/v1/campaigns/{env['ids'][0]}/creatives/at-risk?threshold=0").json()
    assert r["creatives"] and all(c["state"] in ("at_risk_scored", "already_fatigued") for c in r["creatives"])


def test_failed_jobs_retry_then_go_dead(env, monkeypatch):
    pool, cid = env["pool"], env["ids"][0]
    with pool.connection() as conn:
        conn.execute("insert into jobs (kind, campaign_id, max_attempts) values ('refresh_campaign', %s, 2)", (cid,))
        conn.commit()
    monkeypatch.setattr(worker.service, "refresh_campaign", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert "error" in worker.run_once(pool)
    with pool.connection() as conn:
        conn.execute("update jobs set run_after = now() where status = 'queued'")
        conn.commit()
    worker.run_once(pool)
    with pool.connection() as conn:
        job = conn.execute(
            "select status, attempts, last_error from jobs where campaign_id = %s order by id desc limit 1", (cid,)
        ).fetchone()
        conn.execute("delete from jobs where status = 'dead'")
        conn.commit()
    assert job["status"] == "dead" and job["attempts"] == 2 and "boom" in job["last_error"]


def test_concurrent_workers_never_take_the_same_job(env, monkeypatch):
    pool = env["pool"]
    with pool.connection() as conn:
        conn.execute("delete from jobs")
        for cid in env["ids"]:
            conn.execute("insert into jobs (kind, campaign_id, status) values ('refresh_campaign', %s, 'done')", (cid,))
        for i in range(30):
            conn.execute(
                "insert into jobs (kind, campaign_id, status) values ('refresh_campaign', %s, 'queued') on conflict do nothing",
                (env["ids"][i % 3],),
            )
        # partial unique index allows one queued per campaign: add more by using done->queued later
        conn.commit()
    seen, lock = [], threading.Lock()

    def slow(conn, campaign_id, fatigue):
        import time

        time.sleep(0.05)
        return {"ad_sets": 0, "creatives": 0}

    monkeypatch.setattr(worker.service, "refresh_campaign", slow)

    def run():
        while (r := worker.run_once(pool)) is not None:
            with lock:
                seen.append(r["job"])

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(seen) == len(set(seen)) == 3


def test_metrics_expose_queue_health(env):
    m = env["a"].get("/metrics").text
    for name in (
        "cbo_http_request_duration_seconds_bucket",
        "cbo_jobs_queued",
        "cbo_jobs_oldest_queued_seconds",
        "cbo_recommendations_total",
    ):
        assert name in m
