"""Recommendation API.

Advertisers' systems push daily ad-set and creative stats; the API stores them idempotently and
queues a model refresh. Each morning the campaign tool asks for today's budget split. The split is
computed from the latest fitted curves (written by the worker) in a few milliseconds, explained
per ad set, stored, and returned. Every route is tenant-scoped by API key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field, model_validator

from cbo import service
from cbo.db.db import make_pool, migrate
from cbo.optim.allocate import Guardrails

log = logging.getLogger("cbo.api")

PLATFORMS = ("meta", "tiktok", "pinterest", "google")


class AdSetIn(BaseModel):
    id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9._:-]+$")
    platform: str
    name: str = ""
    active: bool = True

    @model_validator(mode="after")
    def check_platform(self):
        if self.platform not in PLATFORMS:
            raise ValueError(f"platform must be one of {PLATFORMS}")
        return self


class CampaignIn(BaseModel):
    name: str = ""
    daily_budget: float = Field(gt=0, le=10_000_000)
    ad_sets: list[AdSetIn] = Field(min_length=1, max_length=50)


class CreativeIn(BaseModel):
    id: str = Field(min_length=1, max_length=120)
    format: str = Field(pattern="^(static|carousel|video)$")
    age_days: int = Field(ge=0)
    frequency: float = Field(ge=0)


class StatIn(BaseModel):
    ad_set_id: str
    day: date
    spend: float = Field(ge=0)
    impressions: int = Field(ge=0)
    clicks: int = Field(ge=0)
    conversions: int = Field(ge=0)
    creative: CreativeIn | None = None

    @model_validator(mode="after")
    def clicks_within_impressions(self):
        if self.clicks > self.impressions:
            raise ValueError("clicks cannot exceed impressions")
        return self


class StatsIn(BaseModel):
    rows: list[StatIn] = Field(min_length=1, max_length=5000)


class GuardrailsIn(BaseModel):
    min_spend: float = Field(10.0, ge=0)
    min_share: float = Field(0.03, ge=0, le=0.5)
    max_change: float = Field(0.30, gt=0, le=5)


class RecommendationIn(BaseModel):
    day: date
    budget: float | None = Field(None, gt=0)
    policy: str = Field("damped_thompson", pattern="^(damped_thompson|posterior_mean)$")
    guardrails: GuardrailsIn = GuardrailsIn()


def build_metrics() -> dict:
    reg = CollectorRegistry()
    return {
        "registry": reg,
        "requests": Counter("cbo_http_requests_total", "HTTP requests", ["route", "method", "code"], registry=reg),
        "latency": Histogram(
            "cbo_http_request_duration_seconds",
            "HTTP latency",
            ["route"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2.5),
            registry=reg,
        ),
        "rows": Counter("cbo_stats_rows_ingested_total", "Daily stat rows stored", registry=reg),
        "recs": Counter("cbo_recommendations_total", "Recommendations served", ["policy", "cold_start"], registry=reg),
        "queue": Gauge("cbo_jobs_queued", "Jobs waiting", registry=reg),
        "lag": Gauge("cbo_jobs_oldest_queued_seconds", "Age of the oldest queued job", registry=reg),
        "dead": Gauge("cbo_jobs_dead", "Jobs that exhausted their retries", registry=reg),
    }


def create_app(database_url: str | None = None) -> FastAPI:
    metrics = build_metrics()
    key_cache: dict[str, tuple[str, float]] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pool = make_pool(database_url)
        migrate(app.state.pool)
        yield
        app.state.pool.close()

    app = FastAPI(title="Campaign budget optimizer", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception(json.dumps({"request_id": rid, "event": "unhandled"}))
            response = JSONResponse({"error": "internal", "request_id": rid}, status_code=500)
        route = getattr(request.scope.get("route"), "path", "unmatched")
        dt = time.perf_counter() - t0
        metrics["requests"].labels(route, request.method, str(response.status_code)).inc()
        metrics["latency"].labels(route).observe(dt)
        response.headers["x-request-id"] = rid
        if route not in ("/health", "/metrics"):
            log.info(
                json.dumps(
                    {
                        "request_id": rid,
                        "method": request.method,
                        "route": route,
                        "status": response.status_code,
                        "ms": round(dt * 1000, 1),
                    }
                )
            )
        return response

    def tenant(authorization: str = Header(default="")) -> str:
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing API key")
        h = hashlib.sha256(authorization[7:].encode()).hexdigest()
        hit = key_cache.get(h)
        if hit and hit[1] > time.monotonic():
            return hit[0]
        with app.state.pool.connection() as conn:
            r = conn.execute(
                "select tenant_id from api_keys where key_hash = %s and revoked_at is null", (h,)
            ).fetchone()
        if not r:
            raise HTTPException(401, "invalid API key")
        key_cache[h] = (r["tenant_id"], time.monotonic() + 60)
        return r["tenant_id"]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        try:
            with app.state.pool.connection(timeout=2) as conn:
                conn.execute("select 1")
        except Exception as e:
            raise HTTPException(503, f"database unavailable: {type(e).__name__}") from e
        return {"status": "ready"}

    @app.get("/metrics")
    def prom():
        with app.state.pool.connection() as conn:
            r = conn.execute(
                """select count(*) filter (where status = 'queued') as queued,
                          coalesce(extract(epoch from now() - min(created_at) filter (where status = 'queued')), 0) as lag,
                          count(*) filter (where status = 'dead') as dead from jobs"""
            ).fetchone()
        metrics["queue"].set(r["queued"])
        metrics["lag"].set(float(r["lag"]))
        metrics["dead"].set(r["dead"])
        return PlainTextResponse(generate_latest(metrics["registry"]).decode(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/campaigns")
    def list_campaigns(t: str = Depends(tenant)):
        with app.state.pool.connection() as conn:
            rows = conn.execute(
                """select c.id, c.name, c.daily_budget::float as daily_budget, count(a.id) as ad_sets,
                          (select max(day) from daily_stats s join ad_sets a2 on a2.id = s.ad_set_id where a2.campaign_id = c.id) as last_day
                   from campaigns c left join ad_sets a on a.campaign_id = c.id and a.active
                   where c.tenant_id = %s group by c.id order by c.id""",
                (t,),
            ).fetchall()
        return {"campaigns": rows}

    @app.put("/v1/campaigns/{campaign_id}")
    def upsert_campaign(campaign_id: str, body: CampaignIn, t: str = Depends(tenant)):
        with app.state.pool.connection() as conn:
            owner = conn.execute("select tenant_id from campaigns where id = %s", (campaign_id,)).fetchone()
            if owner and owner["tenant_id"] != t:
                raise HTTPException(404, f"campaign {campaign_id}")
            conn.execute(
                """insert into campaigns (id, tenant_id, name, daily_budget) values (%s, %s, %s, %s)
                   on conflict (id) do update set name = excluded.name, daily_budget = excluded.daily_budget, updated_at = now()""",
                (campaign_id, t, body.name, body.daily_budget),
            )
            for a in body.ad_sets:
                other = conn.execute("select campaign_id from ad_sets where id = %s", (a.id,)).fetchone()
                if other and other["campaign_id"] != campaign_id:
                    raise HTTPException(409, f"ad set {a.id} belongs to another campaign")
                conn.execute(
                    """insert into ad_sets (id, campaign_id, platform, name, active) values (%s, %s, %s, %s, %s)
                       on conflict (id) do update set platform = excluded.platform, name = excluded.name, active = excluded.active""",
                    (a.id, campaign_id, a.platform, a.name, a.active),
                )
            conn.commit()
            return service.campaign(conn, t, campaign_id)

    @app.post("/v1/campaigns/{campaign_id}/stats")
    def ingest(campaign_id: str, body: StatsIn, t: str = Depends(tenant)):
        """Idempotent: re-sending a day overwrites it (platforms restate yesterday's numbers)."""
        with app.state.pool.connection() as conn:
            camp = _campaign(conn, t, campaign_id)
            known = {s["id"] for s in camp["ad_sets"]}
            unknown = sorted({r.ad_set_id for r in body.rows} - known)
            if unknown:
                raise HTTPException(422, {"error": "unknown ad sets for this campaign", "ad_sets": unknown[:20]})
            with conn.cursor() as cur:
                cur.executemany(
                    """insert into daily_stats (ad_set_id, day, spend, impressions, clicks, conversions) values (%s, %s, %s, %s, %s, %s)
                       on conflict (ad_set_id, day) do update set spend = excluded.spend, impressions = excluded.impressions,
                         clicks = excluded.clicks, conversions = excluded.conversions, ingested_at = now()""",
                    [(r.ad_set_id, r.day, r.spend, r.impressions, r.clicks, r.conversions) for r in body.rows],
                )
                cr = [r for r in body.rows if r.creative]
                if cr:
                    cur.executemany(
                        """insert into creative_daily (creative_id, ad_set_id, day, creative_format, creative_age, frequency, impressions, clicks)
                           values (%s, %s, %s, %s, %s, %s, %s, %s)
                           on conflict (creative_id, day) do update set creative_age = excluded.creative_age, frequency = excluded.frequency,
                             impressions = excluded.impressions, clicks = excluded.clicks""",
                        [
                            (
                                r.creative.id,
                                r.ad_set_id,
                                r.day,
                                r.creative.format,
                                r.creative.age_days,
                                r.creative.frequency,
                                r.impressions,
                                r.clicks,
                            )
                            for r in cr
                        ],
                    )
            conn.execute(
                """insert into jobs (kind, campaign_id) values ('refresh_campaign', %s)
                   on conflict (kind, campaign_id) where status = 'queued' do update set updated_at = now()""",
                (campaign_id,),
            )
            conn.commit()
        metrics["rows"].inc(len(body.rows))
        return {"stored": len(body.rows), "refresh": "queued"}

    @app.post("/v1/campaigns/{campaign_id}/recommendations")
    def recommendation(
        campaign_id: str,
        body: RecommendationIn,
        t: str = Depends(tenant),
        idempotency_key: str | None = Header(default=None),
    ):
        with app.state.pool.connection() as conn:
            camp = _campaign(conn, t, campaign_id)
            if idempotency_key:
                prev = conn.execute(
                    "select allocation from recommendations where campaign_id = %s and idempotency_key = %s",
                    (campaign_id, idempotency_key),
                ).fetchone()
                if prev:
                    return prev["allocation"]
            g = Guardrails(body.guardrails.min_spend, body.guardrails.min_share, body.guardrails.max_change)
            try:
                rec = service.recommend(conn, camp, body.day, body.budget, body.policy, g)
            except ValueError as e:
                raise HTTPException(422, str(e)) from e
            rec["recommendation_id"] = str(uuid.uuid4())
            conn.execute(
                """insert into recommendations (id, campaign_id, day, policy, budget, allocation, model_state_as_of, idempotency_key)
                   values (%s, %s, %s, %s, %s, %s, %s, %s) on conflict (campaign_id, idempotency_key) do nothing""",
                (
                    rec["recommendation_id"],
                    campaign_id,
                    body.day,
                    body.policy,
                    rec["budget"],
                    Jsonb(rec),
                    rec["model_state_as_of"],
                    idempotency_key,
                ),
            )
            conn.commit()
        metrics["recs"].labels(body.policy, str(any(a["cold_start"] for a in rec["ad_sets"])).lower()).inc()
        return rec

    @app.get("/v1/campaigns/{campaign_id}")
    def get_campaign(campaign_id: str, t: str = Depends(tenant)):
        with app.state.pool.connection() as conn:
            camp = _campaign(conn, t, campaign_id)
            camp["daily_budget"] = float(camp["daily_budget"])
            camp["curves"] = service.curve_points(conn, camp)
            camp["recent"] = conn.execute(
                """select s.day, s.ad_set_id, s.spend::float as spend, s.conversions from daily_stats s join ad_sets a on a.id = s.ad_set_id
                   where a.campaign_id = %s and s.day > (select max(day) - 14 from daily_stats s2 join ad_sets a2 on a2.id = s2.ad_set_id where a2.campaign_id = %s)
                   order by s.day, s.ad_set_id""",
                (campaign_id, campaign_id),
            ).fetchall()
            return camp

    @app.get("/v1/campaigns/{campaign_id}/recommendations/latest")
    def latest(campaign_id: str, t: str = Depends(tenant)):
        with app.state.pool.connection() as conn:
            _campaign(conn, t, campaign_id)
            r = conn.execute(
                "select allocation from recommendations where campaign_id = %s order by created_at desc limit 1",
                (campaign_id,),
            ).fetchone()
        if not r:
            raise HTTPException(404, "no recommendation yet")
        return r["allocation"]

    @app.get("/v1/campaigns/{campaign_id}/creatives/at-risk")
    def at_risk(campaign_id: str, threshold: float = 0.5, t: str = Depends(tenant)):
        with app.state.pool.connection() as conn:
            _campaign(conn, t, campaign_id)
            rows = conn.execute(
                """select f.creative_id, f.ad_set_id, f.day, f.probability, f.state, f.model_version from fatigue_scores f join ad_sets a on a.id = f.ad_set_id
                   where a.campaign_id = %s and f.probability >= %s order by f.probability desc""",
                (campaign_id, threshold),
            ).fetchall()
        return {"threshold": threshold, "creatives": rows}

    def _campaign(conn, t, campaign_id):
        try:
            return service.campaign(conn, t, campaign_id)
        except service.NotFound as e:
            raise HTTPException(404, str(e)) from e

    return app
