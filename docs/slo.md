# Service level objectives

The recommendation API is called by the campaign tool every morning for every campaign, and by
people re-planning during the day. If it is slow or down, budgets are not moved and the
advertiser's spend runs on yesterday's split. The objectives below are what we promise; the alert
rules in `ops/prometheus/rules.yml` page when the error budget is burning too fast.

| SLO | Objective (28 days) | Measured by |
|---|---|---|
| Availability | 99.9% of API requests succeed (not 5xx) | `cbo_http_requests_total` |
| Latency | 99% of recommendation requests finish within 300 ms | `cbo_http_request_duration_seconds` for `/v1/campaigns/{campaign_id}/recommendations` |
| Freshness | new stats are reflected in the model within 30 minutes | `cbo_jobs_oldest_queued_seconds` |

Error budget: 0.1% of requests (about 40 minutes of total outage in 28 days).

## Alerting: multi-window burn rates

Following the Google SRE workbook, each SLO pages on two burn rates, each confirmed by a short
window so the alert clears quickly once the problem is fixed:

| Severity | Burn rate | Long window | Short window | Budget spent before the alert fires |
|---|---|---|---|---|
| page | 14.4x | 1 h | 5 min | 2% |
| page | 6x | 6 h | 30 min | 5% |
| ticket | 1x | 3 d | 6 h | 10% |

Latency uses the same scheme with "slower than 300 ms" as the bad event.

## Measured

The load test (`loadtest/locustfile.py`, run in CI against the real API and PostgreSQL) drives a
morning-like mix: 70% recommendations, 20% campaign reads, 10% stats deliveries. Results are in
the README.

## Runbooks

### High error rate
1. `kubectl -n cbo logs deploy/cbo-api` — every line carries the request id returned in `x-request-id`.
2. `/ready` fails when PostgreSQL is unreachable: check the database and the connection pool
   (`cbo_http_requests_total{code="503"}`).
3. Roll back with `kubectl -n cbo rollout undo deploy/cbo-api` if a release caused it.

### Slow recommendations
Recommendations read fitted curves; they never fit on the request path. If latency rises, check
PostgreSQL (slow queries on `daily_stats`), CPU throttling of the API pods, and campaigns with
unusually many ad sets. Scale with the HPA or add replicas.

### Stale model (freshness)
`cbo_jobs_oldest_queued_seconds` above 30 minutes: workers are down or too few. Check
`kubectl -n cbo get pods -l app=cbo-worker`, then `cbo_jobs_dead` — dead jobs carry `last_error`
in the `jobs` table. Fix the cause and requeue with
`update jobs set status = 'queued', attempts = 0, run_after = now() where status = 'dead'`.
Recommendations stay available meanwhile, on the previous fit, and say how old it is
(`model_state_as_of`).

### Dead jobs
Any job in `dead` means a campaign's model is not updating. See above.
