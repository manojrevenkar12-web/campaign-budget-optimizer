"""Background worker: refits response curves and scores creatives when new stats arrive.

Jobs live in PostgreSQL. A worker claims one with FOR UPDATE SKIP LOCKED, so any number of
replicas can run without double work; a claimed job carries a lease (`locked_until`), so a job
whose worker died is picked up again. Failures retry with exponential backoff and end in `dead`
after `max_attempts`, which is alerted on.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time

from prometheus_client import Counter, Histogram, start_http_server
from psycopg_pool import ConnectionPool

from cbo import service
from cbo.fatigue.train import load_champion

log = logging.getLogger("cbo.worker")
LEASE_SECONDS = 300

JOBS = Counter("cbo_worker_jobs_total", "Jobs finished", ["kind", "outcome"])
DURATION = Histogram(
    "cbo_worker_job_duration_seconds", "Job duration", ["kind"], buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60)
)

CLAIM = """
update jobs set status = 'running', attempts = attempts + 1, locked_until = now() + make_interval(secs => %s), updated_at = now()
where id = (
    select id from jobs
    where (status = 'queued' and run_after <= now()) or (status = 'running' and locked_until < now())
    order by run_after, id
    for update skip locked
    limit 1
)
returning id, kind, campaign_id, attempts, max_attempts
"""


def claim(pool: ConnectionPool) -> dict | None:
    with pool.connection() as conn:
        job = conn.execute(CLAIM, (LEASE_SECONDS,)).fetchone()
        conn.commit()
        return job


def finish(pool: ConnectionPool, job: dict, error: str | None) -> None:
    with pool.connection() as conn:
        if error is None:
            conn.execute(
                "update jobs set status = 'done', locked_until = null, last_error = null, updated_at = now() where id = %s",
                (job["id"],),
            )
        elif job["attempts"] >= job["max_attempts"]:
            conn.execute(
                "update jobs set status = 'dead', locked_until = null, last_error = %s, updated_at = now() where id = %s",
                (error[:2000], job["id"]),
            )
        else:
            # back to the queue unless a newer refresh for this campaign is already waiting
            backoff = min(600, 2 ** job["attempts"] * 5)
            conn.execute(
                """update jobs set status = case when exists (select 1 from jobs j where j.kind = jobs.kind and j.campaign_id = jobs.campaign_id and j.status = 'queued') then 'done' else 'queued' end,
                   run_after = now() + make_interval(secs => %s), locked_until = null, last_error = %s, updated_at = now() where id = %s""",
                (backoff, error[:2000], job["id"]),
            )
        conn.commit()


def run_once(pool: ConnectionPool, fatigue=None) -> dict | None:
    job = claim(pool)
    if not job:
        return None
    t0 = time.perf_counter()
    try:
        with pool.connection() as conn:
            result = service.refresh_campaign(conn, job["campaign_id"], fatigue)
            conn.commit()
        finish(pool, job, None)
        JOBS.labels(job["kind"], "done").inc()
        log.info(
            json.dumps(
                {
                    "job": job["id"],
                    "campaign": job["campaign_id"],
                    "result": result,
                    "ms": round((time.perf_counter() - t0) * 1000),
                }
            )
        )
        return {"job": job["id"], **result}
    except Exception as e:  # recorded on the job and retried
        log.exception("job %s failed", job["id"])
        finish(pool, job, f"{type(e).__name__}: {e}")
        JOBS.labels(job["kind"], "failed").inc()
        return {"job": job["id"], "error": str(e)}
    finally:
        DURATION.labels(job["kind"]).observe(time.perf_counter() - t0)


def drain(pool: ConnectionPool, fatigue=None, limit: int = 10_000) -> int:
    n = 0
    while n < limit and run_once(pool, fatigue):
        n += 1
    return n


def main() -> None:
    from cbo.db.db import make_pool, migrate

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pool = make_pool(max_size=4)
    migrate(pool)
    fatigue = load_champion(os.environ.get("CBO_DATA_DIR", "data"))
    log.info("fatigue model %s", fatigue.version if fatigue else "not loaded; creatives will not be scored")
    start_http_server(int(os.environ.get("CBO_WORKER_METRICS_PORT", "9100")))
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    while not stop.is_set():
        if run_once(pool, fatigue) is None:
            stop.wait(1.0)
    pool.close()


if __name__ == "__main__":
    main()
