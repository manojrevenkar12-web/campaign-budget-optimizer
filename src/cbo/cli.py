"""Command line: `cbo serve | worker | migrate | create-tenant | seed | train-fatigue | evaluate`."""

from __future__ import annotations

import argparse
import json
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    p = argparse.ArgumentParser(prog="cbo")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--workers", type=int, default=2)
    sub.add_parser("worker")
    sub.add_parser("drain", help="process queued jobs until the queue is empty, then exit")
    sub.add_parser("migrate")
    t = sub.add_parser("create-tenant")
    t.add_argument("tenant_id")
    sd = sub.add_parser("seed", help="push simulated advertisers through the API")
    sd.add_argument("--api", default="http://localhost:8000")
    sd.add_argument("--key", required=True)
    sd.add_argument("--campaigns", type=int, default=10)
    sd.add_argument("--days", type=int, default=60)
    tf = sub.add_parser("train-fatigue")
    tf.add_argument("--out", default="data")
    tf.add_argument("--campaigns", type=int, default=120)
    tf.add_argument("--days", type=int, default=150)
    ev = sub.add_parser("evaluate", help="budget policies on simulated campaigns vs true demand")
    ev.add_argument("--campaigns", type=int, default=60)
    ev.add_argument("--days", type=int, default=60)
    ev.add_argument("--out", default="data/policy_evaluation.json")
    a = p.parse_args(argv)

    if a.cmd == "serve":
        import uvicorn

        uvicorn.run(
            "cbo.api.app:create_app", factory=True, host="0.0.0.0", port=a.port, workers=a.workers, log_level="warning"
        )
    elif a.cmd == "worker":
        from cbo.worker.worker import main as worker_main

        worker_main()
    elif a.cmd == "drain":
        import os

        from cbo.db.db import make_pool, migrate
        from cbo.fatigue.model import load_champion
        from cbo.worker.worker import drain

        pool = make_pool()
        migrate(pool)
        print(json.dumps({"jobs": drain(pool, load_champion(os.environ.get("CBO_DATA_DIR", "data")))}))
        pool.close()
    elif a.cmd == "migrate":
        from cbo.db.db import make_pool, migrate

        print(json.dumps({"applied": migrate(make_pool())}))
    elif a.cmd == "create-tenant":
        from cbo.db.db import make_pool, migrate
        from cbo.seed import create_tenant

        pool = make_pool()
        migrate(pool)
        print(create_tenant(pool, a.tenant_id))
    elif a.cmd == "seed":
        import httpx

        from cbo.seed import seed

        with httpx.Client(base_url=a.api, headers={"Authorization": f"Bearer {a.key}"}, timeout=60) as c:
            print(json.dumps({"campaigns": seed(c, a.campaigns, a.days)}))
    elif a.cmd == "train-fatigue":
        from cbo.fatigue.train import train

        r = train(a.out, a.campaigns, a.days)
        print(json.dumps({k: r[k] for k in ("test", "auc_difference_ci95", "at_threshold", "gate")}, indent=2))
        return 0 if r["gate"]["promote"] else 5
    elif a.cmd == "evaluate":
        from pathlib import Path

        from cbo.sim.evaluate import evaluate

        res, summary = evaluate(a.campaigns, a.days)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(summary, indent=2))
        res.to_csv(Path(a.out).with_suffix(".csv"), index=False)
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
