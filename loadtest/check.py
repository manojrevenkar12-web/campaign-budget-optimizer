"""Fail the build if the load test breaks the SLOs (p99 latency of recommendations, error rate)."""

import csv
import json
import sys

with open(sys.argv[1]) as fh:
    stats = {r["Name"]: r for r in csv.DictReader(fh)}
rec = stats["/v1/campaigns/{id}/recommendations"]
agg = stats["Aggregated"]
out = {
    "requests": int(agg["Request Count"]),
    "failures": int(agg["Failure Count"]),
    "throughput_rps": round(float(agg["Requests/s"]), 1),
    "recommendation_p50_ms": float(rec["50%"]),
    "recommendation_p95_ms": float(rec["95%"]),
    "recommendation_p99_ms": float(rec["99%"]),
}
print(json.dumps(out, indent=2))
ok = out["failures"] / max(out["requests"], 1) <= 0.001 and out["recommendation_p99_ms"] <= 300
sys.exit(0 if ok else 1)
