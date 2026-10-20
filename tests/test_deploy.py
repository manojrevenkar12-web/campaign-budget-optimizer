import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_cluster_rules_match_the_compose_rules():
    assert yaml.safe_load((ROOT / "deploy/k8s/slo-rules.yaml").read_text())["spec"] == yaml.safe_load(
        (ROOT / "ops/prometheus/rules.yml").read_text()
    )


def test_every_alert_links_an_existing_runbook_section():
    doc = (ROOT / "docs/slo.md").read_text().lower()
    anchors = {
        "#" + re.sub(r"[^a-z0-9 -]", "", line.lstrip("#").strip()).replace(" ", "-")
        for line in doc.splitlines()
        if line.startswith("### ")
    }
    for g in yaml.safe_load((ROOT / "ops/prometheus/rules.yml").read_text())["groups"]:
        for r in g["rules"]:
            if "alert" in r:
                assert r["annotations"]["runbook"].split("slo.md")[1] in anchors, r["alert"]


def test_dashboards_and_rules_use_metrics_the_services_export():
    from cbo.api.app import build_metrics
    from cbo.worker import worker

    names = {m._name for m in build_metrics().values() if hasattr(m, "_name")} | {
        worker.JOBS._name,
        worker.DURATION._name,
    }
    text = (ROOT / "ops/grafana/dashboards/cbo.json").read_text() + (ROOT / "ops/prometheus/rules.yml").read_text()
    used = set(re.findall(r"\b(cbo_[a-z_]+?)(?:_bucket|_count|_sum|_total)?\b", text))
    for u in used:
        assert any(u.startswith(n) or n.startswith(u) for n in names), u
    json.loads((ROOT / "ops/grafana/dashboards/cbo.json").read_text())
