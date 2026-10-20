"""Train, gate and register the creative-fatigue model.

Split by campaign (no campaign in two splits). The gate compares with the rule many teams use —
"replace a creative after two weeks", i.e. score = creative age — and promotes only if the model's
ROC-AUC is better with a 95% bootstrap interval above zero, and its expected calibration error is
small enough that a "70% likely fatigued" alert means roughly that.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from cbo.fatigue.data import FEATURES, generate
from cbo.fatigue.model import FatigueModel, torch_predict, train_torch

MODEL_NAME = "creative_fatigue"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def bootstrap_auc_diff(y, p_model, p_base, groups, n=1000, seed=0) -> list[float]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in uniq}
    diffs = []
    for _ in range(n):
        idx = np.concatenate([by[g] for g in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], p_model[idx]) - roc_auc_score(y[idx], p_base[idx]))
    return [float(x) for x in np.percentile(diffs, [2.5, 97.5])]


def split(frame: pd.DataFrame, seed: int = 0) -> dict[str, pd.DataFrame]:
    camps = np.array(sorted(frame["campaign"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(camps)
    n = len(camps)
    parts = {"train": camps[: int(0.6 * n)], "val": camps[int(0.6 * n) : int(0.8 * n)], "test": camps[int(0.8 * n) :]}
    return {k: frame[frame["campaign"].isin(v)] for k, v in parts.items()}


def train(
    out_dir: str = "data", n_campaigns: int = 120, days: int = 150, register: bool = True, max_ece: float = 0.05
) -> dict:
    frame = generate(n_campaigns, days)
    s = split(frame)
    x = {k: v[FEATURES].to_numpy(float) for k, v in s.items()}
    y = {k: v["label"].to_numpy(float) for k, v in s.items()}
    model = train_torch(x["train"], y["train"], x["val"], y["val"])
    p = model.predict(x["test"])
    parity = float(np.abs(p - torch_predict(model, x["test"])).max())
    base = s["test"]["age_days"].to_numpy(float)
    yt = y["test"]
    report = {
        "rows": {k: len(v) for k, v in s.items()},
        "positive_rate": float(frame["label"].mean()),
        "test": {
            "auc": float(roc_auc_score(yt, p)),
            "pr_auc": float(average_precision_score(yt, p)),
            "ece": ece(yt, p),
            "baseline_auc_age_rule": float(roc_auc_score(yt, base)),
            "baseline_pr_auc_age_rule": float(average_precision_score(yt, base)),
        },
        "auc_difference_ci95": bootstrap_auc_diff(yt, p, base, s["test"]["campaign"].to_numpy()),
        "numpy_torch_max_abs_diff": parity,
        "temperature": model.temperature,
    }
    # alerting at a 50% probability vs the two-week rule: how many fatigued creatives are caught, and how early
    alert = p >= 0.5
    rule = base >= 14
    report["at_threshold"] = {
        "model_precision": float(yt[alert].mean()) if alert.any() else None,
        "model_recall": float(alert[yt == 1].mean()),
        "rule_precision": float(yt[rule].mean()) if rule.any() else None,
        "rule_recall": float(rule[yt == 1].mean()),
    }
    checks = {
        "better_than_age_rule": report["auc_difference_ci95"][0] > 0,
        "calibrated": report["test"]["ece"] <= max_ece,
        "numpy_matches_torch": parity < 1e-5,
    }
    report["gate"] = {"checks": checks, "promote": all(checks.values())}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fatigue_model.json").write_text(json.dumps(model.to_json()))
    (out / "fatigue_report.json").write_text(json.dumps(report, indent=2))
    if register:
        report["registry"] = _register(model, report, out)
        model.version = str(report["registry"]["version"])
        (out / "fatigue_model.json").write_text(json.dumps(model.to_json()))
    return report


def _register(model: FatigueModel, report: dict, out: Path) -> dict:
    import contextlib

    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    uri = os.environ.get("CBO_MLFLOW_URI", f"sqlite:///{out.resolve()}/mlflow.db")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient(uri, uri)
    exp = client.get_experiment_by_name("creative-fatigue")
    exp_id = (
        exp.experiment_id
        if exp
        else client.create_experiment("creative-fatigue", artifact_location=(out / "mlruns").resolve().as_uri())
    )
    with mlflow.start_run(experiment_id=exp_id) as run:
        mlflow.log_params({"hidden": 32, "features": len(FEATURES), "label_threshold": 0.7, "horizon_days": 7})
        mlflow.log_metrics({k: v for k, v in report["test"].items()})
        mlflow.set_tags({"promoted": report["gate"]["promote"]})
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "model.json").write_text(json.dumps(model.to_json()))
            (Path(d) / "report.json").write_text(json.dumps(report, indent=2))
            mlflow.log_artifacts(d, "model")
    with contextlib.suppress(MlflowException):
        client.create_registered_model(MODEL_NAME)
    mv = client.create_model_version(
        MODEL_NAME, f"{client.get_run(run.info.run_id).info.artifact_uri}/model", run_id=run.info.run_id
    )
    if report["gate"]["promote"]:
        client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    return {"version": int(mv.version), "run_id": run.info.run_id, "champion": report["gate"]["promote"]}


def load_champion(out_dir: str = "data") -> FatigueModel | None:
    """The API loads the exported file (mounted from the registry by the deploy job)."""
    p = Path(os.environ.get("CBO_FATIGUE_MODEL", f"{out_dir}/fatigue_model.json"))
    return FatigueModel.from_json(json.loads(p.read_text())) if p.exists() else None
