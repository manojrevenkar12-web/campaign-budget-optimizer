import numpy as np
import pandas as pd

from cbo.fatigue.data import FEATURES, features_from_history, label
from cbo.fatigue.model import FatigueModel


def creative(n, ctr, cid="c1"):
    imps = np.full(n, 50_000)
    return pd.DataFrame(
        {
            "creative_id": cid,
            "day": np.arange(n),
            "platform": "meta",
            "creative_format": "static",
            "creative_age": np.arange(n),
            "frequency": np.arange(n) * 0.5,
            "impressions": imps,
            "clicks": (imps * np.asarray(ctr)).astype(int),
        }
    )


def test_features_never_look_ahead():
    h = creative(20, np.linspace(0.02, 0.01, 20))
    full = features_from_history(h)
    prefix = features_from_history(h.iloc[:12])
    assert np.allclose(full[FEATURES].iloc[:12].to_numpy(), prefix[FEATURES].to_numpy())


def test_label_marks_a_creative_that_will_lose_30_percent():
    ctr = np.r_[np.full(10, 0.02), np.full(10, 0.012)]
    h = creative(20, ctr)
    f = features_from_history(h)
    y = label(f, h)
    assert y.iloc[3] == 0.0  # days 4-10 are still mostly at the peak
    assert y.iloc[8] == 1.0  # days 9-15 are mostly below 70% of it
    assert np.isnan(y.iloc[15])  # fewer than 7 days left to observe


def test_json_round_trip_is_exact():
    rng = np.random.default_rng(0)
    m = FatigueModel(
        rng.normal(size=len(FEATURES)),
        np.abs(rng.normal(size=len(FEATURES))) + 0.5,
        [(rng.normal(size=(8, len(FEATURES))), rng.normal(size=8)), (rng.normal(size=(1, 8)), rng.normal(size=1))],
        1.3,
        "7",
    )
    x = rng.normal(size=(5, len(FEATURES)))
    assert np.allclose(m.predict(x), FatigueModel.from_json(m.to_json()).predict(x))


def test_trained_model_beats_the_age_rule_and_matches_torch(fatigue_model):
    _, report = fatigue_model
    assert report["gate"]["promote"], report["gate"]
    assert report["test"]["auc"] > report["test"]["baseline_auc_age_rule"] + 0.2
    assert report["numpy_torch_max_abs_diff"] < 1e-5
