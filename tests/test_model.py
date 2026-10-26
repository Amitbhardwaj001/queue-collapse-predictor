from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from queue_predictor.config import LAGS
from queue_predictor.features import FEATURES, add_features, to_minute_series
from queue_predictor.predictor import QueuePredictor
from queue_predictor.simulate import QueueSimulator, generate_history


def test_simulation_has_a_lunch_rush():
    df = generate_history(days=10, seed=1)
    by_hour = df.groupby(df["ts"].dt.hour)["queue"].mean()
    assert by_hour[13] > 3 * by_hour[10]
    assert (df["queue"] >= 0).all()


def test_more_counters_shorten_the_queue():
    def lunch_queue(counters):
        sim = QueueSimulator(np.random.default_rng(0), weekday=1, counters=counters)
        return np.mean([sim.step(m) for m in range(12 * 60, 14 * 60)])
    assert lunch_queue(4) < lunch_queue(2)


def test_features_do_not_leak_across_days():
    df = generate_history(days=2, seed=2)
    feats = add_features(df)
    first_rows = feats.groupby(feats["ts"].dt.date).head(1)
    assert first_rows["lag_1"].isna().all()
    assert set(FEATURES) <= set(feats.columns)


def test_short_history_is_padded():
    start = datetime(2026, 1, 5, 12, 0)
    obs = pd.DataFrame({"ts": [start, start + timedelta(minutes=2)], "count": [5, 9]})
    series = to_minute_series(obs)
    assert len(series) >= max(LAGS) + 1
    assert series.iloc[-1] == 9


@pytest.mark.parametrize("queue, counters, expected", [(2, 3, "LOW"), (60, 1, "HIGH")])
def test_risk_levels(queue, counters, expected):
    now = datetime(2026, 1, 5, 13, 0)
    obs = pd.DataFrame({"ts": [now - timedelta(minutes=m) for m in range(12)], "count": [queue] * 12})
    result = QueuePredictor().predict(obs, counters=counters, service_time_sec=45)
    assert result["risk"] == expected
    assert [f["minutes"] for f in result["forecast"]] == [5, 10, 15]
    assert 0 <= result["critical_probability"] <= 1
