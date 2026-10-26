"""Feature engineering shared by training and live prediction."""
import numpy as np
import pandas as pd

from .config import LAGS

FEATURES = (
    ["queue"]
    + [f"lag_{lag}" for lag in LAGS]
    + ["diff_1", "diff_5", "diff_10", "roll_mean_5", "roll_std_5",
       "tod_sin", "tod_cos", "weekday", "is_weekend",
       "counters", "service_rate", "capacity", "wait_now"]
)


def add_features(df):
    """Add model features to per-minute rows (columns: ts, queue, counters, service_rate).

    Lags and rolling statistics are computed within each calendar day so a
    morning never borrows history from the previous evening.
    """
    df = df.sort_values("ts").reset_index(drop=True).copy()
    by_day = df.groupby(df["ts"].dt.date)["queue"]

    for lag in LAGS:
        df[f"lag_{lag}"] = by_day.shift(lag)
    df["diff_1"] = df["queue"] - df["lag_1"]
    df["diff_5"] = df["queue"] - df["lag_5"]
    df["diff_10"] = df["queue"] - df["lag_10"]
    df["roll_mean_5"] = by_day.transform(lambda s: s.rolling(5, min_periods=1).mean())
    df["roll_std_5"] = by_day.transform(lambda s: s.rolling(5, min_periods=2).std()).fillna(0)

    minute_of_day = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    df["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    df["weekday"] = df["ts"].dt.day_of_week
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    df["capacity"] = df["counters"] * df["service_rate"]      # people served per minute
    df["wait_now"] = df["queue"] / df["capacity"]               # minutes for a new arrival
    return df


def to_minute_series(observations):
    """Turn irregular (ts, count) observations into a gap-free per-minute series.

    Gaps are forward-filled. If less than the longest lag of history exists,
    the earliest value is repeated backwards (assume the queue was steady).
    """
    s = observations.set_index("ts")["count"].sort_index().astype(float)
    per_minute = s.resample("1min").mean().ffill()

    needed = max(LAGS) + 1
    if len(per_minute) < needed:
        missing = needed - len(per_minute)
        earlier = pd.date_range(end=per_minute.index[0] - pd.Timedelta(minutes=1), periods=missing, freq="1min")
        per_minute = pd.concat([pd.Series(per_minute.iloc[0], index=earlier), per_minute])
    return per_minute
