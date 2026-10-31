"""Train and evaluate the forecasting models for every domain.

Usage (from the project root):
    python -m queue_predictor.train                  # all domains
    python -m queue_predictor.train --domain web     # one domain
    python -m queue_predictor.train --days 60        # shorter, faster

For each domain it produces models/<domain>_models.joblib containing
  * one XGBoost regressor per forecast horizon predicting the change in backlog
  * one XGBoost classifier giving the probability that the wait reaches the
    domain's critical level within the next main horizon
and models/<domain>_metrics.json with forecast accuracy and an early-warning
study (how many minutes earlier than a reactive alarm the system warns).
"""
import argparse
import json

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

from . import radar
from .config import HIGH_PROBABILITY, RADAR_ALARM_PROBABILITY, metrics_path, model_path
from .domains import DOMAINS
from .features import add_features, feature_names
from .simulate import generate_history

BUSY_FRACTION = 0.25     # "busy" = a new arrival would wait at least this share of the critical wait
EVAL_PATHS = 100         # Monte Carlo paths per minute in the early-warning study


def build_dataset(history, domain):
    df = add_features(history, domain.lags)
    by_day = df.groupby(df["ts"].dt.date)["queue"]
    for h in domain.horizons:
        df[f"target_{h}"] = by_day.shift(-h)

    # Highest backlog over the next main horizon, converted to waiting time.
    main = domain.main_horizon
    future_max = by_day.transform(
        lambda s: s.shift(-1)[::-1].rolling(main, min_periods=main).max()[::-1]
    )
    df["future_max_wait"] = future_max / df["capacity"]
    df = df.dropna(subset=feature_names(domain.lags) + [f"target_{h}" for h in domain.horizons] + ["future_max_wait"])
    df["critical"] = (df["future_max_wait"] >= domain.high_wait).astype(int)
    return df


def _episodes(wait, high, gap):
    """Indices where the wait first reaches the critical level after at least `gap` calmer minutes."""
    above = wait >= high
    starts = []
    last = -10 ** 9
    for i in np.flatnonzero(above):
        if i - last > gap:
            starts.append(i)
        last = i
    return starts


def _score_alarm(alarm, wait, high, window, cooldown=10):
    """Lead time and false alarms of one alarm signal on one day."""
    events = _episodes(wait, high, 2 * window)
    leads = []
    for e in events:
        lo = max(0, e - window)
        hits = np.flatnonzero(alarm[lo:e + 1])
        leads.append(e - (lo + hits[0]) if len(hits) else None)

    false_alarms = true_alarms = 0
    last_active = -10 ** 9
    for j in np.flatnonzero(alarm):
        onset = j - last_active > cooldown
        last_active = j
        if not onset or wait[j] >= high:
            continue
        if any(0 < e - j <= window for e in events):
            true_alarms += 1
        else:
            false_alarms += 1
    return leads, true_alarms, false_alarms


def early_warning_study(domain, test_df, regressors, classifier, features):
    """Compare a reactive alarm with the classifier, the Collapse Radar and both together on unseen days."""
    window = domain.radar_horizon
    X = test_df[features]
    forecasts = {h: np.clip(X["queue"].to_numpy() + regressors[h].predict(X), 0, None) for h in domain.horizons}
    proba = classifier.predict_proba(X)[:, 1]
    wait = (test_df["queue"] / test_df["capacity"]).to_numpy()

    queue = test_df["queue"].to_numpy()
    counters = test_df["counters"].to_numpy()
    rate = test_df["service_rate"].to_numpy()
    radar_probability = np.zeros(len(test_df))
    # Rows with a clearly empty queue and a calm forecast cannot collapse; skip the simulation there.
    worth_checking = (wait >= 0.15 * domain.high_wait) | (proba >= 0.05) | np.any(
        [forecasts[h] / (counters * rate) >= 0.5 * domain.high_wait for h in domain.horizons], axis=0)
    for i in np.flatnonzero(worth_checking):
        result = radar.analyse(domain, float(queue[i]), {h: forecasts[h][i] for h in domain.horizons},
                               int(counters[i]), float(rate[i]), what_if=False, paths=EVAL_PATHS)
        radar_probability[i] = result["collapse_probability"]

    signals = {
        "reactive": wait >= domain.high_wait,
        "classifier": proba >= HIGH_PROBABILITY,
        "radar": radar_probability >= RADAR_ALARM_PROBABILITY,
    }
    signals["system"] = signals["classifier"] | signals["radar"]   # what the dashboard actually shows
    days = test_df["ts"].dt.date.to_numpy()
    totals = {name: {"leads": [], "true": 0, "false": 0} for name in signals}
    for day in np.unique(days):
        idx = np.flatnonzero(days == day)
        for name, alarm in signals.items():
            leads, ok, bad = _score_alarm(alarm[idx], wait[idx], domain.high_wait, window)
            totals[name]["leads"] += leads
            totals[name]["true"] += ok
            totals[name]["false"] += bad

    n_days = len(np.unique(days))
    study = {"window_min": window, "days": n_days, "policies": {}}
    for name, t in totals.items():
        detected = [x for x in t["leads"] if x is not None]
        early = [x for x in detected if x >= 2]
        alarms = t["true"] + t["false"]
        study["policies"][name] = {
            "events": len(t["leads"]),
            "detected_early": len(early),
            "early_rate": round(len(early) / len(t["leads"]), 3) if t["leads"] else None,
            "mean_lead_min": round(float(np.mean(early)), 1) if early else 0.0,
            "false_alarms_per_day": round(t["false"] / n_days, 2),
            "precision": round(t["true"] / alarms, 3) if alarms else None,
        }
    return study


def _auc_before_collapse(test_df, proba, domain):
    not_yet = (test_df["wait_now"] < domain.high_wait).to_numpy()
    labels = test_df["critical"].to_numpy()[not_yet]
    if labels.min() == labels.max():
        return None
    return round(float(roc_auc_score(labels, proba[not_yet])), 3)


def train(domain, days=None, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    days = days or domain.sim_days
    log(f"[{domain.name}] generating {days} days of synthetic history...")
    df = build_dataset(generate_history(days=days, domain=domain), domain)

    # Time-based split: the last 20% of days are held out for testing.
    dates = sorted(df["ts"].dt.date.unique())
    cutoff = dates[int(len(dates) * 0.8)]
    train_df, test_df = df[df["ts"].dt.date < cutoff], df[df["ts"].dt.date >= cutoff]
    features = feature_names(domain.lags)
    X_train, X_test = train_df[features], test_df[features]
    log(f"  training rows: {len(train_df):,}   test rows: {len(test_df):,}")

    regressors, metrics = {}, {"domain": domain.key, "horizons": {}}
    busy = (test_df["wait_now"] >= BUSY_FRACTION * domain.high_wait).to_numpy()
    for h in domain.horizons:
        # Absolute-error loss and strong regularisation: backlog changes are mostly noise, and a
        # squared-error model with deep trees learns that noise and loses to "no change".
        model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03, min_child_weight=20,
                             reg_lambda=10, objective="reg:absoluteerror",
                             subsample=0.9, colsample_bytree=0.9, n_jobs=-1)
        model.fit(X_train, train_df[f"target_{h}"] - train_df["queue"])
        pred = np.clip(X_test["queue"] + model.predict(X_test), 0, None)
        actual = test_df[f"target_{h}"]
        metrics["horizons"][h] = {
            "mae": round(float(mean_absolute_error(actual, pred)), 2),
            "r2": round(float(r2_score(actual, pred)), 3),
            "baseline_mae": round(float(mean_absolute_error(actual, X_test["queue"])), 2),
            "busy_mae": round(float(mean_absolute_error(actual[busy], pred[busy])), 2),
            "busy_baseline_mae": round(float(mean_absolute_error(actual[busy], X_test["queue"][busy])), 2),
        }
        regressors[h] = model
        m = metrics["horizons"][h]
        log(f"  +{h:>2} min  MAE {m['mae']:6.2f}  R2 {m['r2']:.3f}  (naive MAE {m['baseline_mae']:.2f})"
            f"  | busy: MAE {m['busy_mae']:.2f} vs naive {m['busy_baseline_mae']:.2f}")

    classifier = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                               subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
                               eval_metric="logloss")
    classifier.fit(X_train, train_df["critical"])
    proba = classifier.predict_proba(X_test)[:, 1]
    metrics["critical"] = {
        "auc": round(float(roc_auc_score(test_df["critical"], proba)), 3),
        "accuracy": round(float(((proba >= 0.5) == test_df["critical"]).mean()), 3),
        # Much harder and more honest: only moments that are not critical yet, so the model must
        # foresee the collapse instead of recognising one that is already happening.
        "auc_before_collapse": _auc_before_collapse(test_df, proba, domain),
        "positive_rate": round(float(test_df["critical"].mean()), 3),
    }
    metrics["busy_definition"] = f"a new arrival waits at least {BUSY_FRACTION * domain.high_wait:g} min"
    metrics["train_rows"], metrics["test_rows"] = len(train_df), len(test_df)
    metrics["days"] = days
    log(f"  critical-congestion classifier  AUC {metrics['critical']['auc']}  "
        f"accuracy {metrics['critical']['accuracy']}  AUC before collapse {metrics['critical']['auc_before_collapse']}")

    log("  running the early-warning study on the unseen days...")
    metrics["early_warning"] = early_warning_study(domain, test_df, regressors, classifier, features)
    for name, p in metrics["early_warning"]["policies"].items():
        log(f"    {name:<10} events {p['events']:>3}  warned early {p['early_rate']}  "
            f"mean lead {p['mean_lead_min']} min  false alarms/day {p['false_alarms_per_day']}")

    joblib.dump({"regressors": regressors, "classifier": classifier, "features": features},
                model_path(domain.key))
    metrics_path(domain.key).write_text(json.dumps(metrics, indent=2))
    log(f"  saved {model_path(domain.key).name}")
    return metrics


def train_all(days=None, verbose=True):
    return {key: train(domain, days, verbose) for key, domain in DOMAINS.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["all", *DOMAINS], default="all")
    parser.add_argument("--days", type=int, default=None)
    args = parser.parse_args()
    if args.domain == "all":
        train_all(args.days)
    else:
        train(DOMAINS[args.domain], args.days)
