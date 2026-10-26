"""Train the XGBoost forecasting models.

Usage (from the project root):  python -m queue_predictor.train [--days 150]

Produces models/queue_models.joblib containing
  * one regressor per horizon (5/10/15 min) predicting the change in queue length
  * one classifier giving the probability that the queue becomes critically
    congested (waiting time >= HIGH_WAIT_MIN) within the next MAIN_HORIZON minutes
"""
import argparse
import json

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

from .config import HIGH_WAIT_MIN, HORIZONS, MAIN_HORIZON, METRICS_PATH, MODEL_PATH, SIM_DAYS
from .features import FEATURES, add_features
from .simulate import generate_history

BUSY_QUEUE = 10  # accuracy is also reported for moments with at least this many people waiting


def build_dataset(history):
    df = add_features(history)
    by_day = df.groupby(df["ts"].dt.date)["queue"]
    for h in HORIZONS:
        df[f"target_{h}"] = by_day.shift(-h)

    # Highest queue over the next MAIN_HORIZON minutes, converted to waiting time.
    future_max = by_day.transform(
        lambda s: s.shift(-1)[::-1].rolling(MAIN_HORIZON, min_periods=MAIN_HORIZON).max()[::-1]
    )
    df["future_max_wait"] = future_max / df["capacity"]
    df = df.dropna(subset=FEATURES + [f"target_{h}" for h in HORIZONS] + ["future_max_wait"])
    df["critical"] = (df["future_max_wait"] >= HIGH_WAIT_MIN).astype(int)
    return df


def train(days=SIM_DAYS, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    log(f"Generating {days} days of synthetic queue history...")
    df = build_dataset(generate_history(days=days))

    # Time-based split: the last 20% of days are held out for testing.
    dates = sorted(df["ts"].dt.date.unique())
    cutoff = dates[int(len(dates) * 0.8)]
    train_df, test_df = df[df["ts"].dt.date < cutoff], df[df["ts"].dt.date >= cutoff]
    X_train, X_test = train_df[FEATURES], test_df[FEATURES]
    log(f"Training rows: {len(train_df):,}   test rows: {len(test_df):,}")

    regressors, metrics = {}, {"horizons": {}}
    for h in HORIZONS:
        model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9, n_jobs=-1)
        model.fit(X_train, train_df[f"target_{h}"] - train_df["queue"])
        pred = np.clip(X_test["queue"] + model.predict(X_test), 0, None)
        actual = test_df[f"target_{h}"]
        busy = (X_test["queue"] >= BUSY_QUEUE).to_numpy()
        metrics["horizons"][h] = {
            "mae": round(float(mean_absolute_error(actual, pred)), 2),
            "r2": round(float(r2_score(actual, pred)), 3),
            "baseline_mae": round(float(mean_absolute_error(actual, X_test["queue"])), 2),
            "busy_mae": round(float(mean_absolute_error(actual[busy], pred[busy])), 2),
            "busy_baseline_mae": round(float(mean_absolute_error(actual[busy], X_test["queue"][busy])), 2),
        }
        regressors[h] = model
        m = metrics["horizons"][h]
        log(f"  +{h:>2} min  MAE {m['mae']:5.2f} people  R2 {m['r2']:.3f}  "
            f"(naive 'no change' MAE {m['baseline_mae']:.2f})  |  busy periods: "
            f"MAE {m['busy_mae']:.2f} vs naive {m['busy_baseline_mae']:.2f}")

    classifier = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                               subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
                               eval_metric="logloss")
    classifier.fit(X_train, train_df["critical"])
    proba = classifier.predict_proba(X_test)[:, 1]
    metrics["critical"] = {
        "auc": round(float(roc_auc_score(test_df["critical"], proba)), 3),
        "accuracy": round(float(((proba >= 0.5) == test_df["critical"]).mean()), 3),
        "positive_rate": round(float(test_df["critical"].mean()), 3),
    }
    metrics["busy_queue"] = BUSY_QUEUE
    metrics["train_rows"], metrics["test_rows"] = len(train_df), len(test_df)
    metrics["days"] = days
    log(f"  Critical-congestion classifier  AUC {metrics['critical']['auc']}  "
        f"accuracy {metrics['critical']['accuracy']}")

    joblib.dump({"regressors": regressors, "classifier": classifier, "features": FEATURES}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    log(f"Saved models to {MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=SIM_DAYS)
    train(parser.parse_args().days)
