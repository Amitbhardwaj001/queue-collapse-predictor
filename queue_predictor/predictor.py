"""Turns recent queue counts into forecasts, waiting times, risk and advice."""
import math

import joblib
import pandas as pd

from .config import (HIGH_PROBABILITY, HIGH_WAIT_MIN, HORIZONS, MAIN_HORIZON, MEDIUM_PROBABILITY,
                    MEDIUM_WAIT_MIN, MODEL_PATH, TARGET_WAIT_MIN)
from .features import add_features, to_minute_series


class QueuePredictor:
    def __init__(self, path=MODEL_PATH):
        bundle = joblib.load(path)
        self.regressors = bundle["regressors"]
        self.classifier = bundle["classifier"]
        self.features = bundle["features"]

    def predict(self, observations, counters, service_time_sec):
        """observations: DataFrame with columns ts (datetime) and count (people)."""
        series = to_minute_series(observations)
        service_rate = 60.0 / service_time_sec
        frame = pd.DataFrame({"ts": series.index, "queue": series.values,
                              "counters": counters, "service_rate": service_rate})
        row = add_features(frame).iloc[[-1]][self.features]

        current = int(round(observations.sort_values("ts")["count"].iloc[-1]))
        forecast = {h: max(0, int(round(row["queue"].iloc[0] + self.regressors[h].predict(row)[0])))
                    for h in HORIZONS}
        probability = float(self.classifier.predict_proba(row)[0, 1])

        capacity = counters * service_rate
        predicted = forecast[MAIN_HORIZON]
        current_wait = current / capacity
        predicted_wait = predicted / capacity
        risk = self._risk(max(current_wait, predicted_wait), probability)
        needed = max(1, math.ceil(predicted / (service_rate * TARGET_WAIT_MIN)))

        return {
            "as_of": series.index[-1].strftime("%Y-%m-%d %H:%M"),
            "current_queue": current,
            "forecast": [{"minutes": h, "queue": q} for h, q in forecast.items()],
            "predicted_queue": predicted,
            "horizon_min": MAIN_HORIZON,
            "current_wait_min": round(current_wait, 1),
            "predicted_wait_min": round(predicted_wait, 1),
            "critical_probability": round(probability, 3),
            "risk": risk,
            "counters": counters,
            "counters_needed": needed,
            "capacity_per_min": round(capacity, 2),
            "critical_queue": math.ceil(HIGH_WAIT_MIN * capacity),
            "recommendation": self._recommend(risk, predicted, counters, needed),
        }

    @staticmethod
    def _risk(wait, probability):
        if wait >= HIGH_WAIT_MIN or probability >= HIGH_PROBABILITY:
            return "HIGH"
        if wait >= MEDIUM_WAIT_MIN or probability >= MEDIUM_PROBABILITY:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _recommend(risk, predicted, counters, needed):
        extra = needed - counters
        if risk == "HIGH":
            if extra > 0:
                return (f"Open {extra} more counter{'s' if extra > 1 else ''} now - the queue is "
                        f"expected to reach {predicted} people in {MAIN_HORIZON} minutes.")
            return "Queue is critical. Speed up service (pre-orders, token system) and guide people to other counters."
        if risk == "MEDIUM":
            if extra > 0:
                return f"Queue is building. Get staff ready to open {extra} more counter{'s' if extra > 1 else ''}."
            return "Queue is building. Keep monitoring closely."
        if counters > 1 and needed < counters:
            return "Queue is under control. One counter could be closed if staff are needed elsewhere."
        return "Queue is under control. No action needed."
