"""Turns recent backlog counts into forecasts, waiting times, risk, collapse radar and advice."""
import math

import joblib
import pandas as pd

from . import radar
from .config import HIGH_PROBABILITY, MEDIUM_PROBABILITY, model_path
from .domains import QUEUE
from .features import add_features, to_minute_series


def plural(word, n):
    return word if n == 1 else word + "s"


class QueuePredictor:
    def __init__(self, domain=QUEUE, path=None):
        self.domain = domain
        bundle = joblib.load(path or model_path(domain.key))
        self.regressors = bundle["regressors"]
        self.classifier = bundle["classifier"]
        self.features = bundle["features"]

    def predict(self, observations, counters, service_time_sec, what_if=True):
        """observations: DataFrame with columns ts (datetime) and count (items waiting)."""
        d = self.domain
        series = to_minute_series(observations, d.lags)
        service_rate = 60.0 / service_time_sec
        frame = pd.DataFrame({"ts": series.index, "queue": series.values,
                              "counters": counters, "service_rate": service_rate})
        row = add_features(frame, d.lags).iloc[[-1]][self.features]

        current = int(round(observations.sort_values("ts")["count"].iloc[-1]))
        forecast = {h: max(0, int(round(row["queue"].iloc[0] + self.regressors[h].predict(row)[0])))
                    for h in d.horizons}
        probability = float(self.classifier.predict_proba(row)[0, 1])

        capacity = counters * service_rate
        predicted = forecast[d.main_horizon]
        current_wait = current / capacity
        predicted_wait = predicted / capacity
        risk = self._risk(max(current_wait, predicted_wait), probability)
        needed = max(1, math.ceil(predicted / (service_rate * d.target_wait)))
        radar_result = radar.analyse(d, current, forecast, counters, service_rate, what_if=what_if)

        return {
            "domain": d.key,
            "as_of": series.index[-1].strftime("%Y-%m-%d %H:%M"),
            "current_queue": current,
            "forecast": [{"minutes": h, "queue": q} for h, q in forecast.items()],
            "predicted_queue": predicted,
            "horizon_min": d.main_horizon,
            "current_wait_min": round(current_wait, 1),
            "predicted_wait_min": round(predicted_wait, 1),
            "critical_probability": round(probability, 3),
            "risk": risk,
            "counters": counters,
            "counters_needed": needed,
            "capacity_per_min": round(capacity, 2),
            "critical_queue": math.ceil(d.high_wait * capacity),
            "radar": radar_result,
            "recommendation": self._recommend(risk, predicted, counters, needed, radar_result),
        }

    def _risk(self, wait, probability):
        if wait >= self.domain.high_wait or probability >= HIGH_PROBABILITY:
            return "HIGH"
        if wait >= self.domain.medium_wait or probability >= MEDIUM_PROBABILITY:
            return "MEDIUM"
        return "LOW"

    def _recommend(self, risk, predicted, counters, needed, radar_result):
        d = self.domain
        server, item = d.server, d.item
        safe = radar_result.get("safe_servers")
        target = safe if safe is not None else needed
        extra = max(0, target - counters)
        ttc = radar_result["time_to_collapse"]
        when = f" Collapse expected in about {ttc['median']} min." if ttc and ttc["median"] > 0 else ""

        if risk == "HIGH":
            if extra > 0:
                return (f"Add {extra} more {plural(server, extra)} now - {item} are expected to reach "
                        f"{predicted} in {d.main_horizon} minutes.{when}")
            return f"Backlog is critical. {d.action_hint}"
        if risk == "MEDIUM":
            if extra > 0:
                return f"Backlog is building. Get {extra} more {plural(server, extra)} ready.{when}"
            return "Backlog is building. Keep monitoring closely."
        if counters > 1 and needed < counters and radar_result["status"] == "SAFE":
            return f"Under control. One {server} could be released if needed elsewhere."
        return "Under control. No action needed."
