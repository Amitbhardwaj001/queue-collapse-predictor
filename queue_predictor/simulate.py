"""Synthetic canteen queue simulator.

There is no public minute-by-minute queue dataset for a college canteen, so the
models are first trained on simulated days. The arrival pattern (breakfast,
lunch and evening rushes, sudden class-break bursts, quieter weekends) and the
service process (open counters x people served per minute) are modelled on a
real canteen. Once the camera has logged a few weeks of real counts, retrain
on those instead (see README).
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import SIM_DAY_END, SIM_DAY_START, SIM_DAYS

# (centre minute of day, spread in minutes, extra arrivals per minute)
PEAKS = [
    (8 * 60 + 45, 25, 2.0),
    (13 * 60, 35, 4.5),
    (16 * 60 + 30, 30, 2.5),
]
BURST_PROBABILITY = 4 / 780    # about four class-break rushes per day
BALK_START = 30                # beyond this length some people give up and leave


def arrival_rate(minute_of_day, weekday, day_factor=1.0):
    """Expected new arrivals per minute at a given time of day."""
    rate = 0.35
    for centre, spread, amplitude in PEAKS:
        rate += amplitude * np.exp(-0.5 * ((minute_of_day - centre) / spread) ** 2)
    if weekday >= 5:
        rate *= 0.45
    return rate * day_factor


class QueueSimulator:
    """Moves a single queue forward one minute at a time."""

    def __init__(self, rng, weekday, counters=2, service_time_sec=45, day_factor=1.0, queue=0):
        self.rng = rng
        self.weekday = weekday
        self.counters = counters
        self.service_time_sec = service_time_sec
        self.day_factor = day_factor
        self.queue = queue
        self._burst_left = 0
        self._burst_rate = 0.0

    @property
    def service_rate(self):
        """People one counter serves per minute."""
        return 60.0 / self.service_time_sec

    def step(self, minute_of_day):
        rate = arrival_rate(minute_of_day, self.weekday, self.day_factor)
        if self._burst_left == 0 and self.rng.random() < BURST_PROBABILITY:
            self._burst_left = int(self.rng.integers(2, 5))
            self._burst_rate = float(self.rng.uniform(3, 7))
        if self._burst_left > 0:
            rate += self._burst_rate
            self._burst_left -= 1

        arrivals = int(self.rng.poisson(rate))
        if self.queue > BALK_START:
            stay = max(0.4, 1 - (self.queue - BALK_START) / 150)
            arrivals = int(self.rng.binomial(arrivals, stay))
        self.queue += arrivals

        served = min(self.queue, int(self.rng.poisson(self.counters * self.service_rate)))
        self.queue -= served
        return self.queue


def staffed_counters(rng, minute_of_day, weekday):
    """Counters a manager would open for the coming hour, with some under/over-staffing."""
    expected = max(arrival_rate(minute_of_day + m, weekday) for m in (0, 15, 30, 45, 60))
    counters = int(np.ceil(expected / 1.3))
    if rng.random() < 0.35:
        counters += int(rng.choice([-1, 1]))
    return int(np.clip(counters, 1, 4))


def generate_history(days=SIM_DAYS, seed=42):
    """Return one row per simulated minute: ts, queue, counters, service_rate."""
    rng = np.random.default_rng(seed)
    first_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days)
    rows = []
    for d in range(days):
        day = first_day + timedelta(days=d)
        sim = QueueSimulator(
            rng,
            weekday=day.weekday(),
            service_time_sec=float(rng.uniform(40, 75)),
            day_factor=float(rng.lognormal(0, 0.15)),
        )
        for minute in range(SIM_DAY_START, SIM_DAY_END):
            if minute % 60 == 0 or minute == SIM_DAY_START:
                sim.counters = staffed_counters(rng, minute, day.weekday())
            queue = sim.step(minute)
            rows.append((day + timedelta(minutes=minute), queue, sim.counters, sim.service_rate))
    return pd.DataFrame(rows, columns=["ts", "queue", "counters", "service_rate"])


if __name__ == "__main__":
    df = generate_history(days=7)
    print(df.describe())
    print(df.groupby(df.ts.dt.hour)["queue"].mean().round(1))
