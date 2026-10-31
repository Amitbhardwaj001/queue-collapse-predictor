"""Synthetic demand simulator for every domain.

There is no public minute-by-minute backlog dataset for a canteen, a hospital
OPD, an API server or a call centre, so the models are first trained on
simulated days. The arrival pattern (rush hours, sudden bursts, quieter
weekends) and the service process (open servers x items served per minute) are
described per domain in domains.py. Once a deployment has logged a few weeks of
real counts, retrain on those instead (see README).
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .domains import QUEUE


def arrival_rate(minute_of_day, weekday, day_factor=1.0, domain=QUEUE):
    """Expected new arrivals per minute at a given time of day."""
    demand = domain.demand
    rate = demand.base_rate
    for centre, spread, amplitude in demand.peaks:
        rate += amplitude * np.exp(-0.5 * ((minute_of_day - centre) / spread) ** 2)
    if weekday >= 5:
        rate *= demand.weekend_factor
    return rate * day_factor


class QueueSimulator:
    """Moves one backlog forward a minute at a time."""

    def __init__(self, rng, weekday, counters=2, service_time_sec=45, day_factor=1.0, queue=0,
                 domain=QUEUE):
        self.rng = rng
        self.weekday = weekday
        self.counters = counters
        self.service_time_sec = service_time_sec
        self.day_factor = day_factor
        self.queue = queue
        self.domain = domain
        demand = domain.demand
        day_minutes = demand.day_end - demand.day_start
        self._burst_probability = demand.bursts_per_day / day_minutes
        self._burst_left = 0
        self._burst_rate = 0.0

    @property
    def service_rate(self):
        """Items one server handles per minute."""
        return 60.0 / self.service_time_sec

    def step(self, minute_of_day):
        demand = self.domain.demand
        rate = arrival_rate(minute_of_day, self.weekday, self.day_factor, self.domain)
        if self._burst_left == 0 and self.rng.random() < self._burst_probability:
            self._burst_left = int(self.rng.integers(demand.burst_len[0], demand.burst_len[1] + 1))
            self._burst_rate = float(self.rng.uniform(*demand.burst_rate))
        if self._burst_left > 0:
            rate += self._burst_rate
            self._burst_left -= 1

        arrivals = int(self.rng.poisson(rate))
        if self.queue > demand.balk_start:      # long backlog: some arrivals give up or time out
            stay = max(0.4, 1 - (self.queue - demand.balk_start) / (5 * demand.balk_start))
            arrivals = int(self.rng.binomial(arrivals, stay))
        self.queue += arrivals

        served = min(self.queue, int(self.rng.poisson(self.counters * self.service_rate)))
        self.queue -= served
        return self.queue


def staffed_counters(rng, minute_of_day, weekday, domain=QUEUE):
    """Servers a manager would run for the coming hour, with some under/over-staffing."""
    demand = domain.demand
    expected = max(arrival_rate(minute_of_day + m, weekday, domain=domain) for m in (0, 15, 30, 45, 60))
    counters = int(np.ceil(expected / demand.staff_capacity))
    if rng.random() < 0.35:
        counters += int(rng.choice([-1, 1]))
    return int(np.clip(counters, *demand.servers))


def generate_history(days=None, seed=42, domain=QUEUE):
    """Return one row per simulated minute: ts, queue, counters, service_rate."""
    days = days or domain.sim_days
    demand = domain.demand
    rng = np.random.default_rng(seed)
    first_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days)
    rows = []
    for d in range(days):
        day = first_day + timedelta(days=d)
        sim = QueueSimulator(
            rng,
            weekday=day.weekday(),
            service_time_sec=float(rng.uniform(*demand.service_sec)),
            day_factor=float(rng.lognormal(0, 0.15)),
            domain=domain,
        )
        for minute in range(demand.day_start, demand.day_end):
            if minute % 60 == 0 or minute == demand.day_start:
                sim.counters = staffed_counters(rng, minute, day.weekday(), domain)
            queue = sim.step(minute)
            rows.append((day + timedelta(minutes=minute), queue, sim.counters, sim.service_rate))
    return pd.DataFrame(rows, columns=["ts", "queue", "counters", "service_rate"])


if __name__ == "__main__":
    from .domains import DOMAINS
    for dom in DOMAINS.values():
        df = generate_history(days=7, domain=dom)
        print(dom.name)
        print(df.groupby(df.ts.dt.hour)["queue"].mean().round(1).to_string())
