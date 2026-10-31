"""Collapse Radar: time-to-collapse and what-if staffing, for any domain.

The XGBoost models say how large the backlog will be. The radar turns that
into a *decision*:

1. The forecast backlog is converted into an implied arrival rate per minute
   (net change + what the servers clear in the meantime).
2. That demand is replayed through a stochastic queue (Poisson arrivals with an
   uncertain rate, Poisson service) many times for every staffing option.
3. A path *collapses* when the wait of a new arrival reaches the domain's
   critical wait. From the paths we read off the probability of collapse, the
   time until it happens, an uncertainty band for the backlog, and the smallest
   number of servers that keeps the collapse probability low.

The same random numbers are reused for every staffing option, so adding a
server can never look worse than not adding it.
"""
import numpy as np
from scipy.stats import poisson

from .config import RADAR_ALARM_PROBABILITY, RADAR_DEMAND_SPREAD, RADAR_PATHS, RADAR_SAFE_PROBABILITY


def implied_arrival_rates(current, forecast, horizons, capacity):
    """Arrivals per minute between consecutive knots (now, +h1, +h2, ...), one value per minute."""
    knots_t = [0, *horizons]
    knots_q = [current, *[forecast[h] for h in horizons]]
    rates = []
    for (t0, q0), (t1, q1) in zip(zip(knots_t, knots_q), zip(knots_t[1:], knots_q[1:])):
        net = (q1 - q0) / (t1 - t0)
        served = min(capacity, (q0 + q1) / 2)
        rates.extend([max(0.0, net + served)] * (t1 - t0))
    return np.array(rates)


def _simulate(current, rates, servers, service_rate, paths, spread, seed):
    """Backlog of every path at every minute -> array (paths, minutes + 1)."""
    rng = np.random.default_rng(seed)
    minutes = len(rates)
    demand_factor = rng.lognormal(-spread ** 2 / 2, spread, size=(paths, 1))   # mean 1
    arrivals = rng.poisson(rates[None, :] * demand_factor)
    service_draw = rng.random((paths, minutes))
    queue = np.empty((paths, minutes + 1))
    queue[:, 0] = current
    capacity = servers * service_rate
    # Poisson service drawn through its inverse CDF so the draws can be shared by all server counts.
    served_inv = poisson.ppf(service_draw, capacity)
    for t in range(minutes):
        queue[:, t + 1] = queue[:, t] + arrivals[:, t]
        queue[:, t + 1] -= np.minimum(queue[:, t + 1], served_inv[:, t])
    return queue


def _summarise(queue, servers, service_rate, high_wait):
    critical = high_wait * servers * service_rate
    crossed = queue >= critical
    ever = crossed.any(axis=1)
    first = np.where(ever, crossed.argmax(axis=1), -1)
    peak_wait = queue.max(axis=1) / (servers * service_rate)
    return {
        "servers": servers,
        "collapse_probability": float(ever.mean()),
        "median_peak_wait": float(np.median(peak_wait)),
        "p90_peak_wait": float(np.percentile(peak_wait, 90)),
        "first_crossing": first,
        "critical_queue": critical,
    }


def analyse(domain, current, forecast, servers, service_rate, what_if=True, paths=RADAR_PATHS, seed=7):
    """Run the radar. forecast: {minutes ahead: expected backlog}."""
    horizons = list(domain.horizons)
    capacity = servers * service_rate
    rates = implied_arrival_rates(current, forecast, horizons, capacity)

    base = _simulate(current, rates, servers, service_rate, paths, RADAR_DEMAND_SPREAD, seed)
    now = _summarise(base, servers, service_rate, domain.high_wait)
    collapsed_now = current >= now["critical_queue"]
    probability = 1.0 if collapsed_now else now["collapse_probability"]

    crossing = now["first_crossing"][now["first_crossing"] >= 0]
    if collapsed_now:
        time_to_collapse = {"median": 0, "p10": 0, "p90": 0}
    elif len(crossing) >= max(5, 0.1 * paths):
        time_to_collapse = {"median": int(np.median(crossing)),
                            "p10": int(np.percentile(crossing, 10)),
                            "p90": int(np.percentile(crossing, 90))}
    else:
        time_to_collapse = None

    if collapsed_now:
        status = "COLLAPSED"
    elif probability >= RADAR_ALARM_PROBABILITY:
        status = "IMMINENT"
    elif probability >= RADAR_SAFE_PROBABILITY:
        status = "AT_RISK"
    else:
        status = "SAFE"

    fan = [{"t": t, "p10": float(np.percentile(base[:, t], 10)), "p50": float(np.percentile(base[:, t], 50)),
            "p90": float(np.percentile(base[:, t], 90))} for t in range(base.shape[1])]

    result = {
        "status": status,
        "collapse_probability": round(probability, 3),
        "time_to_collapse": time_to_collapse,
        "horizon_min": len(rates),
        "peak_load": round(float(rates.max() / capacity), 2),      # arrival rate / service capacity
        "fan": fan,
    }
    if not what_if:
        return result

    lo = domain.demand.servers[0]
    options = range(max(1, min(lo, servers)), min(domain.max_servers, max(servers + 8, domain.demand.servers[1])) + 1)
    table = []
    for c in options:
        summary = _summarise(_simulate(current, rates, c, service_rate, paths, RADAR_DEMAND_SPREAD, seed),
                             c, service_rate, domain.high_wait)
        crit = summary["critical_queue"]
        p = 1.0 if current >= crit else summary["collapse_probability"]
        table.append({"servers": c, "collapse_probability": round(p, 3),
                      "median_peak_wait": round(summary["median_peak_wait"], 2),
                      "p90_peak_wait": round(summary["p90_peak_wait"], 2)})
    safe = next((row["servers"] for row in table if row["collapse_probability"] <= RADAR_SAFE_PROBABILITY), None)
    result["what_if"] = table
    result["safe_servers"] = safe
    result["current_servers"] = servers
    return result
