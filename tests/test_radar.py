import numpy as np
import pytest

from queue_predictor import radar
from queue_predictor.domains import DOMAINS, QUEUE
from queue_predictor.simulate import QueueSimulator, generate_history

RATE = 60 / 45


def _run(current, forecast, servers=2, what_if=False):
    return radar.analyse(QUEUE, current, forecast, servers, RATE, what_if=what_if)


def test_calm_queue_is_safe():
    result = _run(3, {5: 3, 10: 3, 15: 3})
    assert result["status"] == "SAFE"
    assert result["time_to_collapse"] is None


def test_surge_gives_a_time_to_collapse():
    result = _run(20, {5: 30, 10: 45, 15: 55})
    assert result["status"] in {"IMMINENT", "COLLAPSED"}
    assert 0 < result["time_to_collapse"]["median"] <= QUEUE.radar_horizon
    assert result["time_to_collapse"]["p10"] <= result["time_to_collapse"]["p90"]


def test_already_collapsed():
    critical = int(QUEUE.high_wait * 2 * RATE) + 5
    result = _run(critical, {5: critical, 10: critical, 15: critical})
    assert result["status"] == "COLLAPSED"
    assert result["time_to_collapse"]["median"] == 0


def test_more_servers_never_increase_collapse_probability():
    table = _run(20, {5: 30, 10: 45, 15: 55}, what_if=True)["what_if"]
    probabilities = [row["collapse_probability"] for row in table]
    assert probabilities == sorted(probabilities, reverse=True)


def test_safe_staffing_is_really_safe():
    result = _run(20, {5: 30, 10: 45, 15: 55}, what_if=True)
    assert result["safe_servers"] is not None and result["safe_servers"] > 2
    row = next(r for r in result["what_if"] if r["servers"] == result["safe_servers"])
    assert row["collapse_probability"] <= 0.10


def test_fan_bands_are_ordered():
    for point in _run(10, {5: 12, 10: 14, 15: 16})["fan"]:
        assert point["p10"] <= point["p50"] <= point["p90"]


@pytest.mark.parametrize("key", list(DOMAINS))
def test_every_domain_simulates_and_has_a_rush(key):
    domain = DOMAINS[key]
    df = generate_history(days=6, seed=3, domain=domain)
    assert (df["queue"] >= 0).all()
    assert df["queue"].max() > df["queue"].median() * 3
    sim = QueueSimulator(np.random.default_rng(0), weekday=1, counters=domain.default_servers,
                         service_time_sec=domain.default_service_sec, domain=domain)
    assert sim.step(domain.demo_start) >= 0
