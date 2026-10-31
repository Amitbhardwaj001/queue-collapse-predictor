"""Application domains.

Queue collapse is not only a canteen problem. Anything that has *arrivals*, a
*service capacity* and a *backlog* can collapse: a hospital OPD, an API server
behind a load balancer, a call centre. The forecasting, risk and what-if engine
is the same for all of them; a Domain only describes the vocabulary, the
thresholds and the simulated demand pattern of one application.

Time always advances in one-minute steps. "Wait" is the time a new arrival
would wait: backlog / (servers x service rate).
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Demand:
    """How a simulated day of demand looks."""
    base_rate: float                    # arrivals per minute outside rush hours
    peaks: tuple                        # (centre minute of day, spread in minutes, extra arrivals/min)
    weekend_factor: float
    bursts_per_day: float               # sudden short surges (class break, ambulance, flash sale, outage)
    burst_len: tuple                    # (min, max) minutes
    burst_rate: tuple                   # (min, max) extra arrivals per minute
    balk_start: int                     # beyond this backlog some arrivals give up / time out
    day_start: int                      # minute of day the simulation starts
    day_end: int
    service_sec: tuple                  # (min, max) seconds one server needs per item
    staff_capacity: float               # items/min a manager assumes one server handles
    servers: tuple                      # (min, max) servers the manager may run


@dataclass(frozen=True)
class Domain:
    key: str
    name: str
    tagline: str
    item: str                           # what waits: people / patients / requests / calls
    server: str                         # what serves: counter / doctor / worker / agent
    horizons: tuple                     # forecast horizons in minutes
    medium_wait: float                  # wait (minutes) thresholds
    high_wait: float
    target_wait: float
    default_servers: int
    default_service_sec: float
    max_servers: int                    # upper bound accepted by the settings form
    service_sec_limits: tuple
    demand: Demand
    sim_days: int = 120
    demo_start: int = 12 * 60 + 5       # minute of day the live simulation starts at
    demo_queue: int = 4
    lags: tuple = (1, 2, 3, 5, 10)      # minutes of history used as features
    camera: bool = False                # backlog can be counted from camera footage
    action_hint: str = ""               # what to do when the server count cannot grow further
    locations: tuple = field(default_factory=tuple)   # (name, is_demo)

    @property
    def item_one(self):
        return "person" if self.item == "people" else self.item.rstrip("s")

    @property
    def history_min(self):
        """Minutes of recent history the models need."""
        return max(30, max(self.lags) + 5)

    @property
    def main_horizon(self):
        return self.horizons[1]

    @property
    def radar_horizon(self):
        return self.horizons[-1]

    def public(self):
        """The part of the domain the browser needs."""
        return {
            "key": self.key, "name": self.name, "tagline": self.tagline, "item": self.item,
            "server": self.server, "horizons": list(self.horizons), "main_horizon": self.main_horizon,
            "medium_wait": self.medium_wait, "high_wait": self.high_wait, "target_wait": self.target_wait,
            "max_servers": self.max_servers, "service_sec_limits": list(self.service_sec_limits),
            "camera": self.camera, "item_one": self.item_one,
        }


QUEUE = Domain(
    key="queue", name="Physical queues", tagline="Canteens, banks, ticket counters (camera + YOLOv8)",
    item="people", server="counter", horizons=(5, 10, 15),
    medium_wait=6, high_wait=12, target_wait=8,
    default_servers=2, default_service_sec=45, max_servers=20, service_sec_limits=(5, 900),
    demand=Demand(
        base_rate=0.35,
        peaks=((8 * 60 + 45, 25, 2.0), (13 * 60, 35, 4.5), (16 * 60 + 30, 30, 2.5)),
        weekend_factor=0.45, bursts_per_day=4, burst_len=(2, 5), burst_rate=(3, 7), balk_start=30,
        day_start=7 * 60, day_end=20 * 60, service_sec=(40, 75), staff_capacity=1.3, servers=(1, 4)),
    sim_days=150, camera=True,
    action_hint="Speed up service (pre-orders, token system) and guide people to other counters.",
    locations=(("Main Canteen", False), ("Demo Canteen (simulated live feed)", True)),
)

HOSPITAL = Domain(
    key="hospital", name="Hospital OPD / ER", tagline="Patients waiting for a doctor",
    item="patients", server="doctor", horizons=(15, 30, 60),
    medium_wait=45, high_wait=90, target_wait=60,
    default_servers=4, default_service_sec=540, max_servers=20, service_sec_limits=(120, 1800),
    demand=Demand(
        base_rate=0.12,
        peaks=((10 * 60 + 30, 80, 0.55), (18 * 60 + 30, 70, 0.40)),
        weekend_factor=0.6, bursts_per_day=2, burst_len=(3, 8), burst_rate=(0.6, 1.5), balk_start=45,
        day_start=8 * 60, day_end=22 * 60, service_sec=(420, 660), staff_capacity=0.115, servers=(2, 8)),
    sim_days=120, demo_start=9 * 60 + 40, demo_queue=6, lags=(5, 10, 20, 30, 60),
    action_hint="Call in an on-call doctor, fast-track minor cases and divert walk-ins to another clinic.",
    locations=(("City Hospital OPD (simulated live feed)", True),),
)

WEB = Domain(
    key="web", name="Web / API servers", tagline="Requests waiting for a worker",
    item="requests", server="worker", horizons=(2, 5, 10),
    medium_wait=0.5, high_wait=1.0, target_wait=0.6,
    default_servers=8, default_service_sec=2, max_servers=64, service_sec_limits=(0.2, 60),
    demand=Demand(
        base_rate=40,
        peaks=((9 * 60 + 30, 40, 80), (14 * 60, 50, 100), (20 * 60, 30, 130)),
        weekend_factor=0.7, bursts_per_day=2, burst_len=(3, 8), burst_rate=(60, 180), balk_start=400,
        day_start=8 * 60, day_end=22 * 60, service_sec=(1.5, 3.0), staff_capacity=20, servers=(3, 24)),
    sim_days=120, demo_start=19 * 60 + 45, demo_queue=30,
    action_hint="Scale out workers, enable rate limiting and shed non-critical traffic.",
    locations=(("Payments API (simulated live feed)", True),),
)

CALLS = Domain(
    key="calls", name="Call centre", tagline="Callers waiting for an agent",
    item="calls", server="agent", horizons=(5, 10, 15),
    medium_wait=2, high_wait=5, target_wait=3,
    default_servers=10, default_service_sec=300, max_servers=60, service_sec_limits=(30, 1800),
    demand=Demand(
        base_rate=0.6,
        peaks=((10 * 60 + 30, 50, 2.6), (15 * 60, 45, 2.0)),
        weekend_factor=0.4, bursts_per_day=3, burst_len=(3, 8), burst_rate=(3, 8), balk_start=25,
        day_start=8 * 60, day_end=20 * 60, service_sec=(240, 400), staff_capacity=0.125, servers=(3, 24)),
    sim_days=120, demo_start=10 * 60 + 5, demo_queue=3,
    action_hint="Pull back-office staff onto phones, enable call-backs and play a delay announcement.",
    locations=(("Support Call Centre (simulated live feed)", True),),
)

DOMAINS = {d.key: d for d in (QUEUE, HOSPITAL, WEB, CALLS)}


def get_domain(key):
    return DOMAINS[key]
