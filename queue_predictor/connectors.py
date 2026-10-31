"""Live metric connectors: read the backlog of a real system once a minute.

Two kinds of feed are supported, both over plain HTTP(S):

* ``json``        GET a URL that returns JSON and pick the backlog with a dotted
                  path, e.g. ``data.queue_depth`` or ``items.0.pending``.
* ``prometheus``  run an instant query against a Prometheus server, e.g.
                  ``sum(http_requests_in_flight)``.

Systems that cannot be polled can instead push readings to
``POST /api/ingest/<location_id>``.
"""
import json
import threading
import urllib.parse
import urllib.request
from datetime import datetime

MIN_INTERVAL_SEC = 5
KINDS = ("json", "prometheus")


def validate(config):
    """Return a cleaned connector config or raise ValueError."""
    kind = config.get("kind")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    url = str(config.get("url", "")).strip()
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ValueError("url must start with http:// or https://")
    cleaned = {"kind": kind, "url": url}
    if kind == "json":
        cleaned["path"] = str(config.get("path", "")).strip()
        if not cleaned["path"]:
            raise ValueError("path is required, e.g. data.queue_depth")
    else:
        cleaned["query"] = str(config.get("query", "")).strip()
        if not cleaned["query"]:
            raise ValueError("query is required, e.g. sum(http_requests_in_flight)")
    try:
        cleaned["interval_sec"] = max(MIN_INTERVAL_SEC, float(config.get("interval_sec", 60)))
    except (TypeError, ValueError):
        raise ValueError("interval_sec must be a number")
    return cleaned


def _get_json(url, timeout=8):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def dig(data, path):
    """Follow a dotted path through dicts and lists."""
    for part in path.split("."):
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def read_backlog(config, fetch=_get_json):
    """Read one backlog value (items waiting) from the configured feed."""
    if config["kind"] == "json":
        value = dig(fetch(config["url"]), config["path"])
    else:
        url = config["url"].rstrip("/") + "/api/v1/query?" + urllib.parse.urlencode({"query": config["query"]})
        result = fetch(url)["data"]["result"]
        if not result:
            raise ValueError("The Prometheus query returned no data")
        value = result[0]["value"][1]
    return max(0, int(round(float(value))))


class ConnectorManager:
    """Runs one polling thread per location that has a connector."""

    def __init__(self, on_reading, fetch=_get_json):
        self._on_reading = on_reading      # callback(location_id, ts, count)
        self._fetch = fetch
        self._threads = {}
        self.state = {}                    # location_id -> {"last_value", "last_time", "error"}

    def running(self, location_id):
        entry = self._threads.get(location_id)
        return bool(entry and entry[0].is_alive())

    def start(self, location_id, config):
        self.stop(location_id)
        stop = threading.Event()
        thread = threading.Thread(target=self._loop, args=(location_id, config, stop), daemon=True)
        self._threads[location_id] = (thread, stop)
        self.state[location_id] = {"last_value": None, "last_time": None, "error": None}
        thread.start()

    def stop(self, location_id):
        entry = self._threads.pop(location_id, None)
        if entry:
            entry[1].set()

    def poll_now(self, location_id, config):
        """One immediate read (used by the 'Test' button and by the polling loop)."""
        state = self.state.setdefault(location_id, {"last_value": None, "last_time": None, "error": None})
        try:
            value = read_backlog(config, self._fetch)
        except Exception as exc:   # network, JSON, path, ... - shown in the dashboard
            state["error"] = f"{type(exc).__name__}: {exc}"
            raise
        now = datetime.now().replace(microsecond=0)
        state.update(last_value=value, last_time=now.strftime("%H:%M:%S"), error=None)
        self._on_reading(location_id, now, value)
        return value

    def _loop(self, location_id, config, stop):
        while not stop.is_set():
            try:
                self.poll_now(location_id, config)
            except Exception:
                pass                # already recorded in self.state
            stop.wait(config["interval_sec"])
