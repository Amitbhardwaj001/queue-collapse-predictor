"""SQLite storage for locations, people counts and predictions."""
import json
import sqlite3
from contextlib import contextmanager

import pandas as pd

from .config import DB_PATH, DEFAULT_COUNTERS, DEFAULT_SERVICE_TIME_SEC

TS_FORMAT = "%Y-%m-%d %H:%M:%S"

SCHEMA = """
CREATE TABLE IF NOT EXISTS locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    counters INTEGER NOT NULL,
    service_time_sec REAL NOT NULL,
    is_demo INTEGER NOT NULL DEFAULT 0,
    roi TEXT
);
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    location_id INTEGER NOT NULL REFERENCES locations(id),
    ts TEXT NOT NULL,
    count INTEGER NOT NULL,
    source TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obs_loc_ts ON observations(location_id, ts);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    location_id INTEGER NOT NULL REFERENCES locations(id),
    ts TEXT NOT NULL,
    current_queue INTEGER NOT NULL,
    predicted_queue INTEGER NOT NULL,
    horizon_min INTEGER NOT NULL,
    critical_probability REAL NOT NULL,
    risk TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_loc_ts ON predictions(location_id, ts);
"""


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with connect() as conn:
        conn.executescript(SCHEMA)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(locations)")}
        if "roi" not in columns:  # databases created before the queue-area feature
            conn.execute("ALTER TABLE locations ADD COLUMN roi TEXT")
        for name, demo in (("Main Canteen", 0), ("Demo Canteen (simulated live feed)", 1)):
            conn.execute(
                "INSERT OR IGNORE INTO locations (name, counters, service_time_sec, is_demo) VALUES (?, ?, ?, ?)",
                (name, DEFAULT_COUNTERS, DEFAULT_SERVICE_TIME_SEC, demo),
            )


def _location(row):
    location = dict(row)
    location["roi"] = json.loads(location["roi"]) if location["roi"] else None
    return location


def list_locations():
    with connect() as conn:
        return [_location(r) for r in conn.execute("SELECT * FROM locations ORDER BY id")]


def get_location(location_id):
    with connect() as conn:
        row = conn.execute("SELECT * FROM locations WHERE id = ?", (location_id,)).fetchone()
        return _location(row) if row else None


def update_roi(location_id, roi):
    """roi: [x1, y1, x2, y2] as fractions of the frame size, or None for the whole frame."""
    with connect() as conn:
        conn.execute("UPDATE locations SET roi = ? WHERE id = ?",
                     (json.dumps(roi) if roi else None, location_id))


def update_location(location_id, counters, service_time_sec):
    with connect() as conn:
        conn.execute("UPDATE locations SET counters = ?, service_time_sec = ? WHERE id = ?",
                     (counters, service_time_sec, location_id))


def add_observations(location_id, rows, source):
    """rows: iterable of (datetime, count)."""
    with connect() as conn:
        conn.executemany(
            "INSERT INTO observations (location_id, ts, count, source) VALUES (?, ?, ?, ?)",
            [(location_id, ts.strftime(TS_FORMAT), int(count), source) for ts, count in rows],
        )


def latest_timestamp(location_id):
    with connect() as conn:
        row = conn.execute("SELECT MAX(ts) AS ts FROM observations WHERE location_id = ?",
                           (location_id,)).fetchone()
        return pd.Timestamp(row["ts"]) if row["ts"] else None


def observations_since(location_id, since):
    with connect() as conn:
        df = pd.read_sql_query(
            "SELECT ts, count, source FROM observations WHERE location_id = ? AND ts >= ? ORDER BY ts",
            conn, params=(location_id, since.strftime(TS_FORMAT)),
        )
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def add_prediction(location_id, result):
    with connect() as conn:
        conn.execute(
            "INSERT INTO predictions (location_id, ts, current_queue, predicted_queue, horizon_min, "
            "critical_probability, risk) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (location_id, pd.Timestamp(result["as_of"]).strftime(TS_FORMAT), result["current_queue"],
             result["predicted_queue"], result["horizon_min"], result["critical_probability"], result["risk"]),
        )


def predictions_since(location_id, since):
    with connect() as conn:
        df = pd.read_sql_query(
            "SELECT ts, current_queue, predicted_queue, horizon_min, critical_probability, risk "
            "FROM predictions WHERE location_id = ? AND ts >= ? ORDER BY ts",
            conn, params=(location_id, since.strftime(TS_FORMAT)),
        )
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def clear_location_data(location_id):
    with connect() as conn:
        conn.execute("DELETE FROM observations WHERE location_id = ?", (location_id,))
        conn.execute("DELETE FROM predictions WHERE location_id = ?", (location_id,))
