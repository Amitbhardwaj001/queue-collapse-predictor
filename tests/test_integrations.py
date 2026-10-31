"""Person tracking, live connectors, alerts and the PDF report."""
import io

import cv2
import numpy as np
import pytest

from queue_predictor import alerts, connectors
from queue_predictor.detector import PersonDetector
from queue_predictor.domains import QUEUE
from queue_predictor.tracker import ByteTracker, FlowMeter


# ---------- tracking ----------
def _walker(t, x0, speed):
    x = x0 + speed * t
    return (x, 100, x + 30, 200, 0.9)


def test_tracker_keeps_identities_apart():
    tracker = ByteTracker()
    for step in range(20):
        t = step * 0.4
        visible = tracker.update([_walker(t, 20, 15), _walker(t, 200, -15)], t)
    assert len(visible) == 2
    assert sorted(tr.id for tr in visible) == [1, 2]


def test_weak_detection_keeps_a_track_alive():
    tracker = ByteTracker()
    for step in range(6):
        tracker.update([(50, 100, 80, 200, 0.9)], step * 0.4)
    # the person is half hidden: low-confidence box only, the track must survive
    visible = tracker.update([(51, 100, 81, 200, 0.3)], 2.4)
    assert [tr.id for tr in visible] == [1]


def test_flow_meter_measures_arrivals_departures_and_wait():
    tracker = ByteTracker()
    # 6 people each stand in the queue for 20 s, arriving every 8 s; nobody is there at the start or end
    people = [(10 + 8 * i, 10 + 8 * i + 20) for i in range(6)]
    for step in range(0, 170):
        t = step * 0.5
        boxes = [(10 + 40 * i, 50, 30 + 40 * i, 150, 0.9) for i, (a, b) in enumerate(people) if a <= t < b]
        tracker.update(boxes, t)
    summary = FlowMeter(400, 300, 85).measure(tracker.all_tracks(), servers=2)
    assert summary["arrivals"] == 6 and summary["departures"] == 6
    assert summary["avg_wait_min"] == pytest.approx(20 / 60, abs=0.05)
    assert summary["suggested_service_time_sec"] is not None


def test_people_already_there_at_the_start_are_not_arrivals():
    tracker = ByteTracker()
    for step in range(0, 60):
        t = step * 0.5
        tracker.update([(10, 50, 30, 150, 0.9)], t)            # stands there the whole video
    summary = FlowMeter(400, 300, 30).measure(tracker.all_tracks())
    assert summary["arrivals"] == 0 and summary["departures"] == 0


def _blob_detector():
    """A detector that finds dark rectangles, so the video test needs no neural network."""
    detector = PersonDetector.__new__(PersonDetector)
    detector.backend = "test"

    def detect(frame):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours((grey < 100).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [(x, y, x + w, y + h, 0.9) for x, y, w, h in map(cv2.boundingRect, contours) if w > 8 and h > 20]

    detector.detect = detect
    return detector


def _queue_video(path, people=8, fps=5, seconds=60, stay=15):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (400, 240))
    for frame_no in range(fps * seconds):
        t = frame_no / fps
        frame = np.full((240, 400, 3), 200, np.uint8)
        for i in range(people):
            arrives = 6 + 4 * i
            if arrives <= t < arrives + stay:
                cv2.rectangle(frame, (10 + 45 * i, 100), (35 + 45 * i, 200), (30, 30, 30), -1)
        writer.write(frame)
    writer.release()


def test_track_video_end_to_end(tmp_path):
    video = tmp_path / "queue.avi"
    _queue_video(video)
    samples, duration, flow, image, frame = _blob_detector().track_video(video, roi=None, servers=2)
    assert flow["arrivals"] == 8 and flow["departures"] == 8
    assert 13 <= flow["avg_wait_min"] * 60 <= 17
    assert flow["suggested_service_time_sec"] > 0
    assert samples and duration > 50


def test_track_endpoint_and_apply_flow(client, locations, tmp_path, monkeypatch):
    from queue_predictor import app as app_module
    monkeypatch.setattr(app_module, "get_detector", _blob_detector)
    video = tmp_path / "q.avi"
    _queue_video(video)
    loc_id = locations["main"]["id"]
    with open(video, "rb") as handle:
        response = client.post(f"/api/track/{loc_id}", data={"file": (handle, "q.avi")},
                               content_type="multipart/form-data")
    body = response.get_json()
    assert response.status_code == 200, body
    assert body["flow"]["arrivals"] == 8
    assert body["prediction"]["radar"]["status"]

    status = client.get(f"/api/status/{loc_id}").get_json()
    assert status["flow"]["departures"] == 8
    applied = client.post(f"/api/locations/{loc_id}/apply-flow")
    assert applied.status_code == 200
    assert applied.get_json()["service_time_sec"] == pytest.approx(status["flow"]["suggested_service_time_sec"], abs=0.1)

    assert client.post(f"/api/track/{locations['web']['id']}", data={"file": (io.BytesIO(b"x"), "q.avi")},
                       content_type="multipart/form-data").status_code == 400
    assert client.post(f"/api/track/{loc_id}", data={"file": (io.BytesIO(b"x"), "n.txt")},
                       content_type="multipart/form-data").status_code == 400


# ---------- live connectors ----------
def test_connector_validation():
    ok = connectors.validate({"kind": "json", "url": "http://x/api", "path": "a.b", "interval_sec": 1})
    assert ok["interval_sec"] == connectors.MIN_INTERVAL_SEC
    for bad in ({"kind": "ftp", "url": "http://x"}, {"kind": "json", "url": "file:///etc/passwd", "path": "a"},
                {"kind": "json", "url": "http://x"}, {"kind": "prometheus", "url": "http://x"}):
        with pytest.raises(ValueError):
            connectors.validate(bad)


def test_reading_json_and_prometheus_feeds():
    json_cfg = {"kind": "json", "url": "http://x", "path": "data.items.1.pending"}
    assert connectors.read_backlog(json_cfg, lambda url: {"data": {"items": [{"pending": 1}, {"pending": 41.6}]}}) == 42

    prom_cfg = {"kind": "prometheus", "url": "http://prom:9090/", "query": "sum(inflight)"}
    seen = []

    def fake(url):
        seen.append(url)
        return {"data": {"result": [{"value": [1700000000, "17"]}]}}

    assert connectors.read_backlog(prom_cfg, fake) == 17
    assert seen[0].startswith("http://prom:9090/api/v1/query?query=sum%28inflight%29")
    with pytest.raises(ValueError):
        connectors.read_backlog(prom_cfg, lambda url: {"data": {"result": []}})


def test_connector_endpoints(client, locations):
    from queue_predictor import app as app_module
    loc_id = locations["web"]["id"]
    manager = app_module.connector_manager
    manager._fetch = lambda url: {"depth": 123}
    body = {"kind": "json", "url": "http://feed.local/stats", "path": "depth", "interval_sec": 60}

    test = client.post(f"/api/locations/{loc_id}/connector/test", json=body)
    assert test.status_code == 200 and test.get_json()["value"] == 123

    saved = client.post(f"/api/locations/{loc_id}/connector", json=body).get_json()
    assert saved["config"]["path"] == "depth"
    assert client.get(f"/api/locations/{loc_id}").status_code in (200, 404)   # route may not exist; must not crash
    assert client.get(f"/api/locations/{loc_id}/connector").get_json()["config"]["url"] == "http://feed.local/stats"

    removed = client.delete(f"/api/locations/{loc_id}/connector").get_json()
    assert removed["config"] is None and not removed["running"]
    assert client.post(f"/api/locations/{loc_id}/connector", json={"kind": "json", "url": "ftp://x"}).status_code == 400

    manager._fetch = lambda url: (_ for _ in ()).throw(OSError("feed down"))
    failed = client.post(f"/api/locations/{loc_id}/connector/test", json=body)
    assert failed.status_code == 502 and "feed down" in failed.get_json()["error"]


def test_ingest_api(client, locations):
    loc_id = locations["calls"]["id"]
    client.post(f"/api/demo/{loc_id}/reset")
    ok = client.post(f"/api/ingest/{loc_id}", json={"count": 12})
    assert ok.status_code == 200 and ok.get_json()["stored"] == 1
    assert ok.get_json()["prediction"]["current_queue"] == 12
    many = client.post(f"/api/ingest/{loc_id}", json={"readings": [
        {"ts": "2026-01-05T10:00:00", "count": 5}, {"ts": "2026-01-05T10:01:00", "count": 7}]})
    assert many.get_json()["stored"] == 2
    for bad in ({}, {"count": -3}, {"count": "lots"}, {"readings": [{"ts": "not a date", "count": 1}]}):
        assert client.post(f"/api/ingest/{loc_id}", json=bad).status_code == 400
    assert client.post("/api/ingest/9999", json={"count": 1}).status_code == 404


# ---------- alerts ----------
@pytest.fixture
def fake_senders(monkeypatch):
    sent = []
    for channel in alerts.CHANNELS:
        monkeypatch.setitem(alerts.SENDERS, channel, lambda cfg, text, channel=channel: sent.append((channel, text)))
    alerts.reset()
    yield sent
    alerts.SETTINGS_PATH.unlink(missing_ok=True)
    alerts.reset()


def _prediction(status="IMMINENT", risk="HIGH"):
    return {"risk": risk, "current_queue": 30, "current_wait_min": 9.0, "counters": 2,
            "recommendation": "Add 2 more counters now.",
            "radar": {"status": status, "collapse_probability": 0.8, "time_to_collapse": {"median": 5, "p10": 3, "p90": 8}}}


LOCATION = {"id": 4242, "name": "Test Canteen"}


def test_secrets_are_masked_and_kept(fake_senders):
    alerts.save_settings({"telegram": {"enabled": True, "bot_token": "123:SECRET", "chat_id": "42"}})
    shown = alerts.public_settings()
    assert shown["telegram"]["bot_token"] == alerts.MASK and shown["telegram"]["chat_id"] == "42"
    alerts.save_settings({"telegram": {"bot_token": alerts.MASK, "chat_id": "43"}})
    assert alerts.load_settings()["telegram"]["bot_token"] == "123:SECRET"
    assert alerts.load_settings()["telegram"]["chat_id"] == "43"


def test_alert_goes_to_enabled_channels_only(fake_senders):
    alerts.save_settings({"telegram": {"enabled": True}, "webhook": {"enabled": True, "url": "http://hook"}})
    assert alerts.maybe_alert(LOCATION, QUEUE, _prediction(), background=False) == "IMMINENT"
    assert sorted(c for c, _ in fake_senders) == ["telegram", "webhook"]
    assert "COLLAPSE IMMINENT" in fake_senders[0][1] and "about 5 min" in fake_senders[0][1]


def test_alert_cooldown_and_escalation(fake_senders):
    alerts.save_settings({"webhook": {"enabled": True, "url": "http://hook"}, "cooldown_min": 60})
    assert alerts.maybe_alert(LOCATION, QUEUE, _prediction("IMMINENT"), background=False) == "IMMINENT"
    assert alerts.maybe_alert(LOCATION, QUEUE, _prediction("IMMINENT"), background=False) is None   # repeat: silent
    assert alerts.maybe_alert(LOCATION, QUEUE, _prediction("COLLAPSED"), background=False) == "COLLAPSED"   # worse: alert
    assert alerts.maybe_alert(LOCATION, QUEUE, _prediction("SAFE", "LOW"), background=False) is None       # recovered
    assert alerts.maybe_alert(LOCATION, QUEUE, _prediction("IMMINENT"), background=False) == "IMMINENT"    # again
    assert len(fake_senders) == 3


def test_a_failing_channel_does_not_stop_the_others(fake_senders, monkeypatch):
    def broken(cfg, text):
        raise OSError("network down")

    monkeypatch.setitem(alerts.SENDERS, "telegram", broken)
    alerts.save_settings({"telegram": {"enabled": True}, "webhook": {"enabled": True, "url": "http://hook"}})
    results, sent = alerts.deliver("hello")
    assert sent == ["webhook"] and "network down" in results["telegram"]


def test_alerts_api_and_dashboard_log(client, locations, fake_senders):
    assert client.post("/api/alerts/test").status_code == 400                       # nothing enabled
    client.post("/api/alerts/settings", json={"webhook": {"enabled": True, "url": "http://hook"}})
    assert client.post("/api/alerts/test").get_json()["results"] == {"webhook": "sent"}
    assert client.get("/api/alerts/settings").get_json()["webhook"]["url"] == "http://hook"

    loc_id = locations["calls"]["id"]
    client.post(f"/api/demo/{loc_id}/reset")
    client.post(f"/api/ingest/{loc_id}", json={"count": 500})                       # far beyond collapse
    import time
    for _ in range(20):                                                             # alerts are sent in a thread
        notifications = client.get(f"/api/status/{loc_id}").get_json()["notifications"]
        if notifications:
            break
        time.sleep(0.1)
    assert notifications and notifications[0]["level"] in {"COLLAPSED", "IMMINENT", "HIGH"}
    assert "dashboard" in notifications[0]["channels"] and "webhook" in notifications[0]["channels"]


# ---------- report ----------
def test_report_pdf(client, locations):
    from pypdf import PdfReader

    loc_id = locations["demo"]["id"]
    client.post(f"/api/demo/{loc_id}/reset")
    empty = client.get(f"/api/report/{loc_id}.pdf")
    assert empty.status_code == 200 and empty.data.startswith(b"%PDF")
    assert "No data yet" in PdfReader(io.BytesIO(empty.data)).pages[0].extract_text()

    for _ in range(6):
        client.post(f"/api/demo/{loc_id}/tick")
    response = client.get(f"/api/report/{loc_id}.pdf")
    assert response.mimetype == "application/pdf"
    text = PdfReader(io.BytesIO(response.data)).pages[0].extract_text()
    for expected in ("Queue Collapse Report", "Staffing advice", "Chance of collapse"):
        assert expected in text
    assert client.get("/api/report/9999.pdf").status_code == 404
