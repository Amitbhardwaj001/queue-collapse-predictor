import io

import cv2
import pytest
import numpy as np


def _image_bytes():
    """A plain synthetic frame; detection must run without errors even when nobody is found."""
    frame = np.full((360, 480, 3), 200, np.uint8)
    cv2.rectangle(frame, (200, 80), (260, 300), (40, 40, 40), -1)
    return io.BytesIO(cv2.imencode(".jpg", frame)[1].tobytes())


def test_pages_load(client):
    for url in ("/", "/poster", "/static/app.js", "/static/vendor/chart.umd.min.js"):
        assert client.get(url).status_code == 200, url


def test_image_upload_counts_and_predicts(client, locations):
    loc = locations["main"]
    r = client.post(f"/api/upload/{loc['id']}", data={"file": (_image_bytes(), "queue.jpg")},
                    content_type="multipart/form-data")
    body = r.get_json()
    assert r.status_code == 200, body
    assert body["count"] >= 0
    assert body["prediction"]["risk"] in {"LOW", "MEDIUM", "HIGH"}
    assert client.get(body["image"]).status_code == 200

    preview = client.post(f"/api/locations/{loc['id']}/preview", json={"frame": body["frame"]})
    assert preview.status_code == 200


def test_rejects_unsupported_files(client, locations):
    r = client.post(f"/api/upload/{locations['main']['id']}", data={"file": (io.BytesIO(b"x"), "notes.txt")},
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_queue_area_validation(client, locations):
    loc_id = locations["main"]["id"]
    ok = client.post(f"/api/locations/{loc_id}/roi", json={"roi": [0.1, 0.2, 0.6, 0.9]})
    assert ok.get_json()["roi"] == [0.1, 0.2, 0.6, 0.9]
    assert client.post(f"/api/locations/{loc_id}/roi", json={"roi": [0.1, 0.1, 0.11, 0.5]}).status_code == 400
    assert client.post(f"/api/locations/{loc_id}/roi", json={"roi": None}).get_json()["roi"] is None
    assert client.post(f"/api/locations/{loc_id}/preview", json={"frame": "../app.py"}).status_code == 404


def test_settings_validation(client, locations):
    loc_id = locations["main"]["id"]
    assert client.post(f"/api/locations/{loc_id}/settings", json={"counters": 0, "service_time_sec": 45}).status_code == 400
    assert client.post(f"/api/locations/{loc_id}/settings", json={"counters": 3, "service_time_sec": 40}).status_code == 200


def test_demo_feed_builds_history(client, locations):
    loc_id = locations["demo"]["id"]
    client.post(f"/api/demo/{loc_id}/reset")
    for _ in range(5):
        tick = client.post(f"/api/demo/{loc_id}/tick")
        assert tick.status_code == 200
    status = client.get(f"/api/status/{loc_id}").get_json()
    assert len(status["history"]) == 20          # 15 warm-up minutes + 5 ticks
    assert status["prediction"]["horizon_min"] == 10
    assert client.post(f"/api/demo/{locations['main']['id']}/tick").status_code == 400


def test_domains_endpoint(client):
    keys = {d["key"] for d in client.get("/api/domains").get_json()}
    assert keys == {"queue", "hospital", "web", "calls"}


def test_status_contains_collapse_radar(client, locations):
    loc_id = locations["demo"]["id"]
    client.post(f"/api/demo/{loc_id}/reset")
    for _ in range(3):
        client.post(f"/api/demo/{loc_id}/tick")
    body = client.get(f"/api/status/{loc_id}").get_json()
    radar = body["prediction"]["radar"]
    assert radar["status"] in {"SAFE", "AT_RISK", "IMMINENT", "COLLAPSED"}
    assert radar["what_if"] and radar["fan"]
    assert body["domain"]["key"] == "queue"


@pytest.mark.parametrize("key", ["hospital", "web", "calls"])
def test_other_domains_run_a_live_feed(client, locations, key):
    loc = locations[key]
    client.post(f"/api/demo/{loc['id']}/reset")
    for _ in range(4):
        assert client.post(f"/api/demo/{loc['id']}/tick").status_code == 200
    body = client.get(f"/api/status/{loc['id']}").get_json()
    assert body["prediction"]["domain"] == key
    assert len(body["prediction"]["forecast"]) == 3
    assert client.post(f"/api/upload/{loc['id']}", data={"file": (_image_bytes(), "q.jpg")},
                       content_type="multipart/form-data").status_code == 400


def test_surge_needs_a_running_feed(client, locations):
    loc_id = locations["web"]["id"]
    client.post(f"/api/demo/{loc_id}/reset")
    assert client.post(f"/api/demo/{loc_id}/surge").status_code == 400
    client.post(f"/api/demo/{loc_id}/tick")
    assert client.post(f"/api/demo/{loc_id}/surge").status_code == 200


def test_settings_limits_follow_the_domain(client, locations):
    loc_id = locations["web"]["id"]
    assert client.post(f"/api/locations/{loc_id}/settings", json={"counters": 40, "service_time_sec": 1}).status_code == 200
    assert client.post(f"/api/locations/{loc_id}/settings", json={"counters": 8, "service_time_sec": 0.01}).status_code == 400
