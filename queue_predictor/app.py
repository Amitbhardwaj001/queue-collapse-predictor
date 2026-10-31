"""Flask backend for the AI Queue Collapse Predictor dashboard.

Run from the project root:  python run.py   then open http://127.0.0.1:5000
"""
import json
import threading
from datetime import datetime, timedelta

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from . import database as db
from .config import HISTORY_WINDOW_MIN, RESULT_DIR, UPLOAD_DIR, metrics_path, model_path
from .domains import DOMAINS
from .predictor import QueuePredictor
from .simulate import QueueSimulator

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_TYPES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

db.init_db()
predictors = {}
for _domain in DOMAINS.values():
    if not model_path(_domain.key).exists():
        from .train import train
        train(_domain)
    predictors[_domain.key] = QueuePredictor(_domain)


def load_metrics():
    return {key: json.loads(metrics_path(key).read_text())
            for key in DOMAINS if metrics_path(key).exists()}

_detector = None
_detector_lock = threading.Lock()
_demo = {}  # location_id -> {"sim": QueueSimulator, "clock": datetime}


def get_detector():
    """YOLO takes a few seconds to load, so only do it when first needed."""
    global _detector
    with _detector_lock:
        if _detector is None:
            from .detector import PersonDetector
            _detector = PersonDetector()
        return _detector


def predict_and_store(location):
    latest = db.latest_timestamp(location["id"])
    if latest is None:
        return None
    domain = DOMAINS[location["domain"]]
    observations = db.observations_since(location["id"], latest - timedelta(minutes=domain.history_min))
    result = predictors[location["domain"]].predict(
        observations, location["counters"], location["service_time_sec"], what_if=False)
    db.add_prediction(location["id"], result)
    return result


def location_or_404(location_id):
    location = db.get_location(location_id)
    if location is None:
        return None, (jsonify(error="Unknown location"), 404)
    return location, None


@app.get("/")
def index():
    return render_template("index.html", domains=list(DOMAINS.values()), metrics=load_metrics())


@app.get("/api/domains")
def domains():
    return jsonify([d.public() for d in DOMAINS.values()])


@app.get("/api/locations")
def locations():
    return jsonify(db.list_locations())


@app.post("/api/locations/<int:location_id>/settings")
def update_settings(location_id):
    location, error = location_or_404(location_id)
    if error:
        return error
    data = request.get_json(force=True)
    try:
        counters = int(data["counters"])
        service_time = float(data["service_time_sec"])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="counters and service_time_sec must be numbers"), 400
    domain = DOMAINS[location["domain"]]
    low, high = domain.service_sec_limits
    if not (1 <= counters <= domain.max_servers and low <= service_time <= high):
        return jsonify(error=f"{domain.server.capitalize()}s must be 1-{domain.max_servers} and "
                             f"service time {low:g}-{high:g} seconds"), 400
    db.update_location(location_id, counters, service_time)
    return jsonify(db.get_location(location_id))


@app.get("/api/status/<int:location_id>")
def status(location_id):
    location, error = location_or_404(location_id)
    if error:
        return error
    latest = db.latest_timestamp(location_id)
    if latest is None:
        return jsonify(location=location, domain=DOMAINS[location["domain"]].public(),
                       prediction=None, history=[], past_predictions=[])

    since = latest - timedelta(minutes=max(HISTORY_WINDOW_MIN, DOMAINS[location["domain"]].history_min))
    history = db.observations_since(location_id, since)
    domain = DOMAINS[location["domain"]]
    observations = history[history["ts"] >= latest - timedelta(minutes=domain.history_min)]
    prediction = predictors[location["domain"]].predict(
        observations, location["counters"], location["service_time_sec"])
    past = db.predictions_since(location_id, since - timedelta(minutes=15))

    def minutes_from_now(ts):
        return round((ts - latest).total_seconds() / 60, 2)

    return jsonify(
        location=location,
        domain=DOMAINS[location["domain"]].public(),
        prediction=prediction,
        history=[{"t": minutes_from_now(r.ts), "count": int(r.count), "source": r.source}
                 for r in history.itertuples()],
        past_predictions=[{"t": minutes_from_now(r.ts + timedelta(minutes=int(r.horizon_min))),
                           "queue": int(r.predicted_queue), "risk": r.risk}
                          for r in past.itertuples()],
        alerts=[{"time": r.ts.strftime("%H:%M"), "queue": int(r.current_queue),
                 "predicted": int(r.predicted_queue), "probability": float(r.critical_probability),
                 "risk": r.risk}
                for r in past[past["risk"] != "LOW"].tail(8).iloc[::-1].itertuples()],
    )


@app.post("/api/upload/<int:location_id>")
def upload(location_id):
    location, error = location_or_404(location_id)
    if error:
        return error
    if not DOMAINS[location["domain"]].camera:
        return jsonify(error="Camera footage is only used for physical queues"), 400
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify(error="No file uploaded"), 400

    name = secure_filename(file.filename) or "upload.jpg"
    suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if suffix not in IMAGE_TYPES | VIDEO_TYPES:
        return jsonify(error=f"Unsupported file type '{suffix}'. Use an image or a video."), 400
    path = UPLOAD_DIR / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{name}"
    file.save(path)

    detector = get_detector()
    now = datetime.now().replace(microsecond=0)
    roi = location["roi"]
    try:
        if suffix in IMAGE_TYPES:
            count, image, frame = detector.process_image(path, roi)
            source = request.form.get("source", "image")
            db.add_observations(location_id, [(now, count)], source)
            samples = [{"offset_sec": 0, "count": count}]
        else:
            frames, duration, image, frame = detector.process_video(path, roi)
            start = now - timedelta(seconds=duration)
            db.add_observations(location_id, [(start + timedelta(seconds=o), c) for o, c in frames], "video")
            count = frames[-1][1]
            samples = [{"offset_sec": round(o, 1), "count": c} for o, c in frames]
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    finally:
        if request.form.get("source") == "camera":
            path.unlink(missing_ok=True)

    return jsonify(count=count, detector=detector.backend, image=f"/results/{image}",
                   frame=frame, samples=samples, prediction=predict_and_store(location))


@app.get("/results/<path:name>")
def result_image(name):
    return send_from_directory(RESULT_DIR, name)


@app.post("/api/locations/<int:location_id>/roi")
def update_roi(location_id):
    """Save the queue area. Body: {"roi": [x1, y1, x2, y2] as 0-1 fractions} or {"roi": null}."""
    location, error = location_or_404(location_id)
    if error:
        return error
    roi = request.get_json(force=True).get("roi")
    if roi is not None:
        try:
            roi = [round(min(1.0, max(0.0, float(v))), 4) for v in roi]
        except (TypeError, ValueError):
            return jsonify(error="roi must be four numbers"), 400
        if len(roi) != 4 or roi[2] - roi[0] < 0.02 or roi[3] - roi[1] < 0.02:
            return jsonify(error="The queue area is too small"), 400
    db.update_roi(location_id, roi)
    return jsonify(db.get_location(location_id))


@app.post("/api/locations/<int:location_id>/preview")
def preview_roi(location_id):
    """Recount a previously analysed frame with the current queue area (nothing is stored)."""
    location, error = location_or_404(location_id)
    if error:
        return error
    name = secure_filename(request.get_json(force=True).get("frame", ""))
    path = RESULT_DIR / name
    if not name.startswith("frame_") or not path.exists():
        return jsonify(error="Frame not found - analyse an image first"), 404
    detector = get_detector()
    count, image, _ = detector.process_image(path, location["roi"])
    return jsonify(count=count, detector=detector.backend, image=f"/results/{image}")


@app.get("/poster")
def poster():
    return render_template("poster.html", domains=list(DOMAINS.values()), metrics=load_metrics())


@app.post("/api/demo/<int:location_id>/tick")
def demo_tick(location_id):
    """Advance the simulated live feed by one minute (only for demo locations)."""
    location, error = location_or_404(location_id)
    if error:
        return error
    if not location["is_demo"]:
        return jsonify(error="Live simulation is only available for the demo location"), 400

    state = _demo.get(location_id)
    if state is None:
        db.clear_location_data(location_id)
        # Start just before the rush so the build-up is visible within a minute.
        domain = DOMAINS[location["domain"]]
        clock = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=domain.demo_start)
        sim = QueueSimulator(np.random.default_rng(), weekday=min(clock.weekday(), 4),
                             queue=domain.demo_queue, domain=domain)
        warmup = []
        for _ in range(max(15, max(domain.lags))):  # give the model some history to start with
            sim.counters = location["counters"]
            sim.service_time_sec = location["service_time_sec"]
            warmup.append((clock, sim.step(clock.hour * 60 + clock.minute)))
            clock += timedelta(minutes=1)
        db.add_observations(location_id, warmup, "simulation")
        state = _demo[location_id] = {"sim": sim, "clock": clock}

    sim, clock = state["sim"], state["clock"]
    sim.counters = location["counters"]
    sim.service_time_sec = location["service_time_sec"]
    count = sim.step(clock.hour * 60 + clock.minute)
    db.add_observations(location_id, [(clock, count)], "simulation")
    state["clock"] = clock + timedelta(minutes=1)
    return jsonify(count=count, time=clock.strftime("%H:%M"), prediction=predict_and_store(location))


@app.post("/api/demo/<int:location_id>/surge")
def demo_surge(location_id):
    """Inject a sudden demand surge (class break, ambulance wave, flash sale, outage) into the live feed."""
    location, error = location_or_404(location_id)
    if error:
        return error
    state = _demo.get(location_id)
    if not location["is_demo"] or state is None:
        return jsonify(error="Start the simulated live feed first"), 400
    burst = DOMAINS[location["domain"]].demand.burst_rate
    state["sim"]._burst_left = 4
    state["sim"]._burst_rate = float(burst[1])
    return jsonify(ok=True)


@app.post("/api/demo/<int:location_id>/reset")
def demo_reset(location_id):
    location, error = location_or_404(location_id)
    if error:
        return error
    if not location["is_demo"]:
        return jsonify(error="Only the demo location can be reset"), 400
    _demo.pop(location_id, None)
    db.clear_location_data(location_id)
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(debug=False, threaded=True)
