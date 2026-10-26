"""Point the app at temporary data/model folders and train a small model once per test run."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="queue-predictor-tests-"))
os.environ["QP_DATA_DIR"] = str(_TMP / "data")
os.environ["QP_MODEL_DIR"] = str(_TMP / "models")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the real YOLO weights if they have been downloaded, so detection tests run offline.
_weights = Path(__file__).resolve().parent.parent / "models" / "yolov8n.pt"
if _weights.exists():
    (_TMP / "models").mkdir(parents=True, exist_ok=True)
    shutil.copy(_weights, _TMP / "models" / "yolov8n.pt")

from queue_predictor.train import train  # noqa: E402  (must come after the env vars)

train(days=20, verbose=False)


@pytest.fixture(scope="session")
def client():
    from queue_predictor.app import app
    return app.test_client()


@pytest.fixture(scope="session")
def locations(client):
    locs = client.get("/api/locations").get_json()
    return {"main": next(l for l in locs if not l["is_demo"]), "demo": next(l for l in locs if l["is_demo"])}


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TMP, ignore_errors=True)
