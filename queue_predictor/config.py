"""Paths, thresholds and settings for the whole project."""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
# Runtime data and trained models can be relocated (the tests use temporary folders).
DATA_DIR = Path(os.environ.get("QP_DATA_DIR", PROJECT_DIR / "data"))
MODEL_DIR = Path(os.environ.get("QP_MODEL_DIR", PROJECT_DIR / "models"))
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
DB_PATH = DATA_DIR / "queue.db"
MODEL_PATH = MODEL_DIR / "queue_models.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Person detection
YOLO_WEIGHTS = MODEL_DIR / "yolov8n.pt"
PERSON_CONFIDENCE = 0.35
VIDEO_SAMPLE_SECONDS = 10      # analyse one frame every N seconds of video
VIDEO_MAX_SAMPLES = 360

# Forecasting
HORIZONS = [5, 10, 15]         # minutes ahead
MAIN_HORIZON = 10
LAGS = [1, 2, 3, 5, 10]        # minutes of history used as features
HISTORY_WINDOW_MIN = 60        # minutes of history shown on the dashboard

# Service defaults (editable per location from the dashboard)
DEFAULT_COUNTERS = 2
DEFAULT_SERVICE_TIME_SEC = 45  # average seconds one counter needs per person

# Congestion thresholds, expressed as waiting time in minutes
MEDIUM_WAIT_MIN = 6
HIGH_WAIT_MIN = 12
TARGET_WAIT_MIN = 8            # used to recommend how many counters to open
MEDIUM_PROBABILITY = 0.30
HIGH_PROBABILITY = 0.60

# Synthetic training data
SIM_DAYS = 150
SIM_DAY_START = 7 * 60         # 07:00
SIM_DAY_END = 20 * 60          # 20:00

for folder in (DATA_DIR, MODEL_DIR, UPLOAD_DIR, RESULT_DIR):
    folder.mkdir(parents=True, exist_ok=True)
