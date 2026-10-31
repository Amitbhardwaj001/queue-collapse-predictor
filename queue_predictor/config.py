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


def model_path(domain_key):
    return MODEL_DIR / f"{domain_key}_models.joblib"


def metrics_path(domain_key):
    return MODEL_DIR / f"{domain_key}_metrics.json"


# Person detection
YOLO_WEIGHTS = MODEL_DIR / "yolov8n.pt"
PERSON_CONFIDENCE = 0.35
VIDEO_SAMPLE_SECONDS = 10      # analyse one frame every N seconds of video
VIDEO_MAX_SAMPLES = 360
TRACK_FPS = 3                  # frames per second analysed when tracking people
TRACK_MAX_FRAMES = 2400        # about 13 minutes of video at TRACK_FPS

# Forecasting (horizons and thresholds are defined per domain in domains.py)
LAGS = [1, 2, 3, 5, 10]        # minutes of history used as features
HISTORY_WINDOW_MIN = 60        # minutes of history shown on the dashboard
MEDIUM_PROBABILITY = 0.30      # probability of critical congestion for MEDIUM / HIGH risk
HIGH_PROBABILITY = 0.60

# Collapse Radar (Monte Carlo what-if simulation)
RADAR_PATHS = 500              # simulated futures per staffing option
RADAR_DEMAND_SPREAD = 0.25     # uncertainty of the future arrival rate (log-normal sigma)
RADAR_SAFE_PROBABILITY = 0.10  # a staffing option is "safe" if collapse probability is below this
RADAR_ALARM_PROBABILITY = 0.50 # early warning is raised at this collapse probability

for folder in (DATA_DIR, MODEL_DIR, UPLOAD_DIR, RESULT_DIR):
    folder.mkdir(parents=True, exist_ok=True)
