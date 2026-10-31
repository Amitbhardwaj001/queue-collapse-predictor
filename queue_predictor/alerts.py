"""Collapse alerts: Telegram, email, SMS (Twilio) and webhook, plus the dashboard log.

Settings live in data/alerts.json on the machine running the dashboard. Secrets
are never sent back to the browser; they are shown as ******** and kept when the
form is saved unchanged.

An alert is raised when the Collapse Radar says COLLAPSE IMMINENT / COLLAPSED
or the risk is HIGH. To avoid spamming, a location alerts again only when the
situation gets worse or after the cool-down.
"""
import json
import smtplib
import threading
import time
import urllib.parse
import urllib.request
from base64 import b64encode
from datetime import datetime
from email.message import EmailMessage

from . import database as db
from .config import DATA_DIR

SETTINGS_PATH = DATA_DIR / "alerts.json"
MASK = "********"
SECRET_FIELDS = {"bot_token", "password", "auth_token"}
SEVERITY = {"HIGH": 1, "IMMINENT": 2, "COLLAPSED": 3}

DEFAULTS = {
    "cooldown_min": 5,
    "telegram": {"enabled": False, "bot_token": "", "chat_id": ""},
    "email": {"enabled": False, "host": "", "port": 587, "username": "", "password": "",
              "sender": "", "recipients": ""},
    "sms": {"enabled": False, "account_sid": "", "auth_token": "", "from_number": "", "to_number": ""},
    "webhook": {"enabled": False, "url": ""},
}
CHANNELS = ("telegram", "email", "sms", "webhook")

_lock = threading.Lock()
_last = {}        # location_id -> (severity, time.monotonic())


# ---------- settings ----------
def load_settings():
    settings = json.loads(json.dumps(DEFAULTS))
    if SETTINGS_PATH.exists():
        saved = json.loads(SETTINGS_PATH.read_text())
        settings["cooldown_min"] = saved.get("cooldown_min", settings["cooldown_min"])
        for channel in CHANNELS:
            settings[channel].update(saved.get(channel, {}))
    return settings


def public_settings():
    """Settings for the browser: secrets replaced by a mask."""
    settings = load_settings()
    for channel in CHANNELS:
        for field in SECRET_FIELDS & settings[channel].keys():
            if settings[channel][field]:
                settings[channel][field] = MASK
    return settings


def save_settings(new):
    """Merge settings from the browser; a masked secret keeps the stored value."""
    current = load_settings()
    if "cooldown_min" in new:
        current["cooldown_min"] = max(0, float(new["cooldown_min"]))
    for channel in CHANNELS:
        for field, value in (new.get(channel) or {}).items():
            if field not in current[channel]:
                continue
            if field in SECRET_FIELDS and value == MASK:
                continue
            current[channel][field] = bool(value) if field == "enabled" else (
                int(value) if field == "port" else str(value).strip())
    SETTINGS_PATH.write_text(json.dumps(current, indent=2))
    return public_settings()


# ---------- senders (replaced by fakes in the tests) ----------
def _post(url, data=None, headers=None, json_body=None, timeout=10):
    body = json.dumps(json_body).encode() if json_body is not None else (
        urllib.parse.urlencode(data).encode() if data is not None else None)
    request = urllib.request.Request(url, data=body, headers=headers or {}, method="POST")
    if json_body is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status


def send_telegram(cfg, text):
    _post(f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage",
          data={"chat_id": cfg["chat_id"], "text": text})


def send_email(cfg, text):
    message = EmailMessage()
    message["Subject"] = text.splitlines()[0][:120]
    message["From"] = cfg["sender"] or cfg["username"]
    message["To"] = cfg["recipients"]
    message.set_content(text)
    with smtplib.SMTP(cfg["host"], int(cfg["port"]), timeout=15) as server:
        server.starttls()
        if cfg["username"]:
            server.login(cfg["username"], cfg["password"])
        server.send_message(message)


def send_sms(cfg, text):
    token = b64encode(f"{cfg['account_sid']}:{cfg['auth_token']}".encode()).decode()
    _post(f"https://api.twilio.com/2010-04-01/Accounts/{cfg['account_sid']}/Messages.json",
          data={"From": cfg["from_number"], "To": cfg["to_number"], "Body": text[:600]},
          headers={"Authorization": f"Basic {token}"})


def send_webhook(cfg, text):
    _post(cfg["url"], json_body={"text": text})


SENDERS = {"telegram": send_telegram, "email": send_email, "sms": send_sms, "webhook": send_webhook}


# ---------- dispatch ----------
def deliver(text, settings=None):
    """Send to every enabled channel. Returns ({channel: 'sent' | error}, [sent channels])."""
    settings = settings or load_settings()
    results, sent = {}, []
    for channel in CHANNELS:
        if not settings[channel]["enabled"]:
            continue
        try:
            SENDERS[channel](settings[channel], text)
            results[channel] = "sent"
            sent.append(channel)
        except Exception as exc:
            results[channel] = f"{type(exc).__name__}: {exc}"
    return results, sent


def level_of(prediction):
    """The alert level of a prediction, or None when everything is fine."""
    status = prediction["radar"]["status"]
    if status in ("IMMINENT", "COLLAPSED"):
        return status
    return "HIGH" if prediction["risk"] == "HIGH" else None


def message_for(location, domain, prediction, level):
    r = prediction["radar"]
    ttc = r["time_to_collapse"]
    head = {"COLLAPSED": "COLLAPSED", "IMMINENT": "COLLAPSE IMMINENT", "HIGH": "HIGH RISK"}[level]
    when = ""
    if level == "IMMINENT" and ttc:
        when = f" Collapse expected in about {ttc['median']} min ({round(r['collapse_probability'] * 100)}% likely)."
    return (f"{head}: {location['name']}\n"
            f"{prediction['current_queue']} {domain.item} waiting, wait about {prediction['current_wait_min']} min "
            f"with {prediction['counters']} {domain.server}(s).{when}\n{prediction['recommendation']}")


def maybe_alert(location, domain, prediction, now=None, background=True):
    """Raise an alert for this prediction if it is new or worse. Returns the level alerted or None."""
    level = level_of(prediction)
    settings = load_settings()
    with _lock:
        previous = _last.get(location["id"])
        if level is None:
            _last.pop(location["id"], None)
            return None
        severity = SEVERITY[level]
        fresh = previous is None or severity > previous[0] or \
            time.monotonic() - previous[1] >= settings["cooldown_min"] * 60
        if not fresh:
            return None
        _last[location["id"]] = (severity, time.monotonic())

    text = message_for(location, domain, prediction, level)
    when = now or datetime.now().replace(microsecond=0)

    def work():
        results, sent = deliver(text, settings)
        errors = [f"{c}: {r}" for c, r in results.items() if r != "sent"]
        db.add_alert(location["id"], when, level, text + (f"\n(failed: {'; '.join(errors)})" if errors else ""),
                     ["dashboard", *sent])

    if background:
        threading.Thread(target=work, daemon=True).start()
    else:
        work()
    return level


def reset(location_id=None):
    with _lock:
        if location_id is None:
            _last.clear()
        else:
            _last.pop(location_id, None)
