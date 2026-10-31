"""PDF report: what happened at a location and what staffing to use next."""
from datetime import datetime, timedelta

from fpdf import FPDF

from . import database as db
from .config import HISTORY_WINDOW_MIN

REPORT_HOURS = 24
EPISODE_GAP_MIN = 5
INK, MUTED, ACCENT = (28, 35, 51), (107, 115, 133), (37, 99, 235)
RISK_COLOURS = {"LOW": (22, 163, 74), "MEDIUM": (217, 119, 6), "HIGH": (220, 38, 38)}


def _text(value):
    """The built-in PDF fonts only know Latin-1."""
    return str(value).replace("–", "-").replace("—", "-").encode("latin-1", "replace").decode("latin-1")


class _Pdf(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 6, _text(f"AI Queue Collapse Predictor - page {self.page_no()}"), align="C")


def _heading(pdf, title):
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 8, _text(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*INK)


def _paragraph(pdf, text, size=10, colour=INK):
    pdf.set_font("Helvetica", "", size)
    pdf.set_text_color(*colour)
    pdf.multi_cell(0, 5.5, _text(text), new_x="LMARGIN", new_y="NEXT")


def _table(pdf, header, rows, widths, risk_column=None):
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(235, 239, 247)
    pdf.set_text_color(*INK)
    for title, width in zip(header, widths):
        pdf.cell(width, 7, _text(title), border=0, fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for row in rows:
        for i, (value, width) in enumerate(zip(row, widths)):
            if i == risk_column and value in RISK_COLOURS:
                pdf.set_text_color(*RISK_COLOURS[value])
                pdf.set_font("Helvetica", "B", 9)
            pdf.cell(width, 6.5, _text(value))
            pdf.set_text_color(*INK)
            pdf.set_font("Helvetica", "", 9)
        pdf.ln()


def risk_episodes(predictions):
    """Group consecutive non-LOW predictions into episodes."""
    episodes, current = [], None
    for row in predictions.itertuples():
        if row.risk == "LOW":
            current = None
            continue
        if current and (row.ts - current["end"]) <= timedelta(minutes=EPISODE_GAP_MIN):
            current["end"] = row.ts
            current["peak_queue"] = max(current["peak_queue"], int(row.current_queue))
            current["peak_predicted"] = max(current["peak_predicted"], int(row.predicted_queue))
            if row.risk == "HIGH":
                current["risk"] = "HIGH"
        else:
            current = {"start": row.ts, "end": row.ts, "peak_queue": int(row.current_queue),
                       "peak_predicted": int(row.predicted_queue), "risk": row.risk}
            episodes.append(current)
    return episodes


def build_report(location, domain, predictor, metrics=None, flow=None):
    """Return the report as PDF bytes."""
    latest = db.latest_timestamp(location["id"])
    pdf = _Pdf(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*INK)
    pdf.cell(0, 10, "Queue Collapse Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 6, _text(f"{location['name']}  |  {domain.name}  |  generated {datetime.now():%Y-%m-%d %H:%M}"),
             new_x="LMARGIN", new_y="NEXT")

    if latest is None:
        _heading(pdf, "No data yet")
        _paragraph(pdf, "No readings have been recorded for this location. Start a live feed, upload footage, "
                        "connect a live metric or push readings to the ingest API, then export the report again.")
        return bytes(pdf.output())

    since = latest - timedelta(hours=REPORT_HOURS)
    observations = db.observations_since(location["id"], since)
    predictions = db.predictions_since(location["id"], since)
    notifications = db.alerts_since(location["id"], since, limit=200)
    capacity = location["counters"] * 60.0 / location["service_time_sec"]
    peak = int(observations["count"].max())
    episodes = risk_episodes(predictions)
    high_minutes = int((predictions["risk"] == "HIGH").sum())

    _heading(pdf, f"Summary: the last {REPORT_HOURS} hours of data (to {latest:%Y-%m-%d %H:%M})")
    _table(pdf, ["Readings", f"Peak {domain.item}", "Peak wait", "Minutes at HIGH risk", "Alerts sent"],
           [[len(observations), peak, f"{peak / capacity:.1f} min", high_minutes, len(notifications)]],
           [32, 40, 34, 46, 30])
    _paragraph(pdf, f"Setup: {location['counters']} {domain.server}(s), {location['service_time_sec']:g} s per "
                    f"{domain.item_one} per {domain.server}. Critical wait: {domain.high_wait:g} min.",
               size=9, colour=MUTED)

    _heading(pdf, "Collapse risk episodes")
    if episodes:
        rows = [[e["start"].strftime("%H:%M"), e["end"].strftime("%H:%M"), e["peak_queue"], e["peak_predicted"], e["risk"]]
                for e in episodes[-15:]]
        _table(pdf, ["From", "To", f"Peak {domain.item}", f"Peak forecast", "Risk"], rows, [28, 28, 44, 44, 30],
               risk_column=4)
    else:
        _paragraph(pdf, "No MEDIUM or HIGH risk periods were recorded.")

    _heading(pdf, "Alerts sent")
    if notifications:
        _table(pdf, ["Time", "Level", "Sent to"],
               [[n["ts"][11:16], n["level"], n["channels"].replace(",", ", ")] for n in notifications[:12]],
               [28, 40, 100])
    else:
        _paragraph(pdf, "No alerts were raised.")

    _heading(pdf, "Staffing advice for the current situation")
    window = observations[observations["ts"] >= latest - timedelta(minutes=max(HISTORY_WINDOW_MIN, domain.history_min))]
    prediction = predictor.predict(window, location["counters"], location["service_time_sec"])
    radar = prediction["radar"]
    ttc = radar["time_to_collapse"]
    when = "collapsed already" if radar["status"] == "COLLAPSED" else (
        f"collapse expected in about {ttc['median']} min" if ttc else f"no collapse expected in {radar['horizon_min']} min")
    _paragraph(pdf, f"Now: {prediction['current_queue']} {domain.item} waiting, {prediction['predicted_queue']} expected in "
                    f"{prediction['horizon_min']} min. Status: {radar['status'].replace('_', ' ')} - {when}.")
    _paragraph(pdf, prediction["recommendation"], size=11)
    pdf.ln(1)
    _table(pdf, [f"{domain.server.capitalize()}s", f"Chance of collapse in {radar['horizon_min']} min",
                 "Typical worst wait", "Bad-case worst wait"],
           [[f"{r['servers']}{' (now)' if r['servers'] == radar['current_servers'] else ''}"
             f"{' (safe)' if r['servers'] == radar['safe_servers'] else ''}",
             f"{round(r['collapse_probability'] * 100)}%", f"{r['median_peak_wait']:.1f} min", f"{r['p90_peak_wait']:.1f} min"]
            for r in radar["what_if"]], [42, 60, 44, 44])

    if flow:
        _heading(pdf, "Measured flow (person tracking)")
        _table(pdf, ["Arrivals / min", "Served / min", "Avg wait", "Service time / server", "Utilisation"],
               [[flow["arrival_rate_per_min"], flow["departure_rate_per_min"],
                 f"{flow['avg_wait_min']} min" if flow["avg_wait_min"] is not None else "n/a",
                 f"{flow['suggested_service_time_sec']} s" if flow["suggested_service_time_sec"] else "n/a",
                 f"{round(flow['utilisation'] * 100)}%" if flow["utilisation"] else "n/a"]],
               [34, 34, 34, 50, 30])

    if metrics:
        _heading(pdf, "How reliable are the forecasts?")
        h = metrics["horizons"][str(domain.main_horizon)]
        policies = metrics["early_warning"]["policies"]
        system = policies.get("system", policies["radar"])
        _paragraph(pdf, f"On unseen simulated days the +{domain.main_horizon} min forecast has R2 {h['r2']} and errs by "
                        f"{h['busy_mae']} {domain.item} in busy periods (a 'no change' guess errs by {h['busy_baseline_mae']}). "
                        f"The full system warned early for {round((system['early_rate'] or 0) * 100)}% of collapses, on average "
                        f"{system['mean_lead_min']} min ahead, with {system['false_alarms_per_day']} false alarms per day; "
                        f"a reactive alarm warns 0 min ahead.")
        _paragraph(pdf, "The models are trained on simulated data. Retrain on real readings before relying on them.",
                   size=9, colour=MUTED)
    return bytes(pdf.output())
