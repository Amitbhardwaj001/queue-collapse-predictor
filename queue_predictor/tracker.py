"""Person tracking and real flow measurement from video.

``ByteTracker`` follows every detected person from frame to frame. It uses
ByteTrack's idea: confident detections are matched to existing tracks first,
then the weak detections (people half hidden behind others) get a second
chance to keep a track alive instead of being thrown away.

``FlowMeter`` turns the tracks into the numbers the forecasting engine needs:
how fast people really *arrive*, how fast they are really *served*, and how long
they really wait, so the service speed no longer has to be guessed.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

HIGH_CONFIDENCE = 0.5
MATCH_IOU = 0.2
MIN_HITS = 3                    # frames before a track is trusted (filters flickers)
MAX_LOST_SEC = 4.0              # how long a hidden person is remembered


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    box: tuple
    first_t: float
    last_t: float
    hits: int = 1
    velocity: tuple = (0.0, 0.0, 0.0, 0.0)
    history: list = field(default_factory=list)       # [(t, box)]

    def predicted(self, t):
        dt = t - self.last_t
        return tuple(b + v * dt for b, v in zip(self.box, self.velocity))

    def update(self, box, t):
        dt = max(t - self.last_t, 1e-3)
        new = tuple(0.5 * v + 0.5 * (n - o) / dt for v, n, o in zip(self.velocity, box, self.box))
        self.velocity, self.box, self.last_t = new, tuple(box), t
        self.hits += 1
        self.history.append((t, tuple(box)))


class ByteTracker:
    def __init__(self, max_lost_sec=MAX_LOST_SEC):
        self.max_lost_sec = max_lost_sec
        self.tracks = []                  # active
        self.finished = []                # lost for good
        self._next_id = 1

    def _associate(self, tracks, detections, t):
        """Hungarian matching on IoU. Returns (matches, unmatched track idx, unmatched detection idx)."""
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        cost = np.array([[1 - iou(tr.predicted(t), d[:4]) for d in detections] for tr in tracks])
        rows, cols = linear_sum_assignment(cost)
        matches = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= 1 - MATCH_IOU]
        matched_r = {r for r, _ in matches}
        matched_c = {c for _, c in matches}
        return (matches, [i for i in range(len(tracks)) if i not in matched_r],
                [j for j in range(len(detections)) if j not in matched_c])

    def update(self, detections, t):
        """detections: [(x1, y1, x2, y2, confidence)] at time t (seconds). Returns the visible tracks."""
        strong = [d for d in detections if d[4] >= HIGH_CONFIDENCE]
        weak = [d for d in detections if d[4] < HIGH_CONFIDENCE]

        matches, left_tracks, new_idx = self._associate(self.tracks, strong, t)
        for r, c in matches:
            self.tracks[r].update(strong[c][:4], t)

        # second chance: weak detections rescue tracks the strong ones missed
        remaining = [self.tracks[i] for i in left_tracks]
        matches2, _, _ = self._associate(remaining, weak, t)
        for r, c in matches2:
            remaining[r].update(weak[c][:4], t)

        for j in new_idx:
            track = Track(self._next_id, tuple(strong[j][:4]), t, t)
            track.history.append((t, track.box))
            self._next_id += 1
            self.tracks.append(track)

        alive = []
        for track in self.tracks:
            if t - track.last_t > self.max_lost_sec:
                self.finished.append(track)
            else:
                alive.append(track)
        self.tracks = alive
        return [tr for tr in self.tracks if tr.last_t == t and tr.hits >= MIN_HITS]

    def all_tracks(self):
        return [tr for tr in self.finished + self.tracks if tr.hits >= MIN_HITS]


def _feet_in_roi(box, roi, width, height):
    if not roi:
        return True
    x = (box[0] + box[2]) / 2 / width
    y = box[3] / height
    return roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]


class FlowMeter:
    """Measures arrivals, departures and waiting time from finished tracks."""

    def __init__(self, width, height, duration_sec, roi=None, edge_sec=5.0):
        self.width, self.height, self.duration = width, height, duration_sec
        self.roi, self.edge = roi, edge_sec

    def measure(self, tracks, servers=1):
        entered, left, dwell = 0, 0, []
        counts = {}
        for track in tracks:
            inside = [(t, _feet_in_roi(b, self.roi, self.width, self.height)) for t, b in track.history]
            times = [t for t, ok in inside if ok]
            if not times:
                continue
            t_in, t_out = min(times), max(times)
            for t in times:
                counts[round(t)] = counts.get(round(t), 0) + 1
            was_there_at_start = t_in <= self.edge
            still_there_at_end = t_out >= self.duration - self.edge
            if not was_there_at_start:
                entered += 1
            if not still_there_at_end:
                left += 1
            if not was_there_at_start and not still_there_at_end:
                dwell.append(t_out - t_in)

        minutes = max(self.duration / 60, 1e-6)
        busy_seconds = max(1, sum(1 for c in counts.values() if c > 0))
        busy_minutes = busy_seconds / 60
        departures_per_min = left / busy_minutes if busy_minutes else 0.0
        summary = {
            "duration_sec": round(self.duration, 1),
            "people_tracked": sum(1 for _ in tracks),
            "arrivals": entered,
            "departures": left,
            "arrival_rate_per_min": round(entered / minutes, 2),
            "departure_rate_per_min": round(departures_per_min, 2),
            "avg_wait_min": round(float(np.mean(dwell)) / 60, 2) if dwell else None,
            "wait_samples": len(dwell),
            "avg_in_queue": round(sum(counts.values()) / max(self.duration, 1), 1),
        }
        # Per-server service time = servers / (people served per second while people were waiting).
        if left >= 5 and departures_per_min > 0:
            summary["suggested_service_time_sec"] = round(servers * 60.0 / departures_per_min, 1)
        else:
            summary["suggested_service_time_sec"] = None
        summary["utilisation"] = (round(summary["arrival_rate_per_min"] / summary["departure_rate_per_min"], 2)
                                  if summary["departure_rate_per_min"] else None)
        return summary
