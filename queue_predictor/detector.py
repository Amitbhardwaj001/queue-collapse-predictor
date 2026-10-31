"""People detection and counting for images and videos.

Uses YOLOv8 (class 0 = person). If the ultralytics package or its weights are
unavailable, falls back to OpenCV's built-in HOG pedestrian detector, which is
less accurate but needs no download.

A queue area (ROI) can be set per location: only people whose feet (bottom
centre of the box) are inside it are counted, so passers-by and people sitting
at tables are ignored.
"""
import os
import uuid

import cv2

from .config import (PERSON_CONFIDENCE, RESULT_DIR, TRACK_FPS, TRACK_MAX_FRAMES, VIDEO_MAX_SAMPLES,
                     VIDEO_SAMPLE_SECONDS, YOLO_WEIGHTS)
from .tracker import ByteTracker, FlowMeter

os.environ.setdefault("YOLO_OFFLINE", "1")  # never try to reach the internet during a demo

COUNTED = (40, 200, 90)
IGNORED = (150, 150, 150)
ROI_COLOUR = (255, 170, 0)


def in_roi(box, roi, width, height):
    if not roi:
        return True
    x1, y1, x2, y2 = box[:4]
    feet_x, feet_y = (x1 + x2) / 2 / width, y2 / height
    return roi[0] <= feet_x <= roi[2] and roi[1] <= feet_y <= roi[3]


class PersonDetector:
    def __init__(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(YOLO_WEIGHTS))
            self.backend = "YOLOv8"
        except Exception as exc:  # missing package, missing weights, ...
            print(f"YOLO unavailable ({exc}); using OpenCV HOG detector instead.")
            self.model = cv2.HOGDescriptor()
            self.model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.backend = "OpenCV HOG"

    def detect(self, frame):
        """Return a list of (x1, y1, x2, y2, confidence) boxes for every person in the frame."""
        if self.backend == "YOLOv8":
            result = self.model.predict(frame, classes=[0], conf=PERSON_CONFIDENCE, verbose=False)[0]
            return [(*map(int, box.xyxy[0].tolist()), float(box.conf[0])) for box in result.boxes]

        scale = 800 / max(frame.shape[:2]) if max(frame.shape[:2]) > 800 else 1.0
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
        rects, weights = self.model.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
        return [(int(x / scale), int(y / scale), int((x + w) / scale), int((y + h) / scale), float(c))
                for (x, y, w, h), c in zip(rects, weights) if float(c) >= 0.3]

    def count(self, frame, roi=None):
        """Return (people in the queue area, boxes in the area, boxes outside it)."""
        height, width = frame.shape[:2]
        inside, outside = [], []
        for box in self.detect(frame):
            (inside if in_roi(box, roi, width, height) else outside).append(box)
        return len(inside), inside, outside

    def annotate(self, frame, inside, outside, roi=None):
        out = frame.copy()
        height, width = frame.shape[:2]
        thick = max(2, width // 400)
        font = 0.5 * thick / 2

        if roi:
            p1 = (int(roi[0] * width), int(roi[1] * height))
            p2 = (int(roi[2] * width), int(roi[3] * height))
            shade = out.copy()
            cv2.rectangle(shade, p1, p2, ROI_COLOUR, -1)
            out = cv2.addWeighted(shade, 0.12, out, 0.88, 0)
            cv2.rectangle(out, p1, p2, ROI_COLOUR, thick)
            cv2.putText(out, "QUEUE AREA", (p1[0] + 6, p2[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        font * 1.2, ROI_COLOUR, max(1, thick // 2))

        for boxes, colour in ((outside, IGNORED), (inside, COUNTED)):
            for x1, y1, x2, y2, conf in boxes:
                cv2.rectangle(out, (x1, y1), (x2, y2), colour, thick)
                cv2.putText(out, f"{conf:.2f}", (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            font, colour, max(1, thick // 2))

        label = f"People in queue: {len(inside)}"
        if outside:
            label += f"  (ignored: {len(outside)})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font * 1.4, max(1, thick // 2))
        cv2.rectangle(out, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
        cv2.putText(out, label, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, font * 1.4,
                    (255, 255, 255), max(1, thick // 2))
        return out

    def _save(self, frame, prefix):
        name = f"{prefix}_{uuid.uuid4().hex[:12]}.jpg"
        cv2.imwrite(str(RESULT_DIR / name), frame)
        return name

    def _analyse(self, frame, roi):
        count, inside, outside = self.count(frame, roi)
        return count, self._save(self.annotate(frame, inside, outside, roi), "result"), self._save(frame, "frame")

    def process_image(self, path, roi=None):
        """Returns (count, annotated image name, raw frame name)."""
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError("Could not read the image file.")
        return self._analyse(frame, roi)

    def process_video(self, path, roi=None):
        """Count people every VIDEO_SAMPLE_SECONDS.

        Returns ([(offset_sec, count)], duration_sec, annotated image name, raw frame name),
        where the images show the busiest sampled frame.
        """
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError("Could not read the video file.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step = max(1, int(round(fps * VIDEO_SAMPLE_SECONDS)))

        samples, busiest = [], (-1, None)
        frame_no = 0
        while len(samples) < VIDEO_MAX_SAMPLES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            if not ok:
                break
            count, _, _ = self.count(frame, roi)
            samples.append((frame_no / fps, count))
            if count > busiest[0]:
                busiest = (count, frame)
            frame_no += step
        cap.release()
        if not samples:
            raise ValueError("The video contains no readable frames.")
        duration = max(total / fps, samples[-1][0])
        _, image, raw = self._analyse(busiest[1], roi)
        return samples, duration, image, raw

    def track_video(self, path, roi=None, servers=1):
        """Follow every person through the video and measure the real flow.

        Returns (samples, duration_sec, flow summary, annotated image name, raw frame name);
        samples are (offset_sec, people in the queue area) every VIDEO_SAMPLE_SECONDS.
        """
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError("Could not read the video file.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, int(round(fps / TRACK_FPS)))
        tracker = ByteTracker()
        samples, next_sample, busiest = [], 0.0, (-1, None, None)
        width = height = 0
        frame_no, analysed, last_t = 0, 0, 0.0
        while analysed < TRACK_MAX_FRAMES:
            ok = cap.grab()
            if not ok:
                break
            if frame_no % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                height, width = frame.shape[:2]
                t = frame_no / fps
                detections = self.detect(frame)
                visible = tracker.update(detections, t)
                inside = [d for d in detections if in_roi(d, roi, width, height)]
                outside = [d for d in detections if not in_roi(d, roi, width, height)]
                if t >= next_sample and len(samples) < VIDEO_MAX_SAMPLES:
                    samples.append((t, len(inside)))
                    next_sample += VIDEO_SAMPLE_SECONDS
                if len(inside) > busiest[0]:
                    busiest = (len(inside), frame.copy(), (inside, outside, visible))
                analysed += 1
                last_t = t
            frame_no += 1
        cap.release()
        if not samples:
            raise ValueError("The video contains no readable frames.")

        summary = FlowMeter(width, height, max(last_t, 1.0), roi).measure(tracker.all_tracks(), servers)
        _, frame, (inside, outside, visible) = busiest
        image = self.annotate(frame, inside, outside, roi)
        for track in visible:
            x1, y1 = int(track.box[0]), int(track.box[1])
            cv2.putText(image, f"#{track.id}", (x1, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return (samples, max(last_t, samples[-1][0]), summary,
                self._save(image, "result"), self._save(frame, "frame"))
