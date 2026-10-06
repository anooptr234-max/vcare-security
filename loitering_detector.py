#!/usr/bin/env python3
"""
Loitering detection for retail CCTV using pretrained models — no training needed.

Pipeline:
    YOLOv8n (COCO-pretrained, person class only)
      -> ByteTrack multi-object tracking (persistent IDs)
      -> zone dwell timer -> loitering alert

A person is flagged as loitering when a single tracked ID stays inside the
configured zone continuously for longer than --dwell seconds.

Usage:
    # video file
    python loitering_detector.py --source shop_entrance.mp4 --output annotated.mp4

    # live RTSP camera (deployment)
    python loitering_detector.py --source rtsp://user:pass@192.168.1.10:554/stream --dwell 300

    # self-test of the dwell/alert logic with synthetic detections (no model needed)
    python loitering_detector.py --synthetic-test
"""

import argparse
import datetime
import json
import os
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Zone helpers
# ---------------------------------------------------------------------------

def parse_zone(spec, w, h):
    """'x1,y1,x2,y2,...' (0-1 normalized) -> Nx2 int pixel polygon."""
    vals = [float(v) for v in spec.split(",")]
    if len(vals) < 6 or len(vals) % 2:
        raise ValueError("zone needs at least 3 x,y pairs")
    pts = np.array(vals, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] *= w
    pts[:, 1] *= h
    return pts.astype(np.int32)


def inside_zone(x, y, poly):
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


# ---------------------------------------------------------------------------
# Track state / dwell logic (model-agnostic: works with real or synthetic tracks)
# ---------------------------------------------------------------------------

class DwellTracker:
    """Accumulates per-ID dwell time inside the zone; fires one alert per ID."""

    def __init__(self, dwell_threshold):
        self.dwell_threshold = dwell_threshold
        self.tracks = {}          # id -> {"dwell": float, "alerted": bool}
        self.events = []

    def update(self, detections, dt, t_video, frame, snapshot_dir):
        """
        detections: list of (track_id, x1, y1, x2, y2)
        dt: seconds elapsed since previous frame
        Returns list of newly fired events.
        """
        new_events = []
        seen = set()
        for tid, x1, y1, x2, y2 in detections:
            seen.add(tid)
            st = self.tracks.setdefault(tid, {"dwell": 0.0, "alerted": False})
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if self._inside(cx, cy):
                st["dwell"] += dt
                if st["dwell"] >= self.dwell_threshold and not st["alerted"]:
                    st["alerted"] = True
                    snap = self._snapshot(frame, (x1, y1, x2, y2), snapshot_dir, tid, t_video)
                    ev = {
                        "event": "loitering",
                        "track_id": int(tid),
                        "dwell_seconds": round(st["dwell"], 1),
                        "video_time_sec": round(t_video, 1),
                        "wall_time": datetime.datetime.now().isoformat(timespec="seconds"),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "snapshot": snap,
                    }
                    self.events.append(ev)
                    new_events.append(ev)
            else:
                # left the zone -> dwell resets (continuous-presence rule)
                st["dwell"] = 0.0
        # drop tracks not seen this frame (tracker lost them)
        for tid in [k for k in self.tracks if k not in seen]:
            del self.tracks[tid]
        return new_events

    # set per run
    _poly = None

    def _inside(self, cx, cy):
        return inside_zone(cx, cy, self._poly)

    @staticmethod
    def _snapshot(frame, bbox, snapshot_dir, tid, t_video):
        if not snapshot_dir:
            return None
        os.makedirs(snapshot_dir, exist_ok=True)
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        path = os.path.join(snapshot_dir, f"loitering_track{tid}_t{int(t_video)}s.jpg")
        cv2.imwrite(path, crop)
        return path


# ---------------------------------------------------------------------------
# Detection backends
# ---------------------------------------------------------------------------

class YoloBackend:
    """Pretrained YOLOv8n person detector + ByteTrack."""

    def __init__(self, conf=0.4, imgsz=640):
        from ultralytics import YOLO
        self.model = YOLO("yolov8n.pt")  # auto-downloads COCO-pretrained weights
        self.conf = conf
        self.imgsz = imgsz

    def tracks(self, frame):
        res = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            classes=[0], conf=self.conf, imgsz=self.imgsz, verbose=False,
        )[0]
        out = []
        if res.boxes is not None and res.boxes.id is not None:
            for box, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.cpu().numpy()):
                x1, y1, x2, y2 = box
                out.append((int(tid), x1, y1, x2, y2))
        return out


class SyntheticBackend:
    """
    Test harness: injects fake tracks so the dwell/alert logic can be
    verified end-to-end without a neural net or real footage.
    Scenario: person #1 stands in the zone the whole clip (should alert),
              person #2 walks straight through (should NOT alert).
    """

    def __init__(self, w, h, fps, zone_poly):
        self.w, self.h, self.fps = w, h, fps
        self.zone = zone_poly

    def make_video(self, path, seconds=35):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(path, fourcc, self.fps, (self.w, self.h))
        rng = np.random.default_rng(7)
        n = int(seconds * self.fps)
        zx = self.zone[:, 0].mean()  # zone centre x
        for i in range(n):
            img = np.full((self.h, self.w, 3), 60, np.uint8)
            cv2.polylines(img, [self.zone], True, (0, 255, 0), 2)
            # person 1: loiterer, stands near zone centre with slight jitter
            jx, jy = rng.integers(-4, 5, 2)
            self._person(img, int(zx + jx), int(self.h * 0.62 + jy), (255, 255, 255))
            # person 2: walker, crosses the frame left->right in ~4 s, then gone
            if i < int(4 * self.fps):
                wx = int(self.w * (i / (4 * self.fps)))
                self._person(img, wx, int(self.h * 0.62), (200, 200, 200))
            vw.write(img)
        vw.release()
        return path

    @staticmethod
    def _person(img, cx, cy, color):
        cv2.circle(img, (cx, cy - 55), 14, color, -1)          # head
        cv2.rectangle(img, (cx - 18, cy - 40), (cx + 18, cy + 40), color, -1)  # torso

    def tracks(self, frame_idx):
        dets = []
        # id 1: loiterer bbox (matches drawn figure, ~36x110 px at 640x480)
        zx = int(self.zone[:, 0].mean())
        cy = int(self.h * 0.62)
        dets.append((1, zx - 20, cy - 70, zx + 20, cy + 42))
        # id 2: walker, present only for first ~4 s
        if frame_idx < int(4 * self.fps):
            wx = int(self.w * (frame_idx / (4 * self.fps)))
            dets.append((2, wx - 20, cy - 70, wx + 20, cy + 42))
        return dets


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def annotate(frame, poly, detections, dwell_tracker, active_alerts):
    cv2.polylines(frame, [poly], True, (0, 0, 255) if active_alerts else (0, 255, 0), 2)
    for tid, x1, y1, x2, y2 in detections:
        st = dwell_tracker.tracks.get(tid, {"dwell": 0.0, "alerted": False})
        color = (0, 0, 255) if st["alerted"] else (255, 200, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"id:{tid} {st['dwell']:.1f}s",
                    (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    if active_alerts:
        msg = "LOITERING: " + ", ".join(f"id {e['track_id']} ({e['dwell_seconds']}s)"
                                        for e in active_alerts[-3:])
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (0, 0, 255), -1)
        cv2.putText(frame, msg, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
    return frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Pretrained-model loitering detection")
    ap.add_argument("--source", help="video file or rtsp/http stream URL")
    ap.add_argument("--output", default="annotated.mp4", help="annotated output video")
    ap.add_argument("--events", default="events.json", help="loitering events JSON")
    ap.add_argument("--snapshots", default="snapshots", help="dir for alert snapshots")
    ap.add_argument("--zone", default="0,0,1,0,1,1,0,1",
                    help="polygon x1,y1,x2,y2,... normalized 0-1 (default: full frame)")
    ap.add_argument("--dwell", type=float, default=60,
                    help="seconds inside zone before alert (default 60)")
    ap.add_argument("--conf", type=float, default=0.4, help="detection confidence")
    ap.add_argument("--imgsz", type=int, default=640, help="inference image size")
    ap.add_argument("--synthetic-test", action="store_true",
                    help="self-test with injected detections (no model needed)")
    ap.add_argument("--synthetic-dwell", type=float, default=10,
                    help="dwell threshold used in synthetic test")
    args = ap.parse_args()

    if args.synthetic_test:
        return synthetic_test(args)

    if not args.source:
        ap.error("--source is required (unless --synthetic-test)")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit(f"Could not open source: {args.source}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    poly = parse_zone(args.zone, w, h)
    backend = YoloBackend(conf=args.conf, imgsz=args.imgsz)
    dwell = DwellTracker(dwell_threshold=args.dwell)
    DwellTracker._poly = poly

    vw = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    frame_idx, active_alerts = 0, []
    print(f"Processing {args.source} ({w}x{h} @ {fps:.1f}fps), dwell>{args.dwell}s ...")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_video = frame_idx / fps
            dets = backend.tracks(frame)
            for ev in dwell.update(dets, 1.0 / fps, t_video, frame, args.snapshots):
                active_alerts.append(ev)
                print(f"  ALERT t={ev['video_time_sec']}s track={ev['track_id']} "
                      f"dwell={ev['dwell_seconds']}s")
            vw.write(annotate(frame, poly, dets, dwell, active_alerts))
            frame_idx += 1
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cap.release()
        vw.release()

    with open(args.events, "w") as f:
        json.dump(dwell.events, f, indent=2)
    print(f"Done: {frame_idx} frames, {len(dwell.events)} loitering event(s) -> {args.events}")


def synthetic_test(args):
    """End-to-end logic test with injected detections (no model, no footage)."""
    w, h, fps, seconds = 640, 480, 10, 35
    poly = parse_zone(args.zone, w, h)
    synth = SyntheticBackend(w, h, fps, poly)
    src = "synthetic_test.mp4"
    synth.make_video(src, seconds)

    dwell = DwellTracker(dwell_threshold=args.synthetic_dwell)
    DwellTracker._poly = poly
    cap = cv2.VideoCapture(src)
    vw = cv2.VideoWriter("synthetic_annotated.mp4",
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    frame_idx, active = 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = synth.tracks(frame_idx)
        for ev in dwell.update(dets, 1.0 / fps, frame_idx / fps, frame, "snapshots_test"):
            active.append(ev)
        vw.write(annotate(frame, poly, dets, dwell, active))
        frame_idx += 1
    cap.release()
    vw.release()

    loiter_ids = {e["track_id"] for e in dwell.events}
    ok = loiter_ids == {1} and all(e["dwell_seconds"] >= args.synthetic_dwell
                                   for e in dwell.events)
    print(f"Synthetic test: {len(dwell.events)} event(s), loitering IDs={sorted(loiter_ids)}")
    print("PASS: loiterer (id 1) alerted, walker (id 2) ignored"
          if ok else "FAIL: unexpected alert set")
    with open("events_test.json", "w") as f:
        json.dump(dwell.events, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    main()
