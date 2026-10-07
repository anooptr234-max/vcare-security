#!/usr/bin/env python3
"""
Camera 3 — dining-area occupancy demo.

- Pretrained YOLOv8n + ByteTrack person detection/tracking
- Trip-line at the doorway: crossing in = entry, crossing out = exit
- Live occupancy counter (entries - exits); returns to 0 when the person leaves
- Overlays: timestamp (top-left), "Camera 3" (bottom-left), track IDs

Usage:
    python camera3_pipeline.py --source dining.mp4 --output camera3_annotated.mp4
"""
import argparse
import datetime
import sys

import cv2
import numpy as np

from loitering_detector import YoloBackend
from pipeline import EntryTracker, parse_line
from store import init_db


def annotate(frame, p1, p2, detections, inside, entry, ts_text):
    h, w = frame.shape[:2]
    # entry trip-line (subtle)
    cv2.line(frame, tuple(p1.astype(int)), tuple(p2.astype(int)), (255, 255, 0), 2)

    # timestamp bar, top-left
    cv2.rectangle(frame, (0, 0), (430, 78), (0, 0, 0), -1)
    cv2.putText(frame, ts_text, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Inside: {inside}", (12, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if inside == 0 else (0, 165, 255), 2)

    # person boxes with track IDs
    for tid, x1, y1, x2, y2 in detections:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 2)
        cv2.putText(frame, f"id:{tid}", (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

    # camera name plate, bottom-right (covers the "logi" watermark area)
    cv2.rectangle(frame, (w - 260, h - 58), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "Camera 3", (w - 245, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def main():
    ap = argparse.ArgumentParser(description="Camera 3 dining-area occupancy")
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", default="camera3_annotated.mp4")
    ap.add_argument("--db", default="camera3.db")
    ap.add_argument("--entry-line", default="0.30,0.70,0.70,0.70",
                    help="trip-line x1,y1,x2,y2 normalized; top->bottom = entry. "
                         "Place it where people are large and well-lit on crossing.")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--date", default="2026-10-05", help="overlay date YYYY-MM-DD")
    ap.add_argument("--start-time", default="19:30:00", help="overlay start HH:MM:SS")
    args = ap.parse_args()

    base = datetime.datetime.strptime(f"{args.date} {args.start_time}",
                                      "%Y-%m-%d %H:%M:%S")
    conn = init_db(args.db)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit(f"Could not open source: {args.source}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    p1, p2 = parse_line(args.entry_line, w, h)
    EntryTracker._p1, EntryTracker._p2 = p1, p2
    entry = EntryTracker(conn)
    person = YoloBackend(conf=args.conf, imgsz=args.imgsz, weights=args.weights)

    vw = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    frame_idx = 0
    print(f"Camera 3: {w}x{h} @ {fps:.1f}fps, base time {base}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / fps
            dets = person.tracks(frame)
            entry.update(dets, t)
            inside = max(0, entry.entries - entry.exits)
            ts = (base + datetime.timedelta(seconds=t)).strftime("%d-%m-%Y %H:%M:%S")
            vw.write(annotate(frame, p1, p2, dets, inside, entry, ts))
            frame_idx += 1
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        vw.release()
        conn.close()
    inside = max(0, entry.entries - entry.exits)
    print(f"Done: in={entry.entries} out={entry.exits} inside_now={inside}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
