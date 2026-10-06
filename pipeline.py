#!/usr/bin/env python3
"""
VCARE shop security pipeline — one process, four jobs:

  1. person entering / exiting   (trip-line crossing on tracked persons)
  2. loitering alerts            (zone dwell timer, reused from loitering_detector)
  3. armed-person alerts          (weapon detector, every Nth frame)
  4. visitor analytics            (entries logged to SQLite -> weekly/monthly counts)

Person detection/tracking uses pretrained YOLOv8n + ByteTrack (no training).
The weapon model must be supplied via --weapon-model — see README for how to
obtain/fine-tune one. Without it, weapon detection is disabled (the rest runs).

Usage:
    python pipeline.py --source entrance.mp4 --db security.db --output annotated.mp4
    python pipeline.py --source rtsp://user:pass@192.168.1.10:554/stream --db /data/security.db
    python pipeline.py --synthetic-test     # full self-test, no model needed
"""

import argparse
import os
import sys

import cv2
import numpy as np

from loitering_detector import DwellTracker, YoloBackend, parse_zone, inside_zone
from store import (init_db, log_entry, log_exit, log_event, totals)


# ---------------------------------------------------------------------------
# Entry / exit trip-line
# ---------------------------------------------------------------------------

def parse_line(spec, w, h):
    """'x1,y1,x2,y2' normalized -> two pixel points."""
    x1, y1, x2, y2 = (float(v) for v in spec.split(","))
    return (np.array([x1 * w, y1 * h]), np.array([x2 * w, y2 * h]))


def side_of(pt, p1, p2):
    """Signed side of directed line p1->p2. >0 / <0 / 0."""
    v = p2 - p1
    return float(v[0] * (pt[1] - p1[1]) - v[1] * (pt[0] - p1[0]))


class EntryTracker:
    """Counts entries/exits from trip-line crossings of tracked persons."""

    def __init__(self, conn, flip=False):
        self.conn = conn
        self.flip = flip
        self.prev_side = {}   # tid -> side
        self.open_visits = {}  # tid -> visit row id
        self.entries = 0
        self.exits = 0

    def update(self, detections, t_video):
        for tid, x1, y1, x2, y2 in detections:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            s = side_of(np.array([cx, cy]), self._p1, self._p2)
            if self.flip:
                s = -s
            prev = self.prev_side.get(tid)
            self.prev_side[tid] = s
            if prev is None or prev == 0 or s == 0:
                continue
            if prev < 0 < s:
                self.entries += 1
                self.open_visits[tid] = log_entry(self.conn, tid, t_video)
            elif prev > 0 > s:
                self.exits += 1
                vid = self.open_visits.pop(tid, None)
                if vid is not None:
                    log_exit(self.conn, vid, t_video)

    # set per run
    _p1 = _p2 = None


# ---------------------------------------------------------------------------
# Weapon detection (optional)
# ---------------------------------------------------------------------------

class WeaponDetector:
    """YOLO weapon model. Disabled when no model path is given."""

    def __init__(self, model_path=None, conf=0.5, imgsz=640, every=5):
        self.enabled = bool(model_path)
        self.conf, self.imgsz, self.every = conf, imgsz, every
        self._n = 0
        if self.enabled:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"Weapon model loaded: {model_path} "
                  f"(classes: {list(self.model.names.values())})")

    def scan(self, frame):
        if not self.enabled:
            return []
        self._n += 1
        if self._n % self.every != 0:
            return []
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                 verbose=False)[0]
        out = []
        if res.boxes is not None:
            for box, cls, cf in zip(res.boxes.xyxy.cpu().numpy(),
                                    res.boxes.cls.cpu().numpy(),
                                    res.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = box
                out.append((self.model.names[int(cls)], float(cf),
                            x1, y1, x2, y2))
        return out


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def annotate(frame, p1, p2, poly, detections, dwell, entry, weapon_hits,
             loiter_active):
    cv2.polylines(frame, [poly], True,
                  (0, 0, 255) if loiter_active else (0, 255, 0), 2)
    cv2.line(frame, tuple(p1.astype(int)), tuple(p2.astype(int)), (255, 255, 0), 2)
    cv2.putText(frame, f"in: {entry.entries}  out: {entry.exits}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    for tid, x1, y1, x2, y2 in detections:
        st = dwell.tracks.get(tid, {"dwell": 0.0, "alerted": False})
        color = (0, 0, 255) if st["alerted"] else (255, 200, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"id:{tid} {st['dwell']:.0f}s",
                    (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    y = 58
    if loiter_active:
        cv2.rectangle(frame, (0, 40), (frame.shape[1], 84), (0, 0, 255), -1)
        cv2.putText(frame, "LOITERING ALERT", (12, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y = 108
    for label, cf, x1, y1, x2, y2 in weapon_hits:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        cv2.putText(frame, f"WEAPON {label} {cf:.2f}", (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if weapon_hits:
        cv2.rectangle(frame, (0, y - 18), (frame.shape[1], y + 26), (0, 0, 255), -1)
        cv2.putText(frame, "ARMED PERSON ALERT", (12, y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="VCARE shop security pipeline")
    ap.add_argument("--source", help="video file or rtsp/http stream URL")
    ap.add_argument("--db", default="security.db", help="SQLite db path")
    ap.add_argument("--output", default="annotated.mp4")
    ap.add_argument("--snapshots", default="snapshots")
    ap.add_argument("--entry-line", default="0.05,0.5,0.95,0.5",
                    help="trip-line x1,y1,x2,y2 normalized; - to + side = entry")
    ap.add_argument("--flip-direction", action="store_true",
                    help="swap which crossing direction counts as entry")
    ap.add_argument("--loiter-zone", default="0,0,1,0,1,1,0,1",
                    help="polygon x1,y1,... normalized (default: full frame)")
    ap.add_argument("--dwell", type=float, default=300,
                    help="loitering seconds (default 300)")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--weapon-model", default=None,
                    help="YOLO weapon weights path (enables armed-person alerts)")
    ap.add_argument("--weapon-conf", type=float, default=0.5)
    ap.add_argument("--weapon-every", type=int, default=5,
                    help="run weapon model every Nth frame")
    ap.add_argument("--synthetic-test", action="store_true")
    args = ap.parse_args()

    if args.synthetic_test:
        return synthetic_test()

    if not args.source:
        ap.error("--source is required (unless --synthetic-test)")

    conn = init_db(args.db)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit(f"Could not open source: {args.source}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    p1, p2 = parse_line(args.entry_line, w, h)
    poly = parse_zone(args.loiter_zone, w, h)
    EntryTracker._p1, EntryTracker._p2 = p1, p2
    DwellTracker._poly = poly

    person = YoloBackend(conf=args.conf, imgsz=args.imgsz)
    entry = EntryTracker(conn, flip=args.flip_direction)
    dwell = DwellTracker(dwell_threshold=args.dwell)
    weapons = WeaponDetector(args.weapon_model, args.weapon_conf,
                             args.imgsz, args.weapon_every)
    if not weapons.enabled:
        print("Weapon detection DISABLED (no --weapon-model). See README.")

    vw = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    frame_idx, loiter_active, n_weapons = 0, False, 0
    print(f"Running: entries/exits + loitering>{args.dwell}s "
          f"+ {'weapons' if weapons.enabled else 'no weapons'} -> {args.db}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / fps
            dets = person.tracks(frame)
            entry.update(dets, t)
            for ev in dwell.update(dets, 1.0 / fps, t, frame, args.snapshots):
                loiter_active = True
                log_event(conn, "loitering", ev["track_id"], ev["video_time_sec"],
                          f"dwell {ev['dwell_seconds']}s", ev["snapshot"])
                print(f"  LOITERING t={ev['video_time_sec']}s id={ev['track_id']}")
            hits = weapons.scan(frame)
            for label, cf, x1, y1, x2, y2 in hits:
                n_weapons += 1
                snap = DwellTracker._snapshot(frame, (x1, y1, x2, y2),
                                              args.snapshots, f"w{n_weapons}", t)
                log_event(conn, "weapon", -1, t, f"{label} {cf:.2f}", snap)
                print(f"  WEAPON t={t:.1f}s {label} {cf:.2f}")
            vw.write(annotate(frame, p1, p2, poly, dets, dwell, entry,
                              hits, loiter_active))
            frame_idx += 1
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        vw.release()
        conn.close()
    print(f"Done: in={entry.entries} out={entry.exits} "
          f"loitering={len([1]) if loiter_active else 0} weapon_hits={n_weapons}")
    print("Totals:", totals(init_db(args.db)))


# ---------------------------------------------------------------------------
# Synthetic self-test (no model, no footage)
# ---------------------------------------------------------------------------

def synthetic_test():
    """3 entries, 1 exit, 1 loiterer -> verifies counting + dwell + reporting."""
    w, h, fps, seconds = 640, 480, 10, 45
    db = "test_security.db"
    if os.path.exists(db):
        os.remove(db)
    conn = init_db(db)

    p1, p2 = parse_line("0.05,0.5,0.95,0.5", w, h)   # horizontal mid-frame
    poly = parse_zone("0.05,0.55,0.95,0.55,0.95,1.0,0.05,1.0", w, h)  # lower zone
    EntryTracker._p1, EntryTracker._p2 = p1, p2
    DwellTracker._poly = poly
    entry = EntryTracker(conn)
    dwell = DwellTracker(dwell_threshold=10)

    # script: (id, kind) — walkers cross the line, loiterer stands in zone
    walkers = [  # (id, start_t, x_frac, direction +1 down / -1 up)
        (1, 2, 0.25, +1), (2, 10, 0.50, +1), (3, 18, 0.75, +1), (4, 26, 0.60, -1),
    ]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter("synthetic_pipeline.mp4", fourcc, fps, (w, h))
    rng = np.random.default_rng(3)
    n = int(seconds * fps)
    loiter_active = False
    for i in range(n):
        t = i / fps
        img = np.full((h, w, 3), 60, np.uint8)
        cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), (255, 255, 0), 2)
        cv2.polylines(img, [poly], True, (0, 255, 0), 2)
        dets = []
        for tid, t0, xf, dirc in walkers:
            if t0 <= t <= t0 + 5:  # 5 s to cross the frame vertically
                prog = (t - t0) / 5
                y = h * (0.08 + 0.84 * prog) if dirc > 0 else h * (0.92 - 0.84 * prog)
                x = int(w * xf)
                _draw_person(img, x, int(y))
                dets.append((tid, x - 20, y - 70, x + 20, y + 42))
        # id 5: loiterer, stands in zone whole clip (below the line, never crosses)
        jx, jy = rng.integers(-4, 5, 2)
        lx, ly = int(w * 0.5 + jx), int(h * 0.75 + jy)
        _draw_person(img, lx, ly, (255, 255, 255))
        dets.append((5, lx - 20, ly - 70, lx + 20, ly + 42))

        entry.update(dets, t)
        for ev in dwell.update(dets, 1.0 / fps, t, img, "snapshots_test"):
            loiter_active = True
            log_event(conn, "loitering", ev["track_id"], ev["video_time_sec"],
                      f"dwell {ev['dwell_seconds']}s", ev["snapshot"])
        vw.write(annotate(img, p1, p2, poly, dets, dwell, entry, [], loiter_active))
    vw.release()

    n_loiter = conn.execute("SELECT COUNT(*) FROM events WHERE type='loitering'").fetchone()[0]
    ok = (entry.entries == 3 and entry.exits == 1 and n_loiter == 1)
    print(f"entries={entry.entries} (want 3)  exits={entry.exits} (want 1)  "
          f"loitering events={n_loiter} (want 1)")
    print("SYNTHETIC PIPELINE TEST:", "PASS" if ok else "FAIL")

    # reporting check on the same db
    from store import weekly_counts, monthly_counts
    print("weekly:", weekly_counts(conn), " monthly:", monthly_counts(conn))
    conn.close()
    return 0 if ok else 1


def _draw_person(img, cx, cy, color=(200, 200, 200)):
    cv2.circle(img, (cx, cy - 55), 14, color, -1)
    cv2.rectangle(img, (cx - 18, cy - 40), (cx + 18, cy + 40), color, -1)


if __name__ == "__main__":
    sys.exit(main())
