#!/usr/bin/env python3
"""
Kitchen PPE compliance monitor.

Detects the employee (fixed ID EMP-01), plus hair cap, gloves, mobile phone
and apron/chef dress, and raises visual alarms:
  - NO HAIR CAP   (employee present, no hair cap)
  - NO GLOVES     (employee present, no gloves)
  - PHONE IN USE  (phone overlapping the employee)
  - NO APRON      (employee present, no apron/dress)

Stabilises shaky footage by aligning every frame to a reference still
(translation-only, temporally smoothed), covers the "logi" watermark with a
camera-name plate, and burns in a configurable timestamp.

Usage:
  python ppe_pipeline.py --source ppe_video2.mp4 --output cam5_annotated.mp4 \
      --camera-name "Camera 5" --base-time "2026-10-05 19:46:00" \
      --reference ref_workarea.png --weights ppe_runs/ppe_v1/weights/best.pt
"""
import argparse
import csv
import os
from datetime import datetime, timedelta

import cv2
import numpy as np

CLASS_NAMES = {0: "hair_cap", 1: "glove", 2: "phone", 3: "apron"}
# class ids in the PPE model: 0=hair_cap, 1=glove, 2=phone, 3=apron
# person comes from a separate COCO-pretrained model (reliable, no fine-tune)
EMPLOYEE_ID = "EMP-01"
ALARM_HOLD_FRAMES = 8  # state must persist this long before an alarm flips


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", required=True,
                    help="trained PPE yolov8 weights (4 classes: "
                         "hair_cap, glove, phone, apron)")
    ap.add_argument("--camera-name", default="Camera 5")
    ap.add_argument("--base-time", default="2026-10-05 19:46:00",
                    help="%%Y-%%m-%%d %%H:%%M:%%S")
    ap.add_argument("--reference", default=None,
                    help="still image to align/crop each frame to")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--events", default="ppe_events.csv")
    return ap.parse_args()


def load_reference(path, W, H):
    """Load a still (EXIF-aware) as grayscale at video frame size."""
    from PIL import Image, ImageOps
    ref = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    ref = np.array(ref.resize((W, H), Image.BILINEAR))
    ref = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)


def find_crop_window(ref_gray, frame_gray):
    """Multi-scale template match: locate the still's view inside the frame.
    Returns (bx, by, bw, bh)."""
    H, W = frame_gray.shape
    rh, rw = ref_gray.shape
    best = (0, None)
    for s in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0):
        tw, th = int(rw * s), int(rh * s)
        if tw >= W or th >= H or tw < 64 or th < 64:
            continue
        tmpl = cv2.resize(ref_gray, (tw, th))
        res = cv2.matchTemplate(frame_gray, tmpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (mx, (ml[0], ml[1], tw, th))
    return best[1], best[0]


def refine_window(frame_gray, tmpl, bx, by, bw, bh, pad=30):
    """Small-search-area template refinement around the current window."""
    H, W = frame_gray.shape
    sx1, sy1 = max(0, bx - pad), max(0, by - pad)
    sx2, sy2 = min(W, bx + bw + pad), min(H, by + bh + pad)
    search = frame_gray[sy1:sy2, sx1:sx2]
    if search.shape[0] < bh or search.shape[1] < bw:
        return bx, by
    res = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
    _, _, _, ml = cv2.minMaxLoc(res)
    return sx1 + ml[0], sy1 + ml[1]


def center_in(box, outer, margin=0.0):
    """True if box centre lies inside outer expanded by margin (fraction)."""
    x1, y1, x2, y2 = box
    ox1, oy1, ox2, oy2 = outer
    w, h = ox2 - ox1, oy2 - oy1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return (ox1 - margin * w) <= cx <= (ox2 + margin * w) and \
           (oy1 - margin * h) <= cy <= (oy2 + margin * h)


def main():
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", os.path.expanduser("~/.ultralytics"))
    from ultralytics import YOLO
    person_model = YOLO("yolov8n.pt")   # COCO person detector
    ppe_model = YOLO(args.weights)      # fine-tuned PPE detector

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    base = datetime.strptime(args.base_time, "%Y-%m-%d %H:%M:%S")

    # ---- crop-to-reference: locate the still's view, then track it ----
    crop = None          # (bx, by, bw, bh) in full-frame coords
    tmpl = None
    ref_gray = None
    if args.reference and os.path.exists(args.reference):
        ref_gray = load_reference(args.reference, W, H)
        ok, first = cap.read()
        if not ok:
            raise RuntimeError("cannot read first frame")
        fg = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        crop, score = find_crop_window(ref_gray, fg)
        if crop is None:
            raise RuntimeError("reference still not found in video")
        bx, by, bw, bh = crop
        print(f"crop window: ({bx},{by}) {bw}x{bh}, match={score:.2f}",
              flush=True)
        tmpl = cv2.resize(ref_gray, (bw, bh))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        bx, by, bw, bh = 0, 0, W, H

    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (bw, bh))
    win_x, win_y = float(bx), float(by)   # smoothed window origin
    smooth = 0.35

    # alarm state with debouncing
    alarms = ["NO HAIR CAP", "NO GLOVES", "PHONE IN USE", "NO APRON"]
    active = {a: False for a in alarms}
    pending = {a: 0 for a in alarms}
    events = []
    emp_visible_frames = 0

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        ts = base + timedelta(seconds=t)

        # ---- crop to the reference view (stabilise) ----
        if tmpl is not None:
            fg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            nbx, nby = refine_window(fg, tmpl, int(win_x), int(win_y),
                                     bw, bh)
            win_x += smooth * (nbx - win_x)
            win_y += smooth * (nby - win_y)
            ix, iy = int(round(win_x)), int(round(win_y))
            ix = max(0, min(W - bw, ix))
            iy = max(0, min(H - bh, iy))
            frame = frame[iy:iy + bh, ix:ix + bw].copy()
            cw, ch = bw, bh
        else:
            cw, ch = W, H

        # ---- detect: person (COCO) + PPE items (fine-tuned) ----
        rp = person_model.predict(frame, classes=[0], conf=0.4,
                                  imgsz=args.imgsz, verbose=False)[0]
        persons = []
        if rp.boxes is not None:
            for b in rp.boxes.xyxy.cpu().numpy():
                persons.append(tuple(b))
        persons.sort(key=lambda b: -((b[2] - b[0]) * (b[3] - b[1])))

        res = ppe_model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                                verbose=False)[0]
        dets = {0: [], 1: [], 2: [], 3: []}  # hair_cap, glove, phone, apron
        if res.boxes is not None:
            for b, c, cf in zip(res.boxes.xyxy.cpu().numpy(),
                                res.boxes.cls.cpu().numpy(),
                                res.boxes.conf.cpu().numpy()):
                dets[int(c)].append((tuple(b), float(cf)))

        emp = persons[0] if persons else None
        if emp is not None:
            emp_visible_frames += 1
            ex1, ey1, ex2, ey2 = emp
            head = (ex1, ey1, ex2, ey1 + (ey2 - ey1) * 0.35)

            cap_on = any(center_in(b, head, margin=0.25) for b, _ in dets[0])
            glove_on = any(center_in(b, emp, margin=0.10) for b, _ in dets[1])
            # phone counts as "in use" only if centred on the employee
            # (ignores phones lying on counters)
            phone_use = any(center_in(b, emp, margin=0.05) for b, _ in dets[2])
            apron_on = any(center_in(b, emp, margin=0.05) for b, _ in dets[3])

            cond = {"NO HAIR CAP": not cap_on,
                    "NO GLOVES": not glove_on,
                    "PHONE IN USE": phone_use,
                    "NO APRON": not apron_on}
        else:
            cond = {a: False for a in alarms}

        # ---- debounce alarm flips ----
        for a in alarms:
            if cond[a] == active[a]:
                pending[a] = 0
            else:
                pending[a] += 1
                if pending[a] >= ALARM_HOLD_FRAMES:
                    active[a] = cond[a]
                    pending[a] = 0
                    events.append((f"{t:.1f}",
                                   ts.strftime("%Y-%m-%d %H:%M:%S"), a,
                                   "ON" if active[a] else "OFF"))

        # ---- annotate ----
        cv2.rectangle(frame, (0, 0), (400, 62), (0, 0, 0), -1)
        cv2.putText(frame, ts.strftime("%d-%m-%Y %H:%M:%S"), (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"Employee: {EMPLOYEE_ID}" if emp is not None
                    else "Employee: not in view", (12, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if emp is not None else (160, 160, 160), 2)

        if emp is not None:
            x1, y1, x2, y2 = [int(v) for v in emp]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(frame, EMPLOYEE_ID, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        colors = {0: (0, 255, 0), 1: (0, 255, 0), 2: (0, 165, 255),
                  3: (0, 255, 0)}
        for c in (0, 1, 2, 3):
            for (x1, y1, x2, y2), cf in dets[c]:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colors[c], 2)
                cv2.putText(frame, f"{CLASS_NAMES[c]} {cf:.2f}",
                            (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[c], 2)

        live = [a for a in alarms if active[a]]
        if live:
            msg = "ALARM: " + " | ".join(live)
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.7, 2)
            cv2.rectangle(frame, (10, 70), (10 + tw + 16, 70 + th + 14),
                          (0, 0, 255), -1)
            cv2.putText(frame, msg, (18, 70 + th + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # camera plate bottom-right (covers "logi" when inside the crop)
        cv2.rectangle(frame, (cw - 260, ch - 58), (cw, ch), (0, 0, 0), -1)
        cv2.putText(frame, args.camera_name, (cw - 245, ch - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(frame)
        fi += 1
        if fi % 600 == 0:
            print(f"  {fi}/{n_frames} frames", flush=True)

    cap.release()
    out.release()
    with open(args.events, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_t", "timestamp", "alarm", "state"])
        w.writerows(events)
    print(f"Done: {fi} frames, employee visible in {emp_visible_frames}, "
          f"{len(events)} alarm transitions -> {args.events}")


if __name__ == "__main__":
    main()
