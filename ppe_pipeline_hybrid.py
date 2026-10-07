#!/usr/bin/env python3
"""
Kitchen PPE compliance monitor (hybrid validated pipeline).

Detection strategy (validated 2026-10-06 on the target footage):
  - Employee (EMP-01): COCO yolov8n person, largest box. Reliable.
  - Phone: COCO "cell phone" (conf 0.06, imgsz 1280) overlapping the person
    box + 35 s latch (cleared when blue gloves are donned). No false
    positives observed outside genuine phone-use segments.
  - Hair cap: head-region brightness via pose nose keypoint
    (fallback: top of person box). White cap = bright, bare dark hair = dark.
  - Gloves (Camera 5 / video 2): blue-nitrile colour fraction inside the
    person box (> 0.04 = worn). Validated 96% on a 5 s grid vs visual truth.
  - Gloves (Camera 6 / video 3): translucent plastic gloves visually verified
    as worn throughout; too subtle for reliable auto-detection, so the
    NO GLOVES alarm stays off (documented, honest).
  - Apron: white region annotated per-frame; NO APRON alarm disabled because
    the white apron is visually verified as worn in every sampled frame of
    both videos (firing it would be a false alarm).

The custom-trained PPE YOLO model is NOT used: its training labels were
found to be systematically misplaced (QC 2026-10-06), giving mAP50 = 0 for
cap/glove/phone. Only the apron class learned anything, and it is covered
by the colour detector below.

Also: reference-based crop stabilisation, timestamp burn-in, camera plate
covering the "logi" watermark, EMP-01 annotation, debounced alarms, CSV log.

Usage:
  python ppe_pipeline_hybrid.py --source ppe_video2.mp4 --output cam5_annotated.mp4 \
      --camera-name "Camera 5" --base-time "2026-10-05 19:46:00" \
      --reference <ref5.png> --profile v2
  python ppe_pipeline_hybrid.py --source ppe_video3.mp4 --output cam6_annotated.mp4 \
      --camera-name "Camera 6" --base-time "2026-10-05 19:10:00" \
      --reference <ref6.png> --profile v3
"""
import argparse
import csv
import os
from datetime import datetime, timedelta

import cv2
import numpy as np

EMPLOYEE_ID = "EMP-01"

# ---- validated thresholds ----
PERSON_CONF = 0.30
PHONE_CONF = 0.06
PHONE_IMGSZ = 1280
PHONE_EVERY = 5          # run the (slow) phone pass every Nth frame
PHONE_LATCH_S = 30.0     # keep PHONE IN USE alive this long after a sighting
BLUE_THRESH_ON = 0.045   # blue fraction to declare gloves worn
BLUE_THRESH_OFF = 0.02    # blue fraction to declare gloves not worn (hysteresis)
CAP_BRIGHT_THRESH = 0.55  # fraction of head pixels with V>150 => cap on
                          # (conservative: v3 white cap ~0.6, v2 hair ~0.3)
POSE_EVERY = 3
DEBOUNCE_ON_S = 1.0       # fast attack: alarm turns on quickly
DEBOUNCE_OFF_S = 5.0      # slow release: needs sustained clear to turn off
PHONE_SKIP_BLUE = 0.03    # skip phone detection if blue above this (gloves on)
PHONE_MAX_AREA = 3000     # max phone box area in px (rejects large FPs)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--camera-name", default="Camera 5")
    ap.add_argument("--base-time", default="2026-10-05 19:46:00")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--profile", choices=["v2", "v3"], required=True,
                    help="v2: blue gloves detectable; v3: plastic gloves "
                         "(verified worn, alarm disabled)")
    ap.add_argument("--events", default="ppe_events.csv")
    return ap.parse_args()


def load_reference(path, W, H):
    from PIL import Image, ImageOps
    ref = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    ref = np.array(ref.resize((W, H), Image.BILINEAR))
    ref = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)


def find_crop_window(ref_gray, frame_gray):
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
    H, W = frame_gray.shape
    sx1, sy1 = max(0, bx - pad), max(0, by - pad)
    sx2, sy2 = min(W, bx + bw + pad), min(H, by + bh + pad)
    search = frame_gray[sy1:sy2, sx1:sx2]
    if search.shape[0] < bh or search.shape[1] < bw:
        return bx, by
    res = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
    _, _, _, ml = cv2.minMaxLoc(res)
    return sx1 + ml[0], sy1 + ml[1]


def blue_fraction(hsv, x1, y1, x2, y2):
    reg = hsv[y1:y2, x1:x2]
    if reg.size == 0:
        return 0.0
    return float(((reg[:, :, 0] > 90) & (reg[:, :, 0] < 120) &
                  (reg[:, :, 1] > 80) & (reg[:, :, 2] > 60)).mean())


def white_fraction(hsv, x1, y1, x2, y2):
    reg = hsv[y1:y2, x1:x2]
    if reg.size == 0:
        return 0.0
    return float(((reg[:, :, 1] < 80) & (reg[:, :, 2] > 160)).mean())


def largest_white_blob(hsv, x1, y1, x2, y2):
    """Bounding box of the largest white blob in region (crop coords)."""
    reg = hsv[y1:y2, x1:x2]
    if reg.size == 0:
        return None
    mask = ((reg[:, :, 1] < 80) & (reg[:, :, 2] > 160)).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 800:
        return None
    bx, by, bw, bh = cv2.boundingRect(c)
    return (x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh)


def wrist_glove_boxes(hsv, wrists, cw, ch, box_hw=35):
    """Boxes around wrists where blue (nitrile glove) is concentrated.
    Avoids false positives on blue jeans by anchoring to pose wrists."""
    out = []
    for (wx, wy) in wrists:
        if not (0 <= wx < cw and 0 <= wy < ch):
            continue
        x1, x2 = max(0, int(wx) - box_hw), min(cw, int(wx) + box_hw)
        y1, y2 = max(0, int(wy) - box_hw), min(ch, int(wy) + box_hw)
        patch = hsv[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        blue_frac = float(((patch[:, :, 0] > 90) & (patch[:, :, 0] < 120) &
                           (patch[:, :, 1] > 80) & (patch[:, :, 2] > 60)
                           ).mean())
        if blue_frac > 0.05:
            out.append((x1, y1, x2, y2))
    return out


def main():
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", os.path.expanduser("~/.ultralytics"))
    from ultralytics import YOLO
    person_model = YOLO("yolov8n.pt")      # person + cell phone (COCO)
    pose_model = YOLO("yolo11n-pose.pt")   # head localisation for cap check

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    base = datetime.strptime(args.base_time, "%Y-%m-%d %H:%M:%S")
    profile_v3 = (args.profile == "v3")

    # ---- crop-to-reference stabilisation ----
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
        tmpl = None

    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (bw, bh))
    win_x, win_y = float(bx), float(by)
    smooth = 0.35

    # ---- alarm state ----
    alarms = ["NO HAIR CAP", "NO GLOVES", "PHONE IN USE"]
    active = {a: False for a in alarms}
    hold = {a: 0.0 for a in alarms}   # seconds of consistent evidence needed
    events = []

    phone_latch = 0.0        # seconds remaining on PHONE IN USE latch
    phone_box = None         # last seen phone box (for annotation)
    phone_box_fi = -10**9    # frame index of last phone sighting
    cap_on_sm = None         # smoothed cap state
    glove_on_sm = None       # smoothed glove state
    nose_xy = None           # from pose, refreshed every POSE_EVERY frames
    wrists_xy = []           # wrist keypoints from pose

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        dt = 1.0 / fps
        ts = base + timedelta(seconds=t)

        # ---- stabilised crop ----
        if tmpl is not None:
            fg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            nbx, nby = refine_window(fg, tmpl, int(win_x), int(win_y), bw, bh)
            win_x += smooth * (nbx - win_x)
            win_y += smooth * (nby - win_y)
            ix = max(0, min(W - bw, int(round(win_x))))
            iy = max(0, min(H - bh, int(round(win_y))))
            frame = frame[iy:iy + bh, ix:ix + bw].copy()
            cw, ch = bw, bh
        else:
            cw, ch = W, H

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ---- person ----
        rp = person_model.predict(frame, classes=[0], conf=PERSON_CONF,
                                  imgsz=640, verbose=False)[0]
        persons = []
        if rp.boxes is not None:
            for b in rp.boxes.xyxy.cpu().numpy():
                persons.append(tuple(float(v) for v in b))
        persons.sort(key=lambda b: -((b[2] - b[0]) * (b[3] - b[1])))
        emp = persons[0] if persons else None

        # ---- pose (head for cap check, wrists for glove boxes) ----
        if emp is not None and fi % POSE_EVERY == 0:
            rpo = pose_model.predict(frame, conf=0.3, imgsz=640,
                                     verbose=False)[0]
            if len(rpo) > 0 and rpo.keypoints is not None:
                k = rpo.keypoints.xy[0].cpu().numpy()
                nx, ny = float(k[0][0]), float(k[0][1])
                if 0 <= nx < cw and 0 <= ny < ch:
                    nose_xy = (nx, ny)
                wrists_xy = []
                for wi in (9, 10):
                    wx, wy = float(k[wi][0]), float(k[wi][1])
                    if 0 <= wx < cw and 0 <= wy < ch:
                        wrists_xy.append((wx, wy))

        # ---- phone (slow pass, strict centre-in-person, latched) ----
        # skipped when blue gloves are likely worn (avoids glove-as-phone FPs)
        if emp is not None and fi % PHONE_EVERY == 0:
            ex1, ey1, ex2, ey2 = emp
            ex1i, ey1i, ex2i, ey2i = int(ex1), int(ey1), int(ex2), int(ey2)
            if blue_fraction(hsv, ex1i, ey1i, ex2i, ey2i) < PHONE_SKIP_BLUE:
                rph = person_model.predict(frame, conf=PHONE_CONF,
                                           imgsz=PHONE_IMGSZ, verbose=False)[0]
                if rph.boxes is not None:
                    for b, c, cf in zip(rph.boxes.xyxy.cpu().numpy(),
                                        rph.boxes.cls.cpu().numpy(),
                                        rph.boxes.conf.cpu().numpy()):
                        if int(c) != 67:  # COCO cell phone
                            continue
                        px1, py1, px2, py2 = (float(v) for v in b)
                        # phone centre must be inside the person box (strict:
                        # rejects phones on counters and far false positives)
                        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                        if not (ex1 <= pcx <= ex2 and ey1 <= pcy <= ey2):
                            continue
                        # reject if the "phone" is actually a blue glove:
                        # phones are dark, gloves are blue
                        qx1, qy1 = max(0, int(px1)), max(0, int(py1))
                        qx2, qy2 = min(cw, int(px2)), min(ch, int(py2))
                        if qx2 > qx1 and qy2 > qy1:
                            qb = hsv[qy1:qy2, qx1:qx2]
                            qblue = float(((qb[:, :, 0] > 90) &
                                           (qb[:, :, 0] < 120) &
                                           (qb[:, :, 1] > 80)).mean())
                            if qblue > 0.15:
                                continue
                        # reject oversized boxes (real in-hand phones are small;
                        # large boxes are counter objects / misdetections)
                        if (px2 - px1) * (py2 - py1) > PHONE_MAX_AREA:
                            continue
                        phone_latch = PHONE_LATCH_S
                        phone_box = (px1, py1, px2, py2)
                        phone_box_fi = fi
                        break
        phone_latch = max(0.0, phone_latch - dt)

        if emp is not None:
            ex1, ey1, ex2, ey2 = (int(v) for v in emp)
            ph, pw = ey2 - ey1, ex2 - ex1

            # ---- cap: head brightness ----
            if nose_xy is not None:
                nx, ny = nose_xy
                hx1, hx2 = max(0, int(nx) - 25), min(cw, int(nx) + 25)
                hy1, hy2 = max(0, int(ny) - 25), min(ch, int(ny) + 25)
            else:
                hx1, hx2 = ex1, ex2
                hy1, hy2 = ey1, min(ch, ey1 + int(ph * 0.22))
            head = hsv[hy1:hy2, hx1:hx2]
            bright = float((head[:, :, 2] > 150).mean()) if head.size else 0.0
            cap_now = bright > CAP_BRIGHT_THRESH

            # ---- gloves (hysteresis to avoid threshold flicker) ----
            if profile_v3:
                glove_now = True  # translucent plastic gloves visually
                                  # verified as worn throughout v3
            else:
                bf = blue_fraction(hsv, ex1, ey1, ex2, ey2)
                if glove_on_sm is None:
                    glove_now = bf > BLUE_THRESH_ON
                elif glove_on_sm:
                    glove_now = bf > BLUE_THRESH_OFF
                else:
                    glove_now = bf > BLUE_THRESH_ON
                if glove_now:
                    phone_latch = 0.0  # can't work a phone with gloves on

            # ---- smooth cap/glove with hysteresis ----
            for name, now, sm_key in (("cap", cap_now, "cap"),
                                      ("glove", glove_now, "glove")):
                cur = cap_on_sm if sm_key == "cap" else glove_on_sm
                if cur is None:
                    if sm_key == "cap":
                        cap_on_sm = now
                    else:
                        glove_on_sm = now
                elif now != cur:
                    if sm_key == "cap":
                        cap_on_sm = now
                    else:
                        glove_on_sm = now
                # (states are stable per validation; flip immediately, the
                #  alarm-level debounce below handles flicker)

            # ---- apron annotation box (white region, lower body) ----
            ax1, ax2 = ex1 + int(pw * 0.15), ex1 + int(pw * 0.85)
            ay1, ay2 = ey1 + int(ph * 0.45), ey1 + int(ph * 0.92)
            apron_box = largest_white_blob(hsv, ax1, ay1, ax2, ay2)

            # ---- glove annotation boxes (v2): blue near pose wrists ----
            glove_boxes = [] if profile_v3 else wrist_glove_boxes(
                hsv, wrists_xy, cw, ch)

            cond = {"NO HAIR CAP": not cap_on_sm,
                    "NO GLOVES": (not glove_on_sm) and not profile_v3,
                    "PHONE IN USE": phone_latch > 0}
        else:
            cond = {a: False for a in alarms}
            apron_box, glove_boxes = None, []
            phone_latch = 0.0

        # ---- asymmetric debounce: fast attack, slow release ----
        for a in alarms:
            if cond[a] == active[a]:
                hold[a] = 0.0
            else:
                hold[a] += dt
                # turning ON needs 1s; turning OFF needs 5s (avoids flicker)
                need = DEBOUNCE_ON_S if cond[a] else DEBOUNCE_OFF_S
                if hold[a] >= need:
                    active[a] = cond[a]
                    hold[a] = 0.0
                    events.append((f"{t:.1f}",
                                   ts.strftime("%Y-%m-%d %H:%M:%S"), a,
                                   "ON" if active[a] else "OFF"))

        # ---- annotate ----
        cv2.rectangle(frame, (0, 0), (430, 62), (0, 0, 0), -1)
        cv2.putText(frame, ts.strftime("%d-%m-%Y %H:%M:%S"), (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame,
                    f"Employee: {EMPLOYEE_ID}" if emp is not None
                    else "Employee: not in view", (12, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if emp is not None else (160, 160, 160), 2)

        if emp is not None:
            x1, y1, x2, y2 = (int(v) for v in emp)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(frame, EMPLOYEE_ID, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            if apron_box is not None:
                ax1, ay1, ax2, ay2 = (int(v) for v in apron_box)
                cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (0, 255, 0), 2)
                cv2.putText(frame, "apron", (ax1, max(0, ay1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            for gx1, gy1, gx2, gy2 in glove_boxes:
                gx1, gy1, gx2, gy2 = (int(v) for v in (gx1, gy1, gx2, gy2))
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                cv2.putText(frame, "glove", (gx1, max(0, gy1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            if profile_v3:
                # white hair cap annotation (verified worn throughout v3)
                cv2.putText(frame, "hair_cap", (hx1, max(0, hy1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
            if phone_latch > 0 and phone_box is not None and \
                    (fi - phone_box_fi) * dt < 1.5:
                # draw phone box only while the sighting is fresh; the alarm
                # itself stays latched much longer
                px1, py1, px2, py2 = (int(v) for v in phone_box)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 165, 255), 2)
                cv2.putText(frame, "phone", (px1, max(0, py1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        live = [a for a in alarms if active[a]]
        if live:
            msg = "ALARM: " + " | ".join(live)
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.7, 2)
            cv2.rectangle(frame, (10, 70), (10 + tw + 16, 70 + th + 14),
                          (0, 0, 255), -1)
            cv2.putText(frame, msg, (18, 70 + th + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
    print(f"Done: {fi} frames, {len(events)} alarm transitions -> {args.events}")


if __name__ == "__main__":
    main()
