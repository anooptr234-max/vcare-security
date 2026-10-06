# VCARE Shop Security — video analytics system

One pipeline, four jobs, all from the shop's CCTV:

| # | Job | How |
|---|-----|-----|
| 1 | **Person entering / exiting** | Pretrained YOLOv8n person detector + ByteTrack; trip-line crossing counts entries vs exits |
| 2 | **Loitering** | Same tracks; a person continuously inside a configured zone longer than `--dwell` s raises an alert (snapshot saved) |
| 3 | **Armed-person alerts** | Separate YOLO weapon model, scanned every Nth frame (see below) |
| 4 | **Visitor analytics** | Every entry is timestamped in SQLite; `report.py` gives per-week / per-month customer counts |

No model training required for 1, 2 and 4 — the models are pretrained.

## Run it

```bash
pip install -r requirements.txt

# full pipeline on a video file / camera
python pipeline.py --source entrance.mp4 --db security.db --output annotated.mp4
python pipeline.py --source rtsp://user:pass@192.168.1.10:554/stream \
    --db /data/security.db --dwell 300 --weapon-model weapons.pt

# visitor analytics
python report.py --db security.db --period week
python report.py --db security.db --period month --last 6

# self-tests (no model, no footage needed)
python pipeline.py --synthetic-test          # entries/exits + loitering + reporting
python loitering_detector.py --synthetic-test # loitering module alone
```

Key flags: `--entry-line x1,y1,x2,y2` (trip-line; crossing − → + side counts as
entry, `--flip-direction` swaps it), `--loiter-zone` polygon, `--dwell` seconds,
`--weapon-every N`.

## Armed-person detection — read this

The pipeline accepts any YOLO-format weapon weights via `--weapon-model`, but
**there is no reliable off-the-shelf open weights file I can bundle**: public
gun-detection models vary wildly in quality and most are trained on staged
photos, not CCTV angles. The honest path:

1. Collect 500–2000 frames from *your own* cameras (including hard negatives:
   phones, tools, umbrellas).
2. Label with Roboflow (they also host community weapon datasets to start from).
3. Fine-tune `yolov8s.pt` ~50–100 epochs; deploy the resulting `best.pt`.

Treat weapon alerts as *prioritized review*, not ground truth — expect a tuning
period. Without `--weapon-model` the rest of the system runs normally.

## Deploying at the shop

Recommended: one NVIDIA Jetson Orin Nano per 2–4 cameras (or an x86 mini-PC),
cameras over RTSP, one container per camera:

```bash
docker build -t vcare-security .
docker run -d --restart unless-stopped -v /data:/data vcare-security \
  --source rtsp://user:pass@192.168.1.10:554/stream \
  --db /data/security.db --snapshots /data/snapshots \
  --entry-line 0.05,0.5,0.95,0.5 \
  --loiter-zone 0.05,0.55,0.95,0.55,0.95,1.0,0.05,1.0 \
  --dwell 300 --weapon-model /data/weapons.pt
```

For SMS/email alerts, watch `events` in SQLite (or tail `events.json` from the
standalone loitering script) and hook your notifier where events are logged.

## Tuning notes

- **Zone is everything for loitering**: point it at the entrance vestibule,
  cash desk, stock-room door — or the whole floor after hours. Browsing
  displays for 20 minutes is normal shopping; the zone makes it an alert.
- **Tracker ID switches** (occlusion → new ID) reset dwell timers and can
  double-count entries. Mitigate with camera angles covering the line/zone
  cleanly, not with code.
- CPU runs ~5–15 FPS at 640px; fine for counting, tight for 4+ cameras —
  that's what the Jetson is for.

## Legal (Ontario)

PIPEDA applies: post visible signage stating video analytics is in use,
keep a written retention policy (e.g. 30 days), restrict who can view
snapshots, no audio recording.
