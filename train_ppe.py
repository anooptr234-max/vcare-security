#!/usr/bin/env python3
"""
Train the kitchen PPE detector on a GPU machine.

Dataset (ppe2_dataset/) ships in this repo.
Classes: 0=hair_cap, 1=glove, 2=phone, 3=apron  (person comes from COCO)

Usage:
    pip install ultralytics
    python train_ppe.py            # ~5-10 min on a laptop GPU / Colab T4
Then send back: ppe_runs/ppe2/weights/best.pt
"""
from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":
    repo = Path(__file__).resolve().parent

    # data yaml with absolute paths (works on any machine / Colab)
    data_yaml = repo / "ppe2_runtime.yaml"
    data_yaml.write_text(
        f"""path: {repo / "ppe2_dataset"}
train: images/train
val: images/val
names:
  0: hair_cap
  1: glove
  2: phone
  3: apron
"""
    )

    model = YOLO("yolov8n.pt")  # COCO-pretrained start
    model.train(
        data=str(data_yaml),
        epochs=150,
        imgsz=640,
        batch=16,
        patience=40,
        name="ppe2",
        project=str(repo / "ppe_runs"),
        verbose=True,
        plots=False,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        cls=1.0,
    )
    print(f"Done. Send back {repo / 'ppe_runs' / 'ppe2' / 'weights' / 'best.pt'}")
