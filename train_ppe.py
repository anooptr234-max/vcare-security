#!/usr/bin/env python3
"""
Train the kitchen PPE detector on a GPU machine.

Dataset (ppe2_dataset/) and config (ppe2.yaml) are in this repo.
Classes: 0=hair_cap, 1=glove, 2=phone, 3=apron  (person comes from COCO)

Usage:
    pip install ultralytics
    python train_ppe.py            # ~5-10 min on a laptop GPU
Then send back: ppe_runs/ppe2/weights/best.pt
"""
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")  # COCO-pretrained start
    model.train(
        data="ppe2.yaml",
        epochs=150,
        imgsz=640,
        batch=16,
        patience=40,
        name="ppe2",
        project="ppe_runs",
        plots=False,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        cls=1.0,
    )
    print("Done. Send back ppe_runs/ppe2/weights/best.pt")
