#!/usr/bin/env python3
"""Валидация best.pt на val-сете merged_v1."""
from ultralytics import YOLO

model = YOLO('/home/alex/AntiUAV-Detector/runs/detect/train/runs/merged_v1_L/weights/best.pt')
metrics = model.val(
    data='/home/alex/AntiUAV-Detector/prepare_data/merged_v1/data.yaml',
    imgsz=640,
    batch=16,
    device=0,
    workers=0,
    split='val',
    plots=True,
)
print(f'\n=== Результаты валидации ===')
print(f'Precision: {metrics.box.mp:.4f}')
print(f'Recall:    {metrics.box.mr:.4f}')
print(f'mAP50:     {metrics.box.map50:.4f}')
print(f'mAP50-95:  {metrics.box.map:.4f}')
