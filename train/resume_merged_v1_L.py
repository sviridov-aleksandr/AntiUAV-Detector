#!/usr/bin/env python3
"""Возобновление обучения merged_v1_L с best.pt."""
from ultralytics import YOLO

model = YOLO('/home/alex/AntiUAV-Detector/runs/detect/train/runs/merged_v1_L/weights/best.pt')
model.train(resume=True, workers=0)
