#!/usr/bin/env python3
"""Извлекает координаты bbox на v57 для понимания позиции дрона."""
import os
import sys
import cv2

sys.path.insert(0, 'train')
from ultralytics import YOLO

PROJECT = '/home/alex/AntiUAV-Detector'
WEIGHTS = os.path.join(PROJECT, 'runs/detect/train/runs/merged_v1_L/weights/best.pt')

videos = ['v57.mp4', 'v21.mp4', 'v77.mp4', 'v25.mp4']

for video_name in videos:
    video_path = os.path.join(PROJECT, 'video-FPV/Video', video_name)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    model = YOLO(WEIGHTS)

    print(f'\n=== {video_name} ({total} frames) ===')
    print(f'{"frame":>6s} | {"x1":>5s} {"y1":>5s} {"x2":>5s} {"y2":>5s} | {"w":>4s} {"h":>4s} | {"conf":>5s} | у края?')

    step = max(1, total // 15)
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        results = model.predict(frame, verbose=False, conf=0.15, imgsz=640)
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                bw, bh = x2 - x1, y2 - y1
                edge = []
                if y1 < 60: edge.append('TOP')
                if y2 > h - 60: edge.append('BOTTOM')
                if x1 < 60: edge.append('LEFT')
                if x2 > w - 60: edge.append('RIGHT')
                edge_str = ','.join(edge) if edge else 'нет'
                print(f'{i:6d} | {x1:5.0f} {y1:5.0f} {x2:5.0f} {y2:5.0f} | {bw:4.0f} {bh:4.0f} | {conf:5.2f} | {edge_str}')
        else:
            print(f'{i:6d} | --- нет детекции ---')
    cap.release()
