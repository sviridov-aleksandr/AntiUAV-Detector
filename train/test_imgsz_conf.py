#!/usr/bin/env python3
"""
Тест проблемных видео с разными imgsz и conf.
Сравнивает: 640/0.15 (базовый), 960/0.10, 1280/0.10.
"""
import os
import sys

import cv2

sys.path.insert(0, 'train')
from ultralytics import YOLO

PROJECT = '/home/alex/AntiUAV-Detector'
VIDEO_DIR = os.path.join(PROJECT, 'video-FPV/Video')
WEIGHTS = os.path.join(PROJECT, 'runs/detect/train/runs/merged_v1_L/weights/best.pt')

PROBLEM_VIDEOS = [
    'v57.mp4', 'v64.mp4', 'v4.mp4', 'v46.mp4', 'v19.mp4', 'v13.mp4',
    'v77.mp4', 'v21.mp4', 'v68.mp4', 'v25.mp4', 'v67.mp4', 'v18.mp4',
]

CONFIGS = [
    (640, 0.15, '640/0.15'),
    (960, 0.10, '960/0.10'),
    (1280, 0.10, '1280/0.10'),
]


def count_predict(video_path, model, imgsz, conf):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hits = 0
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, verbose=False, conf=conf, imgsz=imgsz)
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            hits += 1
    cap.release()
    return hits, total


def main():
    model = YOLO(WEIGHTS)

    # Заголовок
    header = f'{"Видео":12s}'
    for _, _, label in CONFIGS:
        header += f' | {label:>10s}'
    header += ' | прирост'
    print(header, flush=True)
    print('-' * 70, flush=True)

    for video_name in PROBLEM_VIDEOS:
        video_path = os.path.join(VIDEO_DIR, video_name)
        row = f'{video_name:12s}'
        results = []
        for imgsz, conf, label in CONFIGS:
            hits, total = count_predict(video_path, model, imgsz, conf)
            pct = hits / total * 100 if total > 0 else 0
            results.append(pct)
            row += f' | {pct:9.1f}%'

        gain = results[-1] - results[0]
        arrow = f' +{gain:.0f}%' if gain > 0 else f' {gain:.0f}%'
        row += f' |{arrow}'
        print(row, flush=True)


if __name__ == '__main__':
    main()
