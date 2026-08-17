#!/usr/bin/env python3
"""Разбор причин расхождения: conf=0.3 vs 0.15, OSD-фильтр."""
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


def count(video_path, model, conf, osd_filter):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_hits = 0
    filtered_hits = 0
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        results = model.predict(frame, verbose=False, conf=conf, imgsz=640)
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            raw_hits += 1
            if osd_filter:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    if not (y1 < 60 or y2 > h - 60 or x1 < 60 or x2 > w - 60):
                        filtered_hits += 1
                        break
    cap.release()
    return raw_hits, filtered_hits, total


def main():
    model = YOLO(WEIGHTS)

    print(f'{"Видео":12s} | {"0.15 raw":>9s} | {"0.15+OSD":>9s} | {"0.30 raw":>9s} | {"0.30+OSD":>9s} | {"batch был":>9s}', flush=True)
    print('-' * 80, flush=True)

    for video_name in PROBLEM_VIDEOS:
        video_path = os.path.join(VIDEO_DIR, video_name)

        r015, f015, total = count(video_path, model, 0.15, True)
        r030, f030, _ = count(video_path, model, 0.30, True)

        print(f'{video_name:12s} | {r015/total*100:8.1f}% | {f015/total*100:8.1f}% | '
              f'{r030/total*100:8.1f}% | {f030/total*100:8.1f}% | '
              f'(conf=0.3+OSD)', flush=True)


if __name__ == '__main__':
    main()
