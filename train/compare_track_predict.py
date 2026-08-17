#!/usr/bin/env python3
"""
Сравнение track vs predict на проблемных видео.
Показывает, теряет ли дрон трекер (track) или модель (predict).
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


def count_predict(video_path, model):
    """Чистая детекция без трекинга."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hits = 0
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, verbose=False, conf=0.15)
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            hits += 1
    cap.release()
    return hits, total


def count_track(video_path):
    """Трекинг с новой моделью для каждого видео."""
    model = YOLO(WEIGHTS)  # свежая модель — сброс трекера
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hits = 0
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, persist=False, verbose=False, conf=0.15,
                              tracker='bytetrack.yaml')
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            hits += 1
    cap.release()
    return hits, total


def main():
    print(f'{"Видео":12s} | {"track %":>8s} | {"predict %":>9s} | вывод', flush=True)
    print('-' * 60, flush=True)

    model = YOLO(WEIGHTS)

    for video_name in PROBLEM_VIDEOS:
        video_path = os.path.join(VIDEO_DIR, video_name)

        track_hits, total = count_track(video_path)
        pred_hits, _ = count_predict(video_path, model)

        track_pct = track_hits / total * 100 if total > 0 else 0
        predict_pct = pred_hits / total * 100 if total > 0 else 0
        verdict = 'ТРЕКЕР ТЕРЯЕТ' if predict_pct > track_pct + 10 else 'объект улетел/мелкий'

        print(f'{video_name:12s} | {track_pct:7.1f}% | {predict_pct:8.1f}% | {verdict}', flush=True)


if __name__ == '__main__':
    main()