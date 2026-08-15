#!/usr/bin/env python3
"""
Пакетный тест детекции YOLO на видео.
Прогоняет все v*.mp4, считает % кадров с детекцией, средний conf, FPS.

Использование:
    python3 train/test_videos.py [model_path] [video_dir]
"""

import cv2
import sys
import time
from pathlib import Path
from ultralytics import YOLO

MODEL = sys.argv[1] if len(sys.argv) > 1 else \
    '/home/alex/AntiUAV-Detector/runs/detect/train/runs/drone_v2-5/weights/best.pt'
VIDEO_DIR = sys.argv[2] if len(sys.argv) > 2 else \
    '/home/alex/AntiUAV-Detector/video-FPV/Video'

OSD_MARGIN = 60
MIN_BBOX_AREA = 500
CONF = 0.3


def test_video(model, video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total_frames = 0
    detected_frames = 0
    confs = []
    fps = 0.0
    fps_time = time.time()
    fps_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        h, w = frame.shape[:2]

        results = model.predict(frame, verbose=False, conf=CONF, imgsz=640)

        detected = False
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls != 0:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                if (y1 < OSD_MARGIN or y2 > h - OSD_MARGIN or
                    x1 < OSD_MARGIN or x2 > w - OSD_MARGIN):
                    continue
                area = (x2 - x1) * (y2 - y1)
                if area < MIN_BBOX_AREA:
                    continue
                detected = True
                confs.append(float(box.conf[0]))

        if detected:
            detected_frames += 1

        fps_count += 1
        if fps_count % 30 == 0:
            fps = 30.0 / (time.time() - fps_time)
            fps_time = time.time()

    cap.release()

    if total_frames == 0:
        return None

    return {
        'frames': total_frames,
        'detected': detected_frames,
        'pct': detected_frames / total_frames * 100,
        'avg_conf': sum(confs) / len(confs) if confs else 0.0,
        'fps': fps,
    }


def main():
    print(f"Модель: {MODEL}")
    print(f"Папка: {VIDEO_DIR}\n")

    model = YOLO(MODEL)
    videos = sorted(Path(VIDEO_DIR).glob('v*.mp4'))

    print(f"{'Видео':<12} {'Кадры':>6} {'Детект':>7} {'%':>7} {'Conf':>7} {'FPS':>7}")
    print("─" * 52)

    results_all = []
    for v in videos:
        r = test_video(model, v)
        if r is None:
            print(f"{v.name:<12}  ОШИБКА")
            continue
        results_all.append((v.name, r))
        print(f"{v.name:<12} {r['frames']:>6} {r['detected']:>7} "
              f"{r['pct']:>6.1f}% {r['avg_conf']:>6.2f} {r['fps']:>6.1f}")

    print("─" * 52)
    if results_all:
        total_frames = sum(r['frames'] for _, r in results_all)
        total_det = sum(r['detected'] for _, r in results_all)
        print(f"{'ИТОГО':<12} {total_frames:>6} {total_det:>7} "
              f"{total_det/total_frames*100:>6.1f}%")


if __name__ == '__main__':
    main()
