#!/usr/bin/env python3
"""
Анализ проблемных видео: извлекает кадры через равные интервалы,
прогоняет YOLO, сохраняет аннотированные кадры для визуального анализа.

Usage:
    python3 train/analyze_failures.py
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, 'train')
from ultralytics import YOLO

PROJECT = '/home/alex/AntiUAV-Detector'
VIDEO_DIR = os.path.join(PROJECT, 'video-FPV/Video')
WEIGHTS = os.path.join(PROJECT, 'runs/detect/train/runs/merged_v1_L/weights/best.pt')
OUTPUT_DIR = os.path.join(PROJECT, 'analysis_failures')
SAMPLE_FRAMES = 8  # кадров из каждого видео

PROBLEM_VIDEOS = [
    'v57.mp4',  # тепловизор, агродрон
    'v64.mp4',  # шум, запись с экрана
    'v4.mp4',   # тепловизор, мелкий объект
    'v46.mp4',  # тепловизор, помехи
    'v19.mp4',  # самолёт, обычно детектировалось
    'v13.mp4',  # самолёт, как v19
    'v77.mp4',  # тепловизор
    'v21.mp4',  # тепловизор, плохое качество
    'v68.mp4',  # дрон в последней четверти
    'v25.mp4',  # хаки на зелёном фоне
    'v67.mp4',  # тепловизор, низкое качество
    'v18.mp4',  # хаки на зелёном фоне
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = YOLO(WEIGHTS)

    print(f'Анализ {len(PROBLEM_VIDEOS)} видео')
    print(f'Кадров на видео: {SAMPLE_FRAMES}')
    print(f'Выход: {OUTPUT_DIR}/')
    print('=' * 70)

    for video_name in PROBLEM_VIDEOS:
        video_path = os.path.join(VIDEO_DIR, video_name)
        if not os.path.exists(video_path):
            print(f'  [SKIP] {video_name} — не найден')
            continue

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Индексы кадров через равные интервалы
        indices = [int(i * total / SAMPLE_FRAMES) for i in range(SAMPLE_FRAMES)]

        video_stem = os.path.splitext(video_name)[0]
        video_out_dir = os.path.join(OUTPUT_DIR, video_stem)
        os.makedirs(video_out_dir, exist_ok=True)

        print(f'\n--- {video_name} ({total} frames, {fps:.0f} FPS) ---')

        for idx_pos, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f'  frame {frame_idx}: не удалось прочитать')
                continue

            h, w = frame.shape[:2]

            # YOLO детекция (без трекинга, чистый detect)
            results = model.predict(frame, verbose=False, conf=0.15, imgsz=640)

            detections = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append((conf, x1, y1, x2, y2))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, f'{conf:.2f}', (int(x1), int(y1) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Также пробуем conf=0.05 для поиска слабых детекций
            results_low = model.predict(frame, verbose=False, conf=0.05, imgsz=640)
            low_dets = []
            if results_low and results_low[0].boxes is not None:
                for box in results_low[0].boxes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    low_dets.append(conf)
                    # Рисуем красным для слабых (которых не было в conf=0.15)
                    if conf < 0.15:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                      (0, 0, 255), 1)
                        cv2.putText(frame, f'{conf:.2f}', (int(x1), int(y2) + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            n_det = len(detections)
            n_low = len(low_dets)
            best_conf = max(d[0] for d in detections) if detections else 0.0

            # Аннотация
            info = (f'frame {frame_idx}/{total} | det={n_det} (conf>0.15) | '
                    f'low={n_low} (conf>0.05) | best={best_conf:.2f}')
            cv2.putText(frame, info, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            out_path = os.path.join(video_out_dir, f'frame_{idx_pos:02d}_{frame_idx:05d}.jpg')
            cv2.imwrite(out_path, frame)

            print(f'  frame {frame_idx:5d}: det={n_det} low={n_low} best={best_conf:.2f} | {out_path}')

        cap.release()

    print(f'\n{"=" * 70}')
    print(f'Готово! Кадры сохранены в {OUTPUT_DIR}/')
    print(f'Зелёные рамки — conf > 0.15')
    print(f'Красные рамки — conf 0.05-0.15 (слабые детекции)')


if __name__ == '__main__':
    main()
