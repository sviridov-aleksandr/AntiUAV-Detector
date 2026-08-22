#!/usr/bin/env python3
"""
Детекция дрона на видеофайле с визуализацией bounding boxes.
Читает v61.mp4, прогоняет через YOLO, рисует рамки и сохраняет
результат в video-FPV/Video/v61_detected.mp4.

Использование:
    python3 train/detect_video_file.py [video_path] [model_path] [output_path]
"""

import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, 'train')
from osd_filter import is_osd_false_positive

# Параметры по умолчанию
VIDEO = sys.argv[1] if len(sys.argv) > 1 else \
    '/home/alex/AntiUAV-Detector/video-FPV/Video/v61.mp4'
MODEL = sys.argv[2] if len(sys.argv) > 2 else \
    '/home/alex/AntiUAV-Detector/train/runs/merged_v1_26L_2c/weights/best.pt'
OUTPUT = sys.argv[3] if len(sys.argv) > 3 else \
    '/home/alex/AntiUAV-Detector/video-FPV/Video/v61_detected.mp4'

OSD_MARGIN = 60
MIN_BBOX_AREA = 500
CONF = 0.3


def main():
    print(f"Видео: {VIDEO}")
    print(f"Модель: {MODEL}")
    print(f"Результат: {OUTPUT}")

    model = YOLO(MODEL)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {VIDEO}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Видеописатель для сохранения результата
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT, fourcc, fps, (width, height))

    frame_count = 0
    detected_frames = 0
    fps_time = time.time()
    cur_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w = frame.shape[:2]

        results = model.predict(frame, verbose=False, conf=CONF, imgsz=640)

        detections = 0
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls != 0:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                if is_osd_false_positive(x1, y1, x2, y2, w, h, margin=OSD_MARGIN):
                    continue

                area = (x2 - x1) * (y2 - y1)
                if area < MIN_BBOX_AREA:
                    continue

                detections += 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                cv2.putText(frame, f"DRONE {conf:.2f}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if detections > 0:
            detected_frames += 1

        # FPS
        if frame_count % 10 == 0:
            cur_fps = 10.0 / (time.time() - fps_time)
            fps_time = time.time()

        # OSD
        status = f"FPS: {cur_fps:.1f} | Detections: {detections} | {w}x{h}"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if detections > 0 else (0, 0, 255), 2)

        writer.write(frame)

        if frame_count % 30 == 0:
            print(f"  {frame_count}/{total} кадров | детекций: {detections}",
                  flush=True)

    cap.release()
    writer.release()

    pct = detected_frames / frame_count * 100 if frame_count else 0
    print(f"\nОбработано кадров: {frame_count}")
    print(f"Кадров с детекцией: {detected_frames} ({pct:.1f}%)")
    print(f"Результат сохранён: {OUTPUT}")


if __name__ == '__main__':
    main()