#!/usr/bin/env python3
"""
Live-детекция дронов с USB-камеры (ноутбук) через YOLO11.
Показывает кадры с bounding boxes в реальном времени.

Использование:
    python3 train/live_detect.py [device] [model_path]
"""

import cv2
import signal
import sys
import time
from pathlib import Path
from ultralytics import YOLO

# Флаг для корректного завершения по сигналу (Ctrl+C, kill)
_running = True


def _handle_signal(signum, frame):
    global _running
    _running = False


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# Параметры по умолчанию
DEVICE = sys.argv[1] if len(sys.argv) > 1 else '/dev/video0'
MODEL = sys.argv[2] if len(sys.argv) > 2 else \
    '/home/alex/AntiUAV-Detector/runs/detect/train/runs/drone_v2-5/weights/best.pt'

# OSD-фильтр (отсек ложных срабатываний в краях кадра)
OSD_MARGIN = 60
MIN_BBOX_AREA = 500


def main():
    print(f"Модель: {MODEL}")
    print(f"Камера: {DEVICE}")

    model = YOLO(MODEL)

    # Пробуем открыть камеру (индекс 0 или путь)
    cap = None
    for dev in [DEVICE, 0]:
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            # Прогрев: читаем несколько кадров
            for _ in range(5):
                ret, _ = cap.read()
                if ret:
                    break
            if ret:
                print(f"Камера открыта: {dev}")
                break
        cap.release()
        cap = None

    if cap is None:
        print(f"Не удалось открыть камеру: {DEVICE}")
        sys.exit(1)

    # Запрашиваем 1280x720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Нажмите 'q' для выхода")

    frame_count = 0
    fps_time = time.time()
    fps = 0.0
    window_start = time.time()  # время создания окна

    while _running:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр")
            break

        h, w = frame.shape[:2]

        # YOLO детекция
        results = model.predict(frame, verbose=False, conf=0.3, imgsz=640)

        detections = 0
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls != 0:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                # OSD-фильтр
                if (y1 < OSD_MARGIN or y2 > h - OSD_MARGIN or
                    x1 < OSD_MARGIN or x2 > w - OSD_MARGIN):
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

        # FPS
        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10.0 / (time.time() - fps_time)
            fps_time = time.time()

        # OSD
        status = f"FPS: {fps:.1f} | Detections: {detections} | {w}x{h}"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if detections > 0 else (0, 0, 255), 2)

        cv2.imshow('Anti-UAV Detector (USB Camera)', frame)

        # Выход по 'q', закрытию окна или сигналу.
        # Проверка видимости окна — только после 2 сек (окно ещё создаётся)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or not _running:
            break
        if time.time() - window_start > 2.0:
            try:
                if cv2.getWindowProperty('Anti-UAV Detector (USB Camera)',
                                         cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Завершено")


if __name__ == '__main__':
    main()
