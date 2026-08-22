#!/usr/bin/env python3
"""
Инференс с адаптивным preprocessing для сложных условий съёмки:
- CLAHE (адаптивная эквализация гистограммы) при низкой яркости/контрасте
- ByteTrack с persist=True для удержания между кадрами
- Пониженный conf-порог (0.25) для сложных условий
- Фильтр ложных срабатываний (OSD + размер bbox)

Использование:
    python3 train/adaptive_detect.py [video_path] [model_path]
"""

import cv2
import sys
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO

sys.path.insert(0, 'train')
from osd_filter import is_osd_false_positive

VIDEO = sys.argv[1] if len(sys.argv) > 1 else \
    '/home/alex/AntiUAV-Detector/video-FPV/Video/555.MP4'
MODEL = sys.argv[2] if len(sys.argv) > 2 else \
    '/home/alex/AntiUAV-Detector/train/runs/merged_v1_26L_2c/weights/best.pt'

CONF = 0.25
OSD_MARGIN = 60
MIN_BBOX_AREA = 400
CLAHE_CLIP = 4.0
CLAHE_GRID = (8, 8)
BRIGHTNESS_LOW = 80
BRIGHTNESS_HIGH = 200
CONTRAST_LOW = 30


def apply_clahe(frame):
    """Адаптивная эквализация гистограммы на LAB-канале L."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def frame_needs_clahe(frame):
    """Определяет, нужен ли CLAHE: низкая яркость, засветка или низкий контраст."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_b = gray.mean()
    std_b = gray.std()
    # Засветка (max > 240) или низкая яркость (< 80) или низкий контраст (< 30)
    return mean_b < BRIGHTNESS_LOW or gray.max() > 240 or std_b < CONTRAST_LOW


def main():
    print(f"Видео: {VIDEO}")
    print(f"Модель: {MODEL}")
    print(f"Conf: {CONF} | CLAHE: clip={CLAHE_CLIP}, grid={CLAHE_GRID}")

    model = YOLO(MODEL)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {VIDEO}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = str(Path(VIDEO).with_name(Path(VIDEO).stem + '_adaptive.mp4'))
    writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

    frame_count = 0
    detected_frames = 0
    clahe_frames = 0
    fps_time = time.time()
    cur_fps = 0.0
    track_id = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w = frame.shape[:2]
        display = frame.copy()

        # Адаптивный preprocessing
        needs_clahe = frame_needs_clahe(frame)
        if needs_clahe:
            proc_frame = apply_clahe(frame)
            clahe_frames += 1
        else:
            proc_frame = frame

        # Трекинг с persist=True (удержание между кадрами)
        results = model.track(proc_frame, persist=True, verbose=False,
                              conf=CONF, tracker='bytetrack.yaml', imgsz=640)

        detections = 0
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls != 0:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                # ID трекинга (если есть)
                tid = int(box.id[0]) if box.id is not None else -1

                if is_osd_false_positive(x1, y1, x2, y2, w, h, margin=OSD_MARGIN):
                    continue

                area = (x2 - x1) * (y2 - y1)
                if area < MIN_BBOX_AREA:
                    continue

                detections += 1
                if tid > 0:
                    track_id = tid

                # Рамка: зелёная при YOLO, жёлтая при трекинге
                color = (0, 255, 0) if conf > 0.5 else (0, 200, 255)
                cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)),
                              color, 2)
                label = f"DRONE {conf:.2f}"
                if tid > 0:
                    label += f" ID:{tid}"
                cv2.putText(display, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if detections > 0:
            detected_frames += 1

        # FPS
        if frame_count % 10 == 0:
            cur_fps = 10.0 / (time.time() - fps_time)
            fps_time = time.time()

        # OSD
        clahe_str = "CLAHE" if needs_clahe else "RAW"
        status = f"FPS:{cur_fps:.0f} | Det:{detections} | {clahe_str} | {w}x{h}"
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if detections > 0 else (0, 0, 255), 2)

        writer.write(display)

        if frame_count % 100 == 0:
            print(f"  {frame_count}/{total} | det:{detections} | "
                  f"clahe:{clahe_str} | track_id:{track_id}", flush=True)

    cap.release()
    writer.release()

    pct = detected_frames / frame_count * 100 if frame_count else 0
    clahe_pct = clahe_frames / frame_count * 100 if frame_count else 0
    print(f"\nОбработано: {frame_count} кадров")
    print(f"Детекций: {detected_frames} ({pct:.1f}%)")
    print(f"CLAHE применён: {clahe_frames} ({clahe_pct:.1f}%)")
    print(f"Результат: {output}")


if __name__ == '__main__':
    main()
