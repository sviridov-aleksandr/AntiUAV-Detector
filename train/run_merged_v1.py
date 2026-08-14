#!/usr/bin/env python3
"""
Обучение YOLO11L на merged_v1 (drone_v2 + Anti-UAV + Seraphim).

Учитель для последующей дистилляции в M/S/N.
Старт с весов drone_v2-5 (best.pt) — продолжение, не с нуля.

Запуск:
    python3 train/run_merged_v1.py
"""

from ultralytics import YOLO


def main():
    # Старт с лучших весов drone_v2-5 (если закончен) или v2-4
    model = YOLO(
        '/home/alex/AntiUAV-Detector/runs/detect/train/runs/drone_v2-5/weights/best.pt'
    )

    model.train(
        data='/home/alex/AntiUAV-Detector/prepare_data/merged_v1/data.yaml',
        epochs=60,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        cache='disk',
        project='train/runs',
        name='merged_v1_L',
        patience=20,
        lr0=0.005,
        warmup_epochs=2.0,
        optimizer='auto',
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=5,
        # Усиление аугментации для мелких объектов
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
    )


if __name__ == '__main__':
    main()
