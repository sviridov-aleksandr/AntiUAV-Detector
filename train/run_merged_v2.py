#!/usr/bin/env python3
"""
Обучение YOLO11L на merged_v2 (drone_v2 + Anti-UAV visible + Anti-UAV IR + Seraphim).

Старт с весов merged_v1_L (учитель v1) — продолжение, не с нуля.
Добавляет IR-кадры (29.7K) для устойчивости к тепловизионным изображениям.

Запуск:
    python3 train/run_merged_v2.py
"""

from ultralytics import YOLO


def main():
    # Старт с лучших весов merged_v1_L (если закончен) или drone_v2-5
    model = YOLO(
        '/home/alex/AntiUAV-Detector/runs/detect/train/runs/merged_v1_L/weights/best.pt'
    )

    model.train(
        data='/home/alex/AntiUAV-Detector/prepare_data/merged_v2/data.yaml',
        epochs=60,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        cache='disk',
        project='train/runs',
        name='merged_v2_L',
        patience=20,
        lr0=0.005,
        warmup_epochs=2.0,
        optimizer='auto',
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=5,
        # Усиление аугментации для мелких и IR-объектов
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
    )


if __name__ == '__main__':
    main()
