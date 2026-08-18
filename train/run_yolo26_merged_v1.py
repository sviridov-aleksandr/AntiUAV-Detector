#!/usr/bin/env python3
"""
Обучение YOLO26L на merged_v1 (drone_v2 + Anti-UAV + Seraphim).

YOLO26: NMS-free end-to-end, STAL (Small-Target-Aware Label Assignment),
Progressive Loss, DFL-free head, MuSGD оптимизатор.

Старт с COCO-претрейна yolo26l.pt — transfer learning, не с нуля.

Запуск:
    python3 train/run_yolo26_merged_v1.py
"""

from ultralytics import YOLO


def main():
    # COCO-претрейн YOLO26L
    model = YOLO('yolo26l.pt')

    model.train(
        data='/home/alex/AntiUAV-Detector/prepare_data/merged_v1/data.yaml',
        epochs=60,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        cache='disk',
        project='train/runs',
        name='merged_v1_26L',
        patience=20,
        lr0=0.005,
        warmup_epochs=2.0,
        optimizer='auto',
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=5,
        # Аугментация аналогично merged_v1_L
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
    )


if __name__ == '__main__':
    main()
