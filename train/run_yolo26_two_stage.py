#!/usr/bin/env python3
"""
Двухстадийное обучение YOLO26L для выбора учителя.

Этап 1: drone_v2 (164K, крупные дроны) — база признаков дрона.
        Старт с COCO-претрейна yolo26l.pt (с нуля, НЕ resume с merged_v1_26L-2).
Этап 2: merged_v1 (268K, +Anti-UAV 22% small + Seraphim 11% tiny) —
        добучение с весов этапа 1. Мелкие цели — сильная сторона YOLO26 (STAL).

Устойчив к отключению электричества: при повторном запуске
продолжает с места остановки (resume last.pt).

Запуск:
    cd /home/alex/AntiUAV-Detector
    nohup python3 train/run_yolo26_two_stage.py > train/two_stage_26L.log 2>&1 &
"""

import os
from pathlib import Path

from ultralytics import YOLO

BASE = '/home/alex/AntiUAV-Detector'
PROJECT = f'{BASE}/train/runs'
EPOCHS_1 = 40          # drone_v2 (крупные дроны, быстрая сходимость)
EPOCHS_2 = 60          # merged_v1 (fine-tune на мелких)
PATIENCE = 20

# Общие параметры — как в merged_v1_L / merged_v1_26L
COMMON = dict(
    imgsz=640,
    batch=16,
    device=0,
    workers=0,
    cache='disk',
    project=PROJECT,
    patience=PATIENCE,
    lr0=0.005,
    warmup_epochs=2.0,
    optimizer='auto',
    mixup=0.0,
    copy_paste=0.0,
    close_mosaic=5,
    scale=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    mosaic=1.0,
)


def last_weights(weights_dir: str):
    """Возвращает last.pt для resume, если обучение шло, иначе None."""
    last = Path(weights_dir) / 'last.pt'
    return str(last) if last.exists() else None


def main():
    # ── ЭТАП 1: drone_v2 ────────────────────────────────────────────
    name1 = 'drone_v2_26L'
    w1 = f'{PROJECT}/{name1}/weights'
    last1 = last_weights(w1)

    if last1:
        print(f'[Этап 1] Найден last.pt — RESUME: {last1}', flush=True)
        model = YOLO(last1)
        model.train(
            data=f'{BASE}/prepare_data/drone_v2/data.yaml',
            epochs=EPOCHS_1,
            name=name1,
            resume=True,
        )
    else:
        print('[Этап 1] Старт с COCO-претрейна yolo26l.pt (drone_v2)', flush=True)
        model = YOLO(f'{BASE}/yolo26l.pt')
        model.train(
            data=f'{BASE}/prepare_data/drone_v2/data.yaml',
            epochs=EPOCHS_1,
            name=name1,
            **COMMON,
        )

    best1 = f'{PROJECT}/{name1}/weights/best.pt'
    if not os.path.exists(best1):
        print(f'[Этап 1] ОШИБКА: {best1} не найден', flush=True)
        return

    print(f'[Этап 1] Завершён. Веса: {best1}', flush=True)

    # ── ЭТАП 2: merged_v1 (старт с best.pt этапа 1) ─────────────────
    name2 = 'merged_v1_26L_2stage'
    w2 = f'{PROJECT}/{name2}/weights'
    last2 = last_weights(w2)

    if last2:
        print(f'[Этап 2] Найден last.pt — RESUME: {last2}', flush=True)
        model = YOLO(last2)
        model.train(
            data=f'{BASE}/prepare_data/merged_v1/data.yaml',
            epochs=EPOCHS_2,
            name=name2,
            resume=True,
        )
    else:
        print(f'[Этап 2] Старт с весов этапа 1: {best1}', flush=True)
        model = YOLO(best1)
        model.train(
            data=f'{BASE}/prepare_data/merged_v1/data.yaml',
            epochs=EPOCHS_2,
            name=name2,
            **COMMON,
        )

    print(f'[Этап 2] Завершён. Веса: {PROJECT}/{name2}/weights/best.pt', flush=True)


if __name__ == '__main__':
    main()
