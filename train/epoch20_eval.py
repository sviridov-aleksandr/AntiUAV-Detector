#!/usr/bin/env python3
"""
Мониторинг обучения merged_v1_L: ждёт 20-ю эпоху, затем запускает
валидацию на val-сете и симуляцию перехвата на тестовых видео.

Запуск (в фоне):
    nohup python3 train/epoch20_eval.py > epoch20_eval.log 2>&1 &
"""

import csv
import os
import subprocess
import sys
import time

from ultralytics import YOLO

PROJECT = '/home/alex/AntiUAV-Detector'
RUN_DIR = os.path.join(PROJECT, 'runs/detect/train/runs/merged_v1_L')
RESULTS_CSV = os.path.join(RUN_DIR, 'results.csv')
WEIGHTS = os.path.join(RUN_DIR, 'weights/best.pt')
TARGET_EPOCH = 20

# Видео для симуляции (разные сценарии)
TEST_VIDEOS = [
    ('video-FPV/Video/v78.mp4', 'pursuit', 4.0, 8.0),
    ('video-FPV/Video/v52.mp4', 'pursuit', 4.0, 8.0),
    ('video-FPV/Video/v78.mp4', 'head_on', 4.0, 8.0),
    ('video-FPV/Video/v78.mp4', 'top_dive', 4.0, 8.0),
]


def get_last_epoch():
    """Читает results.csv, возвращает номер последней завершённой эпохи."""
    if not os.path.exists(RESULTS_CSV):
        return 0
    with open(RESULTS_CSV, newline='') as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return 0
    try:
        return int(rows[-1][0])
    except (ValueError, IndexError):
        return 0


def run_validation():
    """Валидация best.pt на val-сете."""
    print(f'\n{"="*60}')
    print(f'ВАЛИДАЦИЯ best.pt (эпоха {TARGET_EPOCH})')
    print(f'{"="*60}\n')

    model = YOLO(WEIGHTS)
    metrics = model.val(
        data=os.path.join(PROJECT, 'prepare_data/merged_v1/data.yaml'),
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        split='val',
        save_json=False,
        plots=True,
    )

    print(f'\n--- Результаты валидации ---')
    print(f'  Precision: {metrics.box.mp:.4f}')
    print(f'  Recall:    {metrics.box.mr:.4f}')
    print(f'  mAP50:     {metrics.box.map50:.4f}')
    print(f'  mAP50-95:  {metrics.box.map:.4f}')

    return metrics


def run_simulation():
    """Симуляция перехвата на тестовых видео."""
    print(f'\n{"="*60}')
    print('СИМУЛЯЦИЯ ПЕРЕХВАТА')
    print(f'{"="*60}\n')

    for video, strategy, kill_r, intercept_d in TEST_VIDEOS:
        video_path = os.path.join(PROJECT, video)
        if not os.path.exists(video_path):
            print(f'  [SKIP] {video} — файл не найден')
            continue

        print(f'\n--- {video} | strategy={strategy} ---')
        cmd = [
            sys.executable,
            os.path.join(PROJECT, 'train/simulate_intercept.py'),
            video_path,
            strategy,
            str(kill_r),
            str(intercept_d),
            WEIGHTS,
        ]
        result = subprocess.run(
            cmd,
            cwd=PROJECT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        print(result.stdout[-500:] if result.stdout else '(нет вывода)')
        if result.returncode != 0:
            print(f'  [ERROR] returncode={result.returncode}')
            if result.stderr:
                print(f'  stderr: {result.stderr[-300:]}')


def main():
    print(f'Монитор epoch20_eval запущен: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Жду завершения эпохи {TARGET_EPOCH}...')
    print(f'Проверяю results.csv каждые 60 сек.\n')

    while True:
        epoch = get_last_epoch()
        print(f'  [{time.strftime("%H:%M:%S")}] Текущая эпоха: {epoch}')

        if epoch >= TARGET_EPOCH:
            print(f'\n✅ Эпоха {TARGET_EPOCH} достигнута!')
            break

        time.sleep(60)

    # Небольшая пауза, чтобы best.pt точно записался
    time.sleep(5)

    run_validation()
    run_simulation()

    print(f'\n{"="*60}')
    print(f'Готово! {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Лог: {PROJECT}/epoch20_eval.log')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
