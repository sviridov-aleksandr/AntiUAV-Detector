#!/usr/bin/env python3
"""
Запуск полного KD-пайплайна: учитель (L) → студент (M).

Этапы:
1. Генерация soft labels учителем на merged_v1 (268K изображений)
2. Создание pseudo-labels (GT + предсказания учителя, conf > 0.5)
3. Обучение студента YOLO11M с transfer weights + pseudo-labels
4. Сравнение mAP и FPS: учитель vs студент

Запуск (после завершения merged_v1_L):
    python3 train/run_distill.py

Параметры по умолчанию:
    Учитель:   merged_v1_L/weights/best.pt (YOLO11L, 25.3M params)
    Студент:   yolo11m.pt (YOLO11M, 20M params)
    Датасет:   merged_v1 (268K train / 26.8K val)
    Эпохи:     60, batch=16, lr0=0.003
    Pseudo-conf: 0.5
"""

import subprocess
import sys
from pathlib import Path

TEACHER = '/home/alex/AntiUAV-Detector/runs/detect/train/runs/merged_v1_L/weights/best.pt'
STUDENT = 'yolo11m.pt'
DATA = '/home/alex/AntiUAV-Detector/prepare_data/merged_v1/data.yaml'
SOFT_DIR = '/home/alex/AntiUAV-Detector/prepare_data/merged_v1/soft_labels/'
NAME = 'distilled_M'
EPOCHS = 60
BATCH = 16


def main():
    teacher_path = Path(TEACHER)
    if not teacher_path.exists():
        print(f"ОШИБКА: Веса учителя не найдены: {teacher_path}")
        print("Сначала завершите обучение merged_v1_L.")
        sys.exit(1)

    print("=" * 60)
    print("KNOWLEDGE DISTILLATION: YOLO11L → YOLO11M")
    print("=" * 60)
    print(f"  Учитель:  {TEACHER}")
    print(f"  Студент:  {STUDENT}")
    print(f"  Датасет:  {DATA}")
    print(f"  Эпохи:    {EPOCHS}, batch={BATCH}")
    print("=" * 60)

    # Полный пайплайн: generate + train
    cmd = [
        sys.executable, 'train/distill_yolo.py',
        '--stage', 'full',
        '--teacher', TEACHER,
        '--student', STUDENT,
        '--data', DATA,
        '--soft-dir', SOFT_DIR,
        '--epochs', str(EPOCHS),
        '--batch', str(BATCH),
        '--pseudo-conf', '0.5',
        '--name', NAME,
    ]
    print(f"\nКоманда: {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd='/home/alex/AntiUAV-Detector')

    # Сравнение после обучения
    student_weights = f'/home/alex/AntiUAV-Detector/train/runs/{NAME}/weights/best.pt'
    if Path(student_weights).exists():
        print("\n=== Сравнение учитель vs студент ===")
        subprocess.run([
            sys.executable, 'train/distill_yolo.py',
            '--stage', 'compare',
            '--teacher', TEACHER,
            '--student-weights', student_weights,
            '--data', DATA,
        ], cwd='/home/alex/AntiUAV-Detector')


if __name__ == '__main__':
    main()
