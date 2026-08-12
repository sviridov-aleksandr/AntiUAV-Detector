#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обучение модели YOLOv11m для детекции дронов.
Используем датасет DroneTrainDataset (bounding box детекция).
"""

import os
import sys
from pathlib import Path

from ultralytics import YOLO


def train_detection_model(
    yaml_path: str,
    model_name: str = "yolo11m.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: int = 0,
    workers: int = 8,
    project: str = "runs/detect",
    name: str = "exp",
    pretrained: bool = True,
    optimizer: str = "AdamW",
    lr0: float = 0.01,
    lrf: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    warmup_epochs: float = 3.0,
    warmup_momentum: float = 0.8,
    warmup_bias_lr: float = 0.1,
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
    flipud: float = 0.0,
    fliplr: float = 0.5,
    mosaic: float = 1.0,
    mixup: float = 0.0,
    copy_paste: float = 0.0,
    patience: int = 50,
    save: bool = True,
    save_period: int = -1,
    cache: bool = False,
    resume: bool = False,
    amp: bool = True,
    fraction: float = 1.0,
    seed: int = 0,
    close_mosaic: int = 10,
    verbose: bool = True,
) -> YOLO:
    """
    Обучает модель YOLOv11 для детекции.
    
    Args:
        yaml_path: Путь к YAML конфигурации датасета
        model_name: Название базовой модели (yolov11n/s/m/l/x)
        epochs: Количество эпох обучения
        imgsz: Размер входного изображения
        batch: Размер батча (уменьшить если не хватает VRAM)
        device: ID GPU (0 для первой видеокарты, 'cpu' для процессора)
        workers: Количество рабочих потоков для загрузки данных
        project: Папка для сохранения результатов
        name: Название эксперимента
        pretrained: Использовать предобученные веса
        optimizer: Оптимизатор (SGD, Adam, AdamW)
        lr0: Начальная скорость обучения
        lrf: Конечная скорость обучения (доля от lr0)
        momentum: Моментум для SGD
        weight_decay: Weight decay для оптимизатора
        warmup_epochs: Количество эпох для warmup
        warmup_momentum: Начальный моментум для warmup
        warmup_bias_lr: Начальная скорость обучения для bias
        hsv_h: Изменение оттенка (HSV)
        hsv_s: Изменение насыщенности (HSV)
        hsv_v: Изменение яркости (HSV)
        degrees: Случайные повороты (градусы)
        translate: Случайные смещения
        scale: Случайное масштабирование
        shear: Случайный сдвиг
        perspective: Случайная перспектива
        flipud: Вертикальный флип (вероятность)
        fliplr: Горизонтальный флип (вероятность)
        mosaic: Вероятность использования mosaic аугментации
        mixup: Вероятность использования mixup аугментации
        copy_paste: Вероятность использования copy-paste аугментации
        patience: Количество эпох без улучшения до остановки
        save: Сохранять веса модели
        save_period: Сохранять каждые N эпох (-1 = только последний)
        cache: Кэшировать данные в RAM
        resume: Продолжить обучение с последнего чекпоинта
        amp: Автоматическое смешанное плавание (AMP)
        fraction: Доля данных для обучения (0.0-1.0)
        seed: Случайное семя
        close_mosaic: Отключить mosaic за N эпох до конца
        verbose: Вывод подробной информации
    
    Returns:
        Обученная модель
    """
    print("=" * 70)
    print("ОБУЧЕНИЕ МОДЕЛИ YOLOv11m ДЛЯ ДЕТЕКЦИИ ДРОНОВ")
    print("=" * 70)
    print(f"  Датасет: {yaml_path}")
    print(f"  Модель: {model_name}")
    print(f"  Эпохи: {epochs}")
    print(f"  Размер изображения: {imgsz}")
    print(f"  Размер батча: {batch}")
    print(f"  Устройство: {device}")
    print(f"  Оптимизатор: {optimizer}")
    print(f"  LR: {lr0} -> {lrf}")
    print("=" * 70)
    
    # Загружаем предобученную модель
    print("\nЗагрузка модели...")
    model = YOLO(model_name)
    
    # Обучаем модель
    print("\nНачало обучения...")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=project,
        name=name,
        pretrained=pretrained,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        warmup_momentum=warmup_momentum,
        warmup_bias_lr=warmup_bias_lr,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        perspective=perspective,
        flipud=flipud,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
        patience=patience,
        save=save,
        save_period=save_period,
        cache=cache,
        resume=resume,
        amp=amp,
        fraction=fraction,
        seed=seed,
        close_mosaic=close_mosaic,
        verbose=verbose,
    )
    
    print("\n" + "=" * 70)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"  Результаты: {results.results_dict}")
    print("=" * 70)
    
    return model


def main():
    """Главная функция."""
    # Путь к YAML конфигурации датасета
    yaml_path = "/home/alex/AntiUAV-Detector/prepare_data/output/drone_dataset/dataset.yaml"
    
    if not os.path.exists(yaml_path):
        print(f"Ошибка: YAML конфигурация не найдена: {yaml_path}")
        print("Сначала запустите prepare_data/main.py")
        sys.exit(1)
    
    # Параметры обучения
    train_detection_model(
        yaml_path=yaml_path,
        model_name="yolo11m.pt",  # Можно изменить на yolov11n.pt, yolov11s.pt, yolov11l.pt, yolov11x.pt
        epochs=100,
        imgsz=640,
        batch=16,  # Уменьшить до 8 если не хватает VRAM
        device=0,  # Использовать GPU 0
        workers=8,
        project="/home/alex/AntiUAV-Detector/train/runs",
        name="yolov11m_drone",
        pretrained=True,
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        patience=50,
        save=True,
        save_period=-1,
        cache=False,
        resume=False,
        amp=True,
        fraction=1.0,
        seed=0,
        close_mosaic=10,
        verbose=True,
    )


if __name__ == "__main__":
    main()
