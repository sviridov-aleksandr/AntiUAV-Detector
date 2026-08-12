#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Извлечение кадров из видео Anti-UAV-RGBT.
Датасет содержит видео (infrared.mp4, visible.mp4) и JSON лейблы.
Извлекаем кадры и сохраняем в папки train/val/test.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import cv2
from tqdm import tqdm


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    labels: List[str],
    label_key: str,
    frame_interval: int = 30,  # извлекать каждый N-й кадр
) -> int:
    """
    Извлекает кадры из видео и сохраняет в папку.
    
    Args:
        video_path: Путь к видеофайлу
        output_dir: Папка для сохранения кадров
        labels: Список лейблов для этого видео
        label_key: Ключ в JSON для лейблов
        frame_interval: Интервал извлечения кадров (каждый N-й)
    
    Returns:
        Количество извлечённых кадров
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {video_path}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Видео: {video_path}")
    print(f"  Всего кадров: {total_frames}, FPS: {fps:.1f}")
    print(f"  Лейблы: {labels}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    extracted = 0
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Извлекаем каждый N-й кадр для уменьшения объёма данных
        if frame_idx % frame_interval == 0:
            output_path = os.path.join(output_dir, f"{label_key}_{extracted:06d}.jpg")
            cv2.imwrite(output_path, frame)
            extracted += 1
        
        frame_idx += 1
    
    cap.release()
    print(f"  Извлечено кадров: {extracted}")
    return extracted


def prepare_anti_uav_dataset(
    dataset_root: str,
    output_root: str,
    frame_interval: int = 30,
) -> Dict[str, int]:
    """
    Подготавливает датасет Anti-UAV-RGBT для обучения.
    
    Args:
        dataset_root: Корневая папка датасета
        output_root: Корневая папка для выходных данных
        frame_interval: Интервал извлечения кадров
    
    Returns:
        Статистика по извлечённым кадрам
    """
    stats = {"train": 0, "val": 0, "test": 0}
    
    # Загружаем JSON лейблы
    label_files = {
        "train": os.path.join(dataset_root, "label_new", "train.json"),
        "val": os.path.join(dataset_root, "label_new", "val.json"),
        "test": os.path.join(dataset_root, "label_new", "test.json"),
    }
    
    for split, label_file in label_files.items():
        if not os.path.exists(label_file):
            print(f"Предупреждение: файл лейблов не найден {label_file}")
            continue
        
        with open(label_file, "r") as f:
            labels_data = json.load(f)
        
        output_split_dir = os.path.join(output_root, split)
        os.makedirs(output_split_dir, exist_ok=True)
        
        print(f"\nОбработка сплита '{split}': {len(labels_data)} изображений")
        
        for img_key, labels in tqdm(labels_data.items(), desc=f"{split}"):
            # Ищем видеофайлы в папке изображения
            img_dir = os.path.join(dataset_root, split, img_key)
            if not os.path.exists(img_dir):
                continue
            
            # Извлекаем кадры из видимого спектра
            visible_video = os.path.join(img_dir, "visible.mp4")
            if os.path.exists(visible_video):
                count = extract_frames_from_video(
                    visible_video,
                    output_split_dir,
                    labels,
                    img_key,
                    frame_interval,
                )
                stats[split] += count
                
                # Также извлекаем из инфракрасного спектра (опционально)
                ir_video = os.path.join(img_dir, "infrared.mp4")
                if os.path.exists(ir_video):
                    ir_output = os.path.join(output_split_dir, f"{img_key}_ir")
                    count_ir = extract_frames_from_video(
                        ir_video,
                        ir_output,
                        labels,
                        f"{img_key}_ir",
                        frame_interval,
                    )
                    stats[split] += count_ir
    
    return stats


def main():
    """Главная функция."""
    dataset_root = "/home/alex/DataSet/Anti-UAV-RGBT"
    output_root = "/home/alex/AntiUAV-Detector/prepare_data/output/anti_uav"
    
    print("=" * 60)
    print("Подготовка датасета Anti-UAV-RGBT")
    print("=" * 60)
    
    stats = prepare_anti_uav_dataset(
        dataset_root,
        output_root,
        frame_interval=30,  # каждый 30-й кадр
    )
    
    print("\n" + "=" * 60)
    print("Статистика:")
    for split, count in stats.items():
        print(f"  {split}: {count} кадров")
    print(f"  Итого: {sum(stats.values())} кадров")
    print("=" * 60)


if __name__ == "__main__":
    main()
