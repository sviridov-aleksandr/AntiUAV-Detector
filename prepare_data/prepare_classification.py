#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Подготовка датасета Anti-UAV-RGBT для классификации.
Датасет содержит мультилейбл классификацию сценариев полёта.
Классы: FM, LI, LR, OC, OV, SV, TC, TC-EASY, TC-HARD, TC-MID
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

import cv2
from tqdm import tqdm


def prepare_classification_dataset(
    dataset_root: str,
    output_root: str,
    frame_interval: int = 30,
) -> Dict[str, int]:
    """
    Извлекает кадры из видео и организует для классификации.
    
    Args:
        dataset_root: Корневая папка датасета
        output_root: Папка для выходных данных
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
            img_dir = os.path.join(dataset_root, split, img_key)
            if not os.path.exists(img_dir):
                continue
            
            # Извлекаем кадры из видимого спектра
            visible_video = os.path.join(img_dir, "visible.mp4")
            if not os.path.exists(visible_video):
                continue
            
            # Создаём папку для этого изображения
            img_output_dir = os.path.join(output_split_dir, img_key)
            os.makedirs(img_output_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(visible_video)
            if not cap.isOpened():
                continue
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            extracted = 0
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    output_path = os.path.join(img_output_dir, f"frame_{extracted:06d}.jpg")
                    cv2.imwrite(output_path, frame)
                    
                    # Сохраняем лейблы
                    label_path = os.path.join(img_output_dir, "labels.txt")
                    with open(label_path, "w") as f:
                        f.write(" ".join(labels) + "\n")
                    
                    extracted += 1
                
                frame_idx += 1
            
            cap.release()
            stats[split] += extracted
    
    return stats


def create_classification_yaml(
    output_root: str,
    classes: List[str],
) -> str:
    """
    Создаёт YAML конфигурацию для YOLO классификации.
    """
    yaml_content = f"""# Конфигурация для YOLO классификации
# Anti-UAV-RGBT: мультилейбл классификация сценариев

path: {output_root}
train: train
val: val
test: test

# Классы
nc: {len(classes)}
names: {classes}
"""
    
    yaml_path = os.path.join(output_root, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    return yaml_path


def main():
    """Главная функция."""
    dataset_root = "/home/alex/DataSet/Anti-UAV-RGBT"
    output_root = "/home/alex/AntiUAV-Detector/prepare_data/output/anti_uav_class"
    
    classes = ["FM", "LI", "LR", "OC", "OV", "SV", "TC", "TC-EASY", "TC-HARD", "TC-MID"]
    
    print("=" * 60)
    print("Подготовка датасета Anti-UAV-RGBT для классификации")
    print("=" * 60)
    
    stats = prepare_classification_dataset(dataset_root, output_root, frame_interval=30)
    
    print("\n" + "=" * 60)
    print("Статистика:")
    for split, count in stats.items():
        print(f"  {split}: {count} кадров")
    print(f"  Итого: {sum(stats.values())} кадров")
    print("=" * 60)
    
    # Создаём YAML
    yaml_path = create_classification_yaml(output_root, classes)
    print(f"\nYAML конфигурация: {yaml_path}")


if __name__ == "__main__":
    main()
