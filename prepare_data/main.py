#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный скрипт подготовки данных.
Объединяет подготовку Anti-UAV-RGBT и DroneTrainDataset.
"""

import os
import sys
from pathlib import Path

# Добавляем путь к prepare_data
sys.path.insert(0, str(Path(__file__).parent))

from extract_frames import prepare_anti_uav_dataset
from convert_drone_dataset import convert_drone_dataset, create_yaml_config
from prepare_classification import prepare_classification_dataset, create_classification_yaml


def main():
    """Главная функция подготовки данных."""
    base_dir = Path(__file__).parent / "output"
    base_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("ПОДГОТОВКА ДАННЫХ ДЛЯ ANTI-UAV DETECTOR")
    print("=" * 70)
    
    # 1. Anti-UAV-RGBT для классификации
    print("\n[1/3] Подготовка Anti-UAV-RGBT (классификация)...")
    print("-" * 70)
    anti_uav_class_root = base_dir / "anti_uav_class"
    stats_class = prepare_classification_dataset(
        "/home/alex/DataSet/Anti-UAV-RGBT",
        str(anti_uav_class_root),
        frame_interval=30,
    )
    
    classes = ["FM", "LI", "LR", "OC", "OV", "SV", "TC", "TC-EASY", "TC-HARD", "TC-MID"]
    yaml_class = create_classification_yaml(str(anti_uav_class_root), classes)
    
    print(f"\n  Итого кадров: {sum(stats_class.values())}")
    print(f"  YAML: {yaml_class}")
    
    # 2. DroneTrainDataset для детекции
    print("\n[2/3] Подготовка DroneTrainDataset (детекция)...")
    print("-" * 70)
    drone_root = base_dir / "drone_dataset"
    classes_drone = {"drone": 0}
    stats_drone = convert_drone_dataset(
        "/home/alex/DataSet/DroneTrainDataset",
        str(drone_root),
        classes_drone,
    )
    
    yaml_drone = create_yaml_config(
        str(drone_root),
        classes_drone,
        "images/train",
        "images/train",
    )
    
    print(f"\n  Конвертировано: {stats_drone['converted']}")
    print(f"  YAML: {yaml_drone}")
    
    # 3. Итоговая сводка
    print("\n" + "=" * 70)
    print("ИТОГОВАЯ СВОДКА:")
    print("=" * 70)
    print(f"  Anti-UAV-RGBT (классификация): {sum(stats_class.values())} кадров")
    print(f"  DroneTrainDataset (детекция): {stats_drone['converted']} изображений")
    print(f"\n  Готово! Можно переходить к обучению.")
    print("=" * 70)


if __name__ == "__main__":
    main()
