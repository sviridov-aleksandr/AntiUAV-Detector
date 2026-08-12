#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Конвертация датасета DroneTrainDataset из PASCAL VOC XML в YOLO формат.
Датасет содержит bounding box аннотации для класса 'drone'.
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from tqdm import tqdm


def xml_to_yolo(
    xml_path: str,
    img_width: int,
    img_height: int,
    img_depth: int,
    classes: Dict[str, int],
) -> List[List[float]]:
    """
    Конвертирует одну XML аннотацию в список YOLO bbox.
    
    Args:
        xml_path: Путь к XML файлу
        img_width: Ширина изображения
        img_height: Высота изображения
        img_depth: Каналы изображения
        classes: Словарь {class_name: class_id}
    
    Returns:
        Список [class_id, x_center, y_center, width, height] (нормализованные)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    yolo_boxes = []
    
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in classes:
            continue
        
        class_id = classes[class_name]
        
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        
        # Конвертация в YOLO формат (нормализованные)
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        yolo_boxes.append([class_id, x_center, y_center, width, height])
    
    return yolo_boxes


def convert_drone_dataset(
    dataset_root: str,
    output_root: str,
    classes: Dict[str, int],
) -> Dict[str, int]:
    """
    Конвертирует весь датасет DroneTrainDataset в YOLO формат.
    
    Args:
        dataset_root: Корневая папка датасета
        output_root: Папка для выходных данных
        classes: Словарь классов
    
    Returns:
        Статистика конвертации
    """
    images_dir = os.path.join(dataset_root, "Drone_TrainSet")
    xmls_dir = os.path.join(dataset_root, "Drone_TrainSet_XMLs")
    
    # Создаём выходную структуру
    output_images = os.path.join(output_root, "images", "train")
    output_labels = os.path.join(output_root, "labels", "train")
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)
    
    # Получаем список XML файлов
    xml_files = sorted(Path(xmls_dir).glob("*.xml"))
    
    if not xml_files:
        print(f"Ошибка: XML файлы не найдены в {xmls_dir}")
        return {"converted": 0, "errors": 0}
    
    print(f"Найдено {len(xml_files)} XML файлов")
    print(f"Классы: {classes}")
    
    stats = {"converted": 0, "errors": 0, "no_bbox": 0}
    
    for xml_file in tqdm(xml_files, desc="Конвертация"):
        try:
            # Парсим XML для получения размеров изображения
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            size = root.find("size")
            if size is None:
                stats["errors"] += 1
                continue
            
            img_width = int(size.find("width").text)
            img_height = int(size.find("height").text)
            img_depth = int(size.find("depth").text) if size.find("depth") is not None else 3
            
            filename = root.find("filename").text
            img_path = os.path.join(images_dir, filename)
            
            # Проверяем существование изображения
            if not os.path.exists(img_path):
                stats["errors"] += 1
                continue
            
            # Копируем изображение
            img = cv2.imread(img_path)
            if img is None:
                stats["errors"] += 1
                continue
            
            cv2.imwrite(os.path.join(output_images, filename), img)
            
            # Конвертируем аннотации
            yolo_boxes = xml_to_yolo(
                str(xml_file), img_width, img_height, img_depth, classes
            )
            
            if not yolo_boxes:
                stats["no_bbox"] += 1
                # Создаём пустой лейбл (изображение без объектов)
                label_path = os.path.join(output_labels, xml_file.stem + ".txt")
                with open(label_path, "w") as f:
                    pass
            else:
                # Сохраняем в YOLO формате
                label_path = os.path.join(output_labels, xml_file.stem + ".txt")
                with open(label_path, "w") as f:
                    for box in yolo_boxes:
                        f.write(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
            
            stats["converted"] += 1
            
        except Exception as e:
            print(f"Ошибка при обработке {xml_file}: {e}")
            stats["errors"] += 1
    
    return stats


def create_yaml_config(
    output_root: str,
    classes: Dict[str, int],
    train_images: str,
    val_images: str,
) -> str:
    """
    Создаёт YAML конфигурацию для YOLO обучения.
    
    Args:
        output_root: Корневая папка выходных данных
        classes: Словарь классов
        train_images: Относительный путь к train images
        val_images: Относительный путь к val images
    
    Returns:
        Путь к созданному YAML файлу
    """
    yaml_content = f"""# Конфигурация датасета для YOLOv11
# Сгенерировано автоматически

# Путь к корневой папке датасета
path: {output_root}

# Разделение данных
train: {train_images}
val: {val_images}

# Классы
nc: {len(classes)}
names: {list(classes.keys())}
"""
    
    yaml_path = os.path.join(output_root, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    return yaml_path


def main():
    """Главная функция."""
    dataset_root = "/home/alex/DataSet/DroneTrainDataset"
    output_root = "/home/alex/AntiUAV-Detector/prepare_data/output/drone_dataset"
    
    # Классы для детекции
    classes = {"drone": 0}
    
    print("=" * 60)
    print("Конвертация датасета DroneTrainDataset в YOLO формат")
    print("=" * 60)
    
    stats = convert_drone_dataset(dataset_root, output_root, classes)
    
    print("\n" + "=" * 60)
    print("Статистика конвертации:")
    print(f"  Конвертировано: {stats['converted']}")
    print(f"  Ошибки: {stats['errors']}")
    print(f"  Без bbox: {stats['no_bbox']}")
    print("=" * 60)
    
    # Создаём YAML конфигурацию
    yaml_path = create_yaml_config(
        output_root,
        classes,
        "images/train",
        "images/train",  # Пока нет val, используем train
    )
    print(f"\nYAML конфигурация сохранена: {yaml_path}")


if __name__ == "__main__":
    main()
