#!/usr/bin/env python3
"""
Скрипт для объединения двух датасетов:
1. Оригинальный (51К изображений, 1 класс: drone)
2. Svanström (15К изображений, 4 класса: drone, helicopter, airplane, bird)
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from mcos_decoder import load_groundtruth
from tqdm import tqdm

# Пути
ORIGINAL_DATASET = Path("/home/alex/AntiUAV-Detector/prepare_data/output/drone_dataset")
SVANSTROM_DATASET = Path("/home/alex/AntiUAV-Detector/prepare_data/drone_detection_thesis")
OUTPUT_DIR = Path("/home/alex/AntiUAV-Detector/prepare_data/combined_dataset")

# Классы
CLASS_NAMES = ['drone', 'helicopter', 'airplane', 'bird']


def copy_original_dataset():
    """Копирует оригинальный датасет (51К изображений, класс drone)."""
    print("Копирование оригинального датасета...")
    
    # Структура: images/train/ и labels/train/
    src_img_train = ORIGINAL_DATASET / "images" / "train"
    src_lbl_train = ORIGINAL_DATASET / "labels" / "train"
    
    dst_img_train = OUTPUT_DIR / "train" / "images"
    dst_lbl_train = OUTPUT_DIR / "train" / "labels"
    
    dst_img_train.mkdir(parents=True, exist_ok=True)
    dst_lbl_train.mkdir(parents=True, exist_ok=True)
    
    # Копируем изображения
    count = 0
    for img_file in tqdm(src_img_train.glob("*.jpg"), desc="train images"):
        shutil.copy2(img_file, dst_img_train / img_file.name)
        count += 1
    
    # Копируем разметку (все классы уже 0 - drone)
    lbl_count = 0
    for lbl_file in tqdm(src_lbl_train.glob("*.txt"), desc="train labels"):
        shutil.copy2(lbl_file, dst_lbl_train / lbl_file.name)
        lbl_count += 1
    
    print(f"Скопировано {count} изображений и {lbl_count} файлов разметки")


def convert_svanstrom_dataset():
    """Конвертирует датасет Svanström в единый формат."""
    print("Конвертация датасета Svanström...")
    
    # Классы Svanström
    CLASS_MAP = {
        "DRONE": 0,
        "HELICOPTER": 1,
        "AIRPLANE": 2,
        "BIRD": 3,
    }
    
    # Разделение train/val (80/20)
    video_files = []
    for video_file in (SVANSTROM_DATASET / "train" / "visible" / "images").parent.parent.glob("*/images/*.jpg"):
        # Собираем уникальные видео по префиксу
        pass
    
    # Пересобираем из оригинальных видео
    svanstrom_data = Path("/home/alex/DataSet/Drone-detection-dataset-1.0.0/Data")
    video_files = []
    for video_file in (svanstrom_data / "Video_V").glob("*.mp4"):
        labels_file = video_file.with_name(video_file.stem + "_LABELS.mat")
        if labels_file.exists():
            video_files.append((video_file, labels_file, 'visible'))
    
    for video_file in (svanstrom_data / "Video_IR").glob("*.mp4"):
        labels_file = video_file.with_name(video_file.stem + "_LABELS.mat")
        if labels_file.exists():
            video_files.append((video_file, labels_file, 'thermal'))
    
    np.random.seed(42)
    np.random.shuffle(video_files)
    split_idx = int(len(video_files) * 0.8)
    train_files = video_files[:split_idx]
    val_files = video_files[split_idx:]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    for split, files in [('train', train_files), ('val', val_files)]:
        dst_img = OUTPUT_DIR / split / 'images'
        dst_lbl = OUTPUT_DIR / split / 'labels'
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        
        for video_file, labels_file, modality in tqdm(files, desc=f"Конвертация {split}"):
            cap = cv2.VideoCapture(str(video_file))
            if not cap.isOpened():
                continue
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            bboxes = load_groundtruth(str(labels_file))
            
            video_name = video_file.stem
            frame_interval = max(1, int(fps / 5))
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    img_name = f"{video_name}_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(dst_img / img_name), frame)
                    
                    # Собираем все боксы для этого кадра
                    label_lines = []
                    for cls_name, cls_id in CLASS_MAP.items():
                        # Ищем боксы для этого класса
                        bbox = bboxes[frame_idx] if frame_idx < len(bboxes) else None
                        if bbox is not None:
                            h, w = frame.shape[:2]
                            x, y, bw, bh = bbox
                            if bw > 10 and bh > 10:
                                x_center = (x + bw / 2) / w
                                y_center = (y + bh / 2) / h
                                bw_norm = bw / w
                                bh_norm = bh / h
                                label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}")
                    
                    if label_lines:
                        with open(dst_lbl / f"{video_name}_{frame_idx:06d}.txt", 'w') as f:
                            f.write('\n'.join(label_lines))
                
                frame_idx += 1
            
            cap.release()
    
    print("Датасет Svanström конвертирован.")


def create_data_yaml():
    """Создаёт data.yaml для объединённого датасета."""
    yaml_content = f"""# Объединённый датасет
# Оригинальный (51К) + Svanström (15К)

path: {OUTPUT_DIR}
train: {OUTPUT_DIR}/train/images
val: {OUTPUT_DIR}/val/images

nc: 4
names: {CLASS_NAMES}
"""
    with open(OUTPUT_DIR / "data.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"Конфигурация сохранена в {OUTPUT_DIR}/data.yaml")


def main():
    print("Начинаем объединение датасетов...")
    
    # Очищаем выходную директорию
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Копируем оригинальный датасет
    copy_original_dataset()
    
    # Конвертируем Svanström
    convert_svanstrom_dataset()
    
    # Создаём конфиг
    create_data_yaml()
    
    print("Объединение завершено!")


if __name__ == "__main__":
    main()
