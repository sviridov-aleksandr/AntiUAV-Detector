#!/usr/bin/env python3
"""
Скрипт для конвертации Drone Detection Dataset (Svanström, 2020) в формат YOLO.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from mcos_decoder import load_groundtruth
from tqdm import tqdm

# Пути
DATASET_DIR = Path("/home/alex/DataSet/Drone-detection-dataset-1.0.0/Data")
OUTPUT_DIR = Path("/home/alex/AntiUAV-Detector/prepare_data/drone_detection_thesis")

# Классы
CLASS_MAP = {
    "DRONE": 0,
    "HELICOPTER": 1,
    "AIRPLANE": 2,
    "BIRD": 3,
}

TRAIN_RATIO = 0.8


def extract_frames_and_labels(video_path, labels_path, output_img_dir, output_label_dir, prefix):
    """Извлекает кадры и конвертирует разметку."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    bboxes = load_groundtruth(str(labels_path))
    
    # Уникальный префикс из имени видео
    video_name = video_path.stem  # например V_AIRPLANE_001
    extracted_count = 0
    frame_interval = max(1, int(fps / 5))
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            # Уникальное имя файла с префиксом видео
            img_name = f"{video_name}_{frame_idx:06d}.jpg"
            cv2.imwrite(str(output_img_dir / img_name), frame)
            
            bbox = bboxes[frame_idx] if frame_idx < len(bboxes) else None
            if bbox is not None:
                h, w = frame.shape[:2]
                x, y, bw, bh = bbox
                if bw > 10 and bh > 10:
                    x_center = (x + bw / 2) / w
                    y_center = (y + bh / 2) / h
                    bw_norm = bw / w
                    bh_norm = bh / h
                    label_file = output_label_dir / f"{video_name}_{frame_idx:06d}.txt"
                    with open(label_file, 'w') as f:
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}")
            
            extracted_count += 1
        
        frame_idx += 1
    
    cap.release()
    return extracted_count


def convert_dataset():
    print("Начинаем конвертацию...")
    
    for split in ['train', 'val']:
        for modality in ['visible', 'thermal']:
            (OUTPUT_DIR / split / modality / 'images').mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / split / modality / 'labels').mkdir(parents=True, exist_ok=True)
    
    video_files = []
    for video_file in (DATASET_DIR / "Video_V").glob("*.mp4"):
        labels_file = video_file.with_name(video_file.stem + "_LABELS.mat")
        if labels_file.exists():
            video_files.append((video_file, labels_file, 'visible', 'V'))
    
    for video_file in (DATASET_DIR / "Video_IR").glob("*.mp4"):
        labels_file = video_file.with_name(video_file.stem + "_LABELS.mat")
        if labels_file.exists():
            video_files.append((video_file, labels_file, 'thermal', 'IR'))
    
    print(f"Найдено {len(video_files)} видеофайлов")
    
    np.random.seed(42)
    np.random.shuffle(video_files)
    split_idx = int(len(video_files) * TRAIN_RATIO)
    train_files = video_files[:split_idx]
    val_files = video_files[split_idx:]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    total_frames = 0
    for video_file, labels_file, modality, prefix in tqdm(train_files + val_files, desc="Конвертация"):
        split = 'train' if video_file in [v[0] for v in train_files] else 'val'
        output_img_dir = OUTPUT_DIR / split / modality / 'images'
        output_label_dir = OUTPUT_DIR / split / modality / 'labels'
        
        frames = extract_frames_and_labels(video_file, labels_file, output_img_dir, output_label_dir, prefix)
        total_frames += frames
    
    print(f"Всего извлечено кадров: {total_frames}")
    
    yaml_content = f"""path: {OUTPUT_DIR}
train: {OUTPUT_DIR}/train/visible/images
val: {OUTPUT_DIR}/val/visible/images
nc: 4
names: ['drone', 'helicopter', 'airplane', 'bird']
"""
    with open(OUTPUT_DIR / "data.yaml", 'w') as f:
        f.write(yaml_content)
    
    print("Готово!")


if __name__ == "__main__":
    convert_dataset()