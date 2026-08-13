#!/usr/bin/env python3
"""
Объединение трёх датасетов:
1. MyDataSet (85K, 4 класса: People, Drone, Car, Boat)
2. Combined (66K, 1 класс: drone)
Результат: ~150K изображений, 4 класса
"""

import shutil
from pathlib import Path
from tqdm import tqdm

# Пути
MY_DATASET = Path("/home/alex/AntiUAV-Detector/DataSet/MyDataSet")
COMBINED_DATASET = Path("/home/alex/AntiUAV-Detector/prepare_data/combined_dataset")
OUTPUT_DIR = Path("/home/alex/AntiUAV-Detector/prepare_data/all_datasets_v1")

def copy_my_dataset():
    """Копируем MyDataSet (85K изображений)."""
    print("Копирование MyDataSet...")
    
    src_img = MY_DATASET / "images" / "train"
    src_lbl = MY_DATASET / "labels" / "train"
    dst_img = OUTPUT_DIR / "train" / "images"
    dst_lbl = OUTPUT_DIR / "train" / "labels"
    
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    
    # Копируем изображения
    count = 0
    for img_file in tqdm(src_img.glob("*"), desc="Images"):
        if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            shutil.copy2(img_file, dst_img / img_file.name)
            count += 1
    
    # Копируем разметку
    lbl_count = 0
    for lbl_file in tqdm(src_lbl.glob("*"), desc="Labels"):
        if lbl_file.suffix == '.txt':
            shutil.copy2(lbl_file, dst_lbl / lbl_file.name)
            lbl_count += 1
    
    print(f"MyDataSet: {count} изображений, {lbl_count} файлов разметки")

def copy_combined_dataset():
    """Копируем combined_dataset (66K изображений, класс drone=0)."""
    print("Копирование combined_dataset...")
    
    src_img = COMBINED_DATASET / "train" / "images"
    src_lbl = COMBINED_DATASET / "train" / "labels"
    dst_img = OUTPUT_DIR / "train" / "images"
    dst_lbl = OUTPUT_DIR / "train" / "labels"
    
    # Копируем изображения
    count = 0
    for img_file in tqdm(src_img.glob("*"), desc="Combined Images"):
        if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            # Проверяем, нет ли уже такого файла
            if not (dst_img / img_file.name).exists():
                shutil.copy2(img_file, dst_img / img_file.name)
                count += 1
    
    # Копируем разметку (класс drone=0)
    lbl_count = 0
    for lbl_file in tqdm(src_lbl.glob("*"), desc="Combined Labels"):
        if lbl_file.suffix == '.txt':
            if not (dst_lbl / lbl_file.name).exists():
                shutil.copy2(lbl_file, dst_lbl / lbl_file.name)
                lbl_count += 1
    
    print(f"Combined: {count} изображений, {lbl_count} файлов разметки")

def create_data_yaml():
    """Создаём data.yaml для 4 классов."""
    yaml_content = """# Объединённый датасет: MyDataSet + Combined
# Итого: ~150K изображений, 4 класса

path: /home/alex/AntiUAV-Detector/prepare_data/all_datasets_v1
train: /home/alex/AntiUAV-Detector/prepare_data/all_datasets_v1/train/images
val: /home/alex/AntiUAV-Detector/prepare_data/all_datasets_v1/train/images

nc: 4
names: ['People', 'Drone', 'Car', 'Boat']
"""
    with open(OUTPUT_DIR / "data.yaml", 'w') as f:
        f.write(yaml_content)
    
    print("data.yaml создан")

def main():
    print("=" * 60)
    print("ОБЪЕДИНЕНИЕ ДАТАСЕТОВ")
    print("=" * 60)
    
    # Очищаем выходную директорию
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Копируем MyDataSet
    copy_my_dataset()
    
    # Копируем combined_dataset
    copy_combined_dataset()
    
    # Создаём конфиг
    create_data_yaml()
    
    print("=" * 60)
    print("ГОТОВО!")
    print(f"Итого изображений: {len(list((OUTPUT_DIR / 'train' / 'images').glob('*')))}")
    print(f"Итого файлов разметки: {len(list((OUTPUT_DIR / 'train' / 'labels').glob('*')))}")
    print("=" * 60)

if __name__ == "__main__":
    main()
