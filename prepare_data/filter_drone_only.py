#!/usr/bin/env python3
"""
Скрипт для фильтрации датасета: оставляем только класс 'drone' (0).
"""

from pathlib import Path
from tqdm import tqdm
import shutil

COMBINED_DIR = Path("/home/alex/AntiUAV-Detector/prepare_data/combined_dataset")
OUTPUT_DIR = Path("/home/alex/AntiUAV-Detector/prepare_data/drone_only_dataset")

def filter_labels():
    print("Начинаем фильтрацию датасета...")
    
    # Создаем структуру выходной папки
    for split in ['train', 'val']:
        (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Копируем изображения и фильтруем метки
    for split in ['train', 'val']:
        src_img = COMBINED_DIR / split / 'images'
        src_lbl = COMBINED_DIR / split / 'labels'
        dst_img = OUTPUT_DIR / split / 'images'
        dst_lbl = OUTPUT_DIR / split / 'labels'
        
        print(f"Обработка сплита: {split}")
        
        # Сначала копируем все изображения
        for img_file in tqdm(src_img.glob("*.jpg"), desc=f"Копирование {split} images"):
            shutil.copy2(img_file, dst_img / img_file.name)
        
        # Затем фильтруем метки
        for lbl_file in tqdm(src_lbl.glob("*.txt"), desc=f"Фильтрация {split} labels"):
            # Читаем файл
            with open(lbl_file, 'r') as f:
                lines = f.readlines()
            
            # Оставляем только строки с классом 0 (drone)
            drone_lines = [line for line in lines if line.startswith('0 ')]
            
            if drone_lines:
                # Записываем отфильтрованные метки
                with open(dst_lbl / lbl_file.name, 'w') as f:
                    f.writelines(drone_lines)
            else:
                # Если в файле нет дронов, удаляем соответствующее изображение
                img_name = lbl_file.name.replace('.txt', '.jpg')
                if (dst_img / img_name).exists():
                    (dst_img / img_name).unlink()

    # Обновляем data.yaml
    yaml_content = f"""# Датасет только для класса 'drone'
path: {OUTPUT_DIR}
train: {OUTPUT_DIR}/train/images
val: {OUTPUT_DIR}/val/images

nc: 1
names: ['drone']
"""
    with open(OUTPUT_DIR / "data.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"Готово! Датасет сохранен в {OUTPUT_DIR}")
    print("data.yaml обновлен для 1 класса.")

if __name__ == "__main__":
    filter_labels()
