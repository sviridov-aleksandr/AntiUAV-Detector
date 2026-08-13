#!/usr/bin/env python3
"""
Сборка чистого датасета drone_v2 (1 класс: Drone).

Источники:
1. MyDataSet (DataSet/MyDataSet): Drone=1 -> 0. People/Car/Boat - изображения как негативы (без лейблов).
2. combined (prepare_data/combined_dataset): drone=0 -> 0. helicopter/airplane/bird - изображения как негативы.
   train: 78,776 изобр. / val: 13,666 изобр.

Результат: prepare_data/drone_v2/{train,val}/{images,labels} + data.yaml
"""

from pathlib import Path
from collections import Counter
import shutil

MY_DATASET = Path("/home/alex/AntiUAV-Detector/DataSet/MyDataSet")
COMBINED = Path("/home/alex/AntiUAV-Detector/prepare_data/combined_dataset")
OUTPUT = Path("/home/alex/AntiUAV-Detector/prepare_data/drone_v2")


def copy_with_remap(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir,
                    keep_classes, class_map, stats):
    """Копирует изображения; лейблы перемаппит, не-целевые классы убирает.

    Если у изображения нет лейбла с целевым классом — оно копируется как негатив.
    """
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Сопоставление лейблов: имя (без расширения) -> путь к лейблу
    lbl_map = {}
    if src_lbl_dir.exists():
        for lbl in src_lbl_dir.glob("*.txt"):
            lbl_map[lbl.stem] = lbl

    for img in sorted(src_img_dir.iterdir()):
        if not img.is_file():
            continue
        ext = img.suffix.lower()
        if ext not in ('.jpg', '.jpeg', '.png'):
            continue

        # Копируем изображение
        dst_img = dst_img_dir / img.name
        shutil.copy2(img, dst_img)
        stats['images'] += 1

        lbl = lbl_map.get(img.stem)
        if lbl is None:
            continue  # негатив без лейбла

        kept = []
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                if cls in keep_classes:
                    parts[0] = str(class_map[cls])
                    kept.append(" ".join(parts))
                    stats['objects'] += 1
                else:
                    stats['non_target_removed'] += 1

        if kept:
            with open(dst_lbl_dir / (img.stem + ".txt"), 'w') as f:
                f.write("\n".join(kept) + "\n")
            stats['positives'] += 1
        else:
            stats['negatives'] += 1  # изображение осталось как негатив


def main():
    print("=" * 60)
    print("СБОРКА drone_v2 (1 класс: Drone)")
    print("=" * 60)

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    stats = Counter()

    # 1. MyDataSet train: Drone(1) -> 0
    print("\n[1/2] MyDataSet...")
    copy_with_remap(
        MY_DATASET / "images" / "train",
        MY_DATASET / "labels" / "train",
        OUTPUT / "train" / "images",
        OUTPUT / "train" / "labels",
        keep_classes={1},
        class_map={1: 0},
        stats=stats,
    )

    # 2. combined train: drone(0) -> 0
    print("[2/2] combined train...")
    copy_with_remap(
        COMBINED / "train" / "images",
        COMBINED / "train" / "labels",
        OUTPUT / "train" / "images",
        OUTPUT / "train" / "labels",
        keep_classes={0},
        class_map={0: 0},
        stats=stats,
    )

    # 3. combined val: drone(0) -> 0
    print("[3/3] combined val...")
    val_stats = Counter()
    copy_with_remap(
        COMBINED / "val" / "images",
        COMBINED / "val" / "labels",
        OUTPUT / "val" / "images",
        OUTPUT / "val" / "labels",
        keep_classes={0},
        class_map={0: 0},
        stats=val_stats,
    )

    # data.yaml
    yaml_content = f"""# Чистый датасет перехвата дронов (1 класс)
# MyDataSet (Drone) + combined (drone) + негативы (люди, вертолёты, самолёты, птицы)

path: {OUTPUT}
train: {OUTPUT / 'train' / 'images'}
val: {OUTPUT / 'val' / 'images'}

nc: 1
names: ['Drone']
"""
    with open(OUTPUT / "data.yaml", 'w') as f:
        f.write(yaml_content)

    print("\n" + "=" * 60)
    print("ИТОГИ:")
    print(f"  Изображений train: {stats['images']}")
    print(f"  Позитивов (с дроном): {stats['positives']}")
    print(f"  Негативов (без дрона): {stats['negatives']}")
    print(f"  Объектов Drone: {stats['objects']}")
    print(f"  Убрано не-целевых объектов: {stats['non_target_removed']}")
    print(f"  Изображений val: {val_stats['images']}")
    print(f"  Позитивов val: {val_stats['positives']}")
    print(f"  Негативов val: {val_stats['negatives']}")
    print(f"  Объектов val: {val_stats['objects']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
