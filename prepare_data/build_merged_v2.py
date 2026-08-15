#!/usr/bin/env python3
"""
Сборка объединённого датасета merged_v2 (1 класс: Drone).

Источники:
1. drone_v2       (prepare_data/drone_v2)       — 164.7K train / 6.8K val  (средние/крупные + IR)
2. antiuav_yolo   (prepare_data/antiuav_yolo)   — 28.4K train / 11.7K val  (visible, small 22%)
3. antiuav_ir_yolo(prepare_data/antiuav_ir_yolo)— 29.7K train / 12.1K val  (IR, tiny 28%)
4. seraphim       (DataSet/seraphim/repo)       — 75.1K train / 8.3K test  (tiny/small 11%)

Результат: prepare_data/merged_v2/{train,val}/{images,labels} + data.yaml

Имена файлов получают префикс источника (v2_, au_, ai_, se_).
Используются hard links (экономия диска).

Запуск:
    python3 prepare_data/build_merged_v2.py [--dry-run]
"""

import argparse
import os
import shutil
from pathlib import Path
from collections import Counter

SRC = {
    'v2': Path('/home/alex/AntiUAV-Detector/prepare_data/drone_v2'),
    'au': Path('/home/alex/AntiUAV-Detector/prepare_data/antiuav_yolo'),
    'ai': Path('/home/alex/AntiUAV-Detector/prepare_data/antiuav_ir_yolo'),
    'se': Path('/home/alex/DataSet/seraphim/repo'),
}
OUTPUT = Path('/home/alex/AntiUAV-Detector/prepare_data/merged_v2')

# Источник -> (train-сплит, val-сплит): (путь_к_images, путь_к_labels)
SPLITS = {
    'v2': (('train/images', 'train/labels'), ('val/images', 'val/labels')),
    'au': (('images/train', 'labels/train'), ('images/val', 'labels/val')),
    'ai': (('images/train', 'labels/train'), ('images/val', 'labels/val')),
    'se': (('train/images', 'train/labels'), ('test/images', 'test/labels')),
}

IMG_EXTS = {'.jpg', '.jpeg', '.png'}


def link_dataset(src_name, src_img_rel, src_lbl_rel, dst_split, prefix,
                 stats, dry_run):
    """Хардлинки изображений и меток из одного сплита."""
    src_dir = SRC[src_name]
    src_img = src_dir / src_img_rel
    src_lbl = src_dir / src_lbl_rel

    dst_img = OUTPUT / dst_split / 'images'
    dst_lbl = OUTPUT / dst_split / 'labels'
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    if not src_img.exists():
        print(f"  [ПРОПУСК] {src_name}: нет папки {src_img}")
        return

    lbl_map = {}
    if src_lbl.exists():
        for lbl in src_lbl.glob('*.txt'):
            lbl_map[lbl.stem] = lbl

    n_img = n_lbl = n_pos = n_neg = 0
    for img in sorted(src_img.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
            continue

        dst_img_path = dst_img / f"{prefix}_{img.name}"
        n_img += 1

        lbl = lbl_map.get(img.stem)
        if lbl is None:
            # Негатив (нет лейбла) — только изображение
            if not dry_run:
                if not dst_img_path.exists():
                    os.link(img, dst_img_path)
            n_neg += 1
            continue

        dst_lbl_path = dst_lbl / f"{prefix}_{lbl.name}"
        if not dry_run:
            if not dst_img_path.exists():
                os.link(img, dst_img_path)
            if not dst_lbl_path.exists():
                os.link(lbl, dst_lbl_path)
        n_lbl += 1
        n_pos += 1

    stats['images'] += n_img
    stats['positives'] += n_pos
    stats['negatives'] += n_neg
    stats['labels'] += n_lbl
    print(f"  {src_name}/{src_img_rel} -> {dst_split}: "
          f"{n_img} img ({n_pos} pos / {n_neg} neg), {n_lbl} lbl")


def main():
    parser = argparse.ArgumentParser(description='Build merged_v2 dataset')
    parser.add_argument('--dry-run', action='store_true',
                        help='Только посчитать, без создания файлов')
    args = parser.parse_args()

    if OUTPUT.exists() and not args.dry_run:
        shutil.rmtree(OUTPUT)

    print("=" * 60)
    print("СБОРКА merged_v2 (1 класс: Drone)")
    print("  drone_v2 + Anti-UAV visible + Anti-UAV IR + Seraphim")
    print("=" * 60)

    train_stats = Counter()
    val_stats = Counter()

    for src_name, ((tr_img, tr_lbl), (va_img, va_lbl)) in SPLITS.items():
        prefix = src_name
        print(f"\n[{src_name}]")
        link_dataset(src_name, tr_img, tr_lbl, 'train', prefix,
                     train_stats, args.dry_run)
        link_dataset(src_name, va_img, va_lbl, 'val', prefix,
                     val_stats, args.dry_run)

    # data.yaml
    if not args.dry_run:
        yaml_content = f"""# Объединённый датасет merged_v2 (1 класс: Drone)
# drone_v2 (164.7K) + Anti-UAV visible (28.4K) + Anti-UAV IR (29.7K) + Seraphim (75.1K)
# Префиксы: v2_, au_, ai_, se_
# IR-кадры (ai_) — для устойчивости к тепловизионным изображениям

path: {OUTPUT}
train: {OUTPUT / 'train' / 'images'}
val: {OUTPUT / 'val' / 'images'}

nc: 1
names: ['Drone']
"""
        with open(OUTPUT / 'data.yaml', 'w') as f:
            f.write(yaml_content)

    print("\n" + "=" * 60)
    print("ИТОГИ:")
    print(f"  TRAIN: {train_stats['images']} img, "
          f"{train_stats['positives']} pos, {train_stats['negatives']} neg, "
          f"{train_stats['labels']} lbl")
    print(f"  VAL:   {val_stats['images']} img, "
          f"{val_stats['positives']} pos, {val_stats['negatives']} neg, "
          f"{val_stats['labels']} lbl")
    print("=" * 60)


if __name__ == '__main__':
    main()
