#!/usr/bin/env python3
"""
Исправление class ID в объединённом датасете all_datasets_v1 (v3, финальная).

Проблема: MyDataSet использует People=0, Drone=1, Car=2, Boat=3,
а combined использует drone=0, helicopter=1, airplane=2, bird=3.
Скрипт combine_all_datasets.py копировал лейблы без перемаппинга.

Исправление (на месте):
- Строим множество ВСЕХ имён лейбл-файлов MyDataSet из оригинала (85,251 шт:
  frame_*, числовые, obj_train_data/*, UAVs/* и т.д.).
- Файлы из MyDataSet (имя в множестве) — НЕ трогаем (уже правильные).
- Файлы из combined (имя НЕ в множестве): оставляем ТОЛЬКО строки drone (0 -> 1),
  строки helicopter/airplane/bird (1/2/3) удаляем.
  Если лейбл стал пустым — удаляем файл, изображение оставляем как негатив.

ВАЖНО: запускать только на ЧИСТОМ датасете (восстановленном из бэкапа,
до любых предыдущих перемаппингов).
"""

from pathlib import Path
from collections import Counter

OUTPUT_DIR = Path("/home/alex/AntiUAV-Detector/prepare_data/all_datasets_v1")
MY_DATASET_LABELS = Path("/home/alex/AntiUAV-Detector/DataSet/MyDataSet/labels")

REMOVE_CLASSES = {1, 2, 3}  # helicopter, airplane, bird


def collect_my_dataset_names() -> set:
    """Все имена лейбл-файлов MyDataSet (рекурсивно)."""
    names = set()
    for lbl in MY_DATASET_LABELS.rglob("*.txt"):
        names.add(lbl.name)
    return names


def fix_dataset(my_names: set):
    labels_dir = OUTPUT_DIR / "train" / "labels"
    stats = Counter()
    my_skipped = 0
    combined_processed = 0

    for lbl_file in sorted(labels_dir.glob("*.txt")):
        if lbl_file.name in my_names:
            my_skipped += 1
            continue

        combined_processed += 1
        kept_lines = []
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])

                if cls == 0:
                    parts[0] = "1"  # drone -> Drone(1)
                    kept_lines.append(" ".join(parts))
                    stats["drone_remapped"] += 1
                elif cls in REMOVE_CLASSES:
                    stats["non_drone_removed"] += 1
                else:
                    kept_lines.append(" ".join(parts))
                    stats["unexpected_kept"] += 1

        if kept_lines:
            with open(lbl_file, 'w') as f:
                f.write("\n".join(kept_lines) + "\n")
            stats["labels_updated"] += 1
        else:
            lbl_file.unlink()  # изображение оставляем как негатив
            stats["empty_labels_deleted"] += 1

    print(f"Пропущено MyDataSet: {my_skipped}")
    print(f"Обработано combined: {combined_processed}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def validate():
    labels_dir = OUTPUT_DIR / "train" / "labels"
    cls_counter = Counter()
    invalid = 0
    for lbl_file in labels_dir.glob("*.txt"):
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                if cls < 0 or cls > 3:
                    invalid += 1
                cls_counter[cls] += 1

    names = ['People', 'Drone', 'Car', 'Boat']
    print("\nВалидация:")
    print(f"  Лейбл-строк: {sum(cls_counter.values())}")
    for cls, cnt in sorted(cls_counter.items()):
        print(f"  Класс {cls} ({names[cls]}): {cnt}")
    print(f"  Некорректных классов: {invalid}")
    print(f"  Файлов лейблов: {len(list(labels_dir.glob('*.txt')))}")
    print(f"  Изображений .jpg: {len(list((OUTPUT_DIR / 'train' / 'images').glob('*.jpg')))}")


if __name__ == "__main__":
    print("=" * 60)
    print("ИСПРАВЛЕНИЕ CLASS ID В all_datasets_v1 (v3)")
    print("=" * 60)
    my_names = collect_my_dataset_names()
    print(f"MyDataSet лейбл-файлов (оригинал): {len(my_names)}")
    fix_dataset(my_names)
    validate()
    print("=" * 60)
    print("ГОТОВО!")
    print("=" * 60)
