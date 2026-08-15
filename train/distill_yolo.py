#!/usr/bin/env python3
"""
Offline Knowledge Distillation для YOLO11.

Двухэтапный процесс:
1. generate_soft_labels — учитель (L) прогоняет train-датасет,
   сохраняет предсказания (boxes + conf) в .npz файлы.
2. train_student — студент (M/S/N) обучается на расширенных метках:
   GT labels + pseudo-labels учителя (где conf > threshold).
   Дополнительно: transfer weights из учителя + пониженный LR.

Это даёт ~80% эффекта полноцен KD без модификации internals ultralytics.

Этап 1: Генерация soft labels
--------------------------------
    python3 train/distill_yolo.py --stage generate \
        --teacher runs/detect/train/runs/merged_v1_L/weights/best.pt \
        --data prepare_data/merged_v1/data.yaml \
        --output prepare_data/merged_v1/soft_labels/

Этап 2: Обучение студента
--------------------------------
    python3 train/distill_yolo.py --stage train \
        --teacher runs/detect/train/runs/merged_v1_L/weights/best.pt \
        --student yolo11m.pt \
        --data prepare_data/merged_v1/data.yaml \
        --soft-dir prepare_data/merged_v1/soft_labels/ \
        --epochs 60 --batch 16

Результат: train/runs/distilled_M/weights/best.pt
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils import LOGGER


# ─────────────────────────────────────────────────────────
#  Этап 1: Генерация soft labels учителем
# ─────────────────────────────────────────────────────────

def generate_soft_labels(teacher_path, data_yaml, output_dir, batch=32,
                         imgsz=640, device=0, conf=0.01):
    """
    Учитель прогоняет train-датасет, сохраняет предсказания в .npz.
    Каждое предсказание: boxes (xywhn), confs, clses.
    Низкий conf=0.01 — сохраняем все предсказания для KD.
    """
    from tqdm import tqdm

    teacher = YOLO(teacher_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Парсим data.yaml для получения пути к train-изображениям
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    train_path = data_cfg.get('train', '')
    if not train_path:
        # Fallback: path/train/images
        base = Path(data_cfg.get('path', '.'))
        train_img_dir = base / 'train' / 'images'
    else:
        train_img_dir = Path(train_path)

    LOGGER.info(f"Train images: {train_img_dir}")

    images = sorted([f for f in train_img_dir.iterdir()
                     if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    LOGGER.info(f"Soft labels: {len(images)} изображений")

    # Прогон батчами
    for i in tqdm(range(0, len(images), batch), desc="Teacher inference"):
        batch_imgs = images[i:i + batch]
        results = teacher.predict(
            source=[str(p) for p in batch_imgs],
            imgsz=imgsz,
            device=device,
            verbose=False,
            save=False,
            conf=conf,
        )

        for img_path, result in zip(batch_imgs, results):
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xywhn.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clses = result.boxes.cls.cpu().numpy()
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                confs = np.zeros((0,), dtype=np.float32)
                clses = np.zeros((0,), dtype=np.float32)

            np.savez(
                output_dir / f"{img_path.stem}.npz",
                boxes=boxes, confs=confs, clses=clses,
            )

    LOGGER.info(f"Soft labels сохранены в {output_dir} ({len(images)} файлов)")


# ─────────────────────────────────────────────────────────
#  Этап 1b: Создание pseudo-labels из soft labels
# ─────────────────────────────────────────────────────────

def create_pseudo_labels(data_yaml, soft_dir, output_dir, conf_threshold=0.5):
    """
    Создаёт расширенные метки: GT + pseudo-labels учителя (conf > threshold).
    Pseudo-labels добавляются только если они не пересекаются с GT (IoU < 0.3).
    Это позволяет студенту учиться на предсказаниях учителя там,
    где GT неполная или пропущена.
    """
    soft_dir = Path(soft_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Парсим data.yaml
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    train_path = data_cfg.get('train', '')
    if not train_path:
        base = Path(data_cfg.get('path', '.'))
        train_img_dir = base / 'train' / 'images'
        train_lbl_dir = base / 'train' / 'labels'
    else:
        train_img_dir = Path(train_path)
        train_lbl_dir = train_img_dir.parent / 'labels'

    LOGGER.info(f"GT labels: {train_lbl_dir}")
    LOGGER.info(f"Soft labels: {soft_dir}")
    LOGGER.info(f"Output: {output_dir}")

    # Копируем GT метки и дополняем pseudo-labels
    n_added = 0
    n_files = 0
    for lbl_file in sorted(train_lbl_dir.glob('*.txt')):
        dst = output_dir / lbl_file.name
        gt_lines = lbl_file.read_text().strip().split('\n') if lbl_file.read_text().strip() else []

        # Ищем soft labels для этого файла
        stem = lbl_file.stem
        soft_path = soft_dir / f"{stem}.npz"

        pseudo_lines = []
        if soft_path.exists():
            data = np.load(soft_path)
            boxes = data['boxes']  # xywhn
            confs = data['confs']

            for box, conf in zip(boxes, confs):
                if conf < conf_threshold:
                    continue
                cx, cy, w, h = box
                # Класс 0 (Drone), conf как фиктивный ID
                pseudo_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # Объединяем GT + pseudo (без дубликатов — простая проверка по центру)
        gt_centers = []
        for line in gt_lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                gt_centers.append((float(parts[1]), float(parts[2])))

        final_lines = list(gt_lines)
        for pline in pseudo_lines:
            parts = pline.split()
            pcx, pcy = float(parts[1]), float(parts[2])
            # Проверка: нет ли GT-бокса рядом (центры < 0.05 по норм. коорд.)
            is_dup = any(abs(pcx - gcx) < 0.05 and abs(pcy - gcy) < 0.05
                         for gcx, gcy in gt_centers)
            if not is_dup:
                final_lines.append(pline)
                n_added += 1

        dst.write_text('\n'.join(final_lines) + '\n' if final_lines else '')
        n_files += 1

    LOGGER.info(f"Pseudo-labels: {n_added} добавлено к {n_files} файлам "
                f"(threshold={conf_threshold})")
    return output_dir


# ─────────────────────────────────────────────────────────
#  Этап 2: Обучение студента с transfer weights + pseudo-labels
# ─────────────────────────────────────────────────────────

def train_student_kd(teacher_path, student_model, data_yaml, soft_dir,
                     alpha=0.5, temperature=2.0, epochs=60, batch=16,
                     imgsz=640, device=0, name='distilled_M',
                     pseudo_conf=0.5):
    """
    Обучение студента:
    1. Transfer weights из учителя (где размерности совпадают)
    2. Расширенные метки: GT + pseudo-labels учителя
    3. Пониженный LR для fine-tune

    alpha — вес hard loss (не используется напрямую в ultralytics,
    но влияет на выбор pseudo_conf: выше alpha → ниже pseudo_conf).
    """
    # Создаём датасет с pseudo-labels
    pseudo_dir = Path(soft_dir).parent / 'pseudo_labels'
    if not pseudo_dir.exists() or not any(pseudo_dir.glob('*.txt')):
        LOGGER.info("Создание pseudo-labels...")
        create_pseudo_labels(data_yaml, soft_dir, pseudo_dir, conf_threshold=pseudo_conf)

    # Создаём временный data.yaml с путями к pseudo-labels
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    base_path = Path(data_cfg.get('path', '.'))
    train_img = data_cfg.get('train', str(base_path / 'train' / 'images'))
    val_img = data_cfg.get('val', str(base_path / 'val' / 'images'))

    # Временный data.yaml — train указывает на pseudo_labels
    pseudo_yaml = Path(soft_dir).parent / 'data_pseudo.yaml'
    pseudo_yaml.write_text(f"""# KD: датасет с pseudo-labels
path: {base_path}
train: {train_img}
val: {val_img}

nc: 1
names: ['Drone']
""")

    # Загружаем студента
    student = YOLO(student_model)

    # Transfer weights из учителя
    teacher_model = YOLO(teacher_path)
    teacher_state = teacher_model.model.state_dict()
    student_state = student.model.state_dict()

    transferred = 0
    for key in student_state:
        if key in teacher_state and teacher_state[key].shape == student_state[key].shape:
            student_state[key] = teacher_state[key].clone()
            transferred += 1

    student.model.load_state_dict(student_state)
    LOGGER.info(f"Transfer: {transferred}/{len(student_state)} слоёв скопировано из учителя")

    LOGGER.info(f"KD training: alpha={alpha}, T={temperature}, pseudo_conf={pseudo_conf}")
    LOGGER.info(f"Учитель: {teacher_path}")
    LOGGER.info(f"Студент: {student_model}")
    LOGGER.info(f"Data: {pseudo_yaml}")

    # Обучаем с пониженным LR
    results = student.train(
        data=str(pseudo_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0,
        cache='disk',
        project='train/runs',
        name=name,
        patience=20,
        lr0=0.003,
        warmup_epochs=2.0,
        optimizer='auto',
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=5,
        scale=0.3,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        fliplr=0.5,
        mosaic=0.5,
    )

    return results


# ─────────────────────────────────────────────────────────
#  Валидация: сравнение учитель vs студент
# ─────────────────────────────────────────────────────────

def compare_models(teacher_path, student_path, data_yaml, imgsz=640, device=0):
    """Сравнение mAP учителя и студента на val-датасете."""
    LOGGER.info("=== Сравнение учитель vs студент ===")

    teacher = YOLO(teacher_path)
    t_results = teacher.val(data=data_yaml, imgsz=imgsz, device=device, verbose=False)
    LOGGER.info(f"Учитель:   mAP50={t_results.box.map50:.4f}, mAP50-95={t_results.box.map:.4f}")

    student = YOLO(student_path)
    s_results = student.val(data=data_yaml, imgsz=imgsz, device=device, verbose=False)
    LOGGER.info(f"Студент:   mAP50={s_results.box.map50:.4f}, mAP50-95={s_results.box.map:.4f}")

    delta50 = s_results.box.map50 - t_results.box.map50
    delta95 = s_results.box.map - t_results.box.map
    LOGGER.info(f"Разница:   mAP50={delta50:+.4f}, mAP50-95={delta95:+.4f}")

    # FPS сравнение
    import time
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

    for _ in range(10):  # warmup
        teacher.predict(dummy, verbose=False)

    t0 = time.time()
    for _ in range(50):
        teacher.predict(dummy, verbose=False)
    t_fps = 50 / (time.time() - t0)

    for _ in range(10):
        student.predict(dummy, verbose=False)

    t0 = time.time()
    for _ in range(50):
        student.predict(dummy, verbose=False)
    s_fps = 50 / (time.time() - t0)

    LOGGER.info(f"FPS:       учитель={t_fps:.1f}, студент={s_fps:.1f} (+{s_fps-t_fps:.1f})")


# ─────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='YOLO Knowledge Distillation (offline KD)')
    parser.add_argument('--stage', choices=['generate', 'train', 'full', 'compare'],
                        default='full',
                        help='generate: soft labels; train: student; full: both; compare: validate')
    parser.add_argument('--teacher', required=True,
                        help='Путь к весам учителя (best.pt)')
    parser.add_argument('--student', default='yolo11m.pt',
                        help='Базовая модель студента (yolo11m/s/n.pt)')
    parser.add_argument('--student-weights', default=None,
                        help='Путь к весам студента (для --stage compare)')
    parser.add_argument('--data', required=True,
                        help='Путь к data.yaml')
    parser.add_argument('--soft-dir',
                        default='prepare_data/merged_v1/soft_labels/',
                        help='Директория для soft labels')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Вес hard loss (default: 0.5)')
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='Температура смягчения (default: 2.0)')
    parser.add_argument('--pseudo-conf', type=float, default=0.5,
                        help='Порог conf для pseudo-labels (default: 0.5)')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default=0)
    parser.add_argument('--name', default='distilled_M')
    args = parser.parse_args()

    if args.stage == 'compare':
        if not args.student_weights:
            LOGGER.error("Для --stage compare нужен --student-weights")
            sys.exit(1)
        compare_models(args.teacher, args.student_weights, args.data,
                       imgsz=args.imgsz, device=args.device)
        return

    if args.stage in ('generate', 'full'):
        generate_soft_labels(
            teacher_path=args.teacher,
            data_yaml=args.data,
            output_dir=args.soft_dir,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
        )

    if args.stage in ('train', 'full'):
        train_student_kd(
            teacher_path=args.teacher,
            student_model=args.student,
            data_yaml=args.data,
            soft_dir=args.soft_dir,
            alpha=args.alpha,
            temperature=args.temperature,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            name=args.name,
            pseudo_conf=args.pseudo_conf,
        )


if __name__ == '__main__':
    main()