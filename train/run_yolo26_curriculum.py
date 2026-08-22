#!/usr/bin/env python3
"""
Полуавтоматическое curriculum-обучение YOLO26L — этап 2 (merged_v1) в 3 стадии
с нарастающими помехами и тестом на натурных видео после каждой стадии.

Логика:
  Этап 1: drone_v2 (40 эпох) — выполняется исходным run_yolo26_two_stage.py
          (best.pt уже должен существовать).
  Этап 2 (merged_v1) делится на стадии с постепенным усложнением:

    2a (20 эпох): базовая аугментация (scale, hsv, fliplr, mosaic)
    2b (20 эпох): + поворот, перспектива, translate, shear, mixup
    2c (20 эпох): + усиленные помехи (засветка, стирание), фин. шлифовка

После каждой стадии:
  1. Автоматический тест на video-FPV/Video (test_videos.py)
  2. Сравнение % детекций с предыдущей стадией
  3. Подтверждение пользователя на продолжение (полуавтоматический режим)

Устойчив к перезапуску: resume через last.pt внутри стадии;
завершённые стадии помечаются маркером и пропускаются.

Запуск (интерактивный терминал!):
    cd /home/alex/AntiUAV-Detector
    venv/bin/python train/run_yolo26_curriculum.py 2>&1 | tee train/curriculum.log
"""

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path

import albumentations as A
from ultralytics import YOLO
from ultralytics.data import augment as ua_augment

BASE = '/home/alex/AntiUAV-Detector'
PROJECT = f'{BASE}/train/runs'
NAME_1 = 'drone_v2_26L'
TESTS_DIR = f'{BASE}/train/test_results'
VENV_PY = f'{BASE}/venv/bin/python'

# Маркер успешного завершения стадии (кладётся в weights/<стадия>/)
DONE_MARK = 'stage_done.msg'

# Стадии этапа 2. Параметры только для нового старта; при resume берутся из ckpt.
STAGES = [
    dict(
        name='merged_v1_26L_2a',
        epochs=20,
        desc='базовая аугментация: scale/hsv/fliplr/mosaic',
        train=dict(
            imgsz=640, batch=16, device=0, workers=0, cache='disk',
            project=PROJECT, patience=15, lr0=0.005, warmup_epochs=2.0,
            optimizer='auto',
            mixup=0.0, copy_paste=0.0, close_mosaic=5,
            scale=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            fliplr=0.5, mosaic=1.0,
        ),
    ),
    dict(
        name='merged_v1_26L_2b',
        epochs=20,
        desc='+поворот, перспектива, translate, shear, mixup',
        train=dict(
            imgsz=640, batch=16, device=0, workers=0, cache='disk',
            project=PROJECT, patience=15, lr0=0.002, warmup_epochs=1.0,
            optimizer='auto',
            mixup=0.15, copy_paste=0.0, close_mosaic=5,
            scale=0.5, shear=5.0, degrees=15.0, translate=0.2,
            perspective=0.0005,
            hsv_h=0.02, hsv_s=0.75, hsv_v=0.45,
            fliplr=0.5, mosaic=1.0,
        ),
    ),
    dict(
        name='merged_v1_26L_2c',
        epochs=20,
        desc='+усиленные помехи (засветка, стирание), close_mosaic',
        train=dict(
            imgsz=640, batch=16, device=0, workers=0, cache='disk',
            project=PROJECT, patience=15, lr0=0.001, warmup_epochs=1.0,
            optimizer='auto',
            mixup=0.25, copy_paste=0.0, close_mosaic=5,
            scale=0.7, shear=10.0, degrees=25.0, translate=0.3,
            perspective=0.001, erasing=0.5,
            hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
            fliplr=0.5, mosaic=1.0,
        ),
    ),
    dict(
        name='merged_v1_26L_2d',
        epochs=10,
        desc='+albumentations: CLAHE, SunFlare, Fog, GaussNoise, MotionBlur, '
             'JPEGCompression, ISONoise',
        train=dict(
            imgsz=640, batch=16, device=0, workers=0, cache='disk',
            project=PROJECT, patience=15, lr0=0.0005, warmup_epochs=1.0,
            optimizer='auto',
            mixup=0.3, copy_paste=0.0, close_mosaic=5,
            scale=0.7, shear=10.0, degrees=25.0, translate=0.3,
            perspective=0.001, erasing=0.6,
            hsv_h=0.02, hsv_s=0.8, hsv_v=0.6,
            fliplr=0.5, mosaic=1.0,
        ),
        # Кастомные albumentations-трансформации для стадии 2d
        albumentations=[
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5),
                             src_radius=100, src_color=(255, 255, 255), p=0.2),
            A.RandomFog(fog_coef_range=(0.1, 0.4), alpha_coef=0.08, p=0.15),
            A.GaussNoise(std_range=(0.05, 0.2), mean_range=(0.0, 0.0), p=0.2),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.15),
            A.MotionBlur(blur_limit=(3, 10), p=0.2),
            A.ImageCompression(quality_range=(40, 85), p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.3,
                                       contrast_limit=0.3, p=0.25),
            A.RandomGamma(gamma_limit=(70, 130), p=0.2),
        ],
    ),
]


def stage_done(stage):
    return (Path(PROJECT) / stage['name'] / 'weights' / DONE_MARK).exists()


def last_weights(stage):
    last = Path(PROJECT) / stage['name'] / 'weights' / 'last.pt'
    return str(last) if last.exists() else None


def best_weights(stage):
    best = Path(PROJECT) / stage['name'] / 'weights' / 'best.pt'
    return str(best) if best.exists() else None


def train_stage(stage, start_weights):
    """Обучает одну стадию: resume при наличии last.pt, иначе старт с prev best."""
    name = stage['name']
    data = f'{BASE}/prepare_data/merged_v1/data.yaml'
    wdir = Path(PROJECT) / name / 'weights'

    # Если у стадии есть кастомные albumentations — патчим Ultralytics
    custom_alb = stage.get('albumentations')
    if custom_alb:
        _patch_albumentations(custom_alb)
        print(f'[Стадия {name}] Albumentations: {len(custom_alb)} трансформаций',
              flush=True)

    last = last_weights(stage)
    if last and not stage_done(stage):
        print(f'[Стадия {name}] RESUME: {last}', flush=True)
        model = YOLO(last)
        model.train(data=data, epochs=stage['epochs'], name=name, resume=True)
    else:
        print(f'[Стадия {name}] Старт с весов: {start_weights}', flush=True)
        model = YOLO(start_weights)
        model.train(data=data, epochs=stage['epochs'], name=name,
                    **stage['train'])

    (wdir / DONE_MARK).write_text('ok\n')
    print(f'[Стадия {name}] обучение завершено, маркер поставлен', flush=True)


def _patch_albumentations(transforms_list):
    """Monkey-patch: подменяет дефолтные albumentations в Ultralytics."""
    original_init = ua_augment.Albumentations.__init__

    def patched_init(self, p=1.0, transforms=None, flip_idx=None):
        if transforms is None:
            transforms = transforms_list
        original_init(self, p=p, transforms=transforms, flip_idx=flip_idx)

    ua_augment.Albumentations.__init__ = patched_init


RE_ROW = re.compile(
    r'^(\S+\.(?:mp4|mov|avi))\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+)'
)
RE_TOTAL = re.compile(r'^ИТОГО\s+(\d+)\s+(\d+)\s+([\d.]+)%')


def parse_test_output(text):
    rows, total = [], None
    for line in text.splitlines():
        m = RE_ROW.match(line.strip())
        if m:
            rows.append(dict(
                video=m.group(1), frames=int(m.group(2)),
                detected=int(m.group(3)), pct=float(m.group(4)),
                conf=float(m.group(5)), fps=float(m.group(6)),
            ))
            continue
        t = RE_TOTAL.match(line.strip())
        if t:
            total = dict(frames=int(t.group(1)), detected=int(t.group(2)),
                         pct=float(t.group(3)))
    return rows, total


def run_video_test(weights, tag):
    """Запускает test_videos.py, сохраняет вывод и CSV сравнения."""
    os.makedirs(TESTS_DIR, exist_ok=True)
    print(f'\n[Тест] {tag}', flush=True)
    print(f'[Тест] модель: {weights}', flush=True)
    proc = subprocess.run([VENV_PY, 'train/test_videos.py', weights],
                          cwd=BASE, capture_output=True, text=True)
    combined = proc.stdout + proc.stderr
    (Path(TESTS_DIR) / f'video_test_{tag}.txt').write_text(combined, encoding='utf-8')

    rows, total = parse_test_output(combined)
    if not total:
        print('[Тест] ВНИМАНИЕ: не удалось распарсить результат теста', flush=True)
        print(combined[-2000:], flush=True)
        return rows, total

    with open(Path(TESTS_DIR) / f'video_test_{tag}.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['video', 'frames', 'detected',
                                               'pct', 'conf', 'fps'])
        writer.writeheader()
        writer.writerows(rows)

    print(f'[Тест] {tag}: детекций {total["detected"]}/{total["frames"]} = '
          f'{total["pct"]:.1f}%   (CSV: test_results/video_test_{tag}.csv)',
          flush=True)
    return rows, total


def print_comparison(prev, cur, prev_total, cur_total):
    """Сравнение % детекций по каждому видео с предыдущей стадией."""
    print('\n=== Сравнение с предыдущей стадией ===', flush=True)
    prev_map = {r['video']: r['pct'] for r in prev}
    cur_map = {r['video']: r['pct'] for r in cur}

    diffs = [(v, prev_map[v], cur_map[v], cur_map[v] - prev_map[v])
             for v in cur_map if v in prev_map]
    diffs.sort(key=lambda d: d[3])

    print(f'Итого:  {prev_total["pct"]:.1f}%  →  {cur_total["pct"]:.1f}%  '
          f'(Δ {cur_total["pct"] - prev_total["pct"]:+.1f} п.п.)', flush=True)

    wors = [d for d in diffs if d[3] <= -1.0]
    impr = [d for d in diffs if d[3] >= 1.0]
    if wors:
        print('\nУХУДШИЛИСЬ (топ-10):')
        for v, p0, p1, d in wors[:10]:
            print(f'  {v:<12} {p0:5.1f}% → {p1:5.1f}%  (Δ {d:+.1f})', flush=True)
    if impr:
        print('\nУЛУЧШИЛИСЬ (топ-10):')
        for v, p0, p1, d in reversed(impr[-10:]):
            print(f'  {v:<12} {p0:5.1f}% → {p1:5.1f}%  (Δ {d:+.1f})', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start-stage', type=int, default=1,
                    help='Стадия для старта (1=2a, 2=2b, 3=2c, 4=2d)')
    ap.add_argument('--skip-test', action='store_true',
                    help='Пропустить тест на видео')
    ap.add_argument('--skip-confirmation', action='store_true',
                    help='Не спрашивать подтверждение между стадиями')
    args = ap.parse_args()

    best1 = Path(PROJECT) / NAME_1 / 'weights' / 'best.pt'
    if not best1.exists():
        print(f'ОШИБКА: {best1} не найден.', flush=True)
        print('Сначала завершите Этап 1 (run_yolo26_two_stage.py) и только '
              'потом запускайте curriculum.', flush=True)
        return

    print(f'[Curriculum] Этап 1 завершён: {best1}', flush=True)
    print('[Curriculum] Стадий этапа 2: ' +
          ', '.join(f'{s["name"]} ({s["epochs"]} эп., {s["desc"]})'
                    for s in STAGES), flush=True)
    if args.start_stage > 1:
        print(f'[Curriculum] Старт со стадии {args.start_stage} '
              f'({STAGES[args.start_stage - 1]["name"]})', flush=True)

    # Определяем начальные веса: best.pt предыдущей стадии (или этапа 1)
    prev_best = str(best1)
    prev_rows, prev_total = None, None

    # Если старт не с 1-й стадии — берём best.pt предыдущей завершённой стадии
    if args.start_stage > 1:
        prev_stage = STAGES[args.start_stage - 2]
        prev_best_path = best_weights(prev_stage)
        if prev_best_path:
            prev_best = prev_best_path
            print(f'[Curriculum] Веса предыдущей стадии: {prev_best}',
                  flush=True)
        else:
            print(f'[Curriculum] ВНИМАНИЕ: best.pt стадии '
                  f'{prev_stage["name"]} не найден, используем этап 1',
                  flush=True)

    for idx, stage in enumerate(STAGES[args.start_stage - 1:], args.start_stage):
        name = stage['name']
        print(f'\n{"=" * 70}', flush=True)
        print(f'Стадия {idx}/{len(STAGES)}: {name} — {stage["desc"]}', flush=True)
        print('=' * 70, flush=True)

        if stage_done(stage):
            print(f'[Стадия {name}] уже завершена ранее — пропуск', flush=True)
            this_best = best_weights(stage)
        else:
            train_stage(stage, start_weights=prev_best)
            this_best = best_weights(stage)
            if not this_best:
                print(f'[Стадия {name}] ОШИБКА: best.pt не найден после '
                      f'обучения', flush=True)
                return

        if args.skip_test:
            print(f'[Тест] Пропущен (--skip-test)', flush=True)
            rows, total = [], None
        else:
            rows, total = run_video_test(this_best, name)

        if total is None:
            print(f'[Тест] Результат недоступен', flush=True)
        elif prev_total is None:
            print(f'[Тест] Базовый уровень (после Этапа 1): {total["pct"]:.1f}%',
                  flush=True)
        else:
            print_comparison(prev_rows, rows, prev_total, total)

        prev_rows, prev_total = rows, total
        prev_best = this_best

        if idx < len(STAGES):
            nxt = STAGES[idx]['name']
            if args.skip_confirmation:
                print(f'[Curriculum] Автопереход к стадии {nxt} '
                      f'(--skip-confirmation)', flush=True)
            else:
                answer = input(f'\nПродолжить стадию {nxt}? [y/N]: ').strip().lower()
                if answer not in ('y', 'yes', 'д', 'да'):
                    print('[Curriculum] Остановлен. Для продолжения запустите '
                          'скрипт снова — завершённые стадии будут пропущены.',
                          flush=True)
                    return

    print('\n[Curriculum] ВСЕ СТАДИИ ЗАВЕРШЕНЫ!', flush=True)
    print(f'[Curriculum] Финальные веса: {prev_best}', flush=True)


if __name__ == '__main__':
    main()
