#!/usr/bin/env python3
"""
Convert Anti-UAV-RGBT dataset to YOLO format.

Structure:
    Anti-UAV-RGBT/
    ├── train/
    │   ├── 20190926_130341_1_2/
    │   │   ├── visible.mp4     (1920x1080)
    │   │   ├── infrared.mp4    (640x512)
    │   │   ├── visible.json    {exist: [...], gt_rect: [[x,y,w,h], ...]}
    │   │   └── infrared.json
    │   └── ...
    └── label_new/
        ├── train.json
        ├── val.json
        └── test.json

Output: YOLO format
    antiuav_yolo/
    ├── images/train/  images/val/
    ├── labels/train/  labels/val/
    └── data.yaml

Usage:
    python3 convert_antiuav_rgbt.py --input /path/to/Anti-UAV-RGBT --output /path/to/antiuav_yolo --mode visible --frame_step 5
"""

import cv2
import json
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def bbox_to_yolo(x, y, w, h, img_w, img_h):
    """Convert xywh to YOLO format (class cx cy w h, normalized)."""
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    wn = w / img_w
    hn = h / img_h
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    wn = max(0.001, min(1, wn))
    hn = max(0.001, min(1, hn))
    return f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}"


def process_series(series_dir, output_img_dir, output_lbl_dir,
                   mode='visible', frame_step=5, quality=90):
    """Extract frames from one series video with annotations.
    mode: 'visible' (RGB) or 'infrared' (IR).
    frame_step: save every N-th frame (5 = 1/5 of frames).
    """
    video_path = series_dir / f'{mode}.mp4'
    ann_path = series_dir / f'{mode}.json'

    if not video_path.exists() or not ann_path.exists():
        return 0, 0

    with open(ann_path) as f:
        ann = json.load(f)

    exist = ann.get('exist', [])
    gt_rect = ann.get('gt_rect', [])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    series_name = series_dir.name
    saved = 0
    tiny_saved = 0

    for frame_idx in range(0, min(total, len(exist)), frame_step):
        if not exist[frame_idx]:
            continue

        box = gt_rect[frame_idx]
        if not isinstance(box, list) or len(box) != 4:
            continue

        x, y, w, h = box
        if w <= 0 or h <= 0:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Save image
        prefix = 'vis' if mode == 'visible' else 'ir'
        img_name = f"{prefix}_{series_name}_f{frame_idx:05d}.jpg"
        img_path = output_img_dir / img_name
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

        # Save label
        lbl_name = f"{prefix}_{series_name}_f{frame_idx:05d}.txt"
        lbl_path = output_lbl_dir / lbl_name
        yolo_line = bbox_to_yolo(x, y, w, h, img_w, img_h)
        with open(lbl_path, 'w') as f:
            f.write(yolo_line + '\n')

        saved += 1
        if w * h < 1024:  # tiny
            tiny_saved += 1

    cap.release()
    return saved, tiny_saved


def main():
    parser = argparse.ArgumentParser(
        description='Convert Anti-UAV-RGBT to YOLO format')
    parser.add_argument('--input', required=True,
                        help='Anti-UAV-RGBT root directory')
    parser.add_argument('--output', required=True,
                        help='Output YOLO dataset directory')
    parser.add_argument('--mode', choices=['visible', 'infrared', 'both'],
                        default='visible', help='Process RGB, IR, or both')
    parser.add_argument('--frame_step', type=int, default=5,
                        help='Save every N-th frame (default: 5)')
    parser.add_argument('--quality', type=int, default=90,
                        help='JPEG quality (default: 90)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Parallel workers (default: 4)')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    train_dir = input_dir / 'train'

    # Load split info
    label_dir = input_dir / 'label_new'
    val_series = set()
    if (label_dir / 'val.json').exists():
        with open(label_dir / 'val.json') as f:
            val_data = json.load(f)
            val_series = set(val_data.keys())

    # Create output
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Collect series from all split directories
    splits_map = {}  # series_dir -> 'train' or 'val'
    for split_name, split_label in [('train', 'train'), ('val', 'val')]:
        split_dir = input_dir / split_name
        if split_dir.exists():
            for d in sorted(split_dir.iterdir()):
                if d.is_dir():
                    splits_map[d] = split_label

    all_series = sorted(splits_map.keys())
    print(f"Found {len(all_series)} series "
          f"(train: {sum(1 for v in splits_map.values() if v == 'train')}, "
          f"val: {sum(1 for v in splits_map.values() if v == 'val')})")
    print(f"Mode: {args.mode}, frame_step: {args.frame_step}")

    modes = ['visible', 'infrared'] if args.mode == 'both' else [args.mode]

    total_saved = 0
    total_tiny = 0

    for mode in modes:
        print(f"\n--- Processing {mode} ---")
        for series_dir in all_series:
            split = splits_map[series_dir]
            out_img = output_dir / 'images' / split
            out_lbl = output_dir / 'labels' / split

            saved, tiny = process_series(
                series_dir, out_img, out_lbl,
                mode=mode, frame_step=args.frame_step, quality=args.quality)

            total_saved += saved
            total_tiny += tiny

            if saved > 0:
                print(f"  {series_dir.name} [{mode}]: {saved} frames "
                      f"({tiny} tiny) → {split}")

    print(f"\n{'='*60}")
    print(f"Total: {total_saved} frames ({total_tiny} tiny <32x32)")
    print(f"Output: {output_dir}")

    # Count per split
    for split in ['train', 'val']:
        imgs = len(list((output_dir / 'images' / split).glob('*.jpg')))
        print(f"  {split}: {imgs} images")

    # Write data.yaml
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"path: {output_dir.absolute()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n  0: drone\n")
    print(f"data.yaml: {yaml_path}")


if __name__ == '__main__':
    main()
