#!/usr/bin/env python3
"""
Пакетное тестирование best.pt на всех видео из video-FPV/Video/.
Headless-режим (без cv2.imshow), собирает статистику детекции и перехвата.

Usage:
    python3 train/batch_test_videos.py
"""

import csv
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, 'train')
from ultralytics import YOLO
from optical_flow_tracker import OpticalFlowTracker
from target_estimator import TargetEstimator, RangeEstimator, InterceptCalculator
from osd_filter import is_osd_false_positive

PROJECT = '/home/alex/AntiUAV-Detector'
VIDEO_DIR = os.path.join(PROJECT, 'video-FPV/Video')
WEIGHTS = os.path.join(PROJECT, 'runs/detect/train/runs/merged_v1_L/weights/best.pt')
OUTPUT_CSV = os.path.join(PROJECT, 'batch_test_results.csv')
STRATEGY = 'pursuit'
KILL_RADIUS = 4.0
INTERCEPT_DISTANCE = 8.0


class HeadlessVision:
    """SimVision без cv2.imshow — только метрики."""

    def __init__(self, model_path, strategy='pursuit', kill_radius=4.0,
                 intercept_distance=8.0, conf=0.3):
        self.model = YOLO(model_path)
        self.of_tracker = OpticalFlowTracker()
        self.conf = conf
        self.strategy = strategy
        self.kill_radius = kill_radius
        self.intercept_distance = intercept_distance

        range_est = RangeEstimator.from_fov(fov_h_deg=60, image_width_px=1280,
                                            real_size_m=0.35)
        self.estimator = TargetEstimator(focal_px=range_est.focal_px,
                                         real_size_m=0.35, lead_frames=5)
        self.focal_px = range_est.focal_px
        self.intercept_calc = InterceptCalculator(interceptor_speed=15.0)

        self.state = 'SEARCH'
        self.lost_counter = 0
        self.max_lost_frames = 10
        self.strike_triggered = False
        self.of_fallback_counter = 0
        self.of_fallback_frames = 5
        self.detection_source = 'NONE'
        self.last_detection_source = None
        self.stats = {'yolo': 0, 'of': 0, 'strike_frames': 0}
        self.current_bbox = None

    def run_frame(self, frame, dt=1/30):
        h, w = frame.shape[:2]
        center_x, center_y = w / 2, h / 2

        drone_detected = False
        target_x, target_y = 0, 0
        bbox_ratio = 0.0
        self.current_bbox = None

        results = self.model.track(frame, persist=True, verbose=False,
                                   conf=self.conf)
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            best = None
            best_area = 0
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls != 0:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                if is_osd_false_positive(x1, y1, x2, y2, w, h):
                    continue
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best = box
            if best is not None:
                x1, y1, x2, y2 = best.xyxy[0].cpu().numpy()
                target_x, target_y = (x1 + x2) / 2, (y1 + y2) / 2
                bbox_ratio = (x2 - x1) * (y2 - y1) / (w * h)
                drone_detected = True
                self.detection_source = 'YOLO'
                self.of_fallback_counter = 0
                self.stats['yolo'] += 1
                self.current_bbox = (int(x1), int(y1), int(x2), int(y2))

        if not drone_detected:
            if self.of_fallback_counter < self.of_fallback_frames:
                of_result = self.of_tracker.find_target(frame)
                if of_result is not None:
                    x1, y1, x2, y2, cx, cy = of_result
                    target_x, target_y = cx, cy
                    bbox_ratio = (x2 - x1) * (y2 - y1) / (w * h)
                    drone_detected = True
                    self.detection_source = 'OF'
                    self.stats['of'] += 1
                    self.current_bbox = (int(x1), int(y1), int(x2), int(y2))
                self.of_fallback_counter += 1
            else:
                self.of_fallback_counter = 0
                self.of_tracker.reset()

        distance = None

        if drone_detected:
            self.lost_counter = 0
            if self.current_bbox is not None:
                est_info = self.estimator.update(self.current_bbox, dt=dt)
                distance = est_info['distance']

            if distance is not None and distance < self.kill_radius:
                self.state = 'STRIKE'
            elif distance is not None and distance < self.intercept_distance:
                self.state = 'INTERCEPT'
            elif bbox_ratio >= 0.35:
                self.state = 'INTERCEPT'
            else:
                self.state = 'TRACK'
        else:
            self.lost_counter += 1
            if self.lost_counter > self.max_lost_frames:
                self.state = 'SEARCH'
                self.of_tracker.reset()
                self.estimator.reset()
                self.strike_triggered = False

        return self.state, distance

    def reset(self):
        self.state = 'SEARCH'
        self.lost_counter = 0
        self.strike_triggered = False
        self.of_fallback_counter = 0
        self.detection_source = 'NONE'
        self.last_detection_source = None
        self.stats = {'yolo': 0, 'of': 0, 'strike_frames': 0}
        self.of_tracker.reset()
        self.estimator.reset()


def main():
    # Собираем все видео
    videos = sorted([f for f in os.listdir(VIDEO_DIR)
                     if f.lower().endswith(('.mp4', '.mov', '.avi'))])

    print(f'Найдено видео: {len(videos)}')
    print(f'Модель: {WEIGHTS}')
    print(f'Стратегия: {STRATEGY} | kill_radius: {KILL_RADIUS}m | '
          f'intercept_distance: {INTERCEPT_DISTANCE}m')
    print('=' * 80)

    sim = HeadlessVision(WEIGHTS, strategy=STRATEGY,
                         kill_radius=KILL_RADIUS,
                         intercept_distance=INTERCEPT_DISTANCE)

    results = []
    t_total = time.time()

    for idx, video in enumerate(videos, 1):
        video_path = os.path.join(VIDEO_DIR, video)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f'  [{idx}/{len(videos)}] {video} — НЕ ОТКРЫВАЕТСЯ')
            results.append({
                'video': video, 'frames': 0, 'fps': 0,
                'yolo': 0, 'of': 0, 'search': 0, 'track': 0,
                'intercept': 0, 'strike': 0, 'strike_triggered': False,
                'min_distance': None, 'status': 'CANNOT_OPEN'
            })
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dt = 1.0 / fps

        sim.reset()
        states = {}
        min_distance = None
        t_start = time.time()

        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break

            state, dist = sim.run_frame(frame, dt)
            states[state] = states.get(state, 0) + 1

            if dist is not None:
                if min_distance is None or dist < min_distance:
                    min_distance = dist

            if state == 'STRIKE':
                sim.stats['strike_frames'] += 1
                if not sim.strike_triggered:
                    sim.strike_triggered = True

        cap.release()
        elapsed = time.time() - t_start
        processed = i + 1 if total > 0 else 0

        result = {
            'video': video,
            'frames': processed,
            'fps': round(fps, 1),
            'yolo': sim.stats['yolo'],
            'of': sim.stats['of'],
            'search': states.get('SEARCH', 0),
            'track': states.get('TRACK', 0),
            'intercept': states.get('INTERCEPT', 0),
            'strike': states.get('STRIKE', 0),
            'strike_triggered': sim.strike_triggered,
            'min_distance': round(min_distance, 2) if min_distance else None,
            'status': 'OK'
        }
        results.append(result)

        det_pct = (sim.stats['yolo'] / processed * 100) if processed > 0 else 0
        strike_str = f'✅ STRIKE @ {min_distance:.1f}m' if sim.strike_triggered else '❌ нет'
        print(f'  [{idx}/{len(videos)}] {video:20s} | {processed:5d} frames | '
              f'YOLO {sim.stats["yolo"]:4d} ({det_pct:4.1f}%) | OF {sim.stats["of"]:3d} | '
              f'{strike_str} | {elapsed:.1f}s')

    # Запись CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Сводка
    total_videos = len(results)
    ok_videos = sum(1 for r in results if r['status'] == 'OK')
    strike_videos = sum(1 for r in results if r['strike_triggered'])
    total_yolo = sum(r['yolo'] for r in results)
    total_of = sum(r['of'] for r in results)
    total_frames = sum(r['frames'] for r in results)

    print('\n' + '=' * 80)
    print(f'СВОДКА ({time.strftime("%Y-%m-%d %H:%M")})')
    print(f'  Видео обработано: {ok_videos}/{total_videos}')
    print(f'  Подрыв сработал:  {strike_videos}/{ok_videos}')
    print(f'  Всего кадров:    {total_frames}')
    print(f'  YOLO детекций:   {total_yolo} ({total_yolo/total_frames*100:.1f}%)')
    print(f'  OF детекций:     {total_of}')
    print(f'  Средний FPS:     {total_frames/(time.time()-t_total):.1f}')
    print(f'  CSV: {OUTPUT_CSV}')
    print('=' * 80)


if __name__ == '__main__':
    main()
