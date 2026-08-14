#!/usr/bin/env python3
"""
IR Tracker for UAV Interceptor (thermal camera).
Detects drones in thermal/IR imagery as bright hot spots.

Strategy:
1. Convert frame to grayscale (thermal image is single-channel).
2. Adaptive threshold (or fixed) to isolate hot spots (drone motor/body).
3. Morphology (open + close) to remove noise and merge nearby blobs.
4. Connected components → candidates.
5. Filter by size, aspect ratio, intensity, position (edge zones).
6. Track selected target with history smoothing + persistence.

The drone in IR appears as a bright blob (hot motor/electronics) against
a cooler background (sky/ground). This is complementary to YOLO (visible
spectrum) and Optical Flow (motion-based).

Usage (standalone test):
    python3 ir_tracker.py video.mp4 [--save]
"""

import cv2
import numpy as np


class IRTracker:
    def __init__(self,
                 threshold_mode='adaptive',  # 'adaptive' | 'fixed' | 'otsu'
                 fixed_threshold=200,        # for 'fixed' mode (0-255)
                 adaptive_block=51,          # block size for adaptive threshold
                 adaptive_c=15,              # C constant for adaptive threshold
                 min_area=6,                 # Min blob area (px)
                 max_area_ratio=0.05,        # Max blob area relative to frame
                 min_intensity=0.6,          # Min mean intensity in blob (0-1)
                 max_aspect_ratio=4.0,       # Max width/height ratio
                 hist_len=5,                 # History length for smoothing
                 edge_margin=0.06,           # Fraction of frame to exclude
                 min_persistence=3,          # Min consecutive detections
                 max_gap=3):                 # Max missed frames before lost
        self.threshold_mode = threshold_mode
        self.fixed_threshold = fixed_threshold
        self.adaptive_block = adaptive_block
        self.adaptive_c = adaptive_c
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.min_intensity = min_intensity
        self.max_aspect_ratio = max_aspect_ratio
        self.hist_len = hist_len
        self.edge_margin = edge_margin
        self.min_persistence = min_persistence
        self.max_gap = max_gap

        self.positions = []
        self.target_bbox = None
        self.last_mask = None

        # Persistence tracking
        self.persistence_count = 0
        self.gap_count = 0
        self.confirmed = False
        self.last_center = None

    def reset(self):
        self.positions = []
        self.target_bbox = None
        self.last_mask = None
        self.persistence_count = 0
        self.gap_count = 0
        self.confirmed = False
        self.last_center = None

    def _threshold(self, gray):
        """Пороговая обработка для выделения горячих пятен.
        Режимы:
          fixed — фиксированный порог (для стабильного фона)
          otsu — автоматический порог Оцу (для контрастных сцен)
          adaptive — адаптивный порог (для неравномерного фона)"""
        if self.threshold_mode == 'fixed':
            _, mask = cv2.threshold(gray, self.fixed_threshold, 255,
                                    cv2.THRESH_BINARY)
        elif self.threshold_mode == 'otsu':
            _, mask = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:  # adaptive
            mask = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, self.adaptive_block, self.adaptive_c)
        return mask

    def _is_in_edge_zone(self, x, y, bw, bh, frame_w, frame_h):
        margin_x = frame_w * self.edge_margin
        margin_y = frame_h * self.edge_margin
        cx = x + bw / 2
        cy = y + bh / 2
        return (cx < margin_x or cx > frame_w - margin_x or
                cy < margin_y or cy > frame_h - margin_y)

    def find_target(self, frame):
        """Детекция горячей цели в IR-кадре.
        Возвращает (x1, y1, x2, y2, cx, cy) или None.
        Возвращает результат только после min_persistence
        последовательных детекций (подавление ложных срабатываний)."""
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        h, w = gray.shape[:2]

        # Threshold
        mask = self._threshold(gray)

        # Morphology: open (remove noise) + close (merge nearby blobs)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Edge zone exclusion
        mx = int(w * self.edge_margin)
        my = int(h * self.edge_margin)
        mask[0:my, :] = 0
        mask[h-my:h, :] = 0
        mask[:, 0:mx] = 0
        mask[:, w-mx:w] = 0

        self.last_mask = mask

        # Connected components
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        frame_area = h * w

        candidates = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            cx, cy = centroids[i]

            if area < self.min_area:
                continue
            if area > frame_area * self.max_area_ratio:
                continue

            if bw > 0 and bh > 0:
                ar = bw / bh
                if ar > self.max_aspect_ratio or ar < 1.0 / self.max_aspect_ratio:
                    continue

            # Mean intensity in blob (normalized 0-1)
            blob_mask = (labels == i).astype(np.uint8)
            mean_int = cv2.mean(gray, mask=blob_mask)[0] / 255.0
            if mean_int < self.min_intensity:
                continue

            candidates.append((area, x, y, bw, bh, cx, cy, mean_int))

        candidates.sort(reverse=True)

        raw_result = None
        if candidates:
            area, x, y, bw, bh, cx, cy, mean_int = candidates[0]
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w, int(x + bw))
            y2 = min(h, int(y + bh))
            center_x = int(cx)
            center_y = int(cy)
            raw_result = (x1, y1, x2, y2, center_x, center_y)

            # Persistence logic
            if self.last_center is not None:
                dist = np.sqrt((center_x - self.last_center[0])**2 +
                               (center_y - self.last_center[1])**2)
                max_jump = 80  # px
                if dist > max_jump:
                    self.persistence_count = 1
                    self.confirmed = False
                else:
                    self.persistence_count += 1
                    self.gap_count = 0
            else:
                self.persistence_count = 1

            self.last_center = (center_x, center_y)

            if self.persistence_count >= self.min_persistence:
                self.confirmed = True
        else:
            self.gap_count += 1
            if self.gap_count > self.max_gap:
                self.persistence_count = 0
                self.confirmed = False
                self.last_center = None

        # Build result
        result = None
        if self.confirmed and raw_result:
            x1, y1, x2, y2, center_x, center_y = raw_result
            self.positions.append((center_x, center_y))
            if len(self.positions) > self.hist_len:
                self.positions.pop(0)

            if len(self.positions) >= 3:
                pts = np.array(self.positions)
                center_x = int(pts[-3:, 0].mean())
                center_y = int(pts[-3:, 1].mean())

            self.target_bbox = (x1, y1, x2, y2)
            result = (x1, y1, x2, y2, center_x, center_y)
        elif not self.confirmed:
            self.target_bbox = None
            if len(self.positions) > 0:
                self.positions.pop(0)

        return result


def test_on_video(video_path, max_frames=None, save_debug=False):
    """Test IR tracker on a video."""
    import os, time
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return

    tracker = IRTracker()
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)

    detections = 0
    times = []
    print(f"Testing {video_path} ({total} frames)...")

    if save_debug:
        os.makedirs('/tmp/ir_debug', exist_ok=True)

    frame_idx = 0
    while frame_idx < total:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        result = tracker.find_target(frame)
        dt = (time.time() - t0) * 1000
        times.append(dt)

        if result:
            detections += 1
            if save_debug and detections <= 10:
                x1, y1, x2, y2, cx, cy = result
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                cv2.imwrite(f'/tmp/ir_debug/det_{frame_idx}.png', frame)

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    avg_time = sum(times) / len(times) if times else 0
    avg_fps = 1000 / avg_time if avg_time > 0 else 0
    hit_rate = detections / total * 100 if total > 0 else 0
    print(f"  Detections: {detections}/{total} ({hit_rate:.1f}%) | "
          f"{avg_time:.1f} ms/frame ({avg_fps:.0f} FPS)")


if __name__ == '__main__':
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/alex/AntiUAV-Detector/video-FPV/Video/v2.mp4'
    save = '--save' in sys.argv
    test_on_video(video, save_debug=save)
