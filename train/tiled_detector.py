#!/usr/bin/env python3
"""
Tiled Detection for small/distant targets.
Splits frame into overlapping tiles, runs YOLO on each tile,
merges detections back to full-frame coordinates.

Key benefit: a small target (e.g. 20x20 px in 1280x800 frame) becomes
a larger target within a 640x640 tile → YOLO detects it more reliably.

Usage:
    python3 tiled_detector.py video.mp4
"""

import cv2
import numpy as np
import time
import sys

from ultralytics import YOLO


class TiledDetector:
    def __init__(self, model_path, conf_threshold=0.3, tile_size=640,
                 overlap=0.25, min_tile_scale=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_tile_scale = min_tile_scale  # min tile size relative to frame

        self.stats = {'tiles': 0, 'detections': 0, 'roi_dets': 0}

    def _compute_tiles(self, w, h):
        """Compute tile grid with overlap."""
        tile_w = min(self.tile_size, w)
        tile_h = min(self.tile_size, h)

        # If frame smaller than tile, single tile
        if w <= tile_w and h <= tile_h:
            return [(0, 0, w, h)]

        step_x = int(tile_w * (1 - self.overlap))
        step_y = int(tile_h * (1 - self.overlap))
        step_x = max(step_x, 1)
        step_y = max(step_y, 1)

        tiles = []
        y = 0
        while y < h:
            x = 0
            while x < w:
                x1 = x
                y1 = y
                x2 = min(x + tile_w, w)
                y2 = min(y + tile_h, h)
                tiles.append((x1, y1, x2, y2))
                x += step_x
            y += step_y

        # Ensure last tile covers the edge
        if tiles:
            last_x1, last_y1, last_x2, last_y2 = tiles[-1]
            if last_x2 < w:
                tiles.append((w - tile_w, last_y1, w, last_y2))
            if last_y2 < h:
                tiles.append((last_x1, h - tile_h, last_x2, h))
            if last_x2 < w and last_y2 < h:
                tiles.append((w - tile_w, h - tile_h, w, h))

        return tiles

    def detect(self, frame):
        """
        Run tiled detection on frame.
        Returns list of detections: [{bbox: (x1,y1,x2,y2), conf, source}]
        """
        h, w = frame.shape[:2]
        tiles = self._compute_tiles(w, h)
        self.stats['tiles'] += len(tiles)

        detections = []

        for (x1, y1, x2, y2) in tiles:
            tile = frame[y1:y2, x1:x2]
            if tile.size == 0:
                continue

            results = self.model.predict(tile, verbose=False,
                                         conf=self.conf_threshold, imgsz=640)
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    if cls != 0:
                        continue
                    conf = float(box.conf[0])
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    # Map tile coords to full frame
                    fx1 = int(x1 + bx1)
                    fy1 = int(y1 + by1)
                    fx2 = int(x1 + bx2)
                    fy2 = int(y1 + by2)
                    detections.append({
                        'bbox': (fx1, fy1, fx2, fy2),
                        'conf': conf,
                        'source': 'tile',
                    })
                    self.stats['detections'] += 1

        # NMS to remove duplicates from overlapping tiles
        if len(detections) > 1:
            detections = self._nms(detections)

        return detections

    def _nms(self, detections, iou_threshold=0.5):
        """Simple NMS for overlapping tile detections."""
        boxes = np.array([d['bbox'] for d in detections], dtype=np.float32)
        scores = np.array([d['conf'] for d in detections])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter_w = np.maximum(0, xx2 - xx1)
            inter_h = np.maximum(0, yy2 - yy1)
            inter = inter_w * inter_h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= iou_threshold]

        return [detections[i] for i in keep]


def test_on_video(video_path, max_frames=None):
    model_path = '/home/alex/AntiUAV-Detector/runs/detect/train/runs/drone_v2-4/weights/best.pt'
    detector = TiledDetector(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)

    frames_det = 0
    times = []
    print(f"Testing {video_path} ({total} frames)...")

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        detections = detector.detect(frame)
        dt = (time.time() - t0) * 1000
        times.append(dt)

        if detections:
            frames_det += 1

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{det['conf']:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f"frame {i} det={len(detections)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Tiled Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    avg = sum(times) / len(times) if times else 0
    print(f"\nResults ({total} frames):")
    print(f"  Frames with detection: {frames_det}/{total} ({frames_det/total*100:.1f}%)")
    print(f"  Avg time: {avg:.1f} ms ({1000/avg:.0f} FPS)")
    print(f"  Stats: {detector.stats}")


if __name__ == '__main__':
    video = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/alex/AntiUAV-Detector/video-FPV/Video/v3.mp4'
    test_on_video(video)
