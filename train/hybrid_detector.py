#!/usr/bin/env python3
"""
Hybrid Detection: Optical Flow region proposals + YOLO verification.
OF finds moving candidates (cheap), YOLO verifies each in upscaled ROI (accurate).
Key benefit: upscaling small ROIs makes distant targets visible to YOLO.

Usage:
    python3 hybrid_detector.py video.mp4
"""

import cv2
import numpy as np
import sys
import time

sys.path.insert(0, 'train')
from optical_flow_tracker import OpticalFlowTracker
from ultralytics import YOLO


class HybridDetector:
    def __init__(self, model_path, conf_threshold=0.3, roi_target_size=640,
                 roi_pad=2.0, max_candidates=2):
        self.model = YOLO(model_path)
        self.of_tracker = OpticalFlowTracker()
        self.conf_threshold = conf_threshold
        self.roi_target_size = roi_target_size  # YOLO imgsz
        self.roi_pad = roi_pad
        self.max_candidates = max_candidates

        # Stats
        self.stats = {'of_candidates': 0, 'yolo_verified': 0,
                      'yolo_full_frame': 0, 'roi_upscaled': 0}

    def _extract_roi(self, frame, bbox):
        """Extract padded ROI around bbox, resized to YOLO input size."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1

        # Pad around candidate (keep aspect, pad to square-ish)
        pad = max(bw, bh) * (self.roi_pad - 1) / 2
        pad_x = int(pad)
        pad_y = int(pad)
        rx1 = max(0, int(x1 - pad_x))
        ry1 = max(0, int(y1 - pad_y))
        rx2 = min(w, int(x2 + pad_x))
        ry2 = min(h, int(y2 + pad_y))

        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None, None

        roi_h, roi_w = roi.shape[:2]

        # Resize to YOLO input size (640x640), preserving aspect
        scale = self.roi_target_size / max(roi_w, roi_h)
        new_w = int(roi_w * scale)
        new_h = int(roi_h * scale)
        roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Pad to square with black border
        canvas = np.zeros((self.roi_target_size, self.roi_target_size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = roi_resized

        if scale > 1.5:
            self.stats['roi_upscaled'] += 1

        return canvas, (rx1, ry1, scale)

    def detect(self, frame):
        """
        Hybrid detection: OF proposals → YOLO verification on upscaled ROI.
        Falls back to full-frame YOLO only when no OF candidates exist.
        Returns list of confirmed detections.
        """
        h, w = frame.shape[:2]
        detections = []

        # Step 1: OF region proposals
        candidates = self.of_tracker.find_candidates(frame, max_candidates=self.max_candidates)
        self.stats['of_candidates'] += len(candidates)

        # Step 2: YOLO verification on each ROI
        for cand in candidates:
            bbox = cand['bbox']
            roi, meta = self._extract_roi(frame, bbox)
            if roi is None:
                continue

            results = self.model.predict(roi, verbose=False,
                                         conf=self.conf_threshold, imgsz=640)
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                rx1, ry1, scale = meta
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    if cls != 0:
                        continue
                    conf = float(box.conf[0])
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    # Map back to full frame
                    fx1 = int(rx1 + bx1 / scale)
                    fy1 = int(ry1 + by1 / scale)
                    fx2 = int(rx1 + bx2 / scale)
                    fy2 = int(ry1 + by2 / scale)
                    detections.append({
                        'bbox': (fx1, fy1, fx2, fy2),
                        'conf': conf,
                        'source': 'yolo_roi',
                    })
                    self.stats['yolo_verified'] += 1

        # Step 3: Full-frame YOLO only if no OF candidates at all
        if not candidates:
            results = self.model.predict(frame, verbose=False,
                                         conf=self.conf_threshold, imgsz=640)
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    if cls != 0:
                        continue
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'conf': conf,
                        'source': 'yolo_full',
                    })
                    self.stats['yolo_full_frame'] += 1

        return detections


def test_on_video(video_path, max_frames=None):
    model_path = '/home/alex/AntiUAV-Detector/runs/detect/train/runs/drone_v2-4/weights/best.pt'
    detector = HybridDetector(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)

    frames_with_det = 0
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
            frames_with_det += 1

        # Draw
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = (0, 255, 0) if det['source'] == 'yolo_full' else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{det['source']} {det['conf']:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"frame {i} det={len(detections)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Hybrid Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    avg = sum(times) / len(times) if times else 0
    print(f"\nResults ({total} frames):")
    print(f"  Frames with detection: {frames_with_det}/{total} ({frames_with_det/total*100:.1f}%)")
    print(f"  Avg time: {avg:.1f} ms ({1000/avg:.0f} FPS)")
    print(f"  Stats: {detector.stats}")


if __name__ == '__main__':
    video = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/alex/AntiUAV-Detector/video-FPV/Video/v3.mp4'
    test_on_video(video)
