#!/usr/bin/env python3
"""
Optical Flow Tracker for UAV Interceptor.
Dense optical flow (Farneback) for detecting small/fast moving targets
that YOLO cannot detect (distant drones < 10px).

Strategy:
1. Compute dense optical flow between consecutive frames.
2. Compensate global motion (camera ego-motion) via affine transform.
3. Build motion mask (pixels moving above threshold, after compensation).
4. Exclude edge zones (OSD text/indicators).
5. Find connected components in motion mask.
6. Filter candidates by size, speed, aspect ratio, position.
7. Track the selected target across frames with history smoothing.

Usage (standalone test):
    python3 optical_flow_tracker.py video.mp4
"""

import cv2
import numpy as np


class OpticalFlowTracker:
    def __init__(self,
                 flow_scale=0.5,          # Downscale factor for flow computation
                 motion_threshold=1.5,    # Min pixel displacement (absolute px, scaled)
                 min_area=4,              # Min blob area (in scaled px)
                 max_area_ratio=0.03,     # Max blob area relative to frame
                 min_speed=2.0,           # Min mean magnitude in blob (absolute px)
                 hist_len=5,              # History length for smoothing
                 edge_margin=0.08,        # Fraction of frame to exclude on each side
                 min_persistence=3,       # Min consecutive detections to confirm target
                 max_gap=2):              # Max missed frames before target lost
        self.flow_scale = flow_scale
        self.motion_threshold = motion_threshold
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.min_speed = min_speed
        self.hist_len = hist_len
        self.edge_margin = edge_margin
        self.min_persistence = min_persistence
        self.max_gap = max_gap

        self.prev_gray = None
        self.prev_full_gray = None
        self.positions = []        # Recent positions history
        self.target_bbox = None    # Current target bbox (x1,y1,x2,y2)
        self.last_motion = None    # Motion mask (for debug)
        self.affine = None

        # Persistence tracking
        self.persistence_count = 0
        self.gap_count = 0
        self.confirmed = False
        self.last_center = None

    def reset(self):
        self.prev_gray = None
        self.prev_full_gray = None
        self.positions = []
        self.target_bbox = None
        self.affine = None
        self.persistence_count = 0
        self.gap_count = 0
        self.confirmed = False
        self.last_center = None

    def _compute_global_motion(self, prev, curr):
        """Estimate global camera motion via sparse feature matching."""
        if prev is None:
            return None
        s = self.flow_scale
        pv = cv2.resize(prev, None, fx=s, fy=s)
        cv_ = cv2.resize(curr, None, fx=s, fy=s)

        features = cv2.goodFeaturesToTrack(pv, maxCorners=100, qualityLevel=0.01,
                                           minDistance=5)
        if features is None or len(features) < 10:
            return None

        try:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(pv, cv_,
                                                           features, None,
                                                           winSize=(15, 15),
                                                           maxLevel=3)
        except cv2.error:
            return None

        good_prev = features[status.flatten() == 1]
        good_next = next_pts[status.flatten() == 1]
        if len(good_prev) < 10:
            return None

        try:
            self.affine = cv2.estimateAffinePartial2D(good_prev, good_next)[0]
        except cv2.error:
            self.affine = None
        return self.affine

    def _is_in_edge_zone(self, x, y, bw, bh, frame_w, frame_h):
        """Check if blob center is in excluded edge zone (OSD area)."""
        margin_x = frame_w * self.edge_margin
        margin_y = frame_h * self.edge_margin
        cx = x + bw / 2
        cy = y + bh / 2
        return (cx < margin_x or cx > frame_w - margin_x or
                cy < margin_y or cy > frame_h - margin_y)

    def find_candidates(self, frame, max_candidates=5):
        """
        Find ALL moving objects (candidates) in frame via optical flow.
        Returns list of dicts: {bbox: (x1,y1,x2,y2), center: (cx,cy), area, speed}
        Used as region proposals for YOLO verification (small/distant targets).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_full_gray = gray
            return []

        s = self.flow_scale
        small_prev = cv2.resize(self.prev_gray, None, fx=s, fy=s)
        small_curr = cv2.resize(gray, None, fx=s, fy=s)
        small_h, small_w = small_curr.shape[:2]

        # Global motion compensation
        self._compute_global_motion(self.prev_full_gray, gray)
        compensated_prev = small_prev
        if self.affine is not None:
            compensated_prev = cv2.warpAffine(
                small_prev, self.affine,
                (small_w, small_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE)

        # Dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            compensated_prev, small_curr,
            None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Motion mask
        motion = (mag > self.motion_threshold).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel, iterations=1)
        motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Edge zone mask
        mx = int(small_w * self.edge_margin)
        my = int(small_h * self.edge_margin)
        motion[0:my, :] = 0
        motion[small_h-my:small_h, :] = 0
        motion[:, 0:mx] = 0
        motion[:, small_w-mx:small_w] = 0

        self.last_motion = motion

        # Connected components
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(motion)
        frame_area = small_h * small_w

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
                if ar > 5.0 or ar < 0.2:
                    continue

            blob_mask = (labels == i).astype(np.uint8)
            mean_mag = cv2.mean(mag, mask=blob_mask)[0]
            if mean_mag < self.min_speed:
                continue

            # Scale to full resolution
            scale = 1.0 / self.flow_scale
            x1 = max(0, int(x * scale))
            y1 = max(0, int(y * scale))
            x2 = min(w, int((x + bw) * scale))
            y2 = min(h, int((y + bh) * scale))
            cx_full = int(cx * scale)
            cy_full = int(cy * scale)

            candidates.append({
                'bbox': (x1, y1, x2, y2),
                'center': (cx_full, cy_full),
                'area': area,
                'speed': mean_mag,
            })

        # Sort by area (largest first), limit count
        candidates.sort(key=lambda c: c['area'], reverse=True)
        candidates = candidates[:max_candidates]

        self.prev_gray = gray
        self.prev_full_gray = gray
        return candidates

    def find_target(self, frame):
        """
        Detect moving target in frame using optical flow.
        Returns (x1, y1, x2, y2, center_x, center_y) or None.
        Only returns after min_persistence consecutive detections.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_full_gray = gray
            return None

        s = self.flow_scale
        small_prev = cv2.resize(self.prev_gray, None, fx=s, fy=s)
        small_curr = cv2.resize(gray, None, fx=s, fy=s)
        small_h, small_w = small_curr.shape[:2]

        # Global motion compensation
        self._compute_global_motion(self.prev_full_gray, gray)
        compensated_prev = small_prev
        if self.affine is not None:
            compensated_prev = cv2.warpAffine(
                small_prev, self.affine,
                (small_w, small_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE)

        # Dense optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            compensated_prev, small_curr,
            None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Magnitude (absolute pixel displacement)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Motion mask
        motion = (mag > self.motion_threshold).astype(np.uint8) * 255

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel, iterations=1)
        motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Edge zone mask: zero out margins
        mx = int(small_w * self.edge_margin)
        my = int(small_h * self.edge_margin)
        motion[0:my, :] = 0
        motion[small_h-my:small_h, :] = 0
        motion[:, 0:mx] = 0
        motion[:, small_w-mx:small_w] = 0

        self.last_motion = motion

        # Connected components
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(motion)
        frame_area = small_h * small_w

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
                if ar > 5.0 or ar < 0.2:
                    continue

            blob_mask = (labels == i).astype(np.uint8)
            mean_mag = cv2.mean(mag, mask=blob_mask)[0]
            if mean_mag < self.min_speed:
                continue

            candidates.append((area, x, y, bw, bh, cx, cy, mean_mag))

        candidates.sort(reverse=True)

        raw_result = None
        if candidates:
            area, x, y, bw, bh, cx, cy, mean_mag = candidates[0]
            scale = 1.0 / self.flow_scale
            x1 = max(0, int(x * scale))
            y1 = max(0, int(y * scale))
            x2 = min(w, int((x + bw) * scale))
            y2 = min(h, int((y + bh) * scale))
            center_x = int(cx * scale)
            center_y = int(cy * scale)
            raw_result = (x1, y1, x2, y2, center_x, center_y)

            # Persistence logic
            if self.last_center is not None:
                dist = np.sqrt((center_x - self.last_center[0])**2 +
                               (center_y - self.last_center[1])**2)
                max_jump = 100  # px in full scale
                if dist > max_jump:
                    # Target jumped — reset
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

        self.prev_gray = gray
        self.prev_full_gray = gray
        return result


def test_on_video(video_path, max_frames=None, save_debug=False):
    """Test optical flow tracker on a video."""
    import os, time
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return

    tracker = OpticalFlowTracker()
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)

    detections = 0
    times = []
    print(f"Testing {video_path} ({total} frames)...")

    if save_debug:
        os.makedirs('/tmp/of_debug', exist_ok=True)

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
                cv2.imwrite(f'/tmp/of_debug/det_{frame_idx}.png', frame)

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