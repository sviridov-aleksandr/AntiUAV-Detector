#!/usr/bin/env python3
"""
Simulate the full interceptor pipeline on a video file (no ROS2 needed).
Replicates vision_node logic:
- YOLO detection + Optical Flow fallback
- Kalman tracking + lead pursuit
- Range estimation (pinhole)
- State machine: SEARCH → TRACK → INTERCEPT → STRIKE
- Intercept strategy: pursuit (default) / head_on / top_dive
- Simulated cmd_vel output

Usage:
    python3 simulate_intercept.py video.mp4 [strategy] [kill_radius] [intercept_distance]
"""

import cv2
import numpy as np
import sys
import time

sys.path.insert(0, 'train')
from ultralytics import YOLO
from optical_flow_tracker import OpticalFlowTracker
from target_estimator import TargetEstimator, RangeEstimator, InterceptCalculator


class SimVision:
    """Replicates vision_node logic without ROS2."""

    def __init__(self, model_path, strategy='pursuit', kill_radius=4.0,
                 intercept_distance=8.0, conf=0.3):
        self.model = YOLO(model_path)
        self.of_tracker = OpticalFlowTracker()
        self.conf = conf
        self.strategy = strategy
        self.kill_radius = kill_radius
        self.intercept_distance = intercept_distance

        # Camera (v78: 1280x720, FOV 60deg)
        range_est = RangeEstimator.from_fov(fov_h_deg=60, image_width_px=1280,
                                            real_size_m=0.35)
        self.estimator = TargetEstimator(focal_px=range_est.focal_px,
                                         real_size_m=0.35, lead_frames=5)
        self.focal_px = range_est.focal_px
        self.intercept_calc = InterceptCalculator(interceptor_speed=15.0)

        # State
        self.state = 'SEARCH'
        self.lost_counter = 0
        self.max_lost_frames = 10
        self.strike_triggered = False
        self.of_fallback_counter = 0
        self.of_fallback_frames = 5
        self.detection_source = 'NONE'
        self.last_detection_source = None

        # Stats
        self.stats = {'yolo': 0, 'of': 0, 'strike_frames': 0}

    def run_frame(self, frame, dt=1/30):
        """Process one frame, return annotated frame."""
        h, w = frame.shape[:2]
        center_x, center_y = w / 2, h / 2

        drone_detected = False
        target_x, target_y = 0, 0
        bbox_ratio = 0.0
        self.current_bbox = None

        # --- YOLO ---
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
                # OSD filter (basic)
                if y1 < 60 or y2 > h - 60 or x1 < 60 or x2 > w - 60:
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

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                cv2.putText(frame, f"DRONE {bbox_ratio:.2f}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- OF fallback ---
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
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, "OF TRACK", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                self.of_fallback_counter += 1
            else:
                self.of_fallback_counter = 0
                self.of_tracker.reset()

        cmd_vel = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}
        distance = None
        info_text = ""

        if drone_detected:
            self.lost_counter = 0
            if self.detection_source != self.last_detection_source:
                print(f'  [{time.strftime("%H:%M:%S")}] Detection: {self.detection_source}')
                self.last_detection_source = self.detection_source

            # Estimator update with REAL bbox
            est_info = self.estimator.update(self.current_bbox, dt=dt)
            distance = est_info['distance']

            # Lead point
            lead = est_info['lead_point']
            aim_x, aim_y = lead if lead is not None else (target_x, target_y)

            # --- State transition ---
            if distance is not None and distance < self.kill_radius:
                self.state = 'STRIKE'
            elif distance is not None and distance < self.intercept_distance:
                self.state = 'INTERCEPT'
            elif bbox_ratio >= 0.35:
                self.state = 'INTERCEPT'
            else:
                self.state = 'TRACK'

            # --- Command generation per state ---
            if self.state == 'TRACK':
                # PID-like: aim at lead point
                err_x = aim_x - center_x
                err_y = aim_y - center_y
                cmd_vel['yaw'] = np.clip(err_x * 0.003, -0.5, 0.5)
                cmd_vel['z'] = np.clip(err_y * 0.003, -0.5, 0.5)
                cmd_vel['x'] = 0.3 if distance is None or distance > 10 else 0.2

            elif self.state == 'INTERCEPT':
                if self.strategy == 'head_on':
                    err_x = target_x - center_x
                    err_y = target_y - center_y
                    cmd_vel['yaw'] = np.clip(err_x * 0.003, -0.5, 0.5)
                    cmd_vel['z'] = np.clip(err_y * 0.003, -0.5, 0.5)
                    cmd_vel['x'] = 0.45
                    info_text = "HEAD-ON"

                elif self.strategy == 'top_dive':
                    target_upper = target_y < center_y - h * 0.15
                    cmd_vel['yaw'] = np.clip((target_x - center_x) * 0.003, -0.5, 0.5)
                    if target_upper:
                        cmd_vel['z'] = 0.24   # climb
                        cmd_vel['x'] = 0.15
                        info_text = "TOP-DIVE CLIMB"
                    else:
                        cmd_vel['z'] = -np.clip((target_y - center_y) * 0.003, -0.5, 0.5)
                        cmd_vel['x'] = 0.45
                        info_text = "TOP-DIVE DIVE"

                else:  # pursuit
                    vx_px, vy_px = est_info['velocity']
                    fps = 1.0 / dt if dt > 0 else 30.0
                    vel_scale = distance / self.focal_px * fps
                    target_vel_3d = (0.0, vx_px * vel_scale, vy_px * vel_scale)
                    intercept = self.intercept_calc.compute_from_image(
                        bbox_center=(aim_x, aim_y),
                        distance=distance,
                        target_vel_3d=target_vel_3d,
                        focal_px=self.focal_px,
                        image_center=(center_x, center_y))
                    if intercept['success']:
                        yaw_cmd, pitch_cmd = intercept['bearing']
                        cmd_vel['yaw'] = np.clip(yaw_cmd * 2.0, -0.5, 0.5)
                        cmd_vel['z'] = np.clip(-pitch_cmd * 2.0, -0.5, 0.5)
                        cmd_vel['x'] = 0.45
                        ip = intercept['intercept_point']
                        info_text = f"T={intercept['time_to_intercept']:.1f}s ({ip[0]:.0f},{ip[1]:.0f},{ip[2]:.0f})"

            elif self.state == 'STRIKE':
                cmd_vel['x'] = 0.45
                cmd_vel['yaw'] = 0.0
                cmd_vel['z'] = 0.0
                self.stats['strike_frames'] += 1
                if not self.strike_triggered:
                    self.strike_triggered = True
                    print(f'  *** STRIKE TRIGGERED *** dist={distance:.2f}m '
                          f'(kill_radius={self.kill_radius}m)')
                info_text = f"STRIKE! R={distance:.1f}m < {self.kill_radius}m"

            # --- Draw ---
            cv2.line(frame, (int(center_x), int(center_y)),
                     (int(aim_x), int(aim_y)), (255, 0, 0), 2)
            cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(aim_x), int(aim_y)), 4, (255, 255, 0), -1)

            state_color = (0, 255, 0) if self.state == 'TRACK' else (0, 0, 255)
            dist_str = f"{distance:.1f}m" if distance is not None else "N/A"
            status = (f"{self.state} [{self.detection_source}] "
                      f"Vx={cmd_vel['x']:.2f} Vz={cmd_vel['z']:.2f} "
                      f"Yaw={cmd_vel['yaw']:.2f} D={dist_str}")
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
            if info_text:
                cv2.putText(frame, info_text, (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        else:
            self.lost_counter += 1
            if self.lost_counter > self.max_lost_frames:
                self.state = 'SEARCH'
                self.of_tracker.reset()
                self.estimator.reset()
                self.strike_triggered = False
            cv2.putText(frame, self.state, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame, self.state, cmd_vel, distance


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else 'video-FPV/Video/v78.mp4'
    strategy = sys.argv[2] if len(sys.argv) > 2 else 'pursuit'
    kill_radius = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
    intercept_distance = float(sys.argv[4]) if len(sys.argv) > 4 else 8.0

    model_path = 'runs/detect/train/runs/drone_v2-4/weights/best.pt'
    sim = SimVision(model_path, strategy=strategy,
                    kill_radius=kill_radius, intercept_distance=intercept_distance)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"Cannot open {video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dt = 1.0 / fps

    print(f"Simulating {video} ({total} frames, {fps:.0f} FPS)")
    print(f"Strategy: {strategy} | kill_radius: {kill_radius}m | "
          f"intercept_distance: {intercept_distance}m")
    print("-" * 60)

    states = {}
    t_start = time.time()

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        frame, state, cmd, dist = sim.run_frame(frame, dt)
        states[state] = states.get(state, 0) + 1

        # Info overlay
        cv2.putText(frame, f"frame {i}/{total} | {sim.stats['yolo']} YOLO "
                    f"{sim.stats['of']} OF | {strategy}", (w := frame.shape[1]-380, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow(f"Interceptor Sim ({strategy})", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(0)  # pause

    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print("-" * 60)
    print(f"Results ({i+1} frames, {elapsed:.1f}s, {i/elapsed:.0f} FPS):")
    print(f"  States: {states}")
    print(f"  Detections: YOLO={sim.stats['yolo']} OF={sim.stats['of']}")
    print(f"  Strike frames: {sim.stats['strike_frames']}")
    if sim.strike_triggered:
        print("  ✓ BЧ подрыв сработал")


if __name__ == '__main__':
    main()
