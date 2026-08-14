#!/usr/bin/env python3
"""
Range Estimation + Kalman Filter for UAV Interceptor.

1. Range estimation from bbox size (pinhole camera model):
   distance = (real_size * focal_px) / bbox_px

2. Kalman filter (constant velocity model) for target tracking:
   - Predicts target position ahead of time (lead pursuit)
   - Smooths noisy detections
   - Provides velocity estimate for intercept point calculation

State vector: [x, y, vx, vy] (image plane, pixels + px/frame)
"""

import numpy as np


class RangeEstimator:
    """
    Estimate distance to target from bbox size using pinhole model.
    distance = (real_size_m * focal_px) / bbox_px

    Requires calibration:
    - focal_px: camera focal length in pixels (from calibration or FOV)
    - real_size_m: physical size of target (drone) in meters
    """

    def __init__(self, focal_px=800.0, real_size_m=0.35,
                 sensor_width_m=0.0064, image_width_px=1280):
        """
        focal_px: focal length in pixels.
        If unknown, estimate from FOV: focal_px = (image_width/2) / tan(fov_h/2)
        """
        self.focal_px = focal_px
        self.real_size_m = real_size_m
        self.sensor_width_m = sensor_width_m
        self.image_width_px = image_width_px

    @classmethod
    def from_fov(cls, fov_h_deg, image_width_px, real_size_m=0.35):
        """Create estimator from horizontal FOV (degrees)."""
        fov_h_rad = np.radians(fov_h_deg)
        focal_px = (image_width_px / 2) / np.tan(fov_h_rad / 2)
        return cls(focal_px=focal_px, real_size_m=real_size_m,
                   image_width_px=image_width_px)

    def estimate(self, bbox_width_px, bbox_height_px=None):
        """
        Estimate distance from bbox size.
        Uses max dimension (width or height) for robustness.
        Returns distance in meters, or None if bbox too small.
        """
        if bbox_width_px <= 0:
            return None

        # Use the larger dimension (more reliable)
        size_px = bbox_width_px
        if bbox_height_px and bbox_height_px > bbox_width_px:
            size_px = bbox_height_px

        if size_px < 2:  # too small, unreliable
            return None

        distance = (self.real_size_m * self.focal_px) / size_px
        return distance

    def estimate_velocity(self, distance_prev, distance_now, dt):
        """
        Estimate closing velocity (m/s) from distance change.
        Negative = approaching.
        """
        if distance_prev is None or distance_now is None or dt <= 0:
            return 0.0
        return (distance_now - distance_prev) / dt


class KalmanTracker:
    """
    Kalman filter for target tracking in image plane.
    Constant velocity model:
      state = [x, y, vx, vy]
      measurement = [x, y]

    Provides:
    - Smoothed position
    - Velocity estimate (px/frame)
    - Predicted position N frames ahead (lead pursuit)
    """

    def __init__(self, dt=1.0, process_noise=1e-2, measurement_noise=1e-1):
        self.dt = dt

        # State: [x, y, vx, vy]
        self.state = np.zeros(4)
        self.P = np.eye(4) * 100.0  # Initial covariance

        # State transition (constant velocity)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        # Measurement matrix (observe x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        # Process noise (acceleration uncertainty)
        self.Q = np.eye(4) * process_noise

        # Measurement noise
        self.R = np.eye(2) * measurement_noise

        self.initialized = False

    def init(self, x, y):
        """Initialize filter with first measurement."""
        self.state = np.array([x, y, 0.0, 0.0])
        self.P = np.eye(4) * 100.0
        self.initialized = True

    def predict(self):
        """Predict next state (no measurement)."""
        if not self.initialized:
            return None
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2].copy()

    def update(self, x, y):
        """Update with measurement."""
        if not self.initialized:
            self.init(x, y)
            return self.state[:2].copy()

        # Predict
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update
        z = np.array([x, y])
        y_innov = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y_innov
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.state[:2].copy()

    def predict_ahead(self, frames_ahead):
        """
        Predict target position N frames ahead (lead pursuit).
        Returns (x, y) predicted position.
        """
        if not self.initialized:
            return None
        x, y, vx, vy = self.state
        return (x + vx * frames_ahead, y + vy * frames_ahead)

    def get_velocity(self):
        """Return velocity (vx, vy) in px/frame."""
        if not self.initialized:
            return (0.0, 0.0)
        return (self.state[2], self.state[3])

    def get_position(self):
        """Return current smoothed position."""
        if not self.initialized:
            return None
        return (self.state[0], self.state[1])

    def reset(self):
        self.initialized = False
        self.state = np.zeros(4)
        self.P = np.eye(4) * 100.0


class TargetEstimator:
    """
    Combined: range estimation + Kalman tracking + lead pursuit.
    High-level interface for vision_node.
    """

    def __init__(self, focal_px=800.0, real_size_m=0.35,
                 lead_frames=5, process_noise=1e-2, measurement_noise=1e-1):
        self.range_est = RangeEstimator(focal_px=focal_px,
                                        real_size_m=real_size_m)
        self.kalman = KalmanTracker(process_noise=process_noise,
                                    measurement_noise=measurement_noise)
        self.lead_frames = lead_frames

        # History
        self.last_distance = None
        self.distances = []

    def update(self, bbox, dt=1.0):
        """
        Update with new detection bbox (x1, y1, x2, y2).
        Returns dict with:
            - position: (x, y) smoothed
            - velocity: (vx, vy) px/frame
            - distance: meters
            - closing_speed: m/s (negative = approaching)
            - lead_point: (x, y) predicted ahead
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        bw = x2 - x1
        bh = y2 - y1

        # Kalman update
        pos = self.kalman.update(cx, cy)
        vel = self.kalman.get_velocity()

        # Range estimation
        distance = self.range_est.estimate(bw, bh)
        closing_speed = 0.0
        if distance is not None and self.last_distance is not None:
            closing_speed = self.range_est.estimate_velocity(
                self.last_distance, distance, dt)
        self.last_distance = distance

        # Lead point (predicted position ahead)
        lead_point = self.kalman.predict_ahead(self.lead_frames)

        return {
            'position': pos,
            'velocity': vel,
            'distance': distance,
            'closing_speed': closing_speed,
            'lead_point': lead_point,
        }

    def reset(self):
        self.kalman.reset()
        self.last_distance = None
        self.distances = []


if __name__ == '__main__':
    # Self-test
    print("=== Range Estimation ===")
    est = RangeEstimator.from_fov(fov_h_deg=60, image_width_px=1280,
                                  real_size_m=0.35)
    print(f"Focal: {est.focal_px:.0f} px")
    for bbox_px in [100, 50, 20, 10, 5]:
        d = est.estimate(bbox_px)
        print(f"  bbox={bbox_px}px → distance={d:.1f} m")

    print("\n=== Kalman Filter ===")
    kf = KalmanTracker()
    # Simulate target moving right at 5 px/frame
    x = 100
    for i in range(20):
        x += 5
        pos = kf.update(x, 200)
        if i >= 15:
            lead = kf.predict_ahead(5)
            vel = kf.get_velocity()
            print(f"  frame {i}: pos=({pos[0]:.1f},{pos[1]:.1f}) "
                  f"vel=({vel[0]:.1f},{vel[1]:.1f}) lead=({lead[0]:.1f},{lead[1]:.1f})")