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
    Оценка дальности до цели по размеру bbox (модель pinhole camera).
    distance = (real_size_m * focal_px) / bbox_px

    focal_px можно получить из калибровки камеры или вычислить из FOV:
      focal_px = (image_width / 2) / tan(fov_h / 2)
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
    Фильтр Калмана для трекинга цели в плоскости изображения.
    Модель постоянной скорости (constant velocity):
      state = [x, y, vx, vy]
      measurement = [x, y]

    Предоставляет:
    - Сглаженную позицию (фильтрация шумных детекций)
    - Оценку скорости (px/frame)
    - Предсказание позиции на N кадров вперёд (lead pursuit)
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


class InterceptCalculator:
    """
    Compute intercept point (lead pursuit in 3D) for a moving target.

    Given:
    - Target position (relative to interceptor, NED frame)
    - Target velocity (m/s, NED)
    - Interceptor max speed (m/s)

    Solves: find time T such that interceptor can reach the point
    where target will be at time T.

    Pure pursuit: aim at current target position.
    Lead pursuit: aim at predicted intercept point.
    """

    def __init__(self, interceptor_speed=10.0, max_lead_time=10.0):
        self.interceptor_speed = interceptor_speed  # m/s
        self.max_lead_time = max_lead_time  # seconds

    def compute(self, target_pos, target_vel):
        """
        Compute intercept point.

        Args:
            target_pos: (x, y, z) relative position in NED (m)
            target_vel: (vx, vy, vz) target velocity in NED (m/s)

        Returns:
            dict with:
                - intercept_point: (x, y, z) where to intercept
                - time_to_intercept: seconds
                - bearing: (yaw, pitch) angles to intercept point (rad)
                - success: bool
        """
        target_pos = np.array(target_pos, dtype=float)
        target_vel = np.array(target_vel, dtype=float)

        # Distance to target now
        dist_now = np.linalg.norm(target_pos)
        if dist_now < 1e-6:
            return {'success': False, 'reason': 'target at origin'}

        # Solve quadratic: |V_i * T|^2 = |P_t + V_t * T|^2
        # (V_i^2 - V_t^2) * T^2 - 2*(P_t·V_t)*T - |P_t|^2 = 0
        a = self.interceptor_speed**2 - np.dot(target_vel, target_vel)
        b = -2.0 * np.dot(target_pos, target_vel)
        c = -np.dot(target_pos, target_pos)

        T = None
        if abs(a) < 1e-9:
            # Linear case (interceptor speed == target speed)
            if abs(b) > 1e-9:
                T = -c / b
        else:
            disc = b**2 - 4 * a * c
            if disc >= 0:
                sqrt_disc = np.sqrt(disc)
                t1 = (-b + sqrt_disc) / (2 * a)
                t2 = (-b - sqrt_disc) / (2 * a)
                # Choose smallest positive root
                candidates = [t for t in (t1, t2) if t > 0]
                if candidates:
                    T = min(candidates)

        if T is None or T <= 0 or T > self.max_lead_time:
            # Cannot intercept in time — fall back to pure pursuit
            return {
                'success': False,
                'reason': f'no solution (T={T})',
                'intercept_point': tuple(target_pos),
                'time_to_intercept': dist_now / self.interceptor_speed,
                'bearing': self._bearing(target_pos),
            }

        # Intercept point = where target will be at time T
        intercept_point = target_pos + target_vel * T

        # Bearing to intercept point
        bearing = self._bearing(intercept_point)

        return {
            'success': True,
            'intercept_point': tuple(intercept_point),
            'time_to_intercept': T,
            'bearing': bearing,
        }

    def _bearing(self, point):
        """Compute yaw/pitch angles to a point in NED frame."""
        x, y, z = point
        yaw = np.arctan2(y, x)      # horizontal angle
        pitch = np.arctan2(-z, np.sqrt(x**2 + y**2))  # vertical (NED: z down)
        return (yaw, pitch)

    def compute_from_image(self, bbox_center, distance, target_vel_3d,
                           focal_px, image_center):
        """
        Compute intercept from image-plane measurements.
        Converts image position + distance to NED, then computes intercept.

        Args:
            bbox_center: (cx, cy) target center in image (px)
            distance: estimated distance to target (m)
            target_vel_3d: (vx, vy, vz) target velocity in NED (m/s)
            focal_px: camera focal length (px)
            image_center: (cx, cy) image center (px)

        Returns:
            dict with intercept info (see compute())
        """
        cx, cy = bbox_center
        icx, icy = image_center

        # Convert image offset to NED (camera frame: x forward, y right, z down)
        dx_px = cx - icx
        dy_px = cy - icy

        # Angular offsets
        yaw_off = np.arctan2(dx_px, focal_px)
        pitch_off = np.arctan2(dy_px, focal_px)

        # Position in NED (camera frame)
        x = distance * np.cos(pitch_off) * np.cos(yaw_off)
        y = distance * np.cos(pitch_off) * np.sin(yaw_off)
        z = distance * np.sin(pitch_off)

        return self.compute((x, y, z), target_vel_3d)


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

    print("\n=== Intercept Calculation ===")
    calc = InterceptCalculator(interceptor_speed=15.0)
    # Target 50m ahead, moving right at 5 m/s
    result = calc.compute((50, 0, 0), (0, 5, 0))
    print(f"  Target at (50,0,0), vel (0,5,0), interceptor 15 m/s:")
    print(f"    success={result['success']}")
    if result['success']:
        ip = result['intercept_point']
        T = result['time_to_intercept']
        yaw, pitch = result['bearing']
        print(f"    intercept at ({ip[0]:.1f}, {ip[1]:.1f}, {ip[2]:.1f}) in {T:.1f}s")
        print(f"    bearing: yaw={np.degrees(yaw):.1f}° pitch={np.degrees(pitch):.1f}°")

    # Target moving away (harder)
    result2 = calc.compute((50, 0, 0), (10, 0, 0))
    print(f"  Target at (50,0,0), vel (10,0,0) moving away:")
    print(f"    success={result2['success']}")
    if result2['success']:
        ip = result2['intercept_point']
        T = result2['time_to_intercept']
        print(f"    intercept at ({ip[0]:.1f}, {ip[1]:.1f}, {ip[2]:.1f}) in {T:.1f}s")

    # Target too fast (cannot intercept)
    result3 = calc.compute((50, 0, 0), (20, 0, 0))
    print(f"  Target at (50,0,0), vel (20,0,0) faster than interceptor:")
    print(f"    success={result3['success']} reason={result3.get('reason')}")