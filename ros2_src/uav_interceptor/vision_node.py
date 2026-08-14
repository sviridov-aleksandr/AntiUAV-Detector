#!/usr/bin/env python3
"""
ROS2 Node: Visual Servoing for UAV Interceptor with REAL YOLO detection.
Subscribes to camera, detects drone using YOLO, computes error, publishes cmd_vel.
State machine: SEARCH → TRACK → INTERCEPT → STRIKE → LOST → SEARCH.

Hybrid detection:
- YOLO: primary detector (stable, accurate)
- Optical Flow: fallback tracker when YOLO loses target (fast/small targets)
- IR Tracker: thermal fallback for hot targets (drone motor in IR spectrum)

OSD filtering: rejects detections in edge zones and abnormally small boxes.
Approach: proportional forward speed based on distance-to-target (bbox ratio).
Terminal: STRIKE state when target within kill_radius (proximity fuze for BЧ).

Intercept strategies:
- pursuit: lead pursuit (default, intercept point calculation)
- head_on: straight at target, max speed (head-on collision course)
- top_dive: climb above target, then dive down (exploits top vulnerability)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import cv2
import numpy as np
import enum

from ultralytics import YOLO
from uav_interceptor.optical_flow_tracker import OpticalFlowTracker
from uav_interceptor.ir_tracker import IRTracker
from uav_interceptor.target_estimator import TargetEstimator, InterceptCalculator


def ros_image_to_numpy(msg: Image) -> np.ndarray:
    """Convert a ROS sensor_msgs/Image (bgr8) to a numpy BGR array."""
    dtype = np.dtype('uint8')
    n_channels = 3
    frame = np.frombuffer(msg.data, dtype=dtype).reshape(
        (msg.height, msg.width, n_channels))
    return frame.copy()


class State(enum.Enum):
    SEARCH = "SEARCH"
    TRACK = "TRACK"
    INTERCEPT = "INTERCEPT"
    STRIKE = "STRIKE"
    LOST = "LOST"


class PIDController:
    def __init__(self, kp, ki, kd, output_limit, integral_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return max(-self.output_limit, min(self.output_limit, output))

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        # Parameters
        self.declare_parameter('model_path',
            '/home/alex/AntiUAV-Detector/runs/detect/train/runs/drone_v2-4/weights/best.pt')
        self.declare_parameter('target_class', 0)
        self.declare_parameter('conf_threshold', 0.4)
        self.declare_parameter('max_lost_frames', 10)
        self.declare_parameter('show_image', True)

        # PID gains
        self.declare_parameter('pid_pan_kp', 0.05)
        self.declare_parameter('pid_pan_ki', 0.001)
        self.declare_parameter('pid_pan_kd', 0.01)
        self.declare_parameter('pid_tilt_kp', 0.05)
        self.declare_parameter('pid_tilt_ki', 0.001)
        self.declare_parameter('pid_tilt_kd', 0.01)
        self.declare_parameter('pid_output_limit', 0.5)
        self.declare_parameter('pid_integral_limit', 100.0)

        # Approach parameters
        self.declare_parameter('approach_speed', 0.3)
        self.declare_parameter('target_bbox_ratio', 0.15)
        self.declare_parameter('intercept_bbox_ratio', 0.35)

        # OSD filtering
        self.declare_parameter('osd_margin', 60)
        self.declare_parameter('min_bbox_area', 500)

        # Optical flow fallback
        self.declare_parameter('use_optical_flow', True)
        self.declare_parameter('of_fallback_frames', 5)  # frames to try OF after YOLO loss

        # IR tracker fallback (thermal camera)
        self.declare_parameter('use_ir_tracker', True)
        self.declare_parameter('ir_threshold_mode', 'adaptive')  # adaptive|fixed|otsu
        self.declare_parameter('ir_fixed_threshold', 200)
        self.declare_parameter('ir_min_area', 6)
        self.declare_parameter('ir_min_intensity', 0.6)

        # Camera / range estimation
        self.declare_parameter('camera_fov_h', 60.0)     # horizontal FOV (deg)
        self.declare_parameter('drone_size_m', 0.35)     # target physical size (m)
        self.declare_parameter('lead_frames', 5)         # Kalman lead (frames ahead)
        # Intercept
        self.declare_parameter('interceptor_speed', 15.0)  # max speed (m/s)
        self.declare_parameter('intercept_distance', 3.0)  # INTERCEPT trigger (m)
        self.declare_parameter('intercept_strategy', 'pursuit')  # pursuit|head_on|top_dive
        self.declare_parameter('kill_radius', 4.0)         # STRIKE trigger (m)
        self.declare_parameter('strike_publish_cmd', True) # publish MAVLink strike cmd

        model_path = self.get_parameter('model_path').value
        self.target_class = self.get_parameter('target_class').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.max_lost_frames = self.get_parameter('max_lost_frames').value
        self.show_image = self.get_parameter('show_image').value

        self.approach_speed = self.get_parameter('approach_speed').value
        self.target_bbox_ratio = self.get_parameter('target_bbox_ratio').value
        self.intercept_bbox_ratio = self.get_parameter('intercept_bbox_ratio').value
        self.osd_margin = self.get_parameter('osd_margin').value
        self.min_bbox_area = self.get_parameter('min_bbox_area').value

        self.use_optical_flow = self.get_parameter('use_optical_flow').value
        self.of_fallback_frames = self.get_parameter('of_fallback_frames').value

        self.use_ir_tracker = self.get_parameter('use_ir_tracker').value
        self.ir_threshold_mode = self.get_parameter('ir_threshold_mode').value
        self.ir_fixed_threshold = self.get_parameter('ir_fixed_threshold').value
        self.ir_min_area = self.get_parameter('ir_min_area').value
        self.ir_min_intensity = self.get_parameter('ir_min_intensity').value

        # Target estimator (range + Kalman + lead)
        fov_h = self.get_parameter('camera_fov_h').value
        drone_size = self.get_parameter('drone_size_m').value
        lead_frames = self.get_parameter('lead_frames').value
        self.target_estimator = TargetEstimator.from_fov(
            fov_h_deg=fov_h, image_width_px=1280,
            real_size_m=drone_size, lead_frames=lead_frames)

        # Intercept calculator
        self.intercept_calc = InterceptCalculator(
            interceptor_speed=self.get_parameter('interceptor_speed').value)
        self.intercept_distance = self.get_parameter('intercept_distance').value
        self.intercept_strategy = self.get_parameter('intercept_strategy').value
        self.kill_radius = self.get_parameter('kill_radius').value
        self.strike_publish_cmd = self.get_parameter('strike_publish_cmd').value
        self.focal_px = self.target_estimator.range_est.focal_px

        # Strike state
        self.strike_triggered = False
        self.strike_time = None

        # Load YOLO model
        self.get_logger().info(f'Loading model: {model_path}')
        self.model = YOLO(model_path)
        self.get_logger().info('Model loaded successfully.')

        # Optical flow tracker (fallback)
        self.of_tracker = OpticalFlowTracker()
        self.of_fallback_counter = 0

        # IR tracker (thermal fallback)
        self.ir_tracker = IRTracker(
            threshold_mode=self.ir_threshold_mode,
            fixed_threshold=self.ir_fixed_threshold,
            min_area=self.ir_min_area,
            min_intensity=self.ir_min_intensity,
        )
        self.ir_fallback_counter = 0

        # State
        self.state = State.SEARCH
        self.lost_counter = 0
        self.search_angle = 0.0

        # PID Controllers
        out_lim = self.get_parameter('pid_output_limit').value
        int_lim = self.get_parameter('pid_integral_limit').value
        self.pid_pan = PIDController(
            kp=self.get_parameter('pid_pan_kp').value,
            ki=self.get_parameter('pid_pan_ki').value,
            kd=self.get_parameter('pid_pan_kd').value,
            output_limit=out_lim,
            integral_limit=int_lim,
        )
        self.pid_tilt = PIDController(
            kp=self.get_parameter('pid_tilt_kp').value,
            ki=self.get_parameter('pid_tilt_ki').value,
            kd=self.get_parameter('pid_tilt_kd').value,
            output_limit=out_lim,
            integral_limit=int_lim,
        )

        # ROS Interfaces
        video_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, video_qos)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/interceptor/state', 10)
        self.strike_pub = self.create_publisher(String, '/interceptor/strike', 10)

        self.current_image_msg = None
        self.last_time = None
        self.last_detection_source = None

        self.get_logger().info('Vision Node (YOLO + Optical Flow) Started. Waiting for image...')

    def image_callback(self, msg):
        self.current_image_msg = msg
        self.process_frame()

    def is_osd_false_positive(self, x1, y1, x2, y2, w, h):
        """Reject detections in OSD edge zones or abnormally small boxes."""
        if (y1 < self.osd_margin or y2 > h - self.osd_margin or
            x1 < self.osd_margin or x2 > w - self.osd_margin):
            return True
        bbox_area = (x2 - x1) * (y2 - y1)
        if bbox_area < self.min_bbox_area:
            return True
        return False

    def select_target(self, boxes, w, h):
        """Select the largest valid drone bbox (closest target)."""
        best_box = None
        best_area = 0
        for box in boxes:
            cls = int(box.cls[0])
            if cls != self.target_class:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            if self.is_osd_false_positive(x1, y1, x2, y2, w, h):
                continue
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box = box
        return best_box

    def process_frame(self):
        if self.current_image_msg is None:
            return

        try:
            cv_image = ros_image_to_numpy(self.current_image_msg)
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')
            return

        h, w = cv_image.shape[:2]
        center_x, center_y = w / 2, h / 2

        now = self.get_clock().now()
        if self.last_time is None:
            dt = 0.033
        else:
            dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        cmd_vel = Twist()
        drone_detected = False
        target_x, target_y = 0, 0
        track_id = 0
        bbox_ratio = 0.0
        detection_source = "NONE"

        # --- YOLO DETECTION ---
        results = self.model.track(cv_image, persist=True, verbose=False,
                                   conf=self.conf_threshold)

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            box = self.select_target(results[0].boxes, w, h)
            if box is not None:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                target_x = (x1 + x2) / 2
                target_y = (y1 + y2) / 2
                bbox_area = (x2 - x1) * (y2 - y1)
                bbox_ratio = bbox_area / (w * h)
                drone_detected = True
                track_id = int(box.id[0]) if box.id is not None else 0
                detection_source = "YOLO"

                # Reset OF fallback counter
                self.of_fallback_counter = 0
                self.ir_fallback_counter = 0

                cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(cv_image, f"ID:{track_id} DRONE {bbox_ratio:.2f}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- OPTICAL FLOW FALLBACK ---
        if not drone_detected and self.use_optical_flow:
            # Only try OF for a limited number of frames after YOLO loss
            if self.of_fallback_counter < self.of_fallback_frames:
                of_result = self.of_tracker.find_target(cv_image)
                if of_result is not None:
                    x1, y1, x2, y2, cx, cy = of_result
                    target_x, target_y = cx, cy
                    bbox_area = (x2 - x1) * (y2 - y1)
                    bbox_ratio = bbox_area / (w * h)
                    drone_detected = True
                    detection_source = "OPTICAL_FLOW"

                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(cv_image, "OF TRACK", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                self.of_fallback_counter += 1
            else:
                # OF exhausted — reset for next YOLO detection cycle
                self.of_fallback_counter = 0
                self.of_tracker.reset()

        # --- IR TRACKER FALLBACK (thermal) ---
        if not drone_detected and self.use_ir_tracker:
            # Try IR tracker after OF is exhausted (or in parallel as last resort)
            if self.ir_fallback_counter < self.of_fallback_frames:
                ir_result = self.ir_tracker.find_target(cv_image)
                if ir_result is not None:
                    x1, y1, x2, y2, cx, cy = ir_result
                    target_x, target_y = cx, cy
                    bbox_area = (x2 - x1) * (y2 - y1)
                    bbox_ratio = bbox_area / (w * h)
                    drone_detected = True
                    detection_source = "IR"

                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(cv_image, "IR TRACK", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                self.ir_fallback_counter += 1
            else:
                self.ir_fallback_counter = 0
                self.ir_tracker.reset()

        if drone_detected:
            self.lost_counter = 0

            # Log detection source (throttled)
            if detection_source != self.last_detection_source:
                self.get_logger().info(f'Detection source: {detection_source}')
                self.last_detection_source = detection_source

            # Update target estimator (range + Kalman + lead)
            est_info = self.target_estimator.update(
                (int(target_x - 20), int(target_y - 20),
                 int(target_x + 20), int(target_y + 20)),
                dt=dt)

            # Aim at lead point (predicted position ahead) for moving targets
            lead_point = est_info['lead_point']
            if lead_point is not None:
                aim_x, aim_y = lead_point
            else:
                aim_x, aim_y = target_x, target_y

            error_x = aim_x - center_x
            error_y = aim_y - center_y

            # Yaw (pan) from horizontal error (lead pursuit)
            cmd_vel.angular.z = float(self.pid_pan.compute(error_x, dt))
            # Altitude from vertical error
            cmd_vel.linear.z = float(self.pid_tilt.compute(error_y, dt))

            # Distance-based approach
            distance = est_info['distance']
            if distance is not None:
                # Proportional approach: faster when far, slower when close
                approach = self.approach_speed * min(1.0, distance / 20.0)
                cmd_vel.linear.x = float(max(0.1, approach))
            else:
                # Fallback to bbox-ratio approach
                if bbox_ratio >= self.intercept_bbox_ratio:
                    cmd_vel.linear.x = float(self.approach_speed * 1.5)
                elif bbox_ratio >= self.target_bbox_ratio:
                    scale = 1.0 - (bbox_ratio - self.target_bbox_ratio) / (
                        self.intercept_bbox_ratio - self.target_bbox_ratio)
                    cmd_vel.linear.x = float(self.approach_speed * max(0.3, scale))
                else:
                    cmd_vel.linear.x = float(self.approach_speed)

            # State transition based on distance
            if distance is not None and distance < self.kill_radius:
                self.state = State.STRIKE
            elif distance is not None and distance < self.intercept_distance:
                self.state = State.INTERCEPT
            elif bbox_ratio >= self.intercept_bbox_ratio:
                self.state = State.INTERCEPT
            else:
                self.state = State.TRACK

            # --- INTERCEPT: compute intercept point and aim at it ---
            if self.state == State.INTERCEPT and distance is not None:
                # Estimate target 3D velocity from image velocity + distance
                vx_px, vy_px = est_info['velocity']
                fps = 1.0 / dt if dt > 0 else 30.0
                vel_scale = distance / self.focal_px * fps
                target_vel_3d = (0.0, vx_px * vel_scale, vy_px * vel_scale)

                strategy = self.intercept_strategy

                if strategy == 'head_on':
                    # Head-on: aim directly at target, max speed
                    # (closing speed = interceptor + target)
                    err_x = target_x - center_x
                    err_y = target_y - center_y
                    cmd_vel.angular.z = float(np.clip(self.pid_pan.compute(err_x, dt), -0.5, 0.5))
                    cmd_vel.linear.z = float(np.clip(self.pid_tilt.compute(err_y, dt), -0.5, 0.5))
                    cmd_vel.linear.x = float(self.approach_speed * 1.5)
                    cv2.putText(cv_image, "HEAD-ON INTERCEPT", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                elif strategy == 'top_dive':
                    # Top-dive: first climb above target, then dive.
                    # If target is in upper part of frame (we're below), climb.
                    # If target is in lower part (we're above), dive.
                    target_upper = target_y < center_y - h * 0.15
                    if target_upper:
                        # We are below the target — climb
                        cmd_vel.angular.z = float(np.clip(
                            self.pid_pan.compute(target_x - center_x, dt), -0.5, 0.5))
                        cmd_vel.linear.z = float(self.approach_speed * 0.8)  # climb
                        cmd_vel.linear.x = float(self.approach_speed * 0.5)
                        cv2.putText(cv_image, "TOP-DIVE CLIMB", (10, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        # We are above the target — dive down on it
                        err_y = target_y - center_y
                        cmd_vel.angular.z = float(np.clip(
                            self.pid_pan.compute(target_x - center_x, dt), -0.5, 0.5))
                        cmd_vel.linear.z = float(-np.clip(
                            self.pid_tilt.compute(err_y, dt), -0.5, 0.5))  # dive
                        cmd_vel.linear.x = float(self.approach_speed * 1.5)
                        cv2.putText(cv_image, "TOP-DIVE DIVE", (10, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                else:  # pursuit (default)
                    intercept = self.intercept_calc.compute_from_image(
                        bbox_center=(aim_x, aim_y),
                        distance=distance,
                        target_vel_3d=target_vel_3d,
                        focal_px=self.focal_px,
                        image_center=(center_x, center_y))

                    if intercept['success']:
                        yaw_cmd, pitch_cmd = intercept['bearing']
                        cmd_vel.angular.z = float(np.clip(yaw_cmd * 2.0, -0.5, 0.5))
                        cmd_vel.linear.z = float(np.clip(-pitch_cmd * 2.0, -0.5, 0.5))
                        cmd_vel.linear.x = float(self.approach_speed * 1.5)

                        ip = intercept['intercept_point']
                        T = intercept['time_to_intercept']
                        cv2.putText(cv_image,
                            f"INTERCEPT T={T:.1f}s ({ip[0]:.0f},{ip[1]:.0f},{ip[2]:.0f})",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # --- STRIKE: proximity fuze — detonate warhead ---
            if self.state == State.STRIKE:
                cmd_vel.linear.x = float(self.approach_speed * 1.5)
                cmd_vel.angular.z = 0.0
                cmd_vel.linear.z = 0.0

                cv2.putText(cv_image, f"STRIKE! R={distance:.1f}m < {self.kill_radius}m",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Publish strike command once (proximity fuze trigger)
                if not self.strike_triggered:
                    self.strike_triggered = True
                    self.strike_time = now
                    strike_msg = String()
                    strike_msg.data = f"STRIKE dist={distance:.2f}m"
                    self.strike_pub.publish(strike_msg)
                    self.get_logger().warn(
                        f'*** STRIKE TRIGGERED *** distance={distance:.2f}m '
                        f'(kill_radius={self.kill_radius}m)')

            cv2.line(cv_image, (int(center_x), int(center_y)),
                     (int(aim_x), int(aim_y)), (255, 0, 0), 2)
            cv2.circle(cv_image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
            # Draw lead point
            cv2.circle(cv_image, (int(aim_x), int(aim_y)), 4, (255, 255, 0), -1)

            state_color = (0, 255, 0) if self.state == State.TRACK else (0, 0, 255)
            if self.state == State.STRIKE:
                state_color = (0, 0, 255)
            dist_str = f"{distance:.1f}m" if distance is not None else "N/A"
            status_text = (f"{self.state.value} [{detection_source}] "
                           f"ErrX={error_x:.0f} ErrY={error_y:.0f} "
                           f"| Yaw={cmd_vel.angular.z:.2f} Vz={cmd_vel.linear.z:.2f} "
                           f"Vx={cmd_vel.linear.x:.2f} D={dist_str}")
            cv2.putText(cv_image, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)

        else:
            self.lost_counter += 1
            if self.lost_counter > self.max_lost_frames:
                self.state = State.SEARCH
                self.pid_pan.reset()
                self.pid_tilt.reset()
                self.of_tracker.reset()
                self.ir_tracker.reset()
                self.target_estimator.reset()
                self.strike_triggered = False
                self.strike_time = None

            if self.state == State.SEARCH:
                self.search_angle += 0.1
                cmd_vel.angular.z = float(0.5 * np.sin(self.search_angle))
                cv2.putText(cv_image, "SEARCHING...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # LOST: full stop
                cmd_vel.angular.z = 0.0
                cmd_vel.linear.x = 0.0
                cmd_vel.linear.z = 0.0
                cv2.putText(cv_image, "TARGET LOST", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Publish state
        state_msg = String()
        state_msg.data = self.state.value
        self.state_pub.publish(state_msg)

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

        if self.show_image:
            cv2.imshow('UAV Interceptor Vision', cv_image)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()