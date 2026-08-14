#!/usr/bin/env python3
"""
ROS2 Node: Publishes video frames to /camera/image_raw.
Supports multiple video sources:
1. RTSP stream (OpenIPC MC800S-V3 thermal/EO camera)
2. USB camera (/dev/videoN, e.g. Bison FHD)
3. Video file (MP4/AVI for testing)

OpenIPC MC800S-V3 specs:
- SigmaStar SSC338Q + Sony IMX415
- 4K@20fps H.265 / 720p@120fps
- RTSP URL: rtsp://<ip>:554/live (or /stream=0)
- Latency: 60-80ms

Manual Image construction (no cv_bridge dependency to avoid OpenCV conflicts).
Uses BEST_EFFORT QoS for high-throughput video streaming.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
import cv2
import numpy as np
import time


def numpy_to_ros_image(frame: np.ndarray) -> Image:
    """Convert a BGR numpy array to a ROS sensor_msgs/Image."""
    msg = Image()
    msg.height = frame.shape[0]
    msg.width = frame.shape[1]
    msg.encoding = 'bgr8'
    msg.is_bigendian = False
    msg.step = frame.shape[1] * 3
    msg.data = frame.tobytes()
    return msg


class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')

        # Source type: 'file' | 'rtsp' | 'usb'
        self.declare_parameter('source_type', 'file')

        # Video file path (for 'file' mode)
        self.declare_parameter('video_path',
            '/home/alex/AntiUAV-Detector/video-FPV/Video/v2.mp4')

        # RTSP URL (for 'rtsp' mode)
        self.declare_parameter('rtsp_url', 'rtsp://192.168.1.10:554/live')

        # USB camera device (for 'usb' mode)
        self.declare_parameter('usb_device', '/dev/video0')
        self.declare_parameter('usb_width', 640)
        self.declare_parameter('usb_height', 480)
        self.declare_parameter('usb_fps', 30)

        # Common
        self.declare_parameter('target_fps', 30)
        self.declare_parameter('reconnect_interval', 5.0)

        source_type = self.get_parameter('source_type').value
        target_fps = self.get_parameter('target_fps').value
        self.reconnect_interval = self.get_parameter('reconnect_interval').value

        # QoS: BEST_EFFORT for video (drop frames instead of retrying)
        video_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        self.publisher = self.create_publisher(Image, '/camera/image_raw', video_qos)

        self.cap = None
        self.source_label = ''
        self.last_reconnect = 0.0

        if source_type == 'rtsp':
            self._open_rtsp()
        elif source_type == 'usb':
            self._open_usb()
        else:
            self._open_file()

        if self.cap is not None and self.cap.isOpened():
            self.timer = self.create_timer(1.0 / target_fps, self.publish_frame)
            self.get_logger().info(
                f'Video Publisher Started: {self.source_label} ({target_fps:.0f} FPS)')
        else:
            self.get_logger().error(
                f'Cannot open source: {self.source_label}. '
                f'Trying reconnect every {self.reconnect_interval:.0f}s.')
            self.timer = self.create_timer(
                self.reconnect_interval, self._reconnect_and_publish)

    def _open_rtsp(self):
        """Открытие RTSP-потока (камера OpenIPC MC800S-V3).
        FFmpeg backend с BUFFERSIZE=1 для минимальной задержки (60-80ms)."""
        rtsp_url = self.get_parameter('rtsp_url').value
        self.source_label = f'RTSP: {rtsp_url}'

        # FFmpeg backend with low-latency flags for RTSP
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize buffering/latency

    def _open_usb(self):
        """Open USB camera (e.g. /dev/video0)."""
        device = self.get_parameter('usb_device').value
        width = self.get_parameter('usb_width').value
        height = self.get_parameter('usb_height').value
        fps = self.get_parameter('usb_fps').value
        self.source_label = f'USB: {device} {width}x{height}@{fps}'

        self.cap = cv2.VideoCapture(device)
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _open_file(self):
        """Open video file for testing."""
        video_path = self.get_parameter('video_path').value
        self.source_label = f'File: {video_path}'
        self.cap = cv2.VideoCapture(video_path)

    def _reconnect_and_publish(self):
        """Авто-реконект при потере потока (RTSP/USB).
        Пытается переподключиться каждые reconnect_interval секунд."""
        now = time.time()
        if now - self.last_reconnect < self.reconnect_interval:
            return
        self.last_reconnect = now

        source_type = self.get_parameter('source_type').value
        self.get_logger().info(f'Reconnecting to {self.source_label}...')

        if self.cap is not None:
            self.cap.release()

        if source_type == 'rtsp':
            self._open_rtsp()
        elif source_type == 'usb':
            self._open_usb()
        else:
            self._open_file()

        if self.cap is not None and self.cap.isOpened():
            self.get_logger().info(f'Reconnected: {self.source_label}')
            self.timer.cancel()
            target_fps = self.get_parameter('target_fps').value
            self.timer = self.create_timer(1.0 / target_fps, self.publish_frame)
        else:
            self.get_logger().warn(f'Reconnect failed: {self.source_label}')

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            source_type = self.get_parameter('source_type').value
            if source_type == 'file':
                # Loop video file
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    self.get_logger().error('Cannot re-read video file')
                    return
            else:
                # RTSP/USB: stream lost, try reconnect
                self.get_logger().warn(
                    f'Frame read failed ({self.source_label}), reconnecting...')
                self.timer.cancel()
                self.timer = self.create_timer(
                    self.reconnect_interval, self._reconnect_and_publish)
                return

        try:
            msg = numpy_to_ros_image(frame)
            self.publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.cap is not None:
            node.cap.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()