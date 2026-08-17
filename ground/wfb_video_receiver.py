#!/usr/bin/env python3
"""
Наземный приёмник видео с WFB-ng для НСУ (Jetson Orin Nano).

Приём двух видеопотоков с борта через WFB-ng:
1. video0: EO-камера (OpenIPC, H.265) → UDP 5600
2. video1: IR-камера (H.264) → UDP 5601

WFB-ng на земле (gs.cfg) направляет потоки на:
  - gs_video0: connect://127.0.0.1:5600 (EO)
  - gs_video1: connect://127.0.0.1:5601 (IR)

Публикует в ROS2:
  /camera/image_raw     — EO-кадры (BGR8)
  /camera/ir_image_raw  — IR-кадры (BGR8)

Использует GStreamer с аппаратным декодером Jetson (nvv4l2decoder)
для минимальной задержки.

Запуск (на Jetson Orin Nano, НСУ):
  python3 ground/wfb_video_receiver.py \
    --eo-udp-port 5600 \
    --ir-udp-port 5601 \
    --eo-format h265 \
    --ir-format h264 \
    --eo-width 1280 --eo-height 720 \
    --ir-width 640 --ir-height 480

Зависимости (Jetson Orin Nano, JetPack 6.x):
  pip install numpy opencv-python
  # GStreamer + nvv4l2decoder уже в JetPack
  # ROS2 (ros-humble-ros-base)
"""

import argparse
import sys
import time
import threading

import numpy as np

try:
    import cv2
except ImportError:
    print("ОШИБКА: opencv-python не установлен")
    sys.exit(1)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import Image
except ImportError:
    print("ОШИБКА: ROS2 (rclpy) не установлен. Установите ros-humble-ros-base")
    sys.exit(1)


def numpy_to_ros_image(frame: np.ndarray) -> Image:
    """BGR numpy → ROS Image."""
    msg = Image()
    msg.height = frame.shape[0]
    msg.width = frame.shape[1]
    msg.encoding = 'bgr8'
    msg.is_bigendian = False
    msg.step = frame.shape[1] * 3
    msg.data = frame.tobytes()
    return msg


class WFBVideoReceiver(Node):
    """
    Приём видео с борта через WFB-ng (UDP).

    GStreamer pipeline (аппаратный декодер Jetson):
      udpsrc port=<port> ! parse ! nvv4l2decoder ! nvvidconv ! BGRx ! videoconvert ! BGR ! appsink

    Для H.265 (EO): h265parse ! nvv4l2decoder
    Для H.264 (IR): h264parse ! nvv4l2decoder
    """

    def __init__(self):
        super().__init__('wfb_video_receiver')

        # Параметры
        self.declare_parameter('eo_udp_port', 5600)
        self.declare_parameter('ir_udp_port', 5601)
        self.declare_parameter('eo_format', 'h265')
        self.declare_parameter('ir_format', 'h264')
        self.declare_parameter('eo_width', 1280)
        self.declare_parameter('eo_height', 720)
        self.declare_parameter('ir_width', 640)
        self.declare_parameter('ir_height', 480)
        self.declare_parameter('enable_ir', True)
        self.declare_parameter('use_gstreamer', True)

        self.eo_port = self.get_parameter('eo_udp_port').value
        self.ir_port = self.get_parameter('ir_udp_port').value
        self.eo_format = self.get_parameter('eo_format').value
        self.ir_format = self.get_parameter('ir_format').value
        self.eo_width = self.get_parameter('eo_width').value
        self.eo_height = self.get_parameter('eo_height').value
        self.ir_width = self.get_parameter('ir_width').value
        self.ir_height = self.get_parameter('ir_height').value
        self.enable_ir = self.get_parameter('enable_ir').value
        self.use_gstreamer = self.get_parameter('use_gstreamer').value

        # QoS: BEST_EFFORT для видео
        video_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        self.eo_pub = self.create_publisher(Image, '/camera/image_raw', video_qos)
        self.ir_pub = self.create_publisher(Image, '/camera/ir_image_raw', video_qos)

        self.eo_cap = None
        self.ir_cap = None
        self.running = True

        # Открытие EO-потока
        self.eo_cap = self._open_stream(
            self.eo_port, self.eo_format,
            self.eo_width, self.eo_height, 'EO')

        # Открытие IR-потока
        if self.enable_ir:
            self.ir_cap = self._open_stream(
                self.ir_port, self.ir_format,
                self.ir_width, self.ir_height, 'IR')

        # Потоки чтения
        self.eo_thread = threading.Thread(target=self._eo_loop, daemon=True)
        self.eo_thread.start()

        if self.enable_ir and self.ir_cap is not None:
            self.ir_thread = threading.Thread(target=self._ir_loop, daemon=True)
            self.ir_thread.start()

        self.eo_count = 0
        self.ir_count = 0
        self.last_stats = time.time()

        self.stats_timer = self.create_timer(5.0, self._stats_callback)
        self.get_logger().info('WFB Video Receiver запущен')

    def _open_stream(self, udp_port, fmt, width, height, label):
        """Открытие видеопотока через GStreamer (Jetson hw decoder)."""
        if self.use_gstreamer:
            # Аппаратный декодер Jetson
            if fmt == 'h265':
                parse_elem = 'h265parse'
            else:
                parse_elem = 'h264parse'

            pipeline = (
                f'udpsrc port={udp_port} buffer-size=2097152 ! '
                f'{parse_elem} ! '
                f'nvv4l2decoder ! '
                f'nvvidconv ! video/x-raw,format=BGRx ! '
                f'videoconvert ! video/x-raw,format=BGR ! '
                f'appsink drop=true sync=false max-buffers=1'
            )
            self.get_logger().info(f'{label}: GStreamer pipeline (port {udp_port})')

            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                self.get_logger().error(
                    f'{label}: GStreamer не открылся, fallback на FFmpeg')
                cap = cv2.VideoCapture(
                    f'udp://0.0.0.0:{udp_port}', cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        else:
            # Fallback: FFmpeg
            cap = cv2.VideoCapture(
                f'udp://0.0.0.0:{udp_port}', cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

    def _eo_loop(self):
        """Чтение EO-кадров и публикация в ROS2."""
        self.get_logger().info(f'EO-поток запущен (UDP {self.eo_port})')
        while self.running and rclpy.ok():
            if self.eo_cap is None or not self.eo_cap.isOpened():
                time.sleep(0.5)
                self.eo_cap = self._open_stream(
                    self.eo_port, self.eo_format,
                    self.eo_width, self.eo_height, 'EO')
                continue

            ret, frame = self.eo_cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            self.eo_count += 1
            msg = numpy_to_ros_image(frame)
            self.eo_pub.publish(msg)

    def _ir_loop(self):
        """Чтение IR-кадров и публикация в ROS2."""
        self.get_logger().info(f'IR-поток запущен (UDP {self.ir_port})')
        while self.running and rclpy.ok():
            if self.ir_cap is None or not self.ir_cap.isOpened():
                time.sleep(0.5)
                self.ir_cap = self._open_stream(
                    self.ir_port, self.ir_format,
                    self.ir_width, self.ir_height, 'IR')
                continue

            ret, frame = self.ir_cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            self.ir_count += 1
            msg = numpy_to_ros_image(frame)
            self.ir_pub.publish(msg)

    def _stats_callback(self):
        now = time.time()
        dt = now - self.last_stats
        eo_fps = self.eo_count / dt if dt > 0 else 0
        ir_fps = self.ir_count / dt if dt > 0 else 0
        self.get_logger().info(
            f'EO: {eo_fps:.1f} FPS ({self.eo_count} кадров), '
            f'IR: {ir_fps:.1f} FPS ({self.ir_count} кадров)')
        self.eo_count = 0
        self.ir_count = 0
        self.last_stats = now

    def destroy_node(self):
        self.running = False
        if self.eo_cap:
            self.eo_cap.release()
        if self.ir_cap:
            self.ir_cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WFBVideoReceiver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()