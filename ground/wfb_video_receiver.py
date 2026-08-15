#!/usr/bin/env python3
"""
Наземный приёмник видео с WFB-ng для НСУ (Jetson Orin Nano).

Приём двух видеопотоков с борта:
1. video0: EO-камера (OpenIPC, H.265) — основной канал детекции
2. video1: IR-камера (H.264) — thermal fallback / dual-band fusion

Публикует в ROS2:
  /camera/image_raw     — EO-кадры (BGR8)
  /camera/ir_image_raw  — IR-кадры (BGR8)

Использует GStreamer с аппаратным декодером Jetson (nvv4l2decoder)
для минимальной задержки.

Запуск (на Jetson Orin Nano, НСУ):
  python3 ground/wfb_video_receiver.py \
    --eo-pipe /tmp/wfb_rx_video0 \
    --ir-pipe /tmp/wfb_rx_video1 \
    --eo-format h265 \
    --ir-format h264 \
    --eo-width 1280 --eo-height 720 \
    --ir-width 640 --ir-height 480 \
    --target-fps 30

Зависимости (Jetson Orin Nano, JetPack 6.x):
  pip install numpy
  # GStreamer + nvv4l2decoder уже в JetPack
  # ROS2 (ros-humble-ros-base)
"""

import argparse
import sys
import time
import threading
import os
from pathlib import Path

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
    Приём видео с борта через WFB-ng named pipes.

    GStreamer pipeline (аппаратный декодер Jetson):
      filesrc location=<pipe> ! parse ! nvv4l2decoder ! nvvidconv ! BGRx ! videoconvert ! BGR ! appsink

    Для H.265 (EO): h265parse ! nvv4l2decoder
    Для H.264 (IR): h264parse ! nvv4l2decoder
    """

    def __init__(self):
        super().__init__('wfb_video_receiver')

        # Параметры
        self.declare_parameter('eo_pipe', '/tmp/wfb_rx_video0')
        self.declare_parameter('ir_pipe', '/tmp/wfb_rx_video1')
        self.declare_parameter('eo_format', 'h265')
        self.declare_parameter('ir_format', 'h264')
        self.declare_parameter('eo_width', 1280)
        self.declare_parameter('eo_height', 720)
        self.declare_parameter('ir_width', 640)
        self.declare_parameter('ir_height', 480)
        self.declare_parameter('target_fps', 30)
        self.declare_parameter('enable_ir', True)
        self.declare_parameter('use_gstreamer', True)

        self.eo_pipe = self.get_parameter('eo_pipe').value
        self.ir_pipe = self.get_parameter('ir_pipe').value
        self.eo_format = self.get_parameter('eo_format').value
        self.ir_format = self.get_parameter('ir_format').value
        self.eo_width = self.get_parameter('eo_width').value
        self.eo_height = self.get_parameter('eo_height').value
        self.ir_width = self.get_parameter('ir_width').value
        self.ir_height = self.get_parameter('ir_height').value
        self.target_fps = self.get_parameter('target_fps').value
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
            self.eo_pipe, self.eo_format,
            self.eo_width, self.eo_height, 'EO')

        # Открытие IR-потока
        if self.enable_ir:
            self.ir_cap = self._open_stream(
                self.ir_pipe, self.ir_format,
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

    def _open_stream(self, pipe_path, fmt, width, height, label):
        """Открытие видеопотока через GStreamer (Jetson hw decoder)."""
        if not os.path.exists(pipe_path):
            self.get_logger().warn(f'{label}: pipe {pipe_path} не существует, ожидание...')
            # Создаём pipe (WFB-ng может ещё не запущен)
            try:
                os.mkfifo(pipe_path)
            except FileExistsError:
                pass

        if self.use_gstreamer:
            # Аппаратный декодер Jetson
            if fmt == 'h265':
                parse_elem = 'h265parse'
            else:
                parse_elem = 'h264parse'

            pipeline = (
                f'filesrc location={pipe_path} ! '
                f'{parse_elem} ! '
                f'nvv4l2decoder ! '
                f'nvvidconv ! video/x-raw,format=BGRx ! '
                f'videoconvert ! video/x-raw,format=BGR ! '
                f'appsink drop=true sync=false max-buffers=1'
            )
            self.get_logger().info(f'{label}: GStreamer pipeline: {pipeline[:80]}...')

            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                self.get_logger().error(
                    f'{label}: GStreamer не открылся, fallback на FFmpeg')
                cap = cv2.VideoCapture(pipe_path, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        else:
            # Fallback: FFmpeg
            cap = cv2.VideoCapture(pipe_path, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

    def _eo_loop(self):
        """Чтение EO-кадров и публикация в ROS2."""
        self.get_logger().info('EO-поток запущен')
        while self.running and rclpy.ok():
            if self.eo_cap is None or not self.eo_cap.isOpened():
                time.sleep(0.5)
                self.eo_cap = self._open_stream(
                    self.eo_pipe, self.eo_format,
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
        self.get_logger().info('IR-поток запущен')
        while self.running and rclpy.ok():
            if self.ir_cap is None or not self.ir_cap.isOpened():
                time.sleep(0.5)
                self.ir_cap = self._open_stream(
                    self.ir_pipe, self.ir_format,
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
    parser = argparse.ArgumentParser(description='WFB Video Receiver (НСУ)')
    parser.add_argument('--eo-pipe', default='/tmp/wfb_rx_video0')
    parser.add_argument('--ir-pipe', default='/tmp/wfb_rx_video1')
    parser.add_argument('--eo-format', choices=['h265', 'h264'], default='h265')
    parser.add_argument('--ir-format', choices=['h264', 'h265'], default='h264')
    parser.add_argument('--eo-width', type=int, default=1280)
    parser.add_argument('--eo-height', type=int, default=720)
    parser.add_argument('--ir-width', type=int, default=640)
    parser.add_argument('--ir-height', type=int, default=480)
    parser.add_argument('--target-fps', type=int, default=30)
    parser.add_argument('--enable-ir', action='store_true', default=True)
    parser.add_argument('--no-ir', action='store_true')
    parser.add_argument('--no-gstreamer', action='store_true')
    args = parser.parse_args()

    if args.no_ir:
        args.enable_ir = False
    if args.no_gstreamer:
        args.use_gstreamer = False

    rclpy.init(args=None)
    node = WFBVideoReceiver()

    # Переопределение параметров из CLI
    node.eo_pipe = args.eo_pipe
    node.ir_pipe = args.ir_pipe
    node.eo_format = args.eo_format
    node.ir_format = args.ir_format
    node.enable_ir = args.enable_ir

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
