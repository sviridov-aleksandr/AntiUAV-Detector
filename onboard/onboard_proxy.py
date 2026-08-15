#!/usr/bin/env python3
"""
Бортовой прокси для Orange Pi 3 Zero (компаньон-компьютер дрона-перехватчика).

Функции:
1. MAVLink proxy: наземная НСУ ↔ CUAV X7+ Pro (двунаправленный)
   Через WFB-ng: WFB-ng сам пробрасывает MAVLink между serial (CUAV)
   и UDP (НСУ). Этот скрипт — только для диагностики/доп. функций.
2. IR-камера (USB) → кодирование H.264 → UDP 5603 (WFB-ng video1)
3. Телеметрия CUAV → WFB (вниз на НСУ)

Архитектура (WFB-ng):
  ┌──────────┐  UART/USB   ┌──────────────┐  WiFi (WFB-ng)  ┌──────┐
  │ CUAV X7+ │◄──────────►│ Orange Pi 3Z │◄───────────────►│ НСУ  │
  │ Pro (FC) │  MAVLink    │ (proxy)      │   radio link     │(Jetson)│
  └──────────┘             └──────┬───────┘                 └──────┘
                                  │ USB
                           ┌──────┴──────┐
                           │  IR-камера   │
                           └─────────────┘

WFB-ng каналы (см. wfb/drone.cfg):
  - video0 (0x00): EO-камера OpenIPC (H.265) → UDP 5602
  - video1 (0x01): IR-камера (H.264) → UDP 5603
  - mavlink (0x10/0x90): CUAV serial ↔ НСУ UDP 14550 (встроено в WFB-ng)
  - tunnel (0x20/0xa0): IPoWB (SSH)

Запуск (на Orange Pi 3 Zero):
  python3 onboard_proxy.py \
    --ir-device /dev/video0 \
    --ir-width 640 --ir-height 480 --ir-fps 30 \
    --ir-udp-port 5603

Зависимости (Orange Pi 3 Zero, Armbian/Ubuntu):
  pip3 install opencv-python-headless numpy
"""

import argparse
import sys
import time
import socket
import struct
import os

import numpy as np

try:
    import cv2
except ImportError:
    print("ОШИБКА: opencv-python не установлен. Установите: pip3 install opencv-python-headless")
    sys.exit(1)


# ─────────────────────────────────────────────────────────
#  IR Camera → UDP (WFB-ng video1)
# ─────────────────────────────────────────────────────────

class IRVideoRelay:
    """
    Захват IR-камеры (USB) → кодирование H.264 → UDP (WFB-ng).

    WFB-ng слушает UDP 5603 на борту (drone_video1 peer=listen://0.0.0.0:5603)
    и передаёт поток вниз на НСУ.

    Использует GStreamer pipeline для кодирования H.264.
    Fallback: JPEG-кадры с фреймингом (IRJP + размер).
    """

    def __init__(self, ir_device, udp_port=5603, width=640, height=480,
                 fps=30, bitrate=1000000):
        self.ir_device = ir_device
        self.udp_port = udp_port
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.cap = None
        self.writer = None
        self.sock = None
        self.running = False
        self.stats = {'frames_captured': 0, 'frames_sent': 0, 'errors': 0}

    def start(self):
        """Открытие камеры, кодировщика и UDP-сокета."""
        # Открытие USB-камеры
        self.cap = cv2.VideoCapture(self.ir_device)
        if not self.cap.isOpened():
            print(f"[IR] ОШИБКА: камера {self.ir_device} не открыта")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[IR] Камера открыта: {self.ir_device} "
              f"{self.width}x{self.height}@{self.fps}")

        # UDP-сокет для отправки в WFB-ng
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2097152)
        self.sock.connect(('127.0.0.1', self.udp_port))
        print(f"[IR] UDP-сокет: 127.0.0.1:{self.udp_port} (WFB-ng video1)")

        # Кодировщик H.264 через GStreamer
        # На Orange Pi (H618 SoC) — аппаратный H.264 (cedrus264)
        # Fallback — x264enc (программный)
        gst_pipeline = (
            f'appsrc ! video/x-raw,format=BGR,width={self.width},'
            f'height={self.height},framerate={self.fps}/1 ! '
            f'videoconvert ! video/x-raw,format=I420 ! '
            f'x264enc tune=zerolatency bitrate={self.bitrate // 1000} '
            f'speed-preset=ultrafast ! '
            f'video/x-h264,profile=baseline ! '
            f'rtph264pay config-interval=1 pt=96 ! '
            f'udpsink host=127.0.0.1 port={self.udp_port} '
            f'sync=false'
        )

        try:
            self.writer = cv2.VideoWriter(
                gst_pipeline, cv2.CAP_GSTREAMER, 0,
                self.fps, (self.width, self.height))
            if not self.writer.isOpened():
                raise RuntimeError("GStreamer writer не открылся")
            print("[IR] Кодировщик H.264 (GStreamer → UDP) запущен")
        except Exception as e:
            print(f"[IR] GStreamer недоступен ({e}), fallback на JPEG")
            self.writer = None

        self.running = True
        return True

    def run(self):
        """Основной цикл захвата и отправки."""
        if not self.start():
            return

        print("[IR] Ретрансляция запущена")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.stats['errors'] += 1
                time.sleep(0.01)
                continue

            self.stats['frames_captured'] += 1

            if self.writer:
                # GStreamer: BGR-кадр → H.264 → RTP → UDP (WFB-ng)
                self.writer.write(frame)
                self.stats['frames_sent'] += 1
            else:
                # Fallback: JPEG-кадры с фреймингом
                _, jpeg = cv2.imencode('.jpg', frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
                try:
                    header = struct.pack('<4sI', b'IRJP', len(jpeg))
                    self.sock.send(header + jpeg.tobytes())
                    self.stats['frames_sent'] += 1
                except Exception:
                    pass

            # Статистика каждые 5 секунд
            if self.stats['frames_captured'] % (self.fps * 5) == 0:
                print(f"[IR] Кадров: {self.stats['frames_captured']}, "
                      f"отправлено: {self.stats['frames_sent']}, "
                      f"ошибок: {self.stats['errors']}")

        self.stop()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        if self.sock:
            self.sock.close()
        print("[IR] Остановлен")


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Бортовой прокси: Orange Pi 3 Zero (IR video → WFB-ng)')
    parser.add_argument('--ir-device', default='/dev/video0',
                        help='IR-камера USB device')
    parser.add_argument('--ir-width', type=int, default=640)
    parser.add_argument('--ir-height', type=int, default=480)
    parser.add_argument('--ir-fps', type=int, default=30)
    parser.add_argument('--ir-bitrate', type=int, default=1000000,
                        help='H.264 bitrate (bps)')
    parser.add_argument('--ir-udp-port', type=int, default=5603,
                        help='UDP порт для WFB-ng video1 (default: 5603)')
    parser.add_argument('--no-ir', action='store_true',
                        help='Отключить IR-камеру (только диагностика)')
    args = parser.parse_args()

    print("=" * 60)
    print("БОРТОВОЙ ПРОКСИ — Orange Pi 3 Zero")
    print("=" * 60)
    if not args.no_ir:
        print(f"  IR камера:   {args.ir_device} "
              f"{args.ir_width}x{args.ir_height}@{args.ir_fps}")
        print(f"  UDP порт:    {args.ir_udp_port} (WFB-ng video1)")
    print("=" * 60)

    if args.no_ir:
        print("[MAIN] Только диагностика (без IR)")
        print("MAVLink пробрасывается WFB-ng напрямую (serial ↔ UDP)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        ir_relay = IRVideoRelay(
            ir_device=args.ir_device,
            udp_port=args.ir_udp_port,
            width=args.ir_width,
            height=args.ir_height,
            fps=args.ir_fps,
            bitrate=args.ir_bitrate,
        )
        try:
            ir_relay.run()
        except KeyboardInterrupt:
            pass
        finally:
            ir_relay.stop()

    print("\n[MAIN] Остановлен")


if __name__ == '__main__':
    main()