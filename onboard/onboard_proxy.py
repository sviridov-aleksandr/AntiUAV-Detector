#!/usr/bin/env python3
"""
Бортовой прокси для Orange Pi 3 Zero (компаньон-компьютер дрона-перехватчика).

Функции:
1. MAVLink proxy: наземная НСУ ↔ CUAV X7+ Pro (двунаправленный)
2. IR-камера (USB) → кодирование H.264 → отправка вниз через WFB
3. Телеметрия CUAV → WFB (вниз на НСУ)

Архитектура:
  ┌──────────┐  UART/USB   ┌──────────────┐  WiFi (WFB)  ┌──────┐
  │ CUAV X7+ │◄──────────►│ Orange Pi 3Z │◄────────────►│ НСУ  │
  │ Pro (FC) │  MAVLink    │ (proxy)      │   radio link  │(Jetson)│
  └──────────┘             └──────┬───────┘              └──────┘
                                  │ USB
                           ┌──────┴──────┐
                           │  IR-камера   │
                           └─────────────┘

WFB-ng каналы:
  - video0: OpenIPC EO-камера (напрямую через WFB, без Orange Pi)
  - video1: IR-камера (через Orange Pi → WFB)
  - telemetry: CUAV телеметрия (через Orange Pi → WFB)
  - command: MAVLink команды (НСУ → Orange Pi → CUAV)

Запуск (на Orange Pi 3 Zero):
  python3 onboard_proxy.py \
    --fc-device /dev/ttyAMA0 \
    --fc-baud 921600 \
    --wfb-tx-pipe /tmp/wfb_tx_video1 \
    --wfb-rx-pipe /tmp/wfb_rx_command \
    --wfb-tx-tel /tmp/wfb_tx_telemetry \
    --ir-device /dev/video0 \
    --ir-width 640 --ir-height 480 --ir-fps 30

Зависимости (Orange Pi 3 Zero, Armbian/Ubuntu):
  pip3 install pymavlink opencv-python-headless numpy
"""

import argparse
import sys
import time
import threading
import struct
import os
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    print("ОШИБКА: opencv-python не установлен. Установите: pip3 install opencv-python-headless")
    sys.exit(1)

try:
    from pymavlink import mavutil
except ImportError:
    print("ОШИБКА: pymavlink не установлен. Установите: pip3 install pymavlink")
    sys.exit(1)


# ─────────────────────────────────────────────────────────
#  MAVLink Proxy: НСУ ↔ CUAV X7+ Pro
# ─────────────────────────────────────────────────────────

class MavlinkProxy:
    """
    Двунаправленный MAVLink-прокси между НСУ (через WFB) и CUAV X7+ Pro.

    Поток данных:
      НСУ → WFB rx pipe → Orange Pi → serial → CUAV (команды)
      CUAV → serial → Orange Pi → WFB tx pipe → НСУ (телеметрия)
    """

    def __init__(self, fc_device, fc_baud, wfb_rx_pipe, wfb_tx_telemetry):
        self.fc_device = fc_device
        self.fc_baud = fc_baud
        self.wfb_rx_pipe = wfb_rx_pipe      # команды от НСУ
        self.wfb_tx_telemetry = wfb_tx_telemetry  # телеметрия вниз

        self.fc_conn = None
        self.rx_pipe_fd = None
        self.tx_pipe_fd = None
        self.running = False
        self.stats = {'cmds_forwarded': 0, 'tel_forwarded': 0, 'errors': 0}

    def connect_fc(self):
        """Подключение к CUAV X7+ Pro через UART/USB."""
        self.fc_conn = mavutil.mavlink_connection(
            self.fc_device, baud=self.fc_baud,
            force_connected=True)
        print(f"[MAVLink] Подключено к FC: {self.fc_device} @ {self.fc_baud}")

        # Ждём heartbeat
        print("[MAVLink] Ожидание heartbeat...")
        self.fc_conn.wait_heartbeat(timeout=15)
        print(f"[MAVLink] Heartbeat получен! System={self.fc_conn.target_system}")

    def open_pipes(self):
        """Открытие WFB-каналов (named pipes)."""
        # RX pipe: команды от НСУ (создаётся WFB-ng)
        if not os.path.exists(self.wfb_rx_pipe):
            os.mkfifo(self.wfb_rx_pipe)
        self.rx_pipe_fd = os.open(self.wfb_rx_pipe, os.O_RDONLY | os.O_NONBLOCK)
        print(f"[MAVLink] RX pipe открыт: {self.wfb_rx_pipe}")

        # TX pipe: телеметрия вниз (создаётся WFB-ng)
        if not os.path.exists(self.wfb_tx_telemetry):
            os.mkfifo(self.wfb_tx_telemetry)
        self.tx_pipe_fd = os.open(self.wfb_tx_telemetry, os.O_WRONLY | os.O_NONBLOCK)
        print(f"[MAVLink] TX pipe открыт: {self.wfb_tx_telemetry}")

    def run(self):
        """Основной цикл: два потока (commands up, telemetry down)."""
        self.connect_fc()
        self.open_pipes()
        self.running = True

        # Поток 1: команды от НСУ → CUAV
        cmd_thread = threading.Thread(target=self._forward_commands, daemon=True)
        # Поток 2: телеметрия от CUAV → НСУ
        tel_thread = threading.Thread(target=self._forward_telemetry, daemon=True)

        cmd_thread.start()
        tel_thread.start()

        print("[MAVLink] Прокси запущен (commands + telemetry)")

        # Мониторинг
        while self.running:
            time.sleep(5.0)
            print(f"[MAVLink] Статистика: cmds={self.stats['cmds_forwarded']}, "
                  f"tel={self.stats['tel_forwarded']}, "
                  f"err={self.stats['errors']}")

    def _forward_commands(self):
        """Чтение MAVLink команд из WFB rx pipe → отправка на CUAV."""
        buf = bytearray()
        while self.running:
            try:
                data = os.read(self.rx_pipe_fd, 1024)
                if data:
                    buf.extend(data)
                    # Парсим MAVLink-сообщения из буфера
                    while len(buf) >= 2:
                        # MAVLink v2: 0xFD, v1: 0xFE
                        if buf[0] not in (0xFD, 0xFE):
                            buf.pop(0)
                            continue

                        # Определяем длину пакета
                        if buf[0] == 0xFD:
                            # MAVLink v2
                            if len(buf) < 3:
                                break
                            plen = buf[1] + 12  # payload + header + checksum + sig
                        else:
                            # MAVLink v1
                            if len(buf) < 6:
                                break
                            plen = buf[1] + 8  # payload + header + checksum

                        if len(buf) < plen:
                            break

                        # Отправляем raw-пакет на FC
                        packet = bytes(buf[:plen])
                        self.fc_conn.write(packet)
                        self.stats['cmds_forwarded'] += 1
                        del buf[:plen]

            except BlockingIOError:
                time.sleep(0.001)
            except Exception as e:
                self.stats['errors'] += 1
                if self.stats['errors'] % 100 == 1:
                    print(f"[MAVLink] Ошибка forward_commands: {e}")
                time.sleep(0.01)

    def _forward_telemetry(self):
        """Чтение телеметрии от CUAV → отправка в WFB tx pipe."""
        while self.running:
            try:
                msg = self.fc_conn.recv_msg()
                if msg is None:
                    time.sleep(0.001)
                    continue

                # Сериализуем и отправляем вниз
                raw = msg.get_msgbuf()
                try:
                    os.write(self.tx_pipe_fd, raw)
                    self.stats['tel_forwarded'] += 1
                except BlockingIOError:
                    pass  # pipe полон — пропускаем (BEST_EFFORT)
                except BrokenPipeError:
                    # WFB rx ещё не подключён
                    time.sleep(0.1)

            except Exception as e:
                self.stats['errors'] += 1
                if self.stats['errors'] % 100 == 1:
                    print(f"[MAVLink] Ошибка forward_telemetry: {e}")
                time.sleep(0.01)

    def stop(self):
        self.running = False
        if self.rx_pipe_fd:
            os.close(self.rx_pipe_fd)
        if self.tx_pipe_fd:
            os.close(self.tx_pipe_fd)
        if self.fc_conn:
            self.fc_conn.close()


# ─────────────────────────────────────────────────────────
#  IR Camera → WFB (видео вниз)
# ─────────────────────────────────────────────────────────

class IRVideoRelay:
    """
    Захват IR-камеры (USB) → кодирование H.264 → отправка через WFB.

    Использует GStreamer pipeline или OpenCV VideoWriter для кодирования.
    Вывод — raw H.264 NAL units в named pipe (WFB tx video1).
    """

    def __init__(self, ir_device, wfb_tx_pipe, width=640, height=480, fps=30,
                 bitrate=1000000):
        self.ir_device = ir_device
        self.wfb_tx_pipe = wfb_tx_pipe
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.cap = None
        self.writer = None
        self.running = False
        self.stats = {'frames_captured': 0, 'frames_sent': 0, 'errors': 0}

    def start(self):
        """Открытие камеры и кодировщика."""
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

        # Кодировщик H.264 через GStreamer (аппаратный, если доступен)
        # На Orange Pi (H6 SoC) — cedrus264 (аппаратный H.264)
        # Fallback — x264enc (программный)
        gst_pipeline = (
            f'appsrc ! video/x-raw,format=BGR,width={self.width},'
            f'height={self.height},framerate={self.fps}/1 ! '
            f'videoconvert ! video/x-raw,format=I420 ! '
            f'x264enc tune=zerolatency bitrate={self.bitrate // 1000} '
            f'speed-preset=ultrafast ! '
            f'video/x-h264,profile=baseline ! '
            f'appsink drop=true sync=false'
        )

        try:
            self.writer = cv2.VideoWriter(
                gst_pipeline, cv2.CAP_GSTREAMER, 0,
                self.fps, (self.width, self.height))
            if not self.writer.isOpened():
                raise RuntimeError("GStreamer writer не открылся")
            print("[IR] Кодировщик H.264 (GStreamer) запущен")
        except Exception as e:
            print(f"[IR] GStreamer недоступен ({e}), fallback на raw frames")
            self.writer = None

        # Открытие WFB tx pipe
        if not os.path.exists(self.wfb_tx_pipe):
            os.mkfifo(self.wfb_tx_pipe)
        self.pipe_fd = os.open(self.wfb_tx_pipe, os.O_WRONLY | os.O_NONBLOCK)
        print(f"[IR] TX pipe открыт: {self.wfb_tx_pipe}")

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
                # GStreamer: пишем BGR-кадр → H.264 NAL → pipe
                self.writer.write(frame)
            else:
                # Fallback: отправляем JPEG-кадры (без H.264)
                _, jpeg = cv2.imencode('.jpg', frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
                try:
                    # Префикс: magic + размер (для фрейминга на стороне приёмника)
                    header = struct.pack('<4sI', b'IRJP', len(jpeg))
                    os.write(self.pipe_fd, header + jpeg.tobytes())
                    self.stats['frames_sent'] += 1
                except (BlockingIOError, BrokenPipeError):
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
        if hasattr(self, 'pipe_fd') and self.pipe_fd:
            os.close(self.pipe_fd)
        print("[IR] Остановлен")


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Бортовой прокси: Orange Pi 3 Zero (MAVLink + IR video)')
    parser.add_argument('--fc-device', default='/dev/ttyAMA0',
                        help='CUAV X7+ Pro serial device (default: /dev/ttyAMA0)')
    parser.add_argument('--fc-baud', type=int, default=921600,
                        help='MAVLink baudrate (default: 921600)')
    parser.add_argument('--wfb-rx-pipe', default='/tmp/wfb_rx_command',
                        help='WFB rx pipe: команды от НСУ')
    parser.add_argument('--wfb-tx-telemetry', default='/tmp/wfb_tx_telemetry',
                        help='WFB tx pipe: телеметрия вниз')
    parser.add_argument('--wfb-tx-video1', default='/tmp/wfb_tx_video1',
                        help='WFB tx pipe: IR-видео вниз')
    parser.add_argument('--ir-device', default='/dev/video0',
                        help='IR-камера USB device')
    parser.add_argument('--ir-width', type=int, default=640)
    parser.add_argument('--ir-height', type=int, default=480)
    parser.add_argument('--ir-fps', type=int, default=30)
    parser.add_argument('--ir-bitrate', type=int, default=1000000,
                        help='H.264 bitrate (bps)')
    parser.add_argument('--no-ir', action='store_true',
                        help='Отключить IR-камеру (только MAVLink proxy)')
    args = parser.parse_args()

    print("=" * 60)
    print("БОРТОВОЙ ПРОКСИ — Orange Pi 3 Zero")
    print("=" * 60)
    print(f"  FC:          {args.fc_device} @ {args.fc_baud}")
    print(f"  WFB rx cmd:  {args.wfb_rx_pipe}")
    print(f"  WFB tx tel:  {args.wfb_tx_telemetry}")
    if not args.no_ir:
        print(f"  IR камера:   {args.ir_device} "
              f"{args.ir_width}x{args.ir_height}@{args.ir_fps}")
        print(f"  WFB tx vid:  {args.wfb_tx_video1}")
    print("=" * 60)

    # MAVLink прокси (в отдельном потоке)
    proxy = MavlinkProxy(
        fc_device=args.fc_device,
        fc_baud=args.fc_baud,
        wfb_rx_pipe=args.wfb_rx_pipe,
        wfb_tx_telemetry=args.wfb_tx_telemetry,
    )

    proxy_thread = threading.Thread(target=proxy.run, daemon=True)
    proxy_thread.start()

    # IR-видео (в основном потоке)
    if not args.no_ir:
        ir_relay = IRVideoRelay(
            ir_device=args.ir_device,
            wfb_tx_pipe=args.wfb_tx_video1,
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
            proxy.stop()
    else:
        print("[MAIN] Только MAVLink прокси (без IR)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            proxy.stop()

    print("\n[MAIN] Остановлен")


if __name__ == '__main__':
    main()
