# Бортовой прокси — Orange Pi 3 Zero

## Назначение

Минимальный компаньон-компьютер на борту дрона-перехватчика. Не выполняет детекцию — только ретрансляция:

1. **MAVLink proxy**: НСУ ↔ CUAV X7+ Pro (двунаправленный)
2. **IR-видео**: USB-камера → H.264 → WFB (вниз на НСУ)
3. **Телеметрия**: CUAV → WFB (вниз на НСУ)

## Архитектура

```
┌──────────┐  UART/USB   ┌──────────────┐  WiFi (WFB)  ┌──────┐
│ CUAV X7+ │◄──────────►│ Orange Pi 3Z │◄────────────►│ НСУ  │
│ Pro (FC) │  MAVLink    │ (proxy)      │   radio link  │(Jetson)│
└──────────┘             └──────┬───────┘              └──────┘
                                │ USB
                         ┌──────┴──────┐
                         │  IR-камера   │
                         └─────────────┘
```

## WFB-ng каналы

| Канал | Направление | Содержимое | Источник |
|-------|-------------|------------|----------|
| video0 | борт → земля | EO-видео H.265 | OpenIPC (напрямую в WFB) |
| video1 | борт → земля | IR-видео H.264 | Orange Pi (USB → encode → WFB) |
| telemetry | борт → земля | MAVLink телеметрия | CUAV → Orange Pi → WFB |
| command | земля → борт | MAVLink команды | НСУ → WFB → Orange Pi → CUAV |

## Установка на Orange Pi 3 Zero

### Зависимости

```bash
# Armbian / Ubuntu (Orange Pi 3 Zero, ARM64)
sudo apt update
sudo apt install -y python3-pip python3-venv gstreamer1.0-tools \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-libav

python3 -m venv ~/proxy-venv
source ~/proxy-venv/bin/activate
pip install pymavlink opencv-python-headless numpy
```

### UART подключение CUAV X7+ Pro

```
CUAV X7+ Pro TELEM2 ──► Orange Pi 3 Zero UART
  TX  ──►  RX
  RX  ──►  TX
  GND ──►  GND
  5V  ──►  (не подключать, Orange Pi питается отдельно)
```

Параметры ArduPilot:
```
SERIAL2_PROTOCOL = 2    # MAVLink 2
SERIAL2_BAUD = 921      # 921600
```

### WFB-ng настройка

```bash
# Установка wfb-ng на Orange Pi
git clone https://github.com/svpcom/wfb-ng.git
cd wfb-ng
sudo make install

# Конфигурация: /etc/wfb-ng.cfg
# На борту (tx):
#   wifi_channel = 149    # 5.8 GHz
#   stream1 = video0 (OpenIPC, raw H.265)
#   stream2 = video1 (IR, raw H.264 from Orange Pi)
#   stream3 = telemetry (MAVLink from CUAV)
#   stream4 = command (MAVLink from ground, rx)
```

## Запуск

### Полный режим (MAVLink + IR)

```bash
source ~/proxy-venv/bin/activate
python3 onboard_proxy.py \
    --fc-device /dev/ttyAMA0 \
    --fc-baud 921600 \
    --wfb-rx-pipe /tmp/wfb_rx_command \
    --wfb-tx-telemetry /tmp/wfb_tx_telemetry \
    --wfb-tx-video1 /tmp/wfb_tx_video1 \
    --ir-device /dev/video0 \
    --ir-width 640 --ir-height 480 --ir-fps 30
```

### Только MAVLink (без IR-камеры)

```bash
python3 onboard_proxy.py \
    --fc-device /dev/ttyAMA0 \
    --fc-baud 921600 \
    --no-ir
```

### Автозапуск (systemd)

```bash
sudo tee /etc/systemd/system/onboard-proxy.service << 'EOF'
[Unit]
Description=Onboard Proxy (MAVLink + IR Video)
After=network.target

[Service]
Type=simple
User=alex
ExecStart=/home/orangepi/proxy-venv/bin/python3 /home/orangepi/onboard_proxy.py \
    --fc-device /dev/ttyAMA0 --fc-baud 921600
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable onboard-proxy
sudo systemctl start onboard-proxy
```

## Проверка

### MAVLink

```bash
# На НСУ: отправка тестовой команды
mavproxy.py --master=/tmp/wfb_rx_command --baudrate 921600

# Проверка heartbeat
# Должны увидеть: HEARTBEAT {type: 2 (quadcopter), autopilot: 3 (ArduPilot)}
```

### IR-видео

```bash
# На НСУ: приём IR-потока
gst-launch-1.0 filesrc location=/tmp/wfb_rx_video1 \
    ! h264parse ! avdec_h264 ! videoconvert ! autovideosink
```

## Hardware

| Компонент | Спецификация |
|-----------|-------------|
| Orange Pi 3 Zero | H618 SoC, 1GB RAM, ARM64, ~10г |
| UART | CUAV TELEM2 ↔ Orange Pi UART |
| USB | IR-камера (FLIR Lepton / MLX90640 / топлинк) |
| Питание | 5V/2A от BEC (от CUAV или отдельный BEC) |
| WiFi | Внешний адаптер (RTL8812AU), 5.8 GHz, WFB-ng |
