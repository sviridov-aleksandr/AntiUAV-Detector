# WFB-ng — радиоканал для дрона-перехватчика

## Обзор

WFB-ng (WiFi Broadcast Next Generation) — пакетный радиоканал на базе raw WiFi.
Обеспечивает:
- **Видео вниз** (EO + IR) с низкой задержкой
- **MAVLink вверх/вниз** (двунаправленный, для управления)
- **IP-туннель** (SSH, диагностика)

## Архитектура каналов

```
БОРТ (drone)                          ЗЕМЛЯ (gs)
┌─────────────────────┐               ┌─────────────────────┐
│ OpenIPC EO (H.265)  │──video0 0x00──►│ UDP 5600 → ROS2     │
│ Orange Pi IR (H.264)│──video1 0x01──►│ UDP 5601 → ROS2     │
│ CUAV X7+ (serial)   │──mavlink 0x10/0x90──►│ UDP 14550 → ROS2 │
│                     │◄──mavlink 0x90/0x10──│ ROS2 → UDP 14550 │
│ SSH (tunnel)        │──tunnel 0x20/0xa0──►│ 10.5.0.1/24      │
└─────────────────────┘               └─────────────────────┘
```

## Требования к железу

| Компонент | Требование |
|-----------|-----------|
| WiFi-адаптер | **RTL8812AU** или **RTL8812EU** (рекомендуется BL-M8812EU2) |
| Питание | Отдельный BEC 5V/5A (НЕ от USB!) |
| Конденсатор | ≥470uF Low ESR (Panasonic EEUFR1V102) |
| Охлаждение | Радиатор 30x30mm + вентилятор (обязательно!) |
| Антенны | Подключены всегда (без антенн НЕ включать!) |

## Установка

### 1. Драйвер WiFi (на борту и на земле)

```bash
# Для RTL8812AU — патченный драйвер v5.2.20
# Для RTL8812EU — патченный драйвер

# Блэклист стокового драйвера
cat > /etc/modprobe.d/wfb.conf <<EOF
blacklist 88XXau
blacklist 8812au
blacklist rtl8812au
blacklist rtl88x2bs
options 88XXau_wfb rtw_tx_pwr_idx_override=30
EOF

# Для 8812eu:
# options 8812eu rtw_tx_pwr_by_rate=0 rtw_tx_pwr_lmt_enable=0

update-initramfs -k all -u
reboot

# Проверка: версия драйвера должна быть пустой
ethtool -i wlan0
```

### 2. Установка wfb-ng

```bash
# Вариант A: apt-репозиторий
curl -s https://apt.wfb-ng.org/apt.key | sudo apt-key add -
echo "deb https://apt.wfb-ng.org/ $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/wfb-ng.list
sudo apt update
sudo apt install wfb-ng

# Вариант B: из исходников
git clone https://github.com/svpcom/wfb-ng.git
cd wfb-ng
sudo make install
```

### 3. Ключи шифрования

```bash
# Генерируем ключи (один раз!)
wfb_keygen

# Ключи:
#   /etc/gs.key    — на земле (НСУ)
#   /etc/drone.key — на борту (Orange Pi)
# ВАЖНО: не запускать wfb_keygen дважды на одной стороне!
```

### 4. Конфигурация

```bash
# Борт (Orange Pi 3 Zero)
cp wfb/drone.cfg /etc/wifibroadcast.cfg

# Земля (Jetson Orin Nano)
cp wfb/gs.cfg /etc/wifibroadcast.cfg
```

### 5. Настройка интерфейса

```bash
# /etc/default/wifibroadcast — указать имя WiFi-интерфейса
# wlan0 → заменить на фактическое имя (wlan0, wfb0 и т.д.)

# Отключить NetworkManager для wlan
cat >> /etc/NetworkManager/NetworkManager.conf <<EOF
[keyfile]
unmanaged-devices=interface-name:wlan0
EOF

# Отключить wpa_supplicant
sudo systemctl stop wpa_supplicant
sudo systemctl disable wpa_supplicant

# RF-kill
sudo rfkill unblock all
```

### 6. Запуск

```bash
# Борт
sudo systemctl start wifibroadcast@drone

# Земля
sudo systemctl start wifibroadcast@gs

# Мониторинг
wfb-cli gs    # на земле
wfb-cli drone # на борту
```

## Настройка видео на борту

### EO-камера (OpenIPC MC800S-V3)

OpenIPC имеет встроенную поддержку WFB. Настройка через web-интерфейс или SSH:

```bash
# На OpenIPC (SSH)
# Включить WFB-режим
fw_setenv wfb_tx 1
fw_setenv wfb_channel 165
fw_setenv wfb_mcs 1
fw_setenv wfb_fec_k 8
fw_setenv wfb_fec_n 12
fw_setenv wfb_bandwidth 20
reboot
```

### IR-камера (Orange Pi → GStreamer → UDP 5603)

```bash
# На Orange Pi 3 Zero
source ~/proxy-venv/bin/activate
python3 onboard_proxy.py \
    --ir-device /dev/video0 \
    --ir-width 640 --ir-height 480 --ir-fps 30 \
    --ir-udp-port 5603
```

## Настройка MAVLink

### Борт (CUAV X7+ Pro → serial)

```bash
# WFB-ng сам пробрасывает MAVLink между serial и UDP
# В drone.cfg: peer = 'serial:ttyAMA0:921600'
# CUAV TELEM2 → Orange Pi UART (TX↔RX, GND)

# Параметры ArduPilot:
#   SERIAL2_PROTOCOL = 2 (MAVLink 2)
#   SERIAL2_BAUD = 921 (921600)
```

### Земля (НСУ → UDP 14550)

```bash
# WFB-ng направляет MAVLink на UDP 14550
# mavlink_bridge.py слушает UDP 14550 (link_mode=radio)
ros2 run uav_interceptor mavlink_bridge --ros-args \
    -p link_mode:=radio \
    -p mavlink_udp_port:=14550 \
    -p simulation:=false
```

## Проверка

### 1. Связь

```bash
wfb-cli gs
# Должны видеть: RSSI, FEC, потеря пакетов
```

### 2. Видео

```bash
# На земле: приём EO-видео
gst-launch-1.0 udpsrc port=5600 ! h265parse ! nvv4l2decoder ! autovideosink

# Приём IR-видео
gst-launch-1.0 udpsrc port=5601 ! h264parse ! nvv4l2decoder ! autovideosink
```

### 3. MAVLink

```bash
# На земле: проверка телеметрии
mavproxy.py --master=udp:127.0.0.1:14550 --out=udp:127.0.0.1:14551
# Должны видеть HEARTBEAT от CUAV X7+ Pro
```

### 4. Туннель (SSH)

```bash
# На земле
ping 10.5.0.2
ssh orangepi@10.5.0.2
```

## Troubleshooting

| Проблема | Решение |
|----------|---------|
| RF-kill | `sudo rfkill unblock all` |
| Нет связи | Проверить канал (165 на обеих сторонах), регион (BO) |
| Много FEC-ошибок | Плохой USB-кабель, слабое питание, нет конденсатора |
| Перегрев карты | Радиатор + вентилятор обязательны |
| wpa_supplicant мешает | `systemctl stop wpa_supplicant` |
| NetworkManager перехватывает | `unmanaged-devices=interface-name:wlan0` |
| Низкая мощность TX | `rtw_tx_pwr_idx_override=30` (8812au) или `wifi_txpower=3000` (8812eu) |
