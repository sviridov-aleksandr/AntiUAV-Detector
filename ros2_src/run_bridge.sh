#!/usr/bin/env bash
# Запуск MAVLink-моста (терминал 1)
#
# Использование:
#   ./run_bridge.sh                    # direct, /dev/ttyACM0, 115200
#   ./run_bridge.sh radio              # radio mode (WFB-ng, UDP 14550)
#   ./run_bridge.sh direct /dev/ttyACM1 921600
#   ./run_bridge.sh sim                # симуляция без железа
#

# ── Параметры по умолчанию ─────────────────────────────────
LINK_MODE="${1:-direct}"
DEVICE="${2:-/dev/ttyACM0}"
BAUDRATE="${3:-115200}"
SIM="false"

if [[ "$LINK_MODE" == "sim" ]]; then
    LINK_MODE="direct"
    SIM="true"
fi

# ── Окружение ──────────────────────────────────────────────
cd /home/alex/AntiUAV-Detector
source venv/bin/activate
source /opt/ros/lyrical/setup.bash 2>/dev/null || source /opt/ros/humble/setup.bash 2>/dev/null
source ~/aerial_nav_ws/install/setup.bash 2>/dev/null
export PYTHONPATH="$HOME/AntiUAV-Detector/venv/lib/python3.14/site-packages:$PYTHONPATH"

if ! command -v ros2 >/dev/null 2>&1; then
    echo "ОШИБКА: ros2 не найден. Проверь установку ROS2." >&2
    exit 1
fi

echo "=== MAVLink Bridge ==="
echo "  link_mode:  $LINK_MODE"
echo "  device:     $DEVICE"
echo "  baudrate:   $BAUDRATE"
echo "  simulation: $SIM"
echo ""

# ── Запуск ─────────────────────────────────────────────────
exec ros2 run uav_interceptor mavlink_bridge --ros-args \
    -p link_mode:="$LINK_MODE" \
    -p device:="$DEVICE" \
    -p baudrate:="$BAUDRATE" \
    -p simulation:="$SIM"