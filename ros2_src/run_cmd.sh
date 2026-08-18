#!/usr/bin/env bash
# Управление перехватчиком (терминал 2)
#
# Использование:
#   ./run_cmd.sh telemetry          # показать все топики телеметрии (по одному)
#   ./run_cmd.sh mode               # текущий режим
#   ./run_cmd.sh arm                # армирование
#   ./run_cmd.sh disarm             # разоружить
#   ./run_cmd.sh hover              # зависнуть (нулевая скорость)
#   ./run_cmd.sh strike             # подрыв БЧ (DO_SET_SERVO ch6 PWM 2000)
#   ./run_cmd.sh kill               # аварийная остановка
#   ./run_cmd.sh vel 1.0 0.0 0.0    # скорость vx vy vz (м/с)
#   ./run_cmd.sh topics             # список топиков
#

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

CMD="${1:-help}"

case "$CMD" in
    telemetry)
        echo "=== Телеметрия (по одному топику) ==="
        for t in mode altitude heading ground_speed battery link_status gps velocity attitude; do
            echo "--- /telemetry/$t ---"
            timeout 3 ros2 topic echo "/telemetry/$t" --once 2>/dev/null || echo "(нет данных)"
        done
        ;;
    mode)
        ros2 topic echo /telemetry/mode --once
        ;;
    arm)
        echo ">>> ARMING"
        ros2 topic pub --once /interceptor/arm std_msgs/msg/Bool "{data: true}"
        ;;
    disarm)
        echo ">>> DISARMING"
        ros2 topic pub --once /interceptor/arm std_msgs/msg/Bool "{data: false}"
        ;;
    hover)
        echo ">>> HOVER (нулевая скорость)"
        ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
            "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
        ;;
    strike)
        echo ">>> STRIKE (подрыв БЧ)"
        ros2 topic pub --once /interceptor/strike std_msgs/msg/String "{data: 'manual'}"
        ;;
    kill)
        echo ">>> KILL SWITCH"
        ros2 topic pub --once /interceptor/kill std_msgs/msg/Bool "{data: true}"
        ;;
    vel)
        VX="${2:-0.0}"
        VY="${3:-0.0}"
        VZ="${4:-0.0}"
        echo ">>> VELOCITY: vx=$VX vy=$VY vz=$VZ"
        ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
            "{linear: {x: $VX, y: $VY, z: $VZ}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
        ;;
    topics)
        ros2 topic list | grep -E "telemetry|cmd_vel|interceptor"
        ;;
    *)
        echo "Использование: $0 {telemetry|mode|arm|disarm|hover|strike|kill|vel|topics}"
        exit 1
        ;;
esac
