# Наземная станция управления (НСУ) — Jetson Orin Nano

## Назначение

Приём видео с борта, детекция YOLO11L, трекинг, наведение, отправка MAVLink команд на борт.

## Архитектура

```
┌──────── НАЗЕМНАЯ СТАНЦИЯ (НСУ) ──────────────────────┐
│                                                      │
│  ┌─────────┐  ┌──────────┐  ┌────────────┐         │
│  │ WFB rx  │─►│ YOLO11L  │─►│ State Machine│         │
│  │ (EO+IR) │  │ (INT8)   │  │ (PID+Kalman)│         │
│  └─────────┘  └──────────┘  └──────┬─────┘         │
│                                     │                │
│                              ┌──────▼─────┐         │
│                              │ MAVLink tx  │         │
│                              │ (WFB uplink)│         │
│                              └──────┬─────┘         │
│─────────────────────────────────────┼────────────────│
│                                     │                │
│  ┌──────────────────────────────────┴──────────┐    │
│  │ QGroundControl / Mission Planner (мониторинг)│    │
│  └─────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

## Компоненты

| Файл | Роль |
|------|------|
| `ground/wfb_video_receiver.py` | Приём EO+IR видео через WFB, публикация в ROS2 |
| `ros2_src/uav_interceptor/vision_node.py` | YOLO + IR + OF + Fusion + PID + State Machine |
| `ros2_src/uav_interceptor/mavlink_bridge.py` | MAVLink через WFB radio (команды + телеметрия) |
| `ros2_src/uav_interceptor/interceptor.launch.py` | Запуск всех узлов |

## Установка на Jetson Orin Nano

### 1. JetPack + CUDA + TensorRT

```bash
# Проверка JetPack
cat /etc/nv_tegra_release

# CUDA
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
nvcc --version
```

### 2. PyTorch для Jetson (НЕ pip!)

```bash
# JetPack 6.x (Python 3.10, CUDA 12.x)
pip install torch torchvision \
    -f https://developer.download.nvidia.com/compute/redist/jp/v60dp/pytorch/

# Проверка
python -c "import torch; print(torch.cuda.is_available())"
```

### 3. Ultralytics + зависимости

```bash
pip install ultralytics pymavlink opencv-python numpy pyyaml

# TensorRT (уже в JetPack)
python -c "import tensorrt; print(tensorrt.__version__)"
```

### 4. ROS2

```bash
sudo apt install ros-humble-ros-base
source /opt/ros/humble/setup.bash
```

### 5. Сборка ROS2 пакета

```bash
cd ~/aerial_nav_ws
colcon build --packages-select uav_interceptor
mkdir -p install/uav_interceptor/lib/uav_interceptor
cp install/uav_interceptor/bin/{vision_node,video_publisher,mavlink_bridge,wfb_video_receiver} \
   install/uav_interceptor/lib/uav_interceptor/
source install/setup.bash
```

### 6. Конвертация модели в INT8 engine

```bash
# Перенести best.onnx с PC
scp user@pc:/home/alex/AntiUAV-Detector/runs/detect/train/runs/drone_v2-5/weights/best.onnx \
    ~/AntiUAV-Detector/weights/

# INT8 engine (требует калибровочный датасет)
yolo export model=weights/best.onnx format=engine int8=True \
    data=data.yaml imgsz=640

# Или FP16 (без калибровки)
yolo export model=weights/best.onnx format=engine half=True imgsz=640
```

### 7. Power mode

```bash
sudo nvpmodel -m 0    # MAXN
sudo jetson_clocks
```

## Запуск

### Полный пайплайн (WFB radio)

```bash
source /opt/ros/humble/setup.bash
source ~/aerial_nav_ws/install/setup.bash
export PYTHONPATH="$HOME/antiuav-venv/lib/python3.10/site-packages:$PYTHONPATH"

ros2 launch uav_interceptor interceptor.launch.py \
    model_path:=/home/user/AntiUAV-Detector/weights/best.engine \
    link_mode:=radio \
    simulation:=false \
    show_image:=true \
    use_dual_band:=true \
    fusion_mode:=eo_primary \
    link_latency:=0.15
```

### Тестовый режим (без радио, симуляция)

```bash
ros2 launch uav_interceptor interceptor.launch.py \
    model_path:=/home/user/AntiUAV-Detector/weights/best.engine \
    simulation:=true \
    show_image:=true
```

### Direct MAVLink (USB к CUAV, для тестов)

```bash
ros2 launch uav_interceptor interceptor.launch.py \
    model_path:=/home/user/AntiUAV-Detector/weights/best.engine \
    link_mode:=direct \
    device:=/dev/ttyACM0 \
    simulation:=false
```

## WFB-ng настройка (земля)

```bash
# Установка wfb-ng
git clone https://github.com/svpcom/wfb-ng.git
cd wfb-ng
sudo make install

# Конфигурация: /etc/wfb-ng.cfg
# На земле (rx):
#   wifi_channel = 149    # 5.8 GHz (тот же, что на борту)
#   stream1 = video0 (EO, rx)
#   stream2 = video1 (IR, rx)
#   stream3 = telemetry (rx)
#   stream4 = command (tx, MAVLink вверх)
```

## Топики ROS2

| Топик | Тип | Направление | Описание |
|-------|-----|-------------|----------|
| `/camera/image_raw` | Image | wfb_rx → vision | EO-кадры (BGR8) |
| `/camera/ir_image_raw` | Image | wfb_rx → vision | IR-кадры (BGR8) |
| `/cmd_vel` | Twist | vision → mavlink | Команды скорости |
| `/interceptor/strike` | String | vision → mavlink | Команда подрыва БЧ |
| `/interceptor/arm` | Bool | external → mavlink | Арм/дизарм |
| `/interceptor/kill` | Bool | external → mavlink | Kill switch |
| `/interceptor/state` | String | vision → external | Текущее состояние |
| `/telemetry/*` | various | mavlink → external | Телеметрия (GPS, IMU, batt) |

## Latency budget

```
Борт → WFB tx → воздух → WFB rx → decode → YOLO → state → MAVLink tx → воздух → CUAV
 5ms    10ms     5ms      10ms     10ms    15ms    2ms      2ms       5ms     5ms
                                                                              = ~69ms one-way
                                                                              = ~138ms round-trip
```

Компенсация: `link_latency=0.15` → +4-5 lead-кадров в Kalman prediction.

## Производительность

| Модель | FPS (FP16) | FPS (INT8) | VRAM | Рекомендация |
|--------|-----------|-----------|------|--------------|
| YOLO11L | 30-50 | 50-80 | ~4GB | ✅ Основной (INT8) |
| YOLO11M | 40-60 | 60-90 | ~3GB | Запасной |
| YOLO11S | 60-90 | 90-120 | ~1.5GB | Fast |
```