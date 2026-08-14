# Anti-UAV Interceptor — Система автономного перехвата дронов

Система реального времени для обнаружения, трекинга и перехвата БПЛА. Включает детекцию на базе YOLO11, визуальный сервопривод (PID), интеграцию с автопилотом через MAVLink/ROS2 и деплой на NVIDIA Jetson Orin.

## Архитектура

```
Камера → video_publisher → vision_node (YOLO + PID) → /cmd_vel → mavlink_bridge → CUAV X7+ Pro
                              ↓
                        /interceptor/state
```

**Конечный автомат перехвата:**

| Состояние | Описание |
|-----------|----------|
| `SEARCH` | Поиск цели (синусоидальное вращение по yaw) |
| `TRACK` | Цель захвачена, PID удержание в центре + пропорциональное сближение |
| `INTERCEPT` | Цель занимает ≥35% кадра — финальное сближение на максимальной скорости |
| `LOST` | Цель потеряна — полный стоп, переход в SEARCH после N кадров |

## Компоненты

### Детекция (YOLO11l)
- **Модель:** YOLO11l, 1 класс (Drone)
- **Датасет:** `drone_v2` — 164,732 train / 6,833 val изображений
- **Метрики (эпоха 12):** mAP50=0.863, mAP50-95=0.430, Precision=0.827, Recall=0.895
- **TensorRT FP16:** 37.1 FPS на RTX 5080

### Визуальный сервопривод (PID)
- **Pan (yaw):** PID по горизонтальной ошибке (цель — центр кадра)
- **Tilt (высота):** PID по вертикальной ошибке
- **Approach (вперёд):** пропорционально `bbox_ratio` — чем дальше цель, тем выше скорость
- **Anti-windup:** ограничение интеграла и выхода

### Фильтрация ложных срабатываний
- Отбрасывание детектов в краевых зонах (OSD: текст, иконки, прицел)
- Минимальный размер bbox — отсек шум и мелкие артефакты

### ROS2 интеграция
- **video_publisher** — публикация кадров в `/camera/image_raw` (BEST_EFFORT QoS)
- **vision_node** — YOLO + трекинг ByteTrack + PID → `/cmd_vel`, `/interceptor/state`
- **mavlink_bridge** — `/cmd_vel` → MAVLink velocity commands + телеметрия

### MAVLink (CUAV X7+ Pro)
- `set_position_target_local_ned_send` (BODY_OFFSET_NED, velocity + yaw_rate)
- Автоматическая установка режима GUIDED
- Телеметрия: altitude, GPS, heading, attitude, velocity, mode (10 Hz)
- Защита: STOP (hover) при отсутствии команд >1с

## Стек технологий

| Компонент | Версия |
|-----------|--------|
| OS | Ubuntu 26.04 (Linux) |
| Python | 3.14 |
| ROS2 | Lyrical |
| Ultralytics | YOLO11l |
| PyTorch | 2.x (CUDA 13.0) |
| pymavlink | 2.4.49 |
| GPU (PC) | RTX 5080 Laptop 16GB |
| GPU (deploy) | NVIDIA Jetson Orin |
| Автопилот | CUAV X7+ Pro |

## Структура проекта

```
AntiUAV-Detector/
├── train/
│   └── run_drone_v2.py          # Обучение YOLO11l на drone_v2
├── prepare_data/
│   ├── build_drone_v2.py        # Сборка датасета (1 класс Drone + негативы)
│   ├── fix_class_ids.py         # Исправление ID классов
│   ├── combine_datasets.py      # Объединение датасетов
│   └── drone_v2/                # Датасет (data.yaml, train/, val/)
├── export_tensorrt.py           # Экспорт ONNX + TensorRT FP16
├── docs/
│   └── DEPLOY_JETSON.md         # Инструкция деплоя на Jetson Orin
├── runs/detect/train/runs/      # Результаты обучения (weights/, results.csv)
├── video-FPV/                   # Тестовые видео
└── README.md

aerial_nav_ws/                   # ROS2 workspace (отдельно)
└── src/uav_interceptor/
    ├── uav_interceptor/
    │   ├── vision_node.py       # YOLO + PID + state machine
    │   ├── video_publisher.py   # Камера/видео → ROS Image
    │   └── mavlink_bridge.py    # ROS → MAVLink + телеметрия
    ├── launch/
    │   └── interceptor.launch.py
    └── config/
        └── interceptor.rviz
```

## Установка

### 1. Клонирование
```bash
git clone https://github.com/sviridov-aleksandr/AntiUAV-Detector.git
cd AntiUAV-Detector
```

### 2. Виртуальное окружение
```bash
python -m venv venv
source venv/bin/activate
pip install ultralytics pymavlink pyserial
```

### 3. ROS2 workspace
```bash
cd ~/aerial_nav_ws
colcon build --packages-select uav_interceptor
source install/setup.bash
```

## Обучение

```bash
source venv/bin/activate
python train/run_drone_v2.py
```

Параметры (в `run_drone_v2.py`):
- `epochs=50`, `patience=20`, `batch=16`, `imgsz=640`
- `lr0=0.005`, `warmup_epochs=1.0`
- `mixup=0.0`, `copy_paste=0.0` (отключены — вредят для 1 класса)
- `close_mosaic=5`, `cache='disk'`

## Экспорт TensorRT

```bash
source venv/bin/activate
python export_tensorrt.py
```

Результат: `best.onnx` (FP32), `best.fp16.onnx`, `best.engine` (FP16).

**Важно:** `.engine` привязан к GPU — на Jetson пересобирать из `.onnx`.

## Запуск

### Полный пайплайн (с автопилотом)
```bash
source /opt/ros/lyrical/setup.bash
source ~/aerial_nav_ws/install/setup.bash
export PYTHONPATH="$HOME/AntiUAV-Detector/venv/lib/python3.14/site-packages:$PYTHONPATH"

ros2 launch uav_interceptor interceptor.launch.py \
    model_path:=/home/alex/AntiUAV-Detector/runs/detect/train/runs/drone_v2-4/weights/best.pt \
    device:=/dev/ttyACM0 \
    simulation:=false \
    show_image:=true
```

### Simulation (без автопилота)
```bash
ros2 launch uav_interceptor interceptor.launch.py \
    simulation:=true \
    show_image:=true
```

### Параметры launch

| Параметр | По умолчанию | Описание |
|-----------|--------------|----------|
| `video_path` | `.../v2.mp4` | Тестовое видео |
| `device` | `/dev/ttyACM0` | MAVLink serial |
| `simulation` | `false` | Без железа (лог команд) |
| `model_path` | `.../best.pt` | YOLO модель (.pt/.onnx/.engine) |
| `show_image` | `true` | Окно детекции |

### Настройка PID (ROS params)

| Параметр | По умолчанию | Описание |
|-----------|--------------|----------|
| `pid_pan_kp/ki/kd` | 0.05/0.001/0.01 | PID yaw (pan) |
| `pid_tilt_kp/ki/kd` | 0.05/0.001/0.01 | PID высота (tilt) |
| `pid_output_limit` | 0.5 | Лимит выхода PID |
| `approach_speed` | 0.3 | Базовая скорость сближения |
| `target_bbox_ratio` | 0.15 | Порог перехода TRACK→INTERCEPT |
| `intercept_bbox_ratio` | 0.35 | Порог финального сближения |
| `osd_margin` | 60 | Отступ от краёв (фильтр OSD) |
| `min_bbox_area` | 500 | Мин. площадь bbox |
| `conf_threshold` | 0.4 | Порог уверенности YOLO |

## Топики ROS2

| Топик | Тип | QoS | Описание |
|-------|-----|-----|----------|
| `/camera/image_raw` | Image | BEST_EFFORT | Видеопоток |
| `/cmd_vel` | Twist | RELIABLE | Команды движения |
| `/interceptor/state` | String | RELIABLE | SEARCH/TRACK/INTERCEPT/LOST |
| `/telemetry/altitude` | Float64 | RELIABLE | Высота (м) |
| `/telemetry/gps` | NavSatFix | RELIABLE | GPS координаты |
| `/telemetry/heading` | Float64 | RELIABLE | Курс (град) |
| `/telemetry/attitude` | Imu | RELIABLE | Отношение (IMU) |
| `/telemetry/velocity` | Vector3Stamped | RELIABLE | Скорость (м/с) |
| `/telemetry/mode` | String | RELIABLE | Режим автопилота |

## Деплой на Jetson Orin

См. [docs/DEPLOY_JETSON.md](docs/DEPLOY_JETSON.md) — подробная инструкция по переносу, пересборке TensorRT и интеграции ROS2.

## Лицензия

MIT
