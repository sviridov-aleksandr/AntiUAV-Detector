# Anti-UAV Interceptor — Система автономного перехвата дронов

Полнофункциональная система реального времени для обнаружения, трекинга, удержания и поражения БПЛА. Включает мультиспектральную детекцию (YOLO11 + Optical Flow + IR), трекинг с оценкой дальности (Kalman + pinhole), удержание цели (PID), расчёт точки перехвата (lead pursuit), подрыв БЧ (proximity fuze + MAVLink) и деплой на NVIDIA Jetson Orin.

## Архитектура системы

```
                    ┌─────────────────────────────────────────────────┐
                    │              КАМЕРЫ (EO + IR)                    │
                    │  OpenIPC MC800S-V3 (RTSP)  /  USB /  Файл        │
                    └───────────┬───────────────────┬──────────────────┘
                                │                   │
                    /camera/image_raw    /camera/ir_image_raw
                                │                   │
                    ┌───────────▼───────────────────▼──────────────────┐
                    │              VIDEO PUBLISHER                      │
                    │  source_type: file | rtsp | usb                   │
                    │  Авто-реконект, BUFFERSIZE=1 (мин. задержка)      │
                    └───────────────────┬──────────────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────────────┐
                    │              VISION NODE                          │
                    │                                                   │
                    │  ┌─────────┐  ┌──────────┐  ┌────────────────┐   │
                    │  │  YOLO   │  │IR Tracker│  │ Optical Flow   │   │
                    │  │ (EO)    │  │(thermal) │  │ (motion)       │   │
                    │  └────┬────┘  └────┬─────┘  └───────┬────────┘   │
                    │       │            │                 │            │
                    │       └──── FUSION ─┘                 │            │
                    │        eo_primary / ir_primary        │            │
                    │        fused (weighted avg)           │            │
                    │                   │                   │            │
                    │           ┌───────▼───────┐           │            │
                    │           │ TargetEstimator│           │            │
                    │           │ (Kalman+Range) │           │            │
                    │           └───────┬───────┘           │            │
                    │                   │                   │            │
                    │  SEARCH → TRACK → INTERCEPT → STRIKE  │            │
                    │  (PID pan/tilt + proportional approach)            │
                    └──────────┬──────────────────┬──────────────────────┘
                               │                  │
                        /cmd_vel           /interceptor/strike
                               │                  │
                    ┌──────────▼──────────────────▼──────────────────────┐
                    │              MAVLINK BRIDGE                        │
                    │  /cmd_vel → set_position_target_local_ned_send     │
                    │  (BODY_OFFSET_NED, velocity + yaw_rate)            │
                    │  /strike → DO_SET_SERVO (channel 6, PWM 2000)      │
                    │  Телеметрия: alt, GPS, heading, attitude, mode     │
                    └───────────────────────┬───────────────────────────┘
                                            │
                                CUAV X7+ Pro (автопилот)
```

## Конечный автомат перехвата

| Состояние | Условие перехода | Действие |
|-----------|------------------|----------|
| `SEARCH` | Нет цели > N кадров | Синусоидальное сканирование по yaw |
| `TRACK` | Цель обнаружена, dist > intercept_distance | PID удержание + пропорциональное сближение |
| `INTERCEPT` | dist < intercept_distance ИЛИ bbox_ratio ≥ 0.35 | Финальное сближение на макс. скорости, расчёт точки перехвата |
| `STRIKE` | dist < kill_radius (4.0 м) | Подрыв БЧ: `/interceptor/strike` → MAVLink `DO_SET_SERVO` |
| `LOST` | Нет цели > max_lost_frames | Полный стоп, переход в SEARCH |

## Мультиспектральная детекция

### Цепочка обнаружения

```
YOLO (EO, видимый спектр) ──► IR Tracker (тепловизор) ──► Optical Flow (движение) ──► LOST
         │                           │                          │
         └───────── FUSION ──────────┘                          │
                    │                                           │
           eo_primary: EO→IR→OF                                 │
           ir_primary: IR→EO→OF                                 │
           fused: weighted average (IR weight=0.4)              │
                    │                                           │
              TargetEstimator ◄─────────────────────────────────┘
```

### YOLO11l (основной детектор, видимый спектр)
- **Модель:** merged_v1_L (YOLO11L), 1 класс (Drone), 25.3M параметров
- **Датасет merged_v1:** 268,305 train / 26,841 val (drone_v2 + Anti-UAV-RGBT + Seraphim)
- **Метрики (эпоха 19/60):** mAP50=0.968, mAP50-95=0.646
- **TensorRT FP16:** 421 FPS (2.4 мс/кадр) на RTX 5080 Laptop

### IR Tracker (тепловизионный детектор)
- Пороговая обработка (adaptive / fixed / otsu) + морфология
- Connected components → фильтр по размеру, интенсивности, aspect ratio
- Persistence-логика: min_persistence=3 кадров для подтверждения цели
- Источник: отдельный IR-поток (`/camera/ir_image_raw`) или тот же кадр (fallback)

### Optical Flow (fallback по движению)
- Dense Farneback + компенсация глобального движения (affine)
- Motion mask → connected components → фильтр кандидатов
- Persistence + history smoothing (5 кадров)
- Ограничение: `of_fallback_frames=5` кадров после потери YOLO

### Dual-Band Fusion (EO + IR)
- **eo_primary** — EO главное, IR как fallback (день, хорошая видимость)
- **ir_primary** — IR главное, EO как fallback (ночь, туман)
- **fused** — взвешенное среднее позиций EO и IR (всегда, макс. устойчивость)

## Оценка дальности и трекинг

### RangeEstimator (pinhole camera model)
```
distance = (real_size_m × focal_px) / bbox_px
```
- focal_px = 1109 (FOV 60°, 1280px ширина)
- real_size_m = 0.35 (типичный размер дрона)

### KalmanTracker (constant velocity model)
- State: [x, y, vx, vy] (image plane, px + px/frame)
- Predict + Update цикл, сглаживание шумных детекций
- Lead pursuit: предсказание позиции на N кадров вперёд (lead_frames=5)

### InterceptCalculator (lead pursuit в 3D)
- Решение квадратного уравнения: время перехвата T
- Перевод image-plane → NED (через focal_px)
- Bearing: yaw + pitch к точке перехвата
- 3 стратегии: `pursuit` (lead), `head_on` (прямой), `top_dive` (сверху)

## STRIKE — Proximity Fuze (подрыв БЧ)

```
distance < kill_radius (4.0 м)
    ↓
vision_node публикует /interceptor/strike (однократно)
    ↓
mavlink_bridge: MAV_CMD_DO_SET_SERVO (channel=6, PWM=2000)
    ↓
Сервопривод → ударник → детонатор → осколочное облако БЧ
```

## Обучение и датасеты

### Датасеты

| Датасет | Изображений | Особенность | Источник |
|---------|-------------|-------------|----------|
| **drone_v2** | 164,732 train / 6,833 val | Средние/крупные + негативы | MyDataSet + combined |
| **Anti-UAV-RGBT** | 28,439 train / 11,659 val | **22% small (32-64px)**, RGB+IR | ZhaoJ9014 (Google Drive) |
| **Seraphim** | 75,134 train / 8,349 test | **11% tiny (<20px)**, 23 источника | HuggingFace (lgrzybowski) |
| **merged_v1** | **268,305 train / 26,841 val** | Объединение всех трёх | build_merged_v1.py |

### Пайплайн обучения

```
drone_v2 (164K) ──► YOLO11L (v2-5, 50 epochs) ──► best.pt (mAP50=0.897)
                                                        │
merged_v1 (268K) ──► YOLO11L (merged_v1_L, 60 epochs) ◄──┘  ← старт с v2-5
                         │
                    best.pt (mAP50=0.968, эпоха 19/60)
                         │
                    soft labels (.npz)
                         │
                    YOLO11M/S (студент, KD)
                         │
                    TensorRT INT8 → Jetson Orin
```

### Knowledge Distillation (учитель → студент)
- **Учитель:** YOLO11L merged_v1_L (mAP50=0.968)
- **Студент:** YOLO11M/S/N (для Jetson, быстрее и легче)
- **Метод:** offline KD — soft labels учителя + transfer weights + пониженный LR
- **Ожидаемый прирост:** +2-5% mAP для студента по сравнению с обучением с нуля

### Результаты batch-теста (merged_v1_L, 83 видео)

| Метрика | Значение |
|---------|----------|
| **STRIKE (успешный перехват)** | **6 / 83 (7.2%)** |
| Средняя доля детекций | 58.6% |
| Средний FPS | 123 |
| Трекер | ByteTrack |

**STRIKE-видео:** m4.MOV (2.46 мин), v12 (2.17 мин), v22 (3.88 мин), v6 (0.91 мин), v61 (0.37 мин), v62 (0.44 мин)

### Симулятор полного пайплайна
```bash
python3 train/simulate_intercept.py video.mp4 [strategy] [kill_radius] [intercept_distance]
```
Проверен на v78: TRACK 43 → INTERCEPT 50 → STRIKE 55 кадров, подрыв при 3.96 м.

## Эксперименты (НЕ использовать — не дали выигрыша)

| Эксперимент | Результат | Причина |
|-------------|-----------|---------|
| `hybrid_detector.py` (OF→YOLO ROI) | 28-42% (хуже YOLO 82-98%) | ROI теряет контекст |
| `tiled_detector.py` (tiling 640px) | Нет выигрыша, 4-6× медленнее | 117-121ms vs 18-31ms |

**Вывод:** YOLO full-frame + OF fallback — оптимальная архитектура. Для мелких целей нужно добучение, не усложнение детектора.

## Стек технологий

| Компонент | Версия |
|-----------|--------|
| OS (PC) | Ubuntu 26.04 (Linux) |
| Python | 3.14 |
| ROS2 | Lyrical |
| Ultralytics | 8.4.117 (YOLO11) |
| PyTorch | 2.x (CUDA 13.0) |
| OpenCV | 5.0.0 |
| pymavlink | 2.4.49 |
| GPU (PC) | RTX 5080 Laptop 16GB |
| GPU (deploy) | NVIDIA Jetson Orin |
| Автопилот | CUAV X7+ Pro |
| Камера | OpenIPC MC800S-V3 (SSC338Q + IMX415, 4K@20fps, RTSP) |

## Структура проекта

```
AntiUAV-Detector/
├── train/                          # Скрипты обучения и тестов
│   ├── run_drone_v2.py             # Обучение YOLO11L на drone_v2
│   ├── run_merged_v1.py            # Обучение учителя на merged_v1 (268K)
│   ├── distill_yolo.py             # Knowledge Distillation (учитель→студент)
│   ├── simulate_intercept.py       # Симулятор полного пайплайна (без ROS2)
│   ├── batch_test_videos.py        # Пакетный тест всех видео (83 шт.)
│   ├── osd_filter.py               # Умный OSD-фильтр (is_osd_false_positive)
│   ├── analyze_failures.py         # Разбор проблемных видео
│   ├── compare_track_predict.py    # Сравнение трекера и предсказания
│   ├── target_estimator.py         # RangeEstimator + KalmanTracker + InterceptCalculator
│   ├── optical_flow_tracker.py     # Farneback + affine compensation
│   ├── hybrid_detector.py          # [НЕ ИСПОЛЬЗОВАТЬ] OF→YOLO ROI
│   └── tiled_detector.py           # [НЕ ИСПОЛЬЗОВАТЬ] tiling 640px
├── prepare_data/                   # Сборка и конвертация датасетов
│   ├── build_drone_v2.py           # Сборка drone_v2 (1 класс + негативы)
│   ├── build_merged_v1.py          # Объединение drone_v2 + Anti-UAV + Seraphim
│   ├── convert_antiuav_rgbt.py     # Конвертация Anti-UAV-RGBT → YOLO
│   ├── drone_v2/                   # Датасет (164K train)
│   ├── antiuav_yolo/               # Конвертированный Anti-UAV (40K)
│   └── merged_v1/                  # Объединённый датасет (268K train)
├── ros2_src/uav_interceptor/       # ROS2 пакет (зеркало aerial_nav_ws)
│   ├── vision_node.py              # YOLO + IR + OF + Fusion + PID + state machine
│   ├── video_publisher.py          # RTSP / USB / File → ROS Image
│   ├── mavlink_bridge.py           # ROS → MAVLink + телеметрия + STRIKE
│   ├── ir_tracker.py               # Thermal detection (threshold + morphology)
│   ├── optical_flow_tracker.py     # Dense OF + motion compensation
│   ├── target_estimator.py         # Range + Kalman + Intercept
│   └── launch/interceptor.launch.py # Launch-файл со всеми параметрами
├── export_tensorrt.py              # Экспорт ONNX + TensorRT FP16
├── docs/
│   └── DEPLOY_JETSON.md            # Инструкция деплоя на Jetson Orin
├── runs/detect/train/runs/         # Результаты обучения (weights/, results.csv)
├── video-FPV/                      # Тестовые видео (v2, v67-v78)
└── README.md
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
pip install ultralytics pymavlink pyserial opencv-python
```

### 3. ROS2 workspace
```bash
# Копируем ROS2 пакет в workspace
cp -r ros2_src/uav_interceptor ~/aerial_nav_ws/src/

cd ~/aerial_nav_ws
source /opt/ros/lyrical/setup.bash
colcon build --packages-select uav_interceptor

# Workaround: bin → lib
mkdir -p install/uav_interceptor/lib/uav_interceptor
cp install/uav_interceptor/bin/{vision_node,video_publisher,mavlink_bridge} \
   install/uav_interceptor/lib/uav_interceptor/

source install/setup.bash
```

## Обучение

### Этап 1: drone_v2 (базовая модель)
```bash
source venv/bin/activate
python train/run_drone_v2.py
```
Параметры: epochs=50, batch=16, lr0=0.005, imgsz=640, cache=disk

### Этап 2: merged_v1 (учитель, после завершения v2-5)
```bash
python train/run_merged_v1.py
```
Параметры: epochs=60, batch=16, lr0=0.005, старт с best.pt drone_v2-5

### Этап 3: Knowledge Distillation (студент)
```bash
python train/distill_yolo.py \
    --stage full \
    --teacher runs/detect/train/runs/merged_v1_L/weights/best.pt \
    --student yolo11m.pt \
    --data prepare_data/merged_v1/data.yaml \
    --alpha 0.5 --temperature 2.0 \
    --epochs 60 --batch 16
```

## Экспорт TensorRT

```bash
source venv/bin/activate
python export_tensorrt.py [путь_к_model.pt]
```
Результат: `best.onnx` (FP32), `best.engine` (FP16).

**Важно:** `.engine` привязан к GPU — на Jetson пересобирать из `.onnx`.

## Запуск

### Полный пайплайн (с автопилотом)
```bash
source /opt/ros/lyrical/setup.bash
source ~/aerial_nav_ws/install/setup.bash
export PYTHONPATH="$HOME/AntiUAV-Detector/venv/lib/python3.14/site-packages:$PYTHONPATH"

ros2 launch uav_interceptor interceptor.launch.py \
    model_path:=/home/alex/AntiUAV-Detector/runs/detect/train/runs/merged_v1_L/weights/best.pt \
    device:=/dev/ttyACM0 \
    simulation:=false \
    show_image:=true
```

### С RTSP-камерой OpenIPC
```bash
ros2 launch uav_interceptor interceptor.launch.py \
    source_type:=rtsp \
    rtsp_url:=rtsp://192.168.1.10:554/live \
    use_dual_band:=true \
    fusion_mode:=fused
```

### С USB-камерой
```bash
ros2 launch uav_interceptor interceptor.launch.py \
    source_type:=usb \
    usb_device:=/dev/video0
```

### Simulation (без железа)
```bash
ros2 launch uav_interceptor interceptor.launch.py \
    simulation:=true \
    show_image:=true
```

### Симулятор (без ROS2)
```bash
python3 train/simulate_intercept.py video.mp4 pursuit 4.0 8.0
```

## Параметры launch

| Параметр | По умолчанию | Описание |
|-----------|--------------|----------|
| `source_type` | `file` | Источник видео: file | rtsp | usb |
| `video_path` | `.../v2.mp4` | Тестовое видео (source_type=file) |
| `rtsp_url` | `rtsp://192.168.1.10:554/live` | RTSP поток (source_type=rtsp) |
| `usb_device` | `/dev/video0` | USB камера (source_type=usb) |
| `target_fps` | `30` | Целевая частота публикации |
| `device` | `/dev/ttyACM0` | MAVLink serial |
| `simulation` | `false` | Без железа (лог команд) |
| `model_path` | `.../best.pt` | YOLO модель (.pt/.onnx/.engine) |
| `show_image` | `true` | Окно детекции |
| `use_dual_band` | `false` | Включить EO+IR fusion |
| `fusion_mode` | `eo_primary` | eo_primary | ir_primary | fused |
| `ir_topic` | `/camera/ir_image_raw` | ROS-топик IR-камеры |

### Параметры PID и детекции (ROS params)

| Параметр | По умолчанию | Описание |
|-----------|--------------|----------|
| `pid_pan_kp/ki/kd` | 0.05/0.001/0.01 | PID yaw (pan) |
| `pid_tilt_kp/ki/kd` | 0.05/0.001/0.01 | PID высота (tilt) |
| `pid_output_limit` | 0.5 | Лимит выхода PID |
| `approach_speed` | 0.3 | Базовая скорость сближения |
| `intercept_distance` | 3.0 | Порог INTERCEPT (м) |
| `kill_radius` | 4.0 | Порог STRIKE (м) |
| `intercept_strategy` | `pursuit` | pursuit | head_on | top_dive |
| `interceptor_speed` | 15.0 | Макс. скорость перехватчика (м/с) |
| `lead_frames` | 5 | Kalman lead (кадров вперёд) |
| `camera_fov_h` | 60.0 | Горизонтальный FOV (град) |
| `drone_size_m` | 0.35 | Физический размер дрона (м) |
| `conf_threshold` | 0.4 | Порог уверенности YOLO |
| `osd_margin` | 60 | Отступ от краёв (фильтр OSD) |
| `min_bbox_area` | 500 | Мин. площадь bbox |
| `use_optical_flow` | `true` | Optical Flow fallback |
| `use_ir_tracker` | `true` | IR Tracker fallback |
| `ir_threshold_mode` | `adaptive` | adaptive | fixed | otsu |
| `fusion_ir_weight` | 0.4 | Вес IR в fused-режиме (0-1) |
| `strike_servo_channel` | 6 | AUX-канал сервопривода БЧ |
| `strike_servo_pwm` | 2000 | PWM для подрыва |

## Топики ROS2

| Топик | Тип | QoS | Описание |
|-------|-----|-----|----------|
| `/camera/image_raw` | Image | BEST_EFFORT | Видеопоток EO (видимый спектр) |
| `/camera/ir_image_raw` | Image | BEST_EFFORT | Видеопоток IR (тепловизор) |
| `/cmd_vel` | Twist | RELIABLE | Команды движения |
| `/interceptor/state` | String | RELIABLE | SEARCH/TRACK/INTERCEPT/STRIKE/LOST |
| `/interceptor/strike` | String | RELIABLE | Команда подрыва БЧ |
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