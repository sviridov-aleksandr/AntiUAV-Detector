# Anti-UAV Interceptor — Полное описание системы

## 1. Идея: что применяем и как

### Постановка задачи

Автономный перехватчик БПЛА: дрон-перехватчик обнаруживает цель (вражеский дрон) по видеопотоку, удерживает её в кадре, сближается и подрывает боевую часть (БЧ) при достижении дистанции поражения. Всё без оператора — полный автономный цикл от поиска до поражения.

### Концепция

```
Поиск (SEARCH) → Обнаружение → Удержание (TRACK) → Сближение (INTERCEPT) → Поражение (STRIKE)
```

### Что применяем

| Компонент | Технология | Зачем |
|-----------|-----------|-------|
| **Детекция (видимый спектр)** | YOLO11L / YOLO26L | Основной детектор дронов в кадре |
| **Детекция (тепловизор)** | IR Tracker (пороговая обработка) | Дрон в ИК — горячее пятно мотора/электроники |
| **Детекция (движение)** | Optical Flow (Farneback) | Fallback: мелкий дрон, который YOLO не видит |
| **Слияние каналов** | Dual-Band Fusion (EO + IR) | День — EO главное, ночь/туман — IR главное |
| **Оценка дальности** | Pinhole camera model | Дистанция до цели по размеру bbox |
| **Трекинг** | Kalman Filter (constant velocity) | Сглаживание шумных детекций, предсказание позиции |
| **Удержание цели** | PID (pan/tilt) | Держит цель в центре кадра |
| **Расчёт перехвата** | Lead pursuit (квадратное уравнение) | Точка встречи с учётом скорости цели |
| **Управление автопилотом** | MAVLink (ArduPilot) | Команды скорости + yaw_rate, подрыв БЧ через сервопривод |
| **Радиоканал** | WFB-ng (raw WiFi) | Видео вниз, MAVLink вверх/вниз, IP-туннель |
| **Робот-фреймворк** | ROS2 | Связь между узлами: видео → детекция → управление |
| **Деплой** | NVIDIA Jetson Orin + TensorRT | Real-time инференс на борту/земле |

### Шаги автономного цикла

1. **SEARCH** — синусоидальное сканирование по yaw (поиск цели в кадре)
2. **TRACK** — цель обнаружена, PID удерживает её в центре, пропорциональное сближение
3. **INTERCEPT** — дистанция < intercept_distance (8 м), максимальная скорость, расчёт точки перехвата
4. **STRIKE** — дистанция < kill_radius (4 м), подрыв БЧ: MAVLink `DO_SET_SERVO` → сервопривод → ударник → детонатор
5. **LOST** — цель потеряна > N кадров → полный стоп → возврат в SEARCH

---

## 2. От идеи до структуры проекта

### Логическая декомпозиция

Идея разбивается на 5 подсистем:

```
┌─────────────────────────────────────────────────────────┐
│                    СИСТЕМА ПЕРЕХВАТА                      │
├──────────┬──────────┬──────────┬──────────┬──────────────┤
│  ВИДЕО   │  ЗРЕНИЕ  │  ОЦЕНКА  │ УПРАВЛ.  │  РАДИОКАНАЛ  │
│ (захват) │ (детекц.) │ (дальн.) │ (MAVLink) │  (WFB-ng)   │
└──────────┴──────────┴──────────┴──────────┴──────────────┘
```

Каждая подсистема → отдельный модуль/узел:

| Подсистема | Реализация | Расположение |
|-----------|-----------|--------------|
| Захват видео | `video_publisher.py` / `wfb_video_receiver.py` | ROS2 узел |
| Детекция (YOLO + IR + OF) | `vision_node.py` | ROS2 узел |
| Оценка дальности + трекинг | `target_estimator.py` | Библиотека |
| Управление автопилотом | `mavlink_bridge.py` | ROS2 узел |
| Радиоканал | WFB-ng (системные конфиги) | `wfb/` |

Дополнительно — подсистема обучения и тестирования:

| Подсистема | Реализация | Расположение |
|-----------|-----------|--------------|
| Обучение моделей | `run_*.py`, `distill_yolo.py` | `train/` |
| Симулятор (без ROS2) | `simulate_intercept.py` | `train/` |
| Пакетное тестирование | `batch_test_videos.py` | `train/` |
| Подготовка датасетов | `build_*.py`, `convert_*.py` | `prepare_data/` |
| Экспорт моделей | `export_tensorrt.py` | корень |

### Физическая структура

```
AntiUAV-Detector/
├── train/                          # Обучение, тесты, симуляция
│   ├── run_drone_v2.py             # Этап 1: базовая модель (drone_v2, 164K)
│   ├── run_merged_v1.py            # Этап 2: учитель YOLO11L (merged_v1, 268K)
│   ├── run_yolo26_merged_v1.py     # Эксперимент: YOLO26L на merged_v1
│   ├── distill_yolo.py             # Knowledge Distillation (учитель → студент)
│   ├── simulate_intercept.py       # Симулятор полного цикла (без ROS2)
│   ├── batch_test_videos.py        # Пакетный тест 83 видео
│   ├── osd_filter.py               # Умный фильтр ложных детекций (HUD/OSD)
│   ├── analyze_failures.py         # Разбор проблемных видео
│   ├── compare_track_predict.py    # Сравнение трекера и предсказания
│   ├── target_estimator.py         # RangeEstimator + KalmanTracker + InterceptCalc
│   ├── optical_flow_tracker.py     # Dense Farneback + motion compensation
│   ├── live_detect.py              # Детекция с веб-камеры/RTSP (без ROS2)
│   ├── hybrid_detector.py          # [НЕ ИСПОЛЬЗОВАТЬ] OF→YOLO ROI
│   └── tiled_detector.py           # [НЕ ИСПОЛЬЗОВАТЬ] tiling 640px
│
├── prepare_data/                   # Сборка и конвертация датасетов
│   ├── build_drone_v2.py           # Сборка drone_v2 (1 класс + негативы)
│   ├── build_merged_v1.py          # Объединение drone_v2 + Anti-UAV + Seraphim
│   ├── convert_antiuav_rgbt.py     # Конвертация Anti-UAV-RGBT → YOLO
│   ├── drone_v2/                   # Датасет (164K train)
│   ├── antiuav_yolo/               # Anti-UAV-RGBT в YOLO (40K)
│   └── merged_v1/                  # Объединённый датасет (268K train)
│
├── ros2_src/uav_interceptor/       # ROS2 пакет
│   ├── vision_node.py              # YOLO + IR + OF + Fusion + PID + State Machine
│   ├── video_publisher.py          # RTSP / USB / File → ROS Image
│   ├── mavlink_bridge.py           # ROS → MAVLink + телеметрия + STRIKE
│   ├── ir_tracker.py               # Thermal detection (threshold + morphology)
│   ├── optical_flow_tracker.py     # Dense OF + motion compensation
│   ├── target_estimator.py         # Range + Kalman + Intercept
│   ├── launch/interceptor.launch.py # Launch-файл со всеми параметрами
│   ├── setup.py                    # ROS2 package setup
│   └── package.xml                 # ROS2 package manifest
│
├── ground/                         # Наземная НСУ
│   └── wfb_video_receiver.py       # Приём EO+IR видео через WFB-ng → ROS2
│
├── wfb/                            # Радиоканал WFB-ng
│   ├── drone.cfg                   # Конфиг борта (Orange Pi)
│   ├── gs.cfg                      # Конфиг земли (Jetson Orin)
│   ├── setup_telem2.py             # Настройка телеметрии CUAV
│   └── README.md                   # Инструкция по WFB-ng
│
├── export_tensorrt.py              # Экспорт ONNX + TensorRT FP16
├── docs/
│   ├── DEPLOY_JETSON.md            # Инструкция деплоя на Jetson Orin
│   ├── PROJECT_OVERVIEW.md         # Этот документ
│   └── session_log.md              # Лог сессии разработки
├── requirements.txt                # Python-зависимости
└── README.md                       # Краткое описание проекта
```

---

## 3. Как реализуется структура: что используем и почему

### 3.1. Детекция: YOLO (видимый спектр)

**Что:** Ultralytics YOLO11L (merged_v1_L) — основная модель детекции.

**Почему YOLO:**
- Real-time: 421 FPS на RTX 5080 (TensorRT FP16), ~160+ FPS на Jetson Orin
- One-stage: один проход сети, без region proposals
- Хорошо детектирует мелкие объекты (с STAL в YOLO26 — ещё лучше)
- Простой экспорт: ONNX → TensorRT → RKNN

**Почему L (Large), а не M/S/N:**
- L даёт максимальную точность (mAP50=0.968) — это учитель
- Для деплоя на Jetson — дистилляция в M/S (студент)
- L = 25.3M параметров, баланс точности и скорости

**Почему merged_v1 (268K), а не drone_v2 (164K):**
- drone_v2: средние/крупные дроны, мало мелких
- Anti-UAV-RGBT: 22% small (32-64px) — мелкие дроны на дальних дистанциях
- Seraphim: 11% tiny (<20px) — экстремально мелкие цели
- Объединение (merged_v1) покрывает весь спектр размеров

### 3.2. Детекция: IR Tracker (тепловизор)

**Что:** Пороговая обработка (adaptive/fixed/otsu) + морфология + connected components.

**Почему не YOLO для IR:**
- IR-датасет дронов слишком мал для обучения YOLO
- Дрон в ИК — яркое пятно на холодном фоне, простая пороговая обработка работает
- Не требует GPU — экономит ресурсы Jetson
- Дополняет EO: ночью/в тумане YOLO слепнет, IR видит тепло мотора

**Логика:**
```
Грейскейл → threshold → morphology (open+close) → connected components
→ фильтр по размеру/интенсивности/aspect ratio → persistence (3 кадра)
```

### 3.3. Детекция: Optical Flow (fallback по движению)

**Что:** Dense Farneback + компенсация глобального движения (affine).

**Почему:**
- Дрон < 10px — YOLO не видит, но он движется → OF ловит по движению
- Компенсация ego-motion: камера на дрон-перехватчике тоже движется,
  нужно вычесть глобальное движение камеры, чтобы найти локальное движение цели
- Fallback: включается на 5 кадров после потери YOLO, не постоянно

**Почему не трекинг точек (Lucas-Kanade):**
- Dense Farneback даёт плотное поле движения — не нужно знать, где цель
- LK требует начальной позиции (а цель потеряна)

### 3.4. Dual-Band Fusion (EO + IR)

**Что:** Три режима слияния каналов.

| Режим | Логика | Когда |
|-------|--------|-------|
| `eo_primary` | EO главное, IR как fallback | День, хорошая видимость |
| `ir_primary` | IR главное, EO как fallback | Ночь, туман, дым |
| `fused` | Взвешенное среднее позиций EO и IR | Всегда (макс. устойчивость) |

**Почему:** ни один канал не универсален. EO слепнет в темноте, IR не работает днём (фон горячий). Fusion обеспечивает работу 24/7.

### 3.5. Оценка дальности: Pinhole Camera Model

**Что:**
```
distance = (real_size_m × focal_px) / bbox_px
```

**Почему не лидар/радар:**
- Перехватчик — мелкий дрон, нет места для лидара
- Камера уже есть (для детекции), дальность — бесплатный побочный продукт
- Точность ±20% достаточна для порога STRIKE (4 м)

**Параметры:**
- `focal_px = 1109` (FOV 60°, 1280px ширина)
- `real_size_m = 0.35` (типичный размер дрона: размах пропеллеров)

### 3.6. Трекинг: Kalman Filter

**Что:** Constant velocity model, state = [x, y, vx, vy] (image plane).

**Почему:**
- Детекции шумные (bbox дрожит кадр-к-кадру) — Kalman сглаживает
- Предсказание позиции на N кадров вперёд (lead_frames=5) — для расчёта точки перехвата
- Оценка скорости цели — нужна для lead pursuit

### 3.7. Управление: PID + State Machine

**Что:** PID для pan (yaw) и tilt (pitch), пропорциональное сближение по дистанции.

**Почему PID:**
- Простая, проверенная архитектура для visual servoing
- Ошибка = смещение цели от центра кадра → PID → cmd_vel (yaw_rate + forward speed)
- Не требует модели динамики дрона (в отличие от LQR/MPC)

**State Machine:**
```
SEARCH → TRACK → INTERCEPT → STRIKE
           ↑         ↓
          LOST ←─────┘
```

### 3.8. Управление автопилотом: MAVLink

**Что:** `mavlink_bridge.py` переводит ROS2 команды в MAVLink.

| ROS2 топик | MAVLink команда | Действие |
|-----------|----------------|----------|
| `/cmd_vel` | `SET_POSITION_TARGET_LOCAL_NED` | Скорость + yaw_rate (BODY_OFFSET_NED) |
| `/interceptor/strike` | `DO_SET_SERVO` (ch6, PWM 2000) | Подрыв БЧ |

**Почему MAVLink, а не прямой PWM:**
- CUAV X7+ Pro работает на ArduPilot — MAVLink — нативный интерфейс
- `BODY_OFFSET_NED` — команды в системе координат дрона, не глобальной
- Телеметрия (GPS, высота, курс, IMU) — обратно через MAVLink в ROS2

**Почему не OFFBOARD (PX4):**
- ArduPilot (CUAV X7+) — `GUIDED` режим, не OFFBOARD
- `SET_POSITION_TARGET_LOCAL_NED` работает в GUIDED

### 3.9. Радиоканал: WFB-ng

**Что:** Raw WiFi пакетная передача (RTL8812AU/EU), не TCP/IP.

**Почему не обычный WiFi:**
- Обычный WiFi: TCP теряет пакеты, ретрансмиты, задержка растёт
- WFB-ng: FEC (Forward Error Correction), однонаправленная передача видео,
  нет ретрансмитов, задержка фиксирована (~150 мс round-trip)
- Видео вниз (EO + IR), MAVLink вверх/вниз, IP-туннель для SSH

**Архитектура:**
```
БОРТ (Orange Pi 3Z)                    ЗЕМЛЯ (Jetson Orin Nano)
┌────────────────────┐                ┌────────────────────┐
│ OpenIPC EO (H.265) │── video0 ─────►│ UDP 5600 → ROS2    │
│ IR-камера (H.264)  │── video1 ─────►│ UDP 5601 → ROS2    │
│ CUAV X7+ (serial)  │◄─ mavlink ────►│ UDP 14550 → ROS2   │
│                    │── tunnel ─────►│ 10.5.0.1/24 (SSH)  │
└────────────────────┘                └────────────────────┘
```

### 3.10. Робот-фреймворк: ROS2

**Что:** 3 узла, связанные топиками.

```
wfb_video_receiver ──► /camera/image_raw ──► vision_node
                     ──► /camera/ir_image_raw ──►

vision_node ──► /cmd_vel ──► mavlink_bridge
            ──► /interceptor/strike ──►

mavlink_bridge ──► /telemetry/* ──► vision_node
```

**Почему ROS2:**
- Стандартная шина данных: узлы независимы, можно тестировать по отдельности
- QoS: BEST_EFFORT для видео (допускает потерю кадров), RELIABLE для команд
- Launch-файл: один запуск поднимает всю систему
- Экосистема: rviz2, ros2 bag (запись логов полёта)

**Почему не просто Python-скрипт:**
- Симулятор (`simulate_intercept.py`) — для разработки без ROS2
- ROS2 — для реального железа: мульти-процесс, QoS, телеметрия

### 3.11. Обучение: двухэтапный pipeline

```
Этап 1: drone_v2 (164K) → YOLO11L (50 epochs) → drone_v2-5 (mAP50=0.897)
Этап 2: merged_v1 (268K) → YOLO11L (60 epochs, старт с v2-5) → merged_v1_L (mAP50=0.968)
Этап 3: Knowledge Distillation → YOLO11M (студент для Jetson)
```

**Почему двухэтап:**
- drone_v2 — крупные дроны, модель учит базовые признаки
- merged_v1 — добавляются мелкие (Anti-UAV) и экстремально мелкие (Seraphim)
- Старт с v2-5 (а не с COCO) — модель уже знает дроны, быстрее сходится

**Почему Knowledge Distillation:**
- Учитель (L) — максимальная точность, но тяжёлый для Jetson
- Студент (M/S) — легче, быстрее, но чуть менее точный
- KD: pseudo-labels учителя + transfer weights → студент наследует знания
- Offline KD: не требует модификации ultralytics, ~80% эффекта feature-level KD

### 3.12. OSD-фильтр

**Что:** `osd_filter.py` — отсекает ложные детекции в краях кадра (HUD/телеметрия OSD).

**Почему:**
- FPV-камеры накладывают телеметрию (батарея, GPS, скорость) на края кадра
- YOLO иногда детектирует OSD-текст как дрон
- Грубый фильтр (y1 < 60) резал 25% кадра на 240p
- Умный фильтр: пропускает крупные объекты и объекты с центром в безопасной зоне,
  отсекает только мелкие боксы у самых краёв

---

## 4. Программно-аппаратная реализация

### 4.1. Архитектура системы

```
┌─────────────────────────── БОРТ (дрон-перехватчик) ───────────────────────────┐
│                                                                               │
│  ┌─────────────┐   ┌───────────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │ OpenIPC EO  │   │ IR-камера     │   │ Orange Pi 3Z │   │ CUAV X7+ Pro   │  │
│  │ MC800S-V3   │   │ (USB)         │   │ (WFB-ng tx)  │   │ (ArduPilot)    │  │
│  │ 4K@20fps    │   │ 640x480@30    │   │              │   │                │  │
│  │ RTSP/H.265  │   │ H.264         │   │  video0 ◄────┤   │  TELEM2 ──────┤  │
│  │             │   │               │   │  video1 ◄────┤   │  (UART 921600)│  │
│  └──────┬──────┘   └───────┬───────┘   │  mavlink ◄───┤   └───────┬────────┘  │
│         │                  │           │  tunnel ◄───┤           │           │
│         │ H.265            │ H.264     └──────┬───────┘           │           │
│         └──────────────────┴──────────────────┤                   │           │
│                                              │ UART              │           │
│                                              └───────────────────┘           │
│                                                              │ SERVO CH6    │
│                                                              ▼ (PWM 2000)  │
│                                                     ┌────────────────┐      │
│                                                     │ Сервопривод →  │      │
│                                                     │ ударник → БЧ   │      │
│                                                     └────────────────┘      │
└──────────────────────────────────────────┬────────────────────────────────────┘
                                           │
                                    WFB-ng radio link
                                    (RTL8812AU, 5.8 GHz)
                                           │
┌─────────────────────────────── ЗЕМЛЯ (НСУ) ───────────────────────────────────┐
│                                                                               │
│  ┌──────────────┐   ┌───────────────────┐   ┌──────────────────────────────┐  │
│  │ WiFi adapter │   │ Jetson Orin Nano  │   │ ROS2 узлы                    │  │
│  │ RTL8812AU    │   │ (НСУ)             │   │                              │  │
│  │              │   │                   │   │  wfb_video_receiver           │  │
│  │  video0 ────►│   │  UDP 5600 ───────►│   │   → /camera/image_raw        │  │
│  │  video1 ────►│   │  UDP 5601 ───────►│   │   → /camera/ir_image_raw     │  │
│  │  mavlink ──►│   │  UDP 14550 ──────►│   │                              │  │
│  │  tunnel ──►│   │  10.5.0.1/24      │   │  vision_node                  │  │
│  └──────────────┘   └───────────────────┘   │   YOLO + IR + OF + Fusion    │  │
│                                              │   PID + State Machine        │  │
│                                              │   → /cmd_vel                  │  │
│                                              │   → /interceptor/strike       │  │
│                                              │                              │  │
│                                              │  mavlink_bridge               │  │
│                                              │   /cmd_vel → MAVLink → UDP    │  │
│                                              │   /strike → DO_SET_SERVO      │  │
│                                              │   ← telemetry → /telemetry/*  │  │
│                                              └──────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 4.2. Аппаратные компоненты

| Компонент | Модель | Назначение | Характеристики |
|-----------|--------|-----------|----------------|
| **Автопилот** | CUAV X7+ Pro | Управление полётом, БЧ | ArduPilot, IMU, GPS, 8x PWM |
| **EO-камера** | OpenIPC MC800S-V3 | Видимый спектр (детекция) | SSC338Q + IMX415, 4K@20fps, H.265, RTSP |
| **IR-камера** | USB-модуль | Тепловизор (ночь/туман) | 640×480@30fps, H.264 |
| **Бортовой комп** | Orange Pi 3 Zero | WFB-ng передатчик | Мультиплексирование видео + MAVLink |
| **Наземный комп** | NVIDIA Jetson Orin Nano | НСУ: детекция + управление | GPU для TensorRT, ROS2 |
| **WiFi-адаптер** | RTL8812AU/EU | Радиоканал WFB-ng | 5.8 GHz, raw WiFi, FEC |
| **Сервопривод** | Стандартный PWM | Подрыв БЧ | CH6, PWM 2000 = ударник |
| **GPU (PC)** | RTX 5080 Laptop | Обучение, экспорт | 16GB, CUDA 13.0, TensorRT 11.2 |

### 4.3. Программный стек

| Уровень | Технология | Версия | Где |
|---------|-----------|--------|-----|
| **OS (PC)** | Ubuntu | 26.04 | Обучение, разработка |
| **OS (Jetson)** | Ubuntu (JetPack) | 22.04 / JetPack 6.x | Деплой |
| **Python** | CPython | 3.14 (PC), 3.10 (Jetson) | Везде |
| **ML-фреймворк** | PyTorch | 2.13.0+cu130 (PC), 2.1+ (Jetson) | Обучение, инференс |
| **Детекция** | Ultralytics | 8.4.121 (YOLO11 + YOLO26) | Обучение, инференс |
| **Инференс-движок** | TensorRT | 11.2 (PC), 8.x (Jetson) | FP16/INT8 ускорение |
| **CV** | OpenCV | 5.0.0 | IR Tracker, Optical Flow, видео |
| **Робот-фреймворк** | ROS2 | Lyrical (PC), Humble (Jetson) | Связь узлов |
| **Автопилот** | ArduPilot | (на CUAV X7+ Pro) | MAVLink управление |
| **MAVLink** | pymavlink | 2.4.49 | Python ↔ автопилот |
| **Радиоканал** | WFB-ng | latest | Raw WiFi, FEC |

### 4.4. Поток данных в реальном времени

```
[Борт] OpenIPC EO ──H.265──► WFB-ng ──UDP 5600──► [Земля] GStreamer (nvv4l2decoder)
                                                              │
                                                    /camera/image_raw (BGR8)
                                                              │
                                                    ┌─────────▼──────────┐
                                                    │   vision_node       │
                                                    │                     │
                                                    │  YOLO (TensorRT)    │ ← 421 FPS (PC), ~160 FPS (Jetson)
                                                    │  → bbox + conf      │
                                                    │                     │
                                                    │  IR Tracker         │ ← параллельно, CPU
                                                    │  → bbox (thermal)   │
                                                    │                     │
                                                    │  Optical Flow       │ ← fallback, CPU
                                                    │  → bbox (motion)    │
                                                    │                     │
                                                    │  Fusion → target    │
                                                    │  RangeEstimator     │ → distance (м)
                                                    │  KalmanTracker      │ → predicted pos
                                                    │  PID                │ → cmd_vel
                                                    │  State Machine      │ → SEARCH/TRACK/INTERCEPT/STRIKE
                                                    └─────────┬──────────┘
                                                              │
                                              /cmd_vel (Twist)│ /interceptor/strike (String)
                                                              │
                                                    ┌─────────▼──────────┐
                                                    │  mavlink_bridge     │
                                                    │                     │
                                                    │  /cmd_vel →         │
                                                    │  SET_POSITION_      │
                                                    │  TARGET_LOCAL_NED   │ → UDP 14550 → WFB → CUAV
                                                    │  (BODY_OFFSET_NED)  │
                                                    │                     │
                                                    │  /strike →          │
                                                    │  DO_SET_SERVO       │ → CUAV → SERVO CH6 → БЧ
                                                    │  (ch6, PWM 2000)    │
                                                    │                     │
                                                    │  ← telemetry ←      │ ← CUAV → WFB → UDP 14550
                                                    │  GPS, alt, heading  │
                                                    │  → /telemetry/*     │
                                                    └────────────────────┘
```

### 4.5. Задержки (latency budget)

| Этап | Задержка | Где |
|------|---------|-----|
| Захват кадра (EO) | ~50 мс | OpenIPC |
| Кодирование H.265 | ~20 мс | OpenIPC (аппаратное) |
| Радиоканал WFB-ng | ~100-150 мс | RTL8812AU (one-way) |
| Декодирование H.265 | ~10 мс | Jetson (nvv4l2decoder) |
| YOLO инференс | ~6 мс (PC) / ~30 мс (Jetson) | TensorRT FP16 |
| IR Tracker + OF | ~5 мс | CPU |
| PID + State Machine | <1 мс | CPU |
| MAVLink команда | ~50 мс | WFB round-trip |
| **Итого (end-to-end)** | **~250-350 мс** | |

### 4.6. Модели и экспорт

| Модель | Параметры | mAP50 | FPS (PC) | FPS (Jetson) | Формат деплоя |
|--------|-----------|-------|----------|-------------|---------------|
| merged_v1_L (YOLO11L) | 25.3M | 0.968 | 421 | ~160 | TensorRT FP16 |
| merged_v1_26L (YOLO26L) | 26.2M | (обучается) | (TBD) | (TBD) | TensorRT FP16, NMS-free |
| distilled_M (YOLO11M, студент) | ~20M | (TBD) | (TBD) | (TBD) | TensorRT INT8 |

**Цепочка экспорта:**
```
best.pt → ONNX FP32 → TensorRT FP16 (PC) / TensorRT FP16 (Jetson, пересборка)
                     → RKNN INT8 (Orange Pi 5+, резервная платформа)
```

### 4.7. Безопасность

| Механизм | Реализация |
|----------|-----------|
| **Failsafe RTL** | Потеря связи > 5 сек → Return-to-Launch (mavlink_bridge) |
| **Geofence** | Макс. дистанция от точки старта (500 м, настраивается) |
| **Min battery** | < 14V (4S) → запрет армирования |
| **Kill switch** | Немедленная остановка моторов через MAVLink |
| **Pre-flight checks** | Батарея, GPS fix, режим автопилота перед армированием |
| **STRIKE однократно** | Команда подрыва отправляется один раз, повтор блокируется |

---

## 5. Текущий статус и планы

| Этап | Статус |
|------|--------|
| Обучение merged_v1_L (YOLO11L) | ✅ Готов (mAP50=0.968, эпоха 19/60) |
| Экспорт TensorRT FP16 | ✅ Готов (421 FPS) |
| Batch-тест 83 видео | ✅ 6/83 STRIKE, 123 FPS, ByteTrack |
| Обучение YOLO26L | 🔄 В процессе (эпоха 3/60) |
| Knowledge Distillation (студент M) | ⏳ Ожидает выбора учителя (YOLO11L или YOLO26L) |
| Интеграция с CUAV X7+ Pro | ⏳ MAVLink-мост не тестировался на реальном автопилоте |
| Полевые испытания | ⏳ Зависит от интеграции |
