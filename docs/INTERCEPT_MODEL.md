# Модель поведения дрона-перехватчика — декомпозиция

Полное описание конвейера: входные данные → блоки обработки → управляющие воздействия → замыкание контура.

## 1. Общий конвейер

```
[ВХОД] Видео EO ─► Детекция YOLO ─┐
[ВХОД] Видео IR ─► IR-трекер ────┤
[ВХОД] Кадр EO  ─► Optical Flow ─┘──► FUSION ─► Оценка дальности ─► Kalman ─► PID ─► cmd_vel ─► MAVLink ─► Дрон
                                   └─► State Machine (SEARCH/TRACK/INTERCEPT/STRIKE/LOST)
[ВХОД] Телеметрия автопилота (GPS, alt, heading, battery) ◄────────┘
```

## 2. Входные данные системы

| № | Данные | Источник | Формат/частота |
|---|--------|----------|----------------|
| 1 | Видео видимого спектра (EO) | OpenIPC MC800S-V3 / WFB-ng | BGR8, 1280×720, ~30 FPS |
| 2 | Видео тепловизионное (IR) | USB IR-камера | 640×480, ~30 FPS |
| 3 | Телеметрия автопилота | CUAV X7+ (MAVLink) | GPS, alt, heading, battery, mode, link_status (~10 Гц) |
| 4 | Параметры конфигурации | Launch-файл | Пороги, PID, дистанции, стратегии |

## 3. Блоки обработки

### 3.1 Детекция (3 параллельных канала)

| Блок | Вход | Выход | Назначение |
|------|------|-------|-----------|
| YOLO (EO) | Кадр BGR8, conf≥0.4, ByteTrack | box (x1,y1,x2,y2) + track_id + conf | Основной детектор |
| IR-трекер | Кадр IR, threshold adaptive/fixed/otsu + морфология | box горячего пятна (двигатель) | Ночь/туман |
| Optical Flow | Кадр EO, Farneback dense + компенсация ego-motion | box по движению | Fallback на 5 кадров |
| OSD-фильтр | box + размер кадра | отсев ложных (края, <500 px²) | Подавление OSD-артефактов |

### 3.2 Fusion (EO + IR)

| Вход | Выход |
|------|-------|
| box EO + box IR, режим `eo_primary`/`ir_primary`/`fused` | Единая позиция цели (x,y) + box_ratio + источник |

### 3.3 Оценка дальности (RangeEstimator)

- Формула: `distance = real_size_m × focal_px / bbox_px`
- Параметры: focal_px=1109 (FOV 60°, 1280px), real_size=0.35 м
- Выход: дистанция (м), closing_speed (м/с, отрицательная = сближение)

### 3.4 Трекинг (KalmanTracker)

- Модель: constant velocity, state=[x,y,vx,vy]
- Выход: сглаженная позиция, скорость (px/кадр), **lead_point** — предсказание на N кадров вперёд

### 3.5 Расчёт перехвата (InterceptCalculator)

- Вход: позиция цели в NED, скорость цели (м/с), скорость перехватчика (15 м/с)
- Метод: lead pursuit, квадратное уравнение `(V_i²−V_t²)T² − 2(P·V_t)T − |P|² = 0`
- Выход: точка встречи, время до встречи T, пеленг (yaw, pitch)
- Fallback: pure pursuit (прицел в текущую позицию), если решения нет

### 3.6 PID + State Machine (vision_node)

| Вход | Управляющее воздействие |
|------|-------------------------|
| error_x = lead_x − центр кадра | `angular.z` — yaw_rate (PID pan) |
| error_y = lead_y − центр кадра | `linear.z` — Vz (PID tilt) |
| distance (м) | `linear.x` — Vx вперёд, пропорционально дальности |
| distance < kill_radius (4 м) | `/interceptor/strike` (однократно) |

## 4. State Machine

```
SEARCH → TRACK → INTERCEPT → STRIKE
           ↑         ↓
          LOST ←─────┘
```

| Состояние | Условие входа | Поведение |
|-----------|---------------|-----------|
| SEARCH | Нет цели > max_lost_frames (10) | Синусоидальный скан: `angular.z = 0.5·sin(φ)` |
| TRACK | Цель обнаружена, distance ≥ intercept_distance (3 м) | PID удержание в центре + пропорциональное сближение |
| INTERCEPT | distance < 3 м или bbox_ratio ≥ 0.35 | 3 стратегии (см. ниже) |
| STRIKE | distance < kill_radius (4 м) | Однократный подрыв БЧ, продолжение движения |
| LOST | Цель потеряна ≤ max_lost_frames | Полный стоп, затем возврат в SEARCH |

## 5. Стратегии INTERCEPT

| Стратегия | Логика | Управляющие воздействия |
|-----------|--------|-------------------------|
| `pursuit` (default) | Lead pursuit: прицел в точку встречи | `angular.z = clip(yaw_cmd·2, ±0.5)`, `linear.z = clip(−pitch_cmd·2, ±0.5)`, `linear.x = approach·1.5` |
| `head_on` | В лоб, максимальная скорость | Прицел в цель, `linear.x = approach·1.5` |
| `top_dive` | Набор высоты (цель в верхней части) → пикирование сверху | Климб: `linear.z = +0.8·approach`; дайв: `linear.z = −clip(...)` |

## 6. Управляющие воздействия (MAVLink)

| ROS2 топик | MAVLink команда | Действие |
|-----------|----------------|----------|
| `/cmd_vel` | `SET_POSITION_TARGET_LOCAL_NED` (BODY_OFFSET_NED) | Скорость vx, vy, vz + yaw_rate (режим GUIDED) |
| `/interceptor/strike` | `DO_SET_SERVO` ch6, PWM 2000 | Подрыв БЧ (ударник → детонатор) |
| — (geofence) | `MAV_CMD_DO_SET_MODE` → RTL | Возврат при превышении радиуса от home (haversine) |
| — (kill) | `MAV_CMD_COMPONENT_ARM_DISARM` | Немедленная остановка моторов |

## 7. Замыкание контура (телеметрия)

```
mavlink_bridge ← MAVLink (GPS, alt, heading, battery, mode, link_status)
     └─► /telemetry/* ──► vision_node (для отладки и геофенса)
```

## 8. Сводная таблица управляющих воздействий

| Воздействие | Источник | Канал |
|-------------|----------|-------|
| Yaw_rate (поворот) | PID pan / intercept bearing | `cmd_vel.angular.z` |
| Вертикаль (Vz) | PID tilt / dive | `cmd_vel.linear.z` |
| Вперёд (Vx) | Пропорция от дальности | `cmd_vel.linear.x` |
| Подрыв БЧ | Proximity fuze (<4 м), однократно | `DO_SET_SERVO` ch6 |
| Поиск | Синусоидальный скан | `cmd_vel.angular.z` |
| RTL (геофенс/failsafe) | Haversine от home / потеря связи | `DO_SET_MODE` → RTL |
| Kill switch | Ручная команда | `COMPONENT_ARM_DISARM` |

## 9. Ключевые параметры

| Параметр | Значение | Назначение |
|----------|----------|-----------|
| `conf_threshold` | 0.4 | Порог детекции YOLO |
| `max_lost_frames` | 10 | Потеря цели → SEARCH |
| `approach_speed` | 0.3 | Базовая скорость сближения (м/с) |
| `target_bbox_ratio` | 0.15 | Порог TRACK по размеру bbox |
| `intercept_bbox_ratio` | 0.35 | Порог INTERCEPT по bbox |
| `intercept_distance` | 3.0 м | Дистанция перехода в INTERCEPT |
| `kill_radius` | 4.0 м | Дистанция подрыва БЧ |
| `interceptor_speed` | 15.0 м/с | Макс. скорость перехватчика |
| `lead_frames` | 5 | Предсказание Kalman вперёд |
| `link_latency` | 0.15 с | Задержка радиоканала (добавляет lead-кадры) |
| `focal_px` | 1109 | Фокусное расстояние (FOV 60°, 1280px) |
| `real_size_m` | 0.35 м | Физический размер цели |

## 10. Ограничения и риски

- Дальность по pinhole: точность ±20%, при box < 2px — недостоверно
- При STRIKE и kill_radius=4 м > intercept_distance=3 м: STRIKE может наступить раньше INTERCEPT-маневра (проверить порядок порогов)
- Kalman lead без учёта ускорения цели (constant velocity)
- top_dive не проверен в симуляции на всех стратегиях
- Задержка end-to-end ~250–350 мс компенсируется lead_frames (latency×FPS)
