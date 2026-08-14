# Деплой на NVIDIA Jetson Orin

## Обзор

Модель YOLO11L (drone_v2-5, mAP50=0.897) обучена на PC (RTX 5080, CUDA 13.0, TensorRT 11.2) и экспортирована в:

| Файл | Размер | Формат | Назначение |
|------|--------|--------|------------|
| `best.onnx` | 97 MB | ONNX FP32 | **Перенос на Jetson** (универсальный) |
| `best.fp16.onnx` | 49 MB | ONNX FP16 | Альтернатива (меньше, чуть быстрее) |
| `best.engine` | 50 MB | TensorRT FP16 | **Только для PC** (RTX 5080, TensorRT 11.2) |

**На Jetson engine нужно пересобрать из ONNX** — TensorRT на Jetson другой версии (8.x) и архитектуры (ARM64 + SoC GPU).

---

## 1. Перенос файлов на Jetson

```bash
# С PC на Jetson (по SSH)
scp /home/alex/AntiUAV-Detector/runs/detect/train/runs/drone_v2-5/weights/best.onnx \
    user@jetson-ip:/home/user/AntiUAV-Detector/weights/

# FP16 ONNX (опционально, для сравнения точности)
scp /home/alex/AntiUAV-Detector/runs/detect/train/runs/drone_v2-5/weights/best.fp16.onnx \
    user@jetson-ip:/home/user/AntiUAV-Detector/weights/
```

Также перенесите:
- `prepare_data/merged_v1/data.yaml` (для валидации и INT8-калибровки)
- ROS2 пакет `uav_interceptor` (или склонируйте репозиторий с GitHub)

```bash
# Клонирование репозитория на Jetson
ssh user@jetson-ip
git clone https://github.com/sviridov-aleksandr/AntiUAV-Detector.git
```

---

## 2. Установка окружения на Jetson

### 2.1. Системные пакеты

```bash
# JetPack 5.x (Ubuntu 20.04, Python 3.8) или JetPack 6.x (Ubuntu 22.04, Python 3.10)
sudo apt update && sudo apt install -y python3-pip python3-venv

# CUDA (уже в JetPack)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 2.2. Виртуальное окружение

```bash
python3 -m venv ~/antiuav-venv
source ~/antiuav-venv/bin/activate
```

### 2.3. PyTorch для Jetson

**Важно:** обычный `pip install torch` НЕ работает на Jetson. Нужна специальная сборка от NVIDIA.

```bash
# JetPack 5.x (Python 3.8, CUDA 11.4)
pip install torch==2.1.0a0+41361538.nv23.06 \
    torchvision==0.16.0a0+bfce5f2.nv23.06 \
    -f https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/

# JetPack 6.x (Python 3.10, CUDA 12.x) — другая сборка
# См. https://forums.developer.nvidia.com/t/pytorch-for-jetson/
```

### 2.4. Ultralytics и зависимости

```bash
pip install ultralytics pymavlink pyserial opencv-python

# TensorRT (уже в JetPack)
python -c "import tensorrt; print(tensorrt.__version__)"  # 8.x (JetPack 5) или 8.6+ (JetPack 6)
```

### 2.5. ROS2

```bash
# JetPack 5.x → ROS2 Humble, JetPack 6.x → ROS2 Iron/Jazzy
sudo apt install ros-humble-ros-base  # или ros-iron-ros-base
source /opt/ros/humble/setup.bash
```

---

## 3. Конвертация ONNX → TensorRT engine (FP16)

### Вариант A: через Ultralytics (рекомендуется)

```bash
source ~/antiuav-venv/bin/activate
cd ~/AntiUAV-Detector

yolo export model=weights/best.onnx format=engine half=True imgsz=640
# Результат: weights/best.engine (FP16, ~25-50 MB на Jetson)
```

### Вариант B: через trtexec (ручной контроль)

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=weights/best.onnx \
    --saveEngine=weights/best.engine \
    --fp16 \
    --workspace=2048 \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:1x3x640x640
```

### Вариант C: INT8 квантизация (макс. скорость, опционально)

```bash
# Требует калибровочный датасет (200-500 изображений из train)
yolo export model=weights/best.onnx format=engine int8=True \
    data=data.yaml imgsz=640
```

INT8 даёт +2x скорость, но требует калибровки и может потерять ~1-2% mAP.

---

## 4. Проверка engine на Jetson

```bash
source ~/antiuav-venv/bin/activate
python -c "
from ultralytics import YOLO
import time, cv2, numpy as np

model = YOLO('weights/best.engine')

# Бенчмарк на случайном кадре
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
start = time.time()
for _ in range(50):
    model.predict(source=dummy, imgsz=640, device=0, verbose=False)
dt = (time.time() - start) / 50
print(f'Инференс: {1/dt:.1f} FPS')

# Валидация (если есть data.yaml и датасет)
# model.val(data='data.yaml', imgsz=640, device=0)
"
```

Ожидаемая производительность (FP16, imgsz=640, YOLO11L):

| Jetson | FPS (FP16) | FPS (INT8) |
|--------|------------|------------|
| Orin Nano 8GB | 30-50 | 50-80 |
| Orin NX 16GB | 50-80 | 80-120 |
| AGX Orin 64GB | 100-150 | 150-200 |

---

## 5. Сборка ROS2 пакета

```bash
# Копируем пакет в workspace
cp -r ~/AntiUAV-Detector/ros2_src/uav_interceptor ~/aerial_nav_ws/src/

cd ~/aerial_nav_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select uav_interceptor

# Workaround: bin → lib (исполняемые файлы)
mkdir -p install/uav_interceptor/lib/uav_interceptor
cp install/uav_interceptor/bin/{vision_node,video_publisher,mavlink_bridge} \
   install/uav_interceptor/lib/uav_interceptor/

source install/setup.bash
```

---

## 6. Подключение камеры OpenIPC MC800S-V3

Камера OpenIPC MC800S-V3 (SSC338Q + IMX415) подключается по Ethernet и отдаёт RTSP-поток.

### Настройка камеры
1. Подключите камеру к Jetson по Ethernet (или через свитч)
2. Узнайте IP камеры (по умолчанию 192.168.1.10)
3. RTSP URL: `rtsp://192.168.1.10:554/live`

### Проверка потока
```bash
# Тест через ffplay
ffplay -fflags nobuffer -rtsp_transport tcp rtsp://192.168.1.10:554/live

# Тест через GStreamer (для Jetson)
gst-launch-1.0 rtspsrc location=rtsp://192.168.1.10:554/live \
    ! rtph265depay ! h265parse ! nvv4l2decoder \
    ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert \
    ! video/x-raw,format=BGR ! autovideosink
```

### Подключение IR-камеры (для dual-band)
Если используется отдельный тепловизор (например, FLIR Lepton):
- Подключите по USB
- Запустите отдельный `video_publisher` с `source_type:=usb` на топик `/camera/ir_image_raw`

---

## 7. Запуск полного пайплайна

### 7.1. С RTSP-камерой OpenIPC (основной режим)

```bash
source /opt/ros/humble/setup.bash
source ~/aerial_nav_ws/install/setup.bash
export PYTHONPATH="$HOME/antiuav-venv/lib/python3.8/site-packages:$PYTHONPATH"

ros2 launch uav_interceptor interceptor.launch.py \
    model_path:=/home/user/AntiUAV-Detector/weights/best.engine \
    source_type:=rtsp \
    rtsp_url:=rtsp://192.168.1.10:554/live \
    device:=/dev/ttyACM0 \
    simulation:=false \
    show_image:=true \
    use_dual_band:=true \
    fusion_mode:=fused
```

### 7.2. С USB-камерой

```bash
ros2 launch uav_interceptor interceptor.launch.py \
    model_path:=/home/user/AntiUAV-Detector/weights/best.engine \
    source_type:=usb \
    usb_device:=/dev/video0 \
    device:=/dev/ttyACM0 \
    simulation:=false \
    show_image:=true
```

### 7.3. Без автопилота (тест детекции)

```bash
ros2 launch uav_interceptor interceptor.launch.py \
    model_path:=/home/user/AntiUAV-Detector/weights/best.engine \
    source_type:=rtsp \
    rtsp_url:=rtsp://192.168.1.10:554/live \
    simulation:=true \
    show_image:=true
```

---

## 8. Подключение автопилота CUAV X7+ Pro

1. Подключите CUAV X7+ Pro к Jetson через USB (TELEM2 → USB-адаптер)
2. Проверьте устройство: `ls /dev/ttyACM*`
3. Установите параметры в Mission Planner / QGroundControl:
   - `SERIAL2_PROTOCOL` = 2 (MAVLink 2)
   - `SERIAL2_BAUD` = 921 (921600)
   - `ARMING_CHECK` = отключить GPS (если тестируете indoors)
4. Убедитесь, что автопилот переключается в GUIDED при получении команд

---

## 9. Важные замечания

1. **Engine привязан к конкретному SoC**: пересобирайте на каждой Jetson (Orin Nano ≠ Orin NX ≠ AGX Orin).
2. **CUDA 13.0 на PC** vs **CUDA 11.4/12.x на Jetson**: ONNX-файл не зависит от CUDA — переносится без проблем.
3. **FP16 точность**: для детекции дронов обычно достаточно (mAP50 падает на ~0.5-1%).
4. **PyTorch для Jetson**: НЕ устанавливайте через обычный `pip install torch` — только специальная сборка NVIDIA.
5. **Python 3.8 vs 3.10**: JetPack 5.x = Python 3.8, JetPack 6.x = Python 3.10. Пути в `PYTHONPATH` отличаются.
6. **Power Mode**: установите `sudo nvpmodel -m 0` (MAXN) для максимальной производительности.
7. **Jetson Clocks**: `sudo jetson_clocks` для фиксации максимальных частот.

---

## 10. Полный чеклист деплоя

### Подготовка (на PC)
- [x] Модель обучена (drone_v2-5, mAP50=0.897)
- [x] ONNX FP32 экспортирован (`best.onnx`, 97 MB)
- [x] ONNX FP16 экспортирован (`best.fp16.onnx`, 49 MB)
- [x] TensorRT engine для PC собран (`best.engine`, 50 MB)
- [x] `export_tensorrt.py` обновлён
- [x] `DEPLOY_JETSON.md` написан

### Перенос на Jetson
- [ ] `best.onnx` перенесён на Jetson
- [ ] Репозиторий клонирован на Jetson
- [ ] `data.yaml` перенесён (для валидации/INT8)

### Окружение на Jetson
- [ ] JetPack версии проверен (`cat /etc/nv_tegra_release`)
- [ ] CUDA в PATH (`nvcc --version`)
- [ ] PyTorch для Jetson установлен (не pip-версия!)
- [ ] Ultralytics установлен (`python -c "import ultralytics"`)
- [ ] TensorRT проверен (`python -c "import tensorrt"`)
- [ ] ROS2 установлен и sourced

### Engine на Jetson
- [ ] Engine пересобран из ONNX (`yolo export ... format=engine half=True`)
- [ ] Бенчмарк: FPS ≥ 30
- [ ] Тест детекции на тестовом изображении

### ROS2
- [ ] Пакет `uav_interceptor` собран (`colcon build`)
- [ ] Workaround bin→lib применён
- [ ] `vision_node` загружает `best.engine` без ошибок

### Камера
- [ ] OpenIPC MC800S-V3 подключена по Ethernet
- [ ] RTSP-поток проверен (`ffplay rtsp://...`)
- [ ] `video_publisher` публикует `/camera/image_raw`

### Автопилот
- [ ] CUAV X7+ Pro подключен (`/dev/ttyACM0`)
- [ ] MAVLink-соединение установлено
- [ ] GUIDED mode работает
- [ ] `DO_SET_SERVO` (STRIKE) тест (на земле, без БЧ)

### Интеграция
- [ ] Полный пайплайн: камера → детекция → трекинг → MAVLink
- [ ] Dual-band fusion (если есть IR-камера)
- [ ] STRIKE-цепочка: vision_node → mavlink_bridge → сервопривод
- [ ] Полевые испытания
