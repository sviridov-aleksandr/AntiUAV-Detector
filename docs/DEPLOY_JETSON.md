# Деплой на NVIDIA Jetson Orin

## Обзор

Модель обучена на PC (RTX 5080, TensorRT 11.2) и экспортирована в:
- `best.onnx` (FP32, 101 MB) — универсальный, переносится на Jetson
- `best.fp16.onnx` (FP16, 51 MB)
- `best.engine` (FP16, 50 MB) — **привязан к PC**, на Jetson НЕ работает

**На Jetson engine нужно пересобрать из ONNX** (TensorRT на Jetson — 8.x, другая архитектура).

---

## 1. Перенос файлов на Jetson

```bash
# С PC на Jetson (по SSH)
scp /home/alex/AntiUAV-Detector/runs/detect/train/runs/drone_v2/weights/best.onnx \
    user@jetson-ip:/home/user/AntiUAV-Detector/weights/

# Или через USB-накопитель
```

Также перенесите:
- `best.fp16.onnx` (опционально, для сравнения)
- `data.yaml` (для валидации)
- ROS2 пакет `uav_interceptor` (или склонируйте репозиторий)

---

## 2. Установка окружения на Jetson

```bash
# JetPack 5.x (Ubuntu 20.04) — Python 3.8
sudo apt update && sudo apt install -y python3-pip python3-venv

# CUDA (уже в JetPack)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Виртуальное окружение
python3 -m venv ~/antiuav-venv
source ~/antiuav-venv/bin/activate

# PyTorch для Jetson (специальная сборка, НЕ с pip!)
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/
pip install torch==2.1.0a0+41361538.nv23.06 \
    torchvision==0.16.0a0+bfce5f2.nv23.06 \
    -f https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/

# Ultralytics (YOLO11)
pip install ultralytics

# TensorRT (уже в JetPack, проверьте)
python -c "import tensorrt; print(tensorrt.__version__)"  # должно быть 8.x
```

---

## 3. Конвертация ONNX → TensorRT engine (FP16)

### Вариант A: через Ultralytics (рекомендуется)

```bash
source ~/antiuav-venv/bin/activate
cd ~/AntiUAV-Detector

yolo export model=weights/best.onnx format=engine half=True imgsz=640
# Результат: weights/best.engine (FP16, ~25 MB на Jetson)
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

---

## 4. Проверка engine на Jetson

```bash
source ~/antiuav-venv/bin/activate
python -c "
from ultralytics import YOLO
import time

model = YOLO('weights/best.engine')

# Бенчмарк
start = time.time()
for _ in range(50):
    model.predict(source='test.jpg', imgsz=640, device=0, verbose=False)
dt = (time.time() - start) / 50
print(f'Инференс: {1/dt:.1f} FPS')

# Валидация
model.val(data='data.yaml', imgsz=640, device=0)
"
```

Ожидаемая производительность (FP16, imgsz=640):
- **Jetson Orin Nano 8GB**: ~30-60 FPS (YOLO11l)
- **Jetson Orin NX 16GB**: ~60-100 FPS
- **Jetson AGX Orin 64GB**: ~100-150 FPS

---

## 5. Интеграция с ROS2

### Установка ROS2 на Jetson (если ещё нет)

```bash
# JetPack 5.x → ROS2 Humble
sudo apt install ros-humble-ros-base
source /opt/ros/humble/setup.bash
```

### Сборка пакета uav_interceptor

```bash
cd ~/aerial_nav_ws
colcon build --packages-select uav_interceptor
source install/setup.bash
```

### Запуск с TensorRT engine

```bash
export PYTHONPATH="$HOME/antiuav-venv/lib/python3.8/site-packages:$PYTHONPATH"

ros2 launch uav_interceptor interceptor.launch.py \
    model_path:=/home/user/AntiUAV-Detector/weights/best.engine \
    device:=/dev/ttyACM0 \
    simulation:=false \
    show_image:=false
```

---

## 6. Важные замечания

1. **Engine привязан к Jetson**: пересобирайте на каждой конкретной Jetson (разные SoC → разные engine).
2. **JetPack 6.x** (Ubuntu 22.04, TensorRT 8.6+): команды аналогичны, PyTorch для JetPack 6.x — отдельная сборка.
3. **CUDA 13.0 на PC** vs **CUDA 11.4/12.x на Jetson**: ONNX-файл не зависит от CUDA — переносится без проблем.
4. **FP16 точность**: для детекции дронов обычно достаточно (mAP50 падает на ~0.5-1%).
5. **INT8 квантизация** (опционально, +2x скорость): требует калибровочный датасет:
   ```bash
   yolo export model=weights/best.onnx format=engine half=False int8=True \
       data=data.yaml imgsz=640
   ```
6. **Камера на Jetson**: используйте `video_publisher` с `device:=/dev/video0` (GStreamer) вместо тестового видео.

---

## 7. Полный чеклист деплоя

- [ ] `best.onnx` перенесён на Jetson
- [ ] PyTorch для Jetson установлен (не pip-версия!)
- [ ] Ultralytics установлен
- [ ] TensorRT 8.x проверен (`python -c "import tensorrt"`)
- [ ] Engine пересобран на Jetson (`yolo export ... format=engine half=True`)
- [ ] Бенчмарк: FPS ≥ 30
- [ ] ROS2 пакет собран
- [ ] `vision_node` загружает `best.engine` без ошибок
- [ ] Полный пайплайн: камера → детекция → MAVLink