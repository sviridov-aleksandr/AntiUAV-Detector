#!/usr/bin/env python3
"""
Экспорт модели YOLO11 в ONNX (FP32) и TensorRT engine (FP16).

Стратегия:
1. ONNX FP32 — универсальный формат, переносится на Jetson Orin.
2. TensorRT engine FP16 — для текущей PC (RTX 5080, TensorRT 11.2).
   На Jetson engine нужно пересобирать из ONNX (TensorRT 8.x).

Использование:
    python export_tensorrt.py [путь_к_model.pt]
"""

import sys
from pathlib import Path
from ultralytics import YOLO

MODEL = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/alex/AntiUAV-Detector/runs/detect/train/runs/drone_v2-5/weights/best.pt"


def main():
    model_path = Path(MODEL)
    if not model_path.exists():
        print(f"Модель не найдена: {model_path}")
        sys.exit(1)

    print(f"Экспорт модели: {model_path}")
    model = YOLO(str(model_path))

    # 1. ONNX FP32 (универсальный, для Jetson)
    # Экспорт на CPU — не мешает обучению на GPU
    print("\n[1/2] Экспорт в ONNX FP32 (CPU)...")
    onnx_path = model.export(
        format='onnx',
        imgsz=640,
        opset=17,
        simplify=True,
        dynamic=False,
        device='cpu',
    )
    print(f"ONNX сохранён: {onnx_path}")

    # 2. TensorRT engine FP16 (для текущей PC)
    # Требует GPU — запускать только когда обучение закончено
    print("\n[2/2] Экспорт в TensorRT FP16 (GPU)...")
    engine_path = model.export(
        format='engine',
        imgsz=640,
        half=True,
        dynamic=False,
        workspace=4,  # GB
        device=0,
    )
    print(f"Engine сохранён: {engine_path}")

    print("\nГотово!")
    print(f"  ONNX:   {onnx_path}")
    print(f"  Engine: {engine_path}")


if __name__ == '__main__':
    main()
