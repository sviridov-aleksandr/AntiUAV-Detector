from ultralytics import YOLO


def main():
    # Продолжаем с лучших весов предыдущей тренировки
    model = YOLO('runs/detect/train/runs/yolov11m_drone_v3-4/weights/best.pt')

    model.train(
        data='/home/alex/AntiUAV-Detector/prepare_data/output/drone_dataset/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        patience=30,
        close_mosaic=10,
        project='train/runs',
        name='yolov11m_drone_v4',
        cache='ram',
    )


if __name__ == '__main__':
    main()
