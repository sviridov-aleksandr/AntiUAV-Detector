from ultralytics import YOLO


def main():
    model = YOLO('yolo11m.pt')

    model.train(
        data='/home/alex/AntiUAV-Detector/prepare_data/drone_detection_thesis/data.yaml',
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
        name='drone_detection_thesis',
        cache='disk',
    )


if __name__ == '__main__':
    main()
