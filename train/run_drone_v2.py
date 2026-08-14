from ultralytics import YOLO

def main():
    model = YOLO('/home/alex/AntiUAV-Detector/runs/detect/train/runs/drone_v2-4/weights/last.pt')

    model.train(
        data='/home/alex/AntiUAV-Detector/prepare_data/drone_v2/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        cache='disk',
        project='train/runs',
        name='drone_v2',
        patience=20,
        lr0=0.005,
        warmup_epochs=1.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=5,
    )

if __name__ == '__main__':
    main()
