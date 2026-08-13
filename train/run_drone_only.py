from ultralytics import YOLO

def main():
    # Используем большую модель YOLO11l для лучшей точности
    model = YOLO('yolo11l.pt')
    
    model.train(
        data='/home/alex/AntiUAV-Detector/prepare_data/drone_only_dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        cache='disk',
        project='train/runs',
        name='drone_only_v1',
        patience=20,
        mixup=0.1,
        copy_paste=0.1,
        close_mosaic=10,
    )

if __name__ == '__main__':
    main()
