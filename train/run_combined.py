from ultralytics import YOLO

def main():
    model = YOLO('/home/alex/AntiUAV-Detector/runs/detect/train/runs/combined_dataset_v1/weights/best.pt')
    model.train(
        data='/home/alex/AntiUAV-Detector/prepare_data/combined_dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        cache='disk',
        project='train/runs',
        name='combined_dataset_v2',
        resume=True,
        patience=15,
        mixup=0.1,
        copy_paste=0.1,
    )

if __name__ == '__main__':
    main()
