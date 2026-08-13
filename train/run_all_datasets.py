from ultralytics import YOLO

def main():
    model = YOLO('yolo11l.pt')
    
    model.train(
        data='/home/alex/AntiUAV-Detector/prepare_data/all_datasets_v1/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        cache='disk',
        project='train/runs',
        name='all_datasets_v1',
        patience=20,
        mixup=0.1,
        copy_paste=0.1,
        close_mosaic=10,
    )

if __name__ == '__main__':
    main()
