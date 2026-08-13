#!/usr/bin/env python3
"""
Скрипт для тестирования трекинга с предварительной обрезкой OSD.
Обрезает края кадра, где обычно находится интерфейс.
"""

from ultralytics import YOLO
import cv2
import sys


def crop_osd(frame, margin=0.1):
    """
    Обрезает края кадра для удаления OSD.
    margin: доля кадра, которую нужно обрезать (0.1 = 10% со всех сторон).
    """
    h, w = frame.shape[:2]
    y1, y2 = int(h * margin), int(h * (1 - margin))
    x1, x2 = int(w * margin), int(w * (1 - margin))
    return frame[y1:y2, x1:x2]


def track_drone(video_path, model_path, output_path='tracking_cropped_output.mp4', margin=0.1):
    print(f"Загрузка модели: {model_path}")
    model = YOLO(model_path)
    
    print(f"Загрузка видео: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Создаём видеопейрайтер для сохранения результата
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Начинаем трекинг с обрезкой OSD (margin={margin})...")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. Обрезаем OSD
        cropped_frame = crop_osd(frame, margin)
        
        # 2. Запускаем трекинг на обрезанном кадре
        results = model.track(cropped_frame, persist=True, conf=0.4, verbose=False)
        
        # 3. Рисуем результаты на ОРИГИНАЛЬНОМ кадре (с учетом смещения)
        # Нам нужно скорректировать координаты рамок, так как кадр был обрезан
        h_crop, w_crop = cropped_frame.shape[:2]
        y_offset = int(height * margin)
        x_offset = int(width * margin)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Получаем координаты рамки (относительно обрезанного кадра)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Добавляем смещение от обрезки
                    x1 += x_offset
                    y1 += y_offset
                    x2 += x_offset
                    y2 += y_offset
                    
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    track_id = int(box.id[0]) if box.id is not None else 0
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_names = model.names
                    class_name = class_names.get(cls, f"class_{cls}")
                    
                    color = (0, 255, 0)
                    if class_name == 'drone': color = (0, 255, 0)
                    elif class_name == 'helicopter': color = (255, 25, 0)
                    elif class_name == 'airplane': color = (0, 255)
                    elif class_name == 'bird': color = (255, 0)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{track_id} {class_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        out.write(frame)
        frame_count += 1
        
        cv2.imshow('Tracking (Cropped)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Трекинг завершён! Обработано кадров: {frame_count}")
    print(f"Результат сохранён в: {output_path}")


def main():
    if len(sys.argv) < 3:
        print("Использование:")
        print("  python test_tracking_cropped.py <путь_к_модели> <путь_к_видео> [margin]")
        print("  margin: доля обрезки краев (по умолчанию 0.1 = 10%)")
        return
    
    model_path = sys.argv[1]
    video_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else 'tracking_cropped_output.mp4'
    margin = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
    
    track_drone(video_path, model_path, output_path, margin)


if __name__ == '__main__':
    main()
