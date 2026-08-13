#!/usr/bin/env python3
"""
Скрипт для тестирования трекинга (ByteTrack) на текущей модели YOLOv11.
Показывает, насколько стабильно модель держит цель в кадре.
"""

from ultralytics import YOLO
import cv2
import sys


def track_drone(video_path, model_path, output_path='tracking_output.mp4'):
    """
    Запускает трекинг на видео с использованием ByteTrack.
    
    Args:
        video_path: Путь к видеофайлу
        model_path: Путь к весам модели
        output_path: Путь для сохранения результата
    """
    print(f"Загрузка модели: {model_path}")
    model = YOLO(model_path)
    
    print(f"Загрузка видео: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {video_path}")
        return
    
    # Получаем параметры видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Создаём видеопейрайтер для сохранения результата
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Начинаем трекинг... FPS: {fps}, Размер: {width}x{height}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Запускаем трекинг (не просто детекцию!)
        results = model.track(frame, persist=True, conf=0.6, verbose=False)
        
        # Рисуем результаты
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Получаем координаты рамки
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Получаем ID трека
                    track_id = int(box.id[0]) if box.id is not None else 0
                    
                    # Получаем уверенность
                    conf = float(box.conf[0])
                    
                    # Получаем класс
                    cls = int(box.cls[0])
                    class_names = model.names
                    class_name = class_names.get(cls, f"class_{cls}")
                    
                    # Рисуем рамку
                    color = (0, 255, 0)  # Зелёный по умолчанию
                    if class_name == 'drone':
                        color = (0, 255, 0)  # Зелёный для дронов
                    elif class_name == 'helicopter':
                        color = (255, 255, 0)  # Жёлтый для вертолётов
                    elif class_name == 'airplane':
                        color = (0, 255, 255)  # Циан для самолётов
                    elif class_name == 'bird':
                        color = (255, 0, 0)  # Красный для птиц
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Рисуем ID и класс
                    label = f"ID:{track_id} {class_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Сохраняем кадр
        out.write(frame)
        frame_count += 1
        
        # Показываем кадр в реальном времени
        cv2.imshow('Tracking', frame)
        
        # Нажатие 'q' для выхода
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Освобождаем ресурсы
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Трекинг завершён! Обработано кадров: {frame_count}")
    print(f"Результат сохранён в: {output_path}")


def main():
    if len(sys.argv) < 3:
        print("Использование:")
        print("  python test_tracking.py <путь_к_модели> <путь_к_видео>")
        print()
        print("Пример:")
        print("  python test_tracking.py /path/to/best.pt /path/to/video.mp4")
        return
    
    model_path = sys.argv[1]
    video_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else 'tracking_output.mp4'
    
    track_drone(video_path, model_path, output_path)


if __name__ == '__main__':
    main()
