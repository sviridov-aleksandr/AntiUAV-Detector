#!/usr/bin/env python3
"""
Визуальный сервопривод (Visual Servoing) для удержания цели в кадре.
Использует PID-регулятор и конечный автомат (State Machine).
"""

from ultralytics import YOLO
import cv2
import numpy as np
import enum
import os


class State(enum.Enum):
    SEARCH = "SEARCH"
    TRACK = "TRACK"


class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0
        self.deadzone = 2.0

    def compute(self, measured_value):
        error = self.setpoint - measured_value
        
        if abs(error) < self.deadzone:
            return 0.0

        self.integral += error
        derivative = error - self.prev_error
        
        output = (self.kp * error + 
                  self.ki * self.integral + 
                  self.kd * derivative)
        
        self.prev_error = error
        return output


class VisualServoingSystem:
    def __init__(self, model_path, video_path):
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.state = State.SEARCH
        self.lost_counter = 0
        self.max_lost_frames = 10
        
        self.search_angle = 0.0
        self.search_speed = 0.05
        
        self.pid_pan = PIDController(kp=0.05, ki=0.001, kd=0.01)
        self.pid_tilt = PIDController(kp=0.05, ki=0.001, kd=0.01)
        
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter('servoing_output.mp4', fourcc, self.fps, (self.width, self.height))

    def run_search_pattern(self):
        self.search_angle += self.search_speed
        if self.search_angle > 1.0:
            self.search_angle = -1.0
        return self.search_angle

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return False

        h, w = frame.shape[:2]
        center_x, center_y = w / 2, h / 2
        
        results = self.model.track(frame, persist=True, verbose=False)
        drone_detected = False
        target_x, target_y = 0, 0
        
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls == 0: 
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    target_x = (x1 + x2) / 2
                    target_y = (y1 + y2) / 2
                    drone_detected = True
                    
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, "TARGET LOCKED", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break

        if drone_detected:
            self.lost_counter = 0
            self.state = State.TRACK
            
            error_x = target_x - center_x
            error_y = target_y - center_y
            
            cmd_pan = self.pid_pan.compute(target_x)
            cmd_tilt = self.pid_tilt.compute(target_y)
            
            cv2.line(frame, (int(center_x), int(center_y)), (int(target_x), int(target_y)), (255, 0, 0), 2)
            cv2.putText(frame, f"CMD: {cmd_pan:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        else:
            self.lost_counter += 1
            if self.lost_counter > self.max_lost_frames:
                self.state = State.SEARCH
            
            if self.state == State.SEARCH:
                search_offset = self.run_search_pattern() * (w / 4)
                cv2.putText(frame, "SEARCHING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.circle(frame, (int(center_x + search_offset), int(center_y)), 10, (0, 0, 255), -1)
            else:
                cv2.putText(frame, "TARGET LOST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        self.out.write(frame)
        cv2.imshow('Visual Servoing', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
            
        return True

    def run(self):
        print("Запуск визуального сервопривода...")
        while self.process_frame():
            pass
        
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        print("Завершено.")


def main():
    model_path = '/home/alex/AntiUAV-Detector/runs/detect/train/runs/drone_only_v1/weights/best.pt'
    video_path = '/home/alex/AntiUAV-Detector/video-FPV/Video/v2.mp4'
    
    if not os.path.exists(model_path):
        print(f"Модель {model_path} еще не обучена. Используем предыдущую.")
        model_path = '/home/alex/AntiUAV-Detector/runs/detect/train/runs/combined_dataset_v1/weights/best.pt'

    system = VisualServoingSystem(model_path, video_path)
    system.run()

if __name__ == '__main__':
    main()