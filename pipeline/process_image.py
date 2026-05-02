"""
Real-Time Vehicle Detection & Counting Pipeline (Refactored)

Architecture:
Frame → Preprocess → Detect → Filter → Count → Visualize

Features:
- Device auto-detection (CPU/GPU)
- Integrated preprocessing pipeline
- Clean separation of logic
- Structured outputs
- Real-time webcam processing
"""

import cv2
import torch
from ultralytics import YOLO
from collections import Counter

from preprocess import ImagePreprocessor


class VehicleDetectionPipeline:
    def __init__(self, model_path="models/yolov8n.pt", conf_threshold=0.3):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        self.preprocessor = ImagePreprocessor(
            target_size=640,
            enhance=False,
            normalize=False
        )

        self.vehicle_classes = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }

        print(f"[INFO] Running on {self.device}")

    # ---------------------------
    # Core Processing Logic
    # ---------------------------
    def process_frame(self, frame):
        frame = self.preprocessor.process(frame)

        results = self.model(frame, device=self.device)

        boxes = results[0].boxes

        if boxes is None:
            return self._empty_result()

        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        vehicle_types = []

        for cls_id, conf in zip(class_ids, confidences):
            if cls_id in self.vehicle_classes and conf >= self.conf_threshold:
                vehicle_types.append(self.vehicle_classes[cls_id])

        return {
            "total": len(vehicle_types),
            "types": vehicle_types,
            "distribution": dict(Counter(vehicle_types))
        }

    def _empty_result(self):
        return {
            "total": 0,
            "types": [],
            "distribution": {}
        }

    # ---------------------------
    # Visualization
    # ---------------------------
    def draw_overlay(self, frame, result):
        cv2.putText(frame, f"Total Vehicles: {result['total']}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        y_offset = 80
        for v_type, count in result["distribution"].items():
            cv2.putText(frame, f"{v_type}: {count}",
                        (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_offset += 30

        return frame


# ---------------------------
# Webcam Runner
# ---------------------------
def run_webcam_pipeline(camera_index=0):
    pipeline = VehicleDetectionPipeline()

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    print("[INFO] Starting detection... Press ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.process_frame(frame)

        frame = pipeline.draw_overlay(frame, result)

        cv2.imshow("City Eye - Live", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()