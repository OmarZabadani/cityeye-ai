"""
Vehicle Detection & Counting Module (Enhanced)

Features:
- Automatic device selection (CPU/GPU)
- Confidence threshold filtering
- Structured detection output
- Vehicle counting + type distribution
"""

from collections import Counter
from ultralytics import YOLO
import cv2
import torch


# ---------------------------
# Model Loader (Singleton Style)
# ---------------------------
class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.25):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        print(f"[INFO] Using device: {self.device}")

        # COCO vehicle classes
        self.vehicle_classes = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }

    def detect(self, image):
        results = self.model(image, device=self.device)
        return results[0]

    def process_image(self, image_path):
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        result = self.detect(img)

        boxes = result.boxes

        if boxes is None:
            return self._empty_response()

        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        bboxes = boxes.xyxy.cpu().numpy()

        vehicle_types = []
        detections = []

        for cls_id, conf, bbox in zip(class_ids, confidences, bboxes):
            if cls_id in self.vehicle_classes and conf >= self.conf_threshold:
                label = self.vehicle_classes[cls_id]

                vehicle_types.append(label)

                detections.append({
                    "type": label,
                    "confidence": float(conf),
                    "bbox": bbox.tolist()  # [x1, y1, x2, y2]
                })

        return {
            "total_vehicles": len(vehicle_types),
            "vehicle_types": vehicle_types,
            "type_distribution": dict(Counter(vehicle_types)),
            "detections": detections
        }

    def _empty_response(self):
        return {
            "total_vehicles": 0,
            "vehicle_types": [],
            "type_distribution": {},
            "detections": []
        }