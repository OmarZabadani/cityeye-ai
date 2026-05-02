"""
Main Entry Point - Smart Traffic System

This file connects:
- Preprocessing
- YOLO detection
- Traffic simulation logic

Run:
    python main.py
"""

import cv2

from preprocess import ImagePreprocessor
from ultralytics import YOLO

from traffic_logic import TrafficEngine


# =========================================================
# System Controller
# =========================================================
class SmartTrafficSystem:
    def __init__(self):

        # Device setup
        self.device = "cuda"

        # Models / modules
        self.preprocessor = ImagePreprocessor(
            target_size=640,
            enhance=False,
            normalize=False
        )

        self.model = YOLO("models/yolov8n.pt")

        self.engine = TrafficEngine()

        # COCO mapping
        self.vehicle_map = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }

    # -----------------------------------------
    # Step 1: YOLO detection
    # -----------------------------------------
    def detect_vehicles(self, frame):
        results = self.model(frame, device=self.device)

        boxes = results[0].boxes

        if boxes is None:
            return []

        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        detections = []

        for cls_id, conf in zip(class_ids, confidences):
            if cls_id in self.vehicle_map:
                detections.append({
                    "type": self.vehicle_map[cls_id],
                    "confidence": float(conf)
                })

        return detections

    # -----------------------------------------
    # Step 2: Convert lane (simple version)
    # -----------------------------------------
    def assign_lane(self, detections):
        """
        Temporary logic:
        All vehicles go to lane A
        (Later: replace with bbox-based lane detection)
        """

        return "A", detections

    # -----------------------------------------
    # Step 3: Run full pipeline on frame
    # -----------------------------------------
    def process_frame(self, frame):

        # 1. Preprocess
        frame = self.preprocessor.process(frame)

        # 2. Detect vehicles
        detections = self.detect_vehicles(frame)

        # 3. Assign lane
        lane, detections = self.assign_lane(detections)

        # 4. Feed traffic engine
        self.engine.ingest_detections(detections, lane)

        # 5. Run simulation step
        decision = self.engine.step()

        return decision, frame

    # -----------------------------------------
    # Step 4: Run webcam system
    # -----------------------------------------
    def run(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("[ERROR] Camera not found")
            return

        print("[INFO] Smart Traffic System Running... Press ESC to exit")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            decision, frame = self.process_frame(frame)

            # Display result
            text = f"{decision['action']} | {decision['vehicle']} | Lane {decision['lane']}"
            cv2.putText(frame, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

            cv2.imshow("Smart Traffic System", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


# =========================================================
# Entry Point
# =========================================================
if __name__ == "__main__":
    system = SmartTrafficSystem()
    system.run()