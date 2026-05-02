"""
Real-Time Vehicle Detection & Counting Module (Webcam Pipeline)

This module provides a real-time computer vision pipeline for detecting
and counting vehicles from a live webcam stream using YOLOv8.

Features:
- Real-time object detection using YOLOv8 (Ultralytics)
- GPU acceleration support (CUDA) with warm-up optimization
- Continuous frame capture from webcam
- Vehicle filtering based on COCO dataset classes
- Live vehicle counting and type distribution
- On-screen visualization with dynamic overlays
- Efficient model loading (initialized once)

Vehicle Classes (COCO):
- 2 → Car
- 3 → Motorcycle
- 5 → Bus
- 7 → Truck

Pipeline Flow:
    webcam stream → frame capture → YOLO inference → 
    extract detections → filter vehicles → count & categorize → 
    overlay results → display live output

Functions:
- run_webcam_pipeline:
    Starts the real-time detection system using a webcam feed,
    processes each frame, and displays live vehicle statistics.

Notes:
- Requires a working webcam device.
- Assumes YOLOv8 model file is available at "models/yolov8n.pt".
- Uses GPU if available (set to CUDA); may require fallback handling for CPU-only systems.
- Press ESC to terminate the live stream.
"""


import cv2
from ultralytics import YOLO
from collections import Counter

# Load model ONCE (GPU)
model = YOLO("models/yolov8n.pt").to("cuda")

# Vehicle classes
vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


def run_webcam_pipeline(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    print("[INFO] Starting real-time detection... Press ESC to exit")

    # GPU warmup (IMPORTANT)
    dummy = cv2.imread("Data/images/bbb.jpg")
    if dummy is not None:
        model(dummy, device=0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break

        # Run YOLO
        results = model(frame, device=0, imgsz=640)

        boxes = results[0].boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)

        vehicle_types = []
        for cls_id in class_ids:
            if cls_id in vehicle_classes:
                vehicle_types.append(vehicle_classes[cls_id])

        vehicle_count = len(vehicle_types)
        vehicle_type_counts = Counter(vehicle_types)

        # Draw results
        cv2.putText(frame, f"Total Vehicles: {vehicle_count}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        y_offset = 80
        for v_type, count in vehicle_type_counts.items():
            cv2.putText(frame, f"{v_type}: {count}",
                        (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_offset += 30

        # Show frame
        cv2.imshow("City Eye - Live", frame)

        # Exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()