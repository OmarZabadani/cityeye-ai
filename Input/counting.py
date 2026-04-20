"""
Vehicle Detection & Counting Module

This module uses a pretrained YOLOv8 model to detect objects in an image
and estimate the number of vehicles present.

Main functionality:
1. Loads an image from disk.
2. Runs object detection using a YOLOv8 model (yolov8n).
3. Extracts detected object classes from the model output.
4. Counts vehicles based on specific class IDs.

Vehicle classes (COCO dataset):
- 2 → Car
- 3 → Motorcycle
- 5 → Bus
- 7 → Truck

Pipeline flow:
    load image → run YOLO → extract detections → filter vehicle classes → count vehicles

Functions:
- process_number_of_cars: Returns the number of detected vehicles in the input image.

Notes:
- Uses Ultralytics YOLOv8 for detection.
- Assumes the model file (yolov8n.pt) is available locally.
- Returns None if the image cannot be loaded.
"""
import os
from xml.parsers.expat import model
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def process_number_of_cars(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    # Run the YOLO model on the image
    results = model(img)
    # Extract bounding boxes and class IDs
    boxes = results[0].boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)
    # Count the number of vehicles (class IDs 2, 3, 5, 7)
    vehicle_count = sum(1 for cls_id in class_ids if cls_id in [2, 3, 5, 7])
    return vehicle_count
