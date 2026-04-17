import os
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


image_path = r"D:\download\uni\year4\term2\GP2\Data\images\aaa.png"

count = process_number_of_cars(image_path)
if count is not None:
    print(f"Number of vehicles detected: {count}")