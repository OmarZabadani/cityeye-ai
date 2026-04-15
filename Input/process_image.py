import os
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
def process_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    # Run the YOLO model on the image
    results = model(img)
    # Extract bounding boxes and class IDs
    boxes = results[0].boxes
    class_ids = boxes.cls.jpu().numpy().astype(int)
    class_ids = boxes.cls.cpu().numpy().astype(int)