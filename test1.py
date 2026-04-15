from ultralytics import YOLO
import cv2
import os

model = YOLO('yolov8n.pt')

# Target classes (COCO indices for car, motorcycle, bus, truck are 2, 3, 5, 7)
target_ids = [2, 3, 5, 7] 

# Open a video file or camera (0 for webcam)
results = model.track(source="D:\\download\\uni\\year4\\term2\\GP2\\files\\Screenshot 2026-04-15 185210.png", show=True, stream=True, conf=0.3)

os.system('saved_results.txt')  # Save results to a text file

for r in results:
    # Reset counts for the current frame
    current_frame_total = 0
    
    # Filter boxes by the target IDs
    boxes = r.boxes
    for box in boxes:
        if int(box.cls[0]) in target_ids:
            current_frame_total += 1
            
    print(f"Current Traffic Density: {current_frame_total}")
    
    # FUTURE LOGIC:
    # if current_frame_total > 15:
    #     signal.set_green()