from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO

import os
import re
import numpy as np
import cv2


def main():
    # Initialize model directory 
    dir_path = "./edge_server/models"
    
    # Check if directory exists
    if not os.path.exists(dir_path):
        print(f"Error: Model directory not found at {dir_path}")
        print("Please ensure the models directory exists and contains training runs.")
        exit(1)
    
    all_trainings = os.listdir(dir_path)
    latest_training = None
    latest_training_name = None

    # Regex pattern for matching numbers in directory names (e.g., "1_train" -> 1, "1_train2" -> 2)
    pattern = r"train(\d+)?$"

    # Get latest training
    for training in all_trainings:
        # Skip non-directory files
        training_path = os.path.join(dir_path, training)
        if not os.path.isdir(training_path):
            continue
            
        match = re.search(pattern, training)
        if match:
            # If there's a number after "train", use it; otherwise treat as version 1
            version_str = match.group(1)
            version_number = int(version_str) if version_str else 1
            if latest_training is None or latest_training < version_number:
                latest_training = version_number
                latest_training_name = training
    
    # Check if we found any training
    if latest_training_name is None:
        print("Error: No valid training runs found in models directory.")
        exit(1)
    
    # Load latest model
    model_path = os.path.join(dir_path, latest_training_name, "weights", "best.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        exit(1)
    
    personDetector = YOLO(model_path)

    # Open default camera (0 = default, 1 = external)
    cap = cv2.VideoCapture(0)

    # read video capture
    ret, frame = cap.read() 
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    annotated_img = None
    
    while ret:
        ret, frame = cap.read()

        if ret:
            # Detect people
            detections = personDetector(frame)
            
            # Process detections
            for detection in detections:
                annotator = Annotator(detection.plot())
                for box in detection.boxes:
                    coordinates = box.xyxy[0].tolist()
                    class_id = int(box.cls[0].item())
                    label = f'{personDetector.names[class_id]} {box.conf[0].item():.2f}'
                    annotator.box_label(coordinates, label)

                annotated_img = annotator.result()

        # Display resulting frame  
        if annotated_img is not None:
            cv2.imshow('Webcam - Press x to Exit', annotated_img)

        # Press 'x' to exit
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

if __name__ == "__main__":
    main()