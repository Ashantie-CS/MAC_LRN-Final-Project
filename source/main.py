from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO

import os
import re
import numpy as np
import cv2


def main():
    # Initialize model directory
    dir_path = "../model training/runs/detect"
    all_trainings = os.listdir(dir_path)
    latest_training = 0

    # Regex pattern for matching whole number
    pattern = r"[-+]?\d*\.?\d+"

    # Get latest training
    for training in all_trainings:
        match = re.search(pattern, training)
        if match:
            version_number = int(match.group())
            if latest_training < version_number:
                latest_training = version_number
        
    # Load latest model
    personDetector = YOLO(f'../model training/runs/detect/train{latest_training}/weights/best.pt')

    # Open default camera (0 = default, 1 = external)
    cap = cv2.VideoCapture(0)

    # read video capture
    ret, frame = cap.read() 
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while ret:
        ret, frame = cap.read()

        if ret:
            # Detect people
            detections = personDetector(frame)

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
