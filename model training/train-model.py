from ultralytics import YOLO
import torch
torch.cuda.empty_cache()

# Load a model
model = YOLO('yolov8n.yaml')

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="..\data\data.yaml", epochs=100, imgsz=640, batch=16)
