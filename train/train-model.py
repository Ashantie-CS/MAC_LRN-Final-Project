from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    results = model.train(
        data="../data/data.yaml",
        epochs=60,
        imgsz=416,
        batch=16,
        device=0,
    )

if __name__ == "__main__":
    main()
