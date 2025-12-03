"""
Standalone detection script that can be run from any directory.
Expects the model file (best.pt) to be in the current working directory.
Continuously sends detection data to the server for retraining.
"""
from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO
import os
import cv2
import requests
import json
import time
import threading
from io import BytesIO
from PIL import Image
import numpy as np

# Configuration
SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000")
SEND_INTERVAL = float(os.environ.get("SEND_INTERVAL", "5.0"))  # Send data every 5 seconds
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.5"))  # Minimum confidence to send
SEND_DATA = os.environ.get("SEND_DATA", "true").lower() == "true"  # Enable/disable data sending
MODEL_CHECK_INTERVAL = float(os.environ.get("MODEL_CHECK_INTERVAL", "120.0"))  # Check for new models every 2 minutes (120 seconds)

class ModelUpdater:
    """Handles checking for and updating to newer models."""
    
    def __init__(self, server_url, model_path, current_model_name):
        self.server_url = server_url
        self.model_path = model_path
        self.current_model_name = current_model_name
        self.last_check_time = 0
        self.update_lock = threading.Lock()
        self.model_updated = False
        self.new_model_path = None
        self.new_model_name = None
    
    def check_for_update(self):
        """Check if a newer model is available."""
        current_time = time.time()
        if current_time - self.last_check_time < MODEL_CHECK_INTERVAL:
            return False, None, None
        
        with self.update_lock:
            if current_time - self.last_check_time < MODEL_CHECK_INTERVAL:
                return False, None, None
            
            self.last_check_time = current_time
        
        try:
            response = requests.get(
                f"{self.server_url}/check_model_update",
                params={"current_model": self.current_model_name},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("is_newer", False):
                    return True, result.get("latest_model"), result.get("model_path")
            
            return False, None, None
            
        except Exception as e:
            print(f"[Model Update] Error checking for updates: {e}")
            return False, None, None
    
    def download_new_model(self, model_name, model_path_url):
        """Download the new model and set update flag."""
        try:
            print(f"[Model Update] New model available: {model_name}")
            print(f"[Model Update] Downloading new model...")
            
            response = requests.get(
                f"{self.server_url}/download_latest_model",
                timeout=30
            )
            
            if response.status_code == 200:
                # Save to temporary file with .pt extension (YOLO requires .pt)
                model_dir = os.path.dirname(self.model_path)
                temp_path = os.path.join(model_dir, "best_new.pt")
                with open(temp_path, "wb") as f:
                    f.write(response.content)
                
                # Get model name from header
                new_model_name = response.headers.get("X-Model-Name", model_name)
                
                # Set update flag
                with self.update_lock:
                    self.model_updated = True
                    self.new_model_path = temp_path
                    self.new_model_name = new_model_name
                
                print(f"[Model Update] New model downloaded: {new_model_name}")
                return True
            else:
                print(f"[Model Update] Failed to download model: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[Model Update] Error downloading model: {e}")
            return False

class DataSender:
    """Handles sending detection data to the server."""
    
    def __init__(self, server_url):
        self.server_url = server_url
        self.last_send_time = 0
        self.send_lock = threading.Lock()
        
    def send_detection_data(self, frame, detections_list):
        """Send frame and detections to server."""
        if not SEND_DATA:
            return
        
        current_time = time.time()
        if current_time - self.last_send_time < SEND_INTERVAL:
            return
        
        with self.send_lock:
            if current_time - self.last_send_time < SEND_INTERVAL:
                return
            
            try:
                # Convert frame to JPEG
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                img_bytes = BytesIO()
                pil_image.save(img_bytes, format='JPEG')
                img_bytes.seek(0)
                
                # Prepare detections in format: [class_id, x1, y1, x2, y2, confidence]
                detections_json = json.dumps(detections_list)
                
                # Send to server
                files = {'image': ('frame.jpg', img_bytes, 'image/jpeg')}
                data = {'detections': detections_json}
                
                response = requests.post(
                    f"{self.server_url}/upload_detection_data",
                    files=files,
                    data=data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"[Data Sent] Samples collected: {result.get('samples_collected', 0)}")
                    if result.get('should_retrain'):
                        print(f"[Server] Retraining started! PID: {result.get('training_pid')}")
                else:
                    print(f"[Error] Failed to send data: {response.status_code}")
                
                self.last_send_time = current_time
                
            except Exception as e:
                print(f"[Error] Failed to send detection data: {e}")

def main():
    # Look for model in current directory
    model_path = os.path.join(os.getcwd(), "best.pt")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure best.pt is in the current directory.")
        exit(1)
    
    # Get model name from environment variable if available
    model_name = os.environ.get('MODEL_NAME', 'best.pt')
    
    print(f"Loading model: {model_name}")
    print(f"Model file: {model_path}")
    personDetector = YOLO(model_path)
    
    # Initialize data sender and model updater
    data_sender = DataSender(SERVER_URL)
    model_updater = ModelUpdater(SERVER_URL, model_path, model_name)
    
    if SEND_DATA:
        print(f"Data collection enabled: Sending to {SERVER_URL} every {SEND_INTERVAL}s")
        print(f"Minimum confidence: {MIN_CONFIDENCE}")
    else:
        print("Data collection disabled")

    # Open default camera (0 = default, 1 = external)
    cap = cv2.VideoCapture(0)

    # read video capture
    ret, frame = cap.read() 
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    print("Starting detection. Press 'x' to exit.")
    print(f"Model auto-update enabled: Checking every {MODEL_CHECK_INTERVAL}s")
    annotated_img = None
    frame_count = 0
    last_model_check = time.time()
    
    while ret:
        ret, frame = cap.read()

        if ret:
            # Check for model updates periodically (non-blocking, time-based)
            current_time = time.time()
            if current_time - last_model_check >= MODEL_CHECK_INTERVAL:
                last_model_check = current_time
                update_available, new_model_name, new_model_path = model_updater.check_for_update()
                if update_available:
                    # Download new model in background
                    threading.Thread(
                        target=model_updater.download_new_model,
                        args=(new_model_name, new_model_path),
                        daemon=True
                    ).start()
            
            # Check if model update is ready
            if model_updater.model_updated and model_updater.new_model_path:
                try:
                    print(f"[Model Update] Reloading model: {model_updater.new_model_name}")
                    
                    # Load new model from temp file (which has .pt extension)
                    new_detector = YOLO(model_updater.new_model_path)
                    
                    # Replace old model file atomically
                    if os.path.exists(model_updater.model_path):
                        # On Windows, we need to remove old file first, then rename
                        try:
                            os.remove(model_updater.model_path)
                        except PermissionError:
                            # If file is in use, try again next frame
                            print(f"[Model Update] Model file in use, will retry...")
                            continue
                    
                    os.rename(model_updater.new_model_path, model_updater.model_path)
                    
                    # Swap models
                    personDetector = new_detector
                    model_name = model_updater.new_model_name
                    model_updater.current_model_name = model_updater.new_model_name
                    
                    # Clear update flags
                    with model_updater.update_lock:
                        model_updater.model_updated = False
                        model_updater.new_model_path = None
                        model_updater.new_model_name = None
                    
                    print(f"[Model Update] Successfully updated to model: {model_name}")
                except Exception as e:
                    print(f"[Model Update] Error reloading model: {e}")
                    import traceback
                    traceback.print_exc()
                    # Clean up failed update
                    if os.path.exists(model_updater.new_model_path):
                        try:
                            os.remove(model_updater.new_model_path)
                        except:
                            pass
                    with model_updater.update_lock:
                        model_updater.model_updated = False
                        model_updater.new_model_path = None
                        model_updater.new_model_name = None
            
            # Detect people
            detections = personDetector(frame)
            
            # Collect detections for sending
            detections_list = []
            
            # Process detections
            for detection in detections:
                annotator = Annotator(detection.plot())
                for box in detection.boxes:
                    confidence = box.conf[0].item()
                    
                    # Only include detections above minimum confidence
                    if confidence >= MIN_CONFIDENCE:
                        coordinates = box.xyxy[0].tolist()
                        class_id = int(box.cls[0].item())
                        label = f'{personDetector.names[class_id]} {confidence:.2f}'
                        annotator.box_label(coordinates, label)
                        
                        # Add to list for sending: [class_id, x1, y1, x2, y2, confidence]
                        detections_list.append([
                            class_id,
                            float(coordinates[0]),
                            float(coordinates[1]),
                            float(coordinates[2]),
                            float(coordinates[3]),
                            float(confidence)
                        ])

                annotated_img = annotator.result()
            
            # Send data to server if we have detections
            if detections_list and SEND_DATA:
                # Send in background thread to avoid blocking
                threading.Thread(
                    target=data_sender.send_detection_data,
                    args=(frame.copy(), detections_list),
                    daemon=True
                ).start()
            
            frame_count += 1

        # Display resulting frame  
        if annotated_img is not None:
            cv2.imshow('Webcam - Press x to Exit', annotated_img)

        # Press 'x' to exit
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

if __name__ == "__main__":
    main()

