# server/main.py
import os, json, time, sqlite3, subprocess, re, sys, platform, logging, traceback
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
MODELS_DIR = os.path.join(ROOT, "models")
META_PATH = os.path.join(MODELS_DIR, "meta.json")
DB_PATH = os.path.join(ROOT, "server.db")
TRAIN_LOCK_FILE = os.path.join(ROOT, ".training_lock")
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "dataset", "train", "images")
TRAIN_LABELS_DIR = os.path.join(DATA_DIR, "dataset", "train", "labels")
VALID_IMAGES_DIR = os.path.join(DATA_DIR, "dataset", "valid", "images")
VALID_LABELS_DIR = os.path.join(DATA_DIR, "dataset", "valid", "labels")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)
os.makedirs(VALID_IMAGES_DIR, exist_ok=True)
os.makedirs(VALID_LABELS_DIR, exist_ok=True)

# Configuration for continuous learning
COLLECTION_CONFIG = {
    "samples_before_retrain": 50,  # Number of new samples before triggering retraining
    "train_split": 0.8,  # 80% for training, 20% for validation
}

app = FastAPI(title="YOLOv8 Training & Model Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# SQLite helpers
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS uploads (id INTEGER PRIMARY KEY, filename TEXT, metadata TEXT, ts DATETIME)''')
    c.execute('''CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY, payload TEXT, ts DATETIME)''')
    c.execute('''CREATE TABLE IF NOT EXISTS collected_samples (id INTEGER PRIMARY KEY, image_path TEXT, label_path TEXT, ts DATETIME, used_for_training INTEGER DEFAULT 0)''')
    conn.commit()
    conn.close()
init_db()

def get_collected_sample_count():
    """Get count of samples collected but not yet used for training."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM collected_samples WHERE used_for_training = 0")
    count = c.fetchone()[0]
    conn.close()
    return count

def mark_samples_as_used():
    """Mark all collected samples as used for training."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE collected_samples SET used_for_training = 1 WHERE used_for_training = 0")
    conn.commit()
    conn.close()

def is_training_in_progress():
    """Check if training is currently in progress."""
    if not os.path.exists(TRAIN_LOCK_FILE):
        return False
    
    try:
        with open(TRAIN_LOCK_FILE, "r") as f:
            pid_str = f.read().strip()
            if not pid_str:
                # Empty lock file, remove it
                os.remove(TRAIN_LOCK_FILE)
                return False
            pid = int(pid_str)
        
        # Check if process is still running
        if platform.system() == "Windows":
            # On Windows, use tasklist to check if process exists
            try:
                result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                # If process is found, tasklist will return a line with the PID
                if str(pid) in result.stdout:
                    return True
                else:
                    # Process doesn't exist, remove stale lock file
                    os.remove(TRAIN_LOCK_FILE)
                    return False
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError, Exception) as e:
                # Fallback: try os.kill with broader exception handling
                try:
                    os.kill(pid, 0)
                    return True
                except Exception:
                    # Process doesn't exist, remove stale lock file
                    if os.path.exists(TRAIN_LOCK_FILE):
                        os.remove(TRAIN_LOCK_FILE)
                    return False
        else:
            # On Unix-like systems
            try:
                os.kill(pid, 0)
                return True  # Process exists
            except (OSError, ProcessLookupError, ValueError):
                # Process doesn't exist, remove stale lock file
                if os.path.exists(TRAIN_LOCK_FILE):
                    os.remove(TRAIN_LOCK_FILE)
                return False
    except (ValueError, FileNotFoundError, IOError, OSError) as e:
        # Invalid lock file or error reading it, remove it
        logger.warning(f"Error checking training lock file: {e}")
        if os.path.exists(TRAIN_LOCK_FILE):
            try:
                os.remove(TRAIN_LOCK_FILE)
            except Exception:
                pass
        return False
    except Exception as e:
        # Catch any other unexpected errors
        logger.warning(f"Unexpected error checking training status: {e}")
        return False

def set_training_lock(pid):
    """Create a lock file with the training process PID."""
    with open(TRAIN_LOCK_FILE, "w") as f:
        f.write(str(pid))

def clear_training_lock():
    """Remove the training lock file."""
    if os.path.exists(TRAIN_LOCK_FILE):
        os.remove(TRAIN_LOCK_FILE)

def save_collected_sample(image_path, label_path):
    """Save record of collected sample."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO collected_samples (image_path, label_path, ts) VALUES (?, ?, ?)", 
              (image_path, label_path, datetime.utcnow()))
    conn.commit()
    conn.close()

def save_upload(filename, metadata):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO uploads (filename, metadata, ts) VALUES (?, ?, ?)", (filename, json.dumps(metadata), datetime.utcnow()))
    conn.commit(); conn.close()

def save_log(payload):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO logs (payload, ts) VALUES (?, ?)", (json.dumps(payload), datetime.utcnow()))
    conn.commit(); conn.close()

# model metadata
def get_meta():
    if not os.path.exists(META_PATH):
        meta = {"version": 0, "model_path": None, "created_at": None}
        with open(META_PATH, "w") as f:
            json.dump(meta, f)
        return meta
    with open(META_PATH, "r") as f:
        return json.load(f)

def update_meta(version, model_path):
    meta = {"version": version, "model_path": model_path, "created_at": datetime.utcnow().isoformat()}
    with open(META_PATH, "w") as f:
        json.dump(meta, f)
    return meta

# Helper function to find available training models
def find_available_models():
    """
    Finds all available training models in the models directory.
    Returns a list of tuples: [(model_name, model_path, version_number), ...] sorted by version (newest first)
    """
    if not os.path.exists(MODELS_DIR):
        return []
    
    all_trainings = os.listdir(MODELS_DIR)
    available_models = []

    # Regex pattern for matching numbers in directory names (e.g., "1_train" -> 1, "1_train2" -> 2)
    pattern = r"train(\d+)?$"

    # Get all available trainings
    for training in all_trainings:
        # Skip non-directory files
        training_path = os.path.join(MODELS_DIR, training)
        if not os.path.isdir(training_path):
            continue
            
        match = re.search(pattern, training)
        if match:
            # If there's a number after "train", use it; otherwise treat as version 1
            version_str = match.group(1)
            version_number = int(version_str) if version_str else 1
            
            # Check if model file exists
            model_path = os.path.join(MODELS_DIR, training, "weights", "best.pt")
            if os.path.exists(model_path):
                available_models.append((training, model_path, version_number))
    
    # Sort by version number (newest first)
    available_models.sort(key=lambda x: x[2], reverse=True)
    return available_models

def find_latest_model():
    """
    Finds the latest training model in the models directory.
    Returns a tuple (model_path, model_name) if found, (None, None) otherwise.
    """
    available_models = find_available_models()
    if not available_models:
        return None, None
    
    # Return the latest (first in sorted list)
    model_name, model_path, _ = available_models[0]
    return model_path, model_name

def find_any_available_model():
    """
    Finds any available training model (latest first, then falls back to older ones).
    Returns a tuple (model_path, model_name) if found, (None, None) otherwise.
    """
    available_models = find_available_models()
    if not available_models:
        return None, None
    
    # Return the latest (first in sorted list)
    model_name, model_path, _ = available_models[0]
    return model_path, model_name

@app.post("/upload_data")
async def upload_data(file: UploadFile = File(...), metadata: str = Form(...)):
    ts = int(time.time()*1000)
    filename = f"{ts}_{file.filename}"
    path = os.path.join(RAW_DIR, filename)
    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)
    md = json.loads(metadata)
    save_upload(filename, md)
    return {"status":"ok", "filename": filename}

@app.post("/upload_logs")
async def upload_logs(payload: dict):
    save_log(payload)
    return {"status": "ok"}

def convert_to_yolo_format(box, img_width, img_height):
    """
    Convert bounding box from pixel coordinates to YOLO format.
    YOLO format: class_id center_x center_y width height (all normalized 0-1)
    """
    x1, y1, x2, y2 = box
    
    # Calculate center and dimensions
    center_x = (x1 + x2) / 2.0 / img_width
    center_y = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # Ensure values are within [0, 1]
    center_x = max(0, min(1, center_x))
    center_y = max(0, min(1, center_y))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return center_x, center_y, width, height

@app.post("/upload_detection_data")
async def upload_detection_data(
    image: UploadFile = File(...),
    detections: str = Form(...)  # JSON string of detections
):
    """
    Receives detection data from edge device:
    - image: The captured frame
    - detections: JSON array of [class_id, x1, y1, x2, y2, confidence]
    """
    try:
        # Parse detections
        try:
            detections_list = json.loads(detections)
        except json.JSONDecodeError as e:
            return JSONResponse(
                {"status": "error", "message": f"Invalid JSON in detections: {str(e)}", "error_type": "json_parse"},
                status_code=400
            )
        
        if not isinstance(detections_list, list):
            return JSONResponse(
                {"status": "error", "message": "Detections must be a list", "error_type": "invalid_format"},
                status_code=400
            )
        
        # Read image
        try:
            image_bytes = await image.read()
            if len(image_bytes) == 0:
                return JSONResponse(
                    {"status": "error", "message": "Empty image file", "error_type": "empty_image"},
                    status_code=400
                )
            img = Image.open(io.BytesIO(image_bytes))
            img_width, img_height = img.size
        except Exception as e:
            return JSONResponse(
                {"status": "error", "message": f"Failed to read/process image: {str(e)}", "error_type": "image_error"},
                status_code=400
            )
        
        # Generate unique filename
        ts = int(time.time() * 1000)
        image_filename = f"{ts}.jpg"
        label_filename = f"{ts}.txt"
        
        # Determine if this goes to train or valid (80/20 split)
        import random
        is_train = random.random() < COLLECTION_CONFIG["train_split"]
        
        if is_train:
            image_path = os.path.join(TRAIN_IMAGES_DIR, image_filename)
            label_path = os.path.join(TRAIN_LABELS_DIR, label_filename)
        else:
            image_path = os.path.join(VALID_IMAGES_DIR, image_filename)
            label_path = os.path.join(VALID_LABELS_DIR, label_filename)
        
        # Save image
        try:
            img.save(image_path, "JPEG")
        except Exception as e:
            return JSONResponse(
                {"status": "error", "message": f"Failed to save image: {str(e)}", "error_type": "save_image", "path": image_path},
                status_code=500
            )
        
        # Convert detections to YOLO format and save label file
        try:
            with open(label_path, "w") as f:
                for det in detections_list:
                    if len(det) >= 5:
                        try:
                            class_id = int(det[0])
                            x1, y1, x2, y2 = float(det[1]), float(det[2]), float(det[3]), float(det[4])
                            
                            # Convert to YOLO format
                            cx, cy, w, h = convert_to_yolo_format([x1, y1, x2, y2], img_width, img_height)
                            
                            # Write YOLO format: class_id center_x center_y width height
                            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                        except (ValueError, IndexError, TypeError) as e:
                            # Skip invalid detections but continue processing
                            print(f"Warning: Skipping invalid detection: {det}, error: {e}")
                            continue
        except Exception as e:
            return JSONResponse(
                {"status": "error", "message": f"Failed to save label file: {str(e)}", "error_type": "save_label", "path": label_path},
                status_code=500
            )
        
        # Save record
        try:
            save_collected_sample(image_path, label_path)
        except Exception as e:
            # Log error but don't fail the request
            print(f"Warning: Failed to save sample record to database: {e}")
        
        # Check if we should trigger retraining (check BEFORE marking as used)
        try:
            sample_count = get_collected_sample_count()
            should_retrain = sample_count >= COLLECTION_CONFIG["samples_before_retrain"]
            
            # Check if training is already in progress
            training_in_progress = is_training_in_progress()
        except Exception as e:
            print(f"Warning: Failed to check training status: {e}")
            sample_count = 0
            should_retrain = False
            training_in_progress = False
        
        response = {
            "status": "ok",
            "image_path": image_path,
            "label_path": label_path,
            "samples_collected": sample_count,
            "should_retrain": should_retrain and not training_in_progress,
            "training_in_progress": training_in_progress
        }
        
        if should_retrain and not training_in_progress:
            try:
                # Mark samples as used BEFORE starting training to avoid duplicate triggers
                mark_samples_as_used()
                # Trigger retraining in background
                proc = subprocess.Popen(["python", os.path.join(ROOT, "train.py")])
                set_training_lock(proc.pid)
                response["retraining_started"] = True
                response["training_pid"] = proc.pid
                response["samples_collected"] = 0  # Reset count after marking as used
            except Exception as e:
                print(f"Warning: Failed to start retraining: {e}")
                response["retraining_error"] = str(e)
        elif should_retrain and training_in_progress:
            response["message"] = "Training already in progress, skipping new training"
        
        return response
        
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Unexpected error in upload_detection_data: {error_trace}")
        return JSONResponse(
            {"status": "error", "message": str(e), "error_type": "unexpected"},
            status_code=500
        )

@app.get("/collection_status")
def collection_status():
    """Get status of data collection."""
    sample_count = get_collected_sample_count()
    training_in_progress = is_training_in_progress()
    return {
        "samples_collected": sample_count,
        "samples_needed": COLLECTION_CONFIG["samples_before_retrain"],
        "ready_for_retrain": sample_count >= COLLECTION_CONFIG["samples_before_retrain"] and not training_in_progress,
        "training_in_progress": training_in_progress
    }

@app.get("/check_model_update")
def check_model_update(current_model: str = None):
    """
    Check if a newer model is available.
    Returns the latest model name and info for comparison.
    """
    latest_model_path, latest_model_name = find_any_available_model()
    
    if not latest_model_path:
        return {
            "available": False,
            "message": "No models available"
        }
    
    # If no current model specified, return latest
    if not current_model:
        return {
            "available": True,
            "latest_model": latest_model_name,
            "model_path": latest_model_path,
            "is_newer": True
        }
    
    # Compare versions
    available_models = find_available_models()
    current_version = None
    latest_version = None
    
    pattern = r"train(\d+)?$"
    for model_name, _, version in available_models:
        if model_name == current_model:
            current_version = version
        if model_name == latest_model_name:
            latest_version = version
    
    is_newer = (current_version is None) or (latest_version is not None and latest_version > current_version)
    
    return {
        "available": True,
        "current_model": current_model,
        "latest_model": latest_model_name,
        "model_path": latest_model_path,
        "is_newer": is_newer,
        "current_version": current_version,
        "latest_version": latest_version
    }

@app.get("/latest_model")
def latest_model():
    return get_meta()

@app.get("/download_model/{version}")
def download_model(version: int):
    model_file = os.path.join(MODELS_DIR, str(version), "model.tflite")
    if not os.path.exists(model_file):
        return JSONResponse({"error":"not found"}, status_code=404)
    return FileResponse(model_file, filename=f"model_v{version}.tflite", media_type="application/octet-stream")

@app.get("/download_latest_model")
def download_latest_model():
    """
    Downloads the latest trained model (best.pt) to the client's current directory.
    Falls back to previous models if latest is not available.
    """
    latest_model_path, model_name = find_any_available_model()
    if not latest_model_path:
        return JSONResponse({"error": "No trained model found"}, status_code=404)
    
    logger.info(f"Serving model: {model_name}")
    return FileResponse(
        latest_model_path, 
        filename="best.pt", 
        media_type="application/octet-stream",
        headers={"X-Model-Name": model_name}
    )

def run_in_new_terminal(script_path):
    """
    Runs a Python script in a new terminal window.
    Works on Windows, Linux, and macOS.
    """
    script_path = os.path.abspath(script_path)
    script_dir = os.path.dirname(script_path)
    
    if platform.system() == "Windows":
        # Windows: use start command to open new cmd window
        # /k keeps the window open after execution
        cmd = f'start "Detection" cmd /k "cd /d "{script_dir}" && python "{script_path}"'
        proc = subprocess.Popen(cmd, shell=True)
    elif platform.system() == "Darwin":  # macOS
        # macOS: use osascript to open new Terminal window
        cmd = f'osascript -e \'tell application "Terminal" to do script "cd \\"{script_dir}\\" && python3 \\"{script_path}\\""\''
        proc = subprocess.Popen(cmd, shell=True)
    else:  # Linux
        # Linux: try different terminal emulators
        terminals = ["gnome-terminal", "xterm", "konsole", "terminator"]
        for term in terminals:
            try:
                if term == "gnome-terminal":
                    cmd = [term, "--", "bash", "-c", f"cd '{script_dir}' && python3 '{script_path}'; exec bash"]
                elif term == "xterm":
                    cmd = [term, "-e", "bash", "-c", f"cd '{script_dir}' && python3 '{script_path}'; exec bash"]
                elif term == "konsole":
                    cmd = [term, "-e", "bash", "-c", f"cd '{script_dir}' && python3 '{script_path}'; exec bash"]
                elif term == "terminator":
                    cmd = [term, "-e", f"cd '{script_dir}' && python3 '{script_path}'; exec bash"]
                
                proc = subprocess.Popen(cmd)
                break
            except FileNotFoundError:
                continue
        else:
            # Fallback: run in background if no terminal found
            proc = subprocess.Popen(["python3", script_path], cwd=script_dir)
    
    return proc

@app.post("/trigger_train")
def trigger_train():
    # First, check if we can use any available training model (latest first, then fallback)
    latest_model_path, model_name = find_any_available_model()
    
    if latest_model_path:
        logger.info(f"Model available for detection: {model_name}")
        # Return a cross-platform Python script that downloads and runs detection
        # This works on Windows, Linux, and macOS
        script_path = os.path.join(ROOT, "get_and_run_detection.py")
        
        if os.path.exists(script_path):
            # Return the Python script - user can pipe to python or save and run
            return FileResponse(
                script_path,
                filename="get_and_run_detection.py",
                media_type="text/x-python"
            )
        else:
            # Fallback: return JSON with instructions
            return JSONResponse({
                "status": "model_available",
                "model_path": latest_model_path,
                "model_name": model_name,
                "instructions": {
                    "download_model": "curl -X GET http://localhost:8000/download_latest_model -o best.pt",
                    "download_script": "curl -X GET http://localhost:8000/download_detection_script -o run_detection.py",
                    "run": "python run_detection.py"
                },
                "message": f"Model available ({model_name}). Download and run detection locally."
            })
    else:
        # If no model found, check if training is already in progress
        if is_training_in_progress():
            return JSONResponse({
                "status": "training_in_progress",
                "message": "Training is already in progress, please wait"
            })
        
        # Start training
        proc = subprocess.Popen(["python", os.path.join(ROOT, "train.py")])
        set_training_lock(proc.pid)
        return JSONResponse({
            "status": "training_started",
            "pid": proc.pid,
            "message": "No model found, starting training"
        })

@app.get("/download_detection_script")
def download_detection_script():
    """
    Downloads the standalone detection script that can run from any directory.
    """
    script_path = os.path.join(ROOT, "run_detection.py")
    if not os.path.exists(script_path):
        return JSONResponse({"error": "Detection script not found"}, status_code=404)
    
    return FileResponse(
        script_path,
        filename="run_detection.py",
        media_type="text/x-python"
    )
