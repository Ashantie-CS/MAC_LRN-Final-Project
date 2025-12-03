# server/main.py
import os, json, time, sqlite3, subprocess, re, sys, platform, logging, traceback, tempfile, atexit
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_python_cmd():
    """Get the correct Python command based on the platform."""
    if platform.system() == "Darwin":  # macOS
        return "python3"
    else:
        return "python"

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
    "samples_before_retrain": 5,  # Number of new samples before triggering retraining
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
    """Check if training is currently in progress by checking the lock file.
    
    Returns:
        tuple: (is_training, version) where version is the model version being trained,
               or None if version is not available
    """
    if not os.path.exists(TRAIN_LOCK_FILE):
        return False, None
    
    try:
        # Check if lock file is recent (within last 24 hours)
        # If older, assume it's stale and training has finished/crashed
        lock_age = time.time() - os.path.getmtime(TRAIN_LOCK_FILE)
        if lock_age > 86400:  # 24 hours
            logger.warning("Training lock file is very old, assuming stale")
            try:
                os.remove(TRAIN_LOCK_FILE)
            except:
                pass
            return False, None
        
        # Try to read version from lock file
        version = None
        try:
            with open(TRAIN_LOCK_FILE, "r") as f:
                lock_data = json.load(f)
                version = lock_data.get("version")
        except (json.JSONDecodeError, KeyError, ValueError):
            # Old format or invalid JSON, try reading as plain text
            try:
                with open(TRAIN_LOCK_FILE, "r") as f:
                    content = f.read().strip()
                    # If it's just a number (old format), ignore version
            except:
                pass
        
        # Lock file exists and is recent, training is likely in progress
        # The training script itself will clear the lock when it finishes
        return True, version
    except Exception as e:
        logger.warning(f"Failed to check training lock file: {e}")
        return False, None

def set_training_lock(version=None):
    """Create a lock file to indicate training is in progress.
    
    Args:
        version: The version number of the model being trained (optional).
                 If provided, stored in the lock file to identify which model to exclude.
    """
    # Store timestamp and version number
    lock_data = {
        "timestamp": int(time.time()),
        "version": version
    }
    with open(TRAIN_LOCK_FILE, "w") as f:
        json.dump(lock_data, f)

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

# Helper function to find available training models
def find_available_models():
    """
    Finds all available training models in the models directory.
    Only includes models that are fully trained (training lock is not active for them).
    Returns a list of tuples: [(model_name, model_path, version_number), ...] sorted by version (newest first)
    """
    if not os.path.exists(MODELS_DIR):
        return []
    
    # Check if training is currently in progress and get the version being trained
    training_in_progress, training_version = is_training_in_progress()
    
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
                # Check if training is complete for this model
                # Training creates a .training_complete marker file when it finishes
                train_dir = os.path.join(MODELS_DIR, training)
                completion_marker = os.path.join(train_dir, ".training_complete")
                
                # If training is in progress and this is the model being trained, exclude it
                if training_in_progress and training_version is not None and version_number == training_version:
                    logger.info(f"Skipping {training} (version {version_number}) - currently being trained")
                    continue
                
                # CRITICAL: Only include models that have completed training (have the completion marker)
                # YOLO creates best.pt during the first epoch and updates it throughout training,
                # so we MUST check for the completion marker to ensure training is fully done
                if os.path.exists(completion_marker):
                    # Model has completion marker - training is fully complete, safe to serve
                    available_models.append((training, model_path, version_number))
                else:
                    # Model doesn't have completion marker - training is either:
                    # 1. Currently in progress (best.pt exists but training not done)
                    # 2. An older completed model that finished before we added completion markers
                    
                    # Check if this is the model currently being trained
                    is_currently_training = training_in_progress and training_version is not None and version_number == training_version
                    
                    if is_currently_training:
                        # This is the model currently being trained - NEVER include it
                        # even though best.pt exists (it's being updated during training)
                        logger.info(f"Skipping {training} (version {version_number}) - currently being trained (no completion marker)")
                        continue
                    
                    # Not currently being trained, but no completion marker
                    # If this model is OLDER than the one being trained, it must be complete
                    # (otherwise a newer training wouldn't have started)
                    if training_in_progress and training_version is not None:
                        if version_number < training_version:
                            # This is an older model - if training for a newer version started,
                            # this one must have completed. Create marker and include it.
                            try:
                                os.makedirs(train_dir, exist_ok=True)
                                with open(completion_marker, "w") as f:
                                    f.write(str(int(time.time())))
                                logger.info(f"Created completion marker for {training} (version {version_number}) - older than training version {training_version}")
                                available_models.append((training, model_path, version_number))
                            except Exception as e:
                                logger.warning(f"Failed to create completion marker for {training}: {e}")
                                # Still include it - it's older than the one being trained, so it's complete
                                logger.info(f"Including {training} (version {version_number}) - older than training version {training_version}")
                                available_models.append((training, model_path, version_number))
                        else:
                            # This model version is >= the one being trained, but no marker
                            # It might be incomplete or from a failed training
                            logger.info(f"Skipping {training} (version {version_number}) - no completion marker and version >= training version {training_version}")
                    elif not training_in_progress:
                        # No training in progress, so this model must be complete
                        # Create marker retroactively for backward compatibility
                        try:
                            os.makedirs(train_dir, exist_ok=True)
                            with open(completion_marker, "w") as f:
                                f.write(str(int(time.time())))
                            logger.info(f"Created completion marker for {training} (version {version_number}) - no training in progress")
                            available_models.append((training, model_path, version_number))
                        except Exception as e:
                            logger.warning(f"Failed to create completion marker for {training}: {e}")
                            # Still include it if no training is in progress (backward compatibility)
                            available_models.append((training, model_path, version_number))
                    else:
                        # Training is in progress but we don't know which version
                        # Be conservative: don't include models without markers
                        logger.info(f"Skipping {training} (version {version_number}) - no completion marker and training in progress (unknown version)")
    
    # Sort by version number (newest first)
    available_models.sort(key=lambda x: x[2], reverse=True)
    return available_models

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
            training_in_progress, _ = is_training_in_progress()
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
                # Get the next version number that will be used for training
                # We need to read it the same way train.py does
                meta_path = os.path.join(MODELS_DIR, "meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                        next_version = meta.get("version", 0) + 1
                else:
                    next_version = 1
                
                # Trigger retraining in a new terminal window to show progress
                train_script = os.path.join(ROOT, "train.py")
                set_training_lock(next_version)  # Set lock with version number
                proc = run_in_new_terminal(train_script, window_title="Training Progress")
                response["retraining_started"] = True
                response["samples_collected"] = 0  # Reset count after marking as used
                response["message"] = "Training started in new terminal window"
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
    training_in_progress, _ = is_training_in_progress()
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
    # Get all available models (only completed ones)
    available_models = find_available_models()
    
    if not available_models:
        return {
            "available": False,
            "message": "No completed models available"
        }
    
    # Get the latest available model
    latest_model_name, latest_model_path, latest_version = available_models[0]  # Already sorted by version
    
    logger.info(f"Available models: {[m[0] for m in available_models]}")
    logger.info(f"Latest available model: {latest_model_name} (version {latest_version})")
    
    # If no current model specified, return latest
    if not current_model:
        return {
            "available": True,
            "latest_model": latest_model_name,
            "model_path": latest_model_path,
            "is_newer": True,
            "latest_version": latest_version
        }
    
    # Compare versions
    current_version = None
    pattern = r"train(\d+)?$"
    
    # Find current model version in available models
    for model_name, _, version in available_models:
        if model_name == current_model:
            current_version = version
            break
    
    # If current model not found in available models, it might be an old/incomplete model
    if current_version is None:
        # Try to extract version from current_model name
        match = re.search(pattern, current_model)
        if match:
            version_str = match.group(1)
            current_version = int(version_str) if version_str else 1
        else:
            current_version = 0  # Unknown model, treat as version 0
    
    is_newer = latest_version > current_version if current_version is not None else True
    
    logger.info(f"Model update check: current={current_model} (v{current_version}), latest={latest_model_name} (v{latest_version}), is_newer={is_newer}")
    
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

def run_in_new_terminal(script_path, window_title="Process"):
    """
    Runs a Python script in a new terminal window.
    Works on Windows, Linux, and macOS.
    """
    script_path = os.path.abspath(script_path)
    script_dir = os.path.dirname(script_path)
    python_cmd = get_python_cmd()
    
    if platform.system() == "Windows":
        # Windows: use start command to open new cmd window
        # /k keeps the window open after execution
        cmd = f'start "{window_title}" cmd /k "cd /d "{script_dir}" && {python_cmd} "{script_path}"'
        proc = subprocess.Popen(cmd, shell=True)
    elif platform.system() == "Darwin":  # macOS
        # macOS: use osascript to open new Terminal window with proper output handling
        # Create a temporary shell script that ensures unbuffered output
        temp_script = tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False)
        temp_script.write(f'''#!/bin/bash
cd "{script_dir}"
export PYTHONUNBUFFERED=1
{python_cmd} -u "{script_path}"
echo ""
echo "Training completed. Press Enter to close this window..."
read
''')
        temp_script.close()
        os.chmod(temp_script.name, 0o755)
        
        # Register cleanup function
        def cleanup_temp_script():
            try:
                if os.path.exists(temp_script.name):
                    os.remove(temp_script.name)
            except:
                pass
        atexit.register(cleanup_temp_script)
        
        # Use osascript to open Terminal with the script
        # Properly escape the path for AppleScript
        script_path_escaped = temp_script.name.replace('"', '\\"')
        applescript = f'tell application "Terminal" to do script "bash \\"{script_path_escaped}\\""'
        proc = subprocess.Popen(['osascript', '-e', applescript])
    else:  # Linux
        # Linux: try different terminal emulators
        terminals = ["gnome-terminal", "xterm", "konsole", "terminator"]
        for term in terminals:
            try:
                if term == "gnome-terminal":
                    cmd = [term, "--title", window_title, "--", "bash", "-c", f"cd '{script_dir}' && {python_cmd} '{script_path}'; exec bash"]
                elif term == "xterm":
                    cmd = [term, "-T", window_title, "-e", "bash", "-c", f"cd '{script_dir}' && {python_cmd} '{script_path}'; exec bash"]
                elif term == "konsole":
                    cmd = [term, "--title", window_title, "-e", "bash", "-c", f"cd '{script_dir}' && {python_cmd} '{script_path}'; exec bash"]
                elif term == "terminator":
                    cmd = [term, "-e", f"cd '{script_dir}' && {python_cmd} '{script_path}'; exec bash"]
                
                proc = subprocess.Popen(cmd)
                break
            except FileNotFoundError:
                continue
        else:
            # Fallback: run in background if no terminal found
            proc = subprocess.Popen([python_cmd, script_path], cwd=script_dir)
    
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
        training_in_progress, _ = is_training_in_progress()
        if training_in_progress:
            return JSONResponse({
                "status": "training_in_progress",
                "message": "Training is already in progress, please wait"
            })
        
        # Start training in a new terminal window to show progress
        train_script = os.path.join(ROOT, "train.py")
        set_training_lock()  # Set lock before starting training
        proc = run_in_new_terminal(train_script, window_title="Training Progress")
        return JSONResponse({
            "status": "training_started",
            "message": "No model found, starting training in new terminal window"
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
