# server/yolo_train.py
import os, json, subprocess, shutil, sys, platform, time
from datetime import datetime

def get_python_cmd():
    """Get the correct Python command based on the platform."""
    if platform.system() == "Darwin":  # macOS
        return "python3"
    else:
        return "python"





ROOT = os.path.dirname(__file__)
DATASET_DIR = os.path.join(ROOT, "data", "dataset")
MODELS_DIR = os.path.join(ROOT, "models")
META_PATH = os.path.join(MODELS_DIR, "meta.json")
TRAIN_LOCK_FILE = os.path.join(ROOT, ".training_lock")
os.makedirs(MODELS_DIR, exist_ok=True)

# Adjust these params as needed
EPOCHS = 1
BATCH = 8  # Reduced from 12 to help prevent memory issues
IMGSZ = 640
MODEL_PRETRAINED = "yolov8n.pt"  # ultralytics base

# user must prepare a data.yaml pointing to train/val directories
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")  # create this with paths to images/labels

def get_training_device():
    """
    Get the appropriate device for training based on the platform.
    Returns 'mps' for M1 Macs, 'cpu' for other Macs, or '0' for CUDA GPUs.
    """
    if platform.system() == "Darwin":  # macOS
        # Check if MPS (Metal Performance Shaders) is available (M1/M2 Macs)
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except (ImportError, AttributeError):
            # If torch is not available or MPS check fails, use cpu
            return "cpu"
    else:
        # For Windows/Linux, try CUDA first, fallback to CPU
        try:
            import torch
            if torch.cuda.is_available():
                return "0"  # CUDA device 0
            else:
                return "cpu"
        except ImportError:
            # If torch is not available, default to 0 (will be handled by YOLO)
            return "0"

def get_next_version():
    if not os.path.exists(META_PATH):
        return 1
    meta = json.load(open(META_PATH))
    return meta.get("version", 0) + 1

def run_training():
    print("=" * 60)
    print("Starting Training Process")
    print("=" * 60)
    
    # Validate data.yaml exists
    if not os.path.exists(DATA_YAML):
        error_msg = f"ERROR: Data configuration file not found at {DATA_YAML}"
        print(error_msg)
        raise FileNotFoundError(error_msg)
    print(f"✓ Data config found: {DATA_YAML}")
    
    # Validate dataset directories exist
    try:
        with open(DATA_YAML, 'r') as f:
            import yaml
            data_config = yaml.safe_load(f)
            train_path = data_config.get('train', '')
            val_path = data_config.get('val', '')
            if train_path and not os.path.exists(train_path):
                print(f"WARNING: Train path in data.yaml doesn't exist: {train_path}")
            if val_path and not os.path.exists(val_path):
                print(f"WARNING: Val path in data.yaml doesn't exist: {val_path}")
    except Exception as e:
        print(f"WARNING: Could not validate data.yaml paths: {e}")
    
    v = get_next_version()
    out_dir = os.path.join(MODELS_DIR, f"run_{v}")  # ultralytics will write results to runs/detect/train by default, but we set project
    # Get appropriate device for training
    device = get_training_device()
    print(f"✓ Using device: {device}")
    
    # Use ultralytics CLI (must be installed in environment)
    cmd = [
        "yolo", "detect", "train",
        "model="+MODEL_PRETRAINED,
        "data="+DATA_YAML,
        f"device={device}",
        f"epochs={EPOCHS}",
        f"batch={BATCH}",
        f"imgsz={IMGSZ}",
        "project="+MODELS_DIR,
        "name=" + f"{v}_train"
    ]
    print("=" * 60)
    print("Running training command:")
    print(" ".join(cmd))
    print("=" * 60)
    print(f"Training will save to: {os.path.join(MODELS_DIR, f'{v}_train')}")
    print()
    
    # Run training without capturing output so it shows in real-time in the terminal
    # This allows the user to see progress in the terminal window
    print("Starting training process...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"❌ Training command execution failed: {e}")
        raise RuntimeError(f"Training command execution failed: {e}")
    
    print()
    print("=" * 60)
    if result.returncode != 0:
        error_code = result.returncode
        # Windows error code 3221225477 (0xC0000005) is an access violation
        if error_code == 3221225477 or error_code == -1073741819:
            print(f"❌ Training crashed with access violation (error code {error_code})")
            print("This usually indicates:")
            print("  1. GPU/CUDA memory issue - try reducing batch size")
            print("  2. Corrupted data or model files")
            print("  3. GPU driver issue - try updating drivers")
            print("  4. Out of memory - try reducing image size or batch size")
        else:
            print(f"❌ Training command failed with return code {error_code}")
        print("=" * 60)
        raise RuntimeError(f"Training failed with return code {error_code}. Check output above.")
    else:
        print("✓ Training command completed successfully")
        # Give a moment for file system to sync
        time.sleep(1)
    print("=" * 60)
    
    # ultralytics saves best.pt under models/<project>/<name>/weights/best.pt
    # Find the produced best.pt
    train_dir = os.path.join(MODELS_DIR, f"{v}_train")
    candidate_dir = os.path.join(train_dir, "weights")
    
    print()
    print("=" * 60)
    print("Searching for trained model...")
    print(f"Training directory: {train_dir}")
    print(f"Expected weights directory: {candidate_dir}")
    print("=" * 60)
    
    # First, check the expected location
    best_pt = os.path.join(candidate_dir, "best.pt")
    if os.path.exists(best_pt):
        print(f"✓ Found model at expected location: {best_pt}")
        return best_pt, v
    
    # Try last.pt in the same location
    last_pt = os.path.join(candidate_dir, "last.pt")
    if os.path.exists(last_pt):
        print(f"✓ Found last.pt at: {last_pt}")
        return last_pt, v
    
    # Search the entire training directory for any .pt files
    print(f"⚠ Model not found in expected location. Searching entire training directory...")
    found_pt_files = []
    
    if os.path.exists(train_dir):
        print(f"Training directory contents:")
        try:
            for item in os.listdir(train_dir):
                item_path = os.path.join(train_dir, item)
                if os.path.isdir(item_path):
                    print(f"  [DIR]  {item}/")
                else:
                    print(f"  [FILE] {item}")
        except Exception as e:
            print(f"  Error listing directory: {e}")
        
        print()
        print("Searching for .pt files...")
        for root, dirs, files in os.walk(train_dir):
            for f in files:
                if f.endswith(".pt"):
                    found_path = os.path.join(root, f)
                    found_pt_files.append(found_path)
                    print(f"  Found: {found_path}")
    
    # Also check if ultralytics saved to runs/detect/train (default location)
    runs_dir = os.path.join(ROOT, "runs", "detect")
    if os.path.exists(runs_dir):
        print()
        print(f"Also checking default ultralytics location: {runs_dir}")
        for root, dirs, files in os.walk(runs_dir):
            for f in files:
                if f.endswith(".pt") and ("best" in f.lower() or "last" in f.lower()):
                    found_path = os.path.join(root, f)
                    found_pt_files.append(found_path)
                    print(f"  Found: {found_path}")
    
    if found_pt_files:
        # Prefer best.pt, then last.pt, then any .pt file
        best_files = [f for f in found_pt_files if "best" in os.path.basename(f).lower()]
        if best_files:
            print(f"✓ Using best.pt: {best_files[0]}")
            return best_files[0], v
        
        last_files = [f for f in found_pt_files if "last" in os.path.basename(f).lower()]
        if last_files:
            print(f"✓ Using last.pt: {last_files[0]}")
            return last_files[0], v
        
        # Use the first found .pt file
        print(f"✓ Using found model: {found_pt_files[0]}")
        return found_pt_files[0], v
    
    # No model found
    print()
    print("=" * 60)
    print("❌ ERROR: No trained model file (.pt) found!")
    print("=" * 60)
    print("Possible reasons:")
    print("  1. Training failed or was interrupted")
    print("  2. Training didn't complete successfully")
    print("  3. Model was saved to an unexpected location")
    print()
    print(f"Expected location: {candidate_dir}/best.pt")
    print(f"Training directory exists: {os.path.exists(train_dir)}")
    if os.path.exists(train_dir):
        print(f"Training directory contents: {os.listdir(train_dir)}")
    print("=" * 60)
    raise FileNotFoundError(
        f"No trained .pt file found after training.\n"
        f"Expected at: {candidate_dir}/best.pt\n"
        f"Training directory: {train_dir}\n"
        f"Please check the training output above for errors."
    )

def update_meta(version, model_path):
    meta = {"version": version, "model_path": model_path, "created_at": datetime.utcnow().isoformat()}
    open(META_PATH, "w").write(json.dumps(meta))
    print("Updated meta:", meta)

def clear_training_lock():
    """Remove the training lock file."""
    if os.path.exists(TRAIN_LOCK_FILE):
        os.remove(TRAIN_LOCK_FILE)

def main():
    v = None
    pt = None
    try:
        pt, v = run_training()
        update_meta(v, pt)
        print("Training complete. Model at:", pt)
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Always clear the lock file when training finishes (success or failure)
        clear_training_lock()

if __name__ == "__main__":
    main()
