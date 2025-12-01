# server/yolo_train.py
import os, json, subprocess, shutil
from datetime import datetime

ROOT = os.path.dirname(__file__)
DATASET_DIR = os.path.join(ROOT, "data", "dataset")
MODELS_DIR = os.path.join(ROOT, "models")
META_PATH = os.path.join(MODELS_DIR, "meta.json")
os.makedirs(MODELS_DIR, exist_ok=True)

# Adjust these params as needed
EPOCHS = 50
BATCH = 16
IMGSZ = 640
MODEL_PRETRAINED = "yolov8n.pt"  # ultralytics base

# user must prepare a data.yaml pointing to train/val directories
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")  # create this with paths to images/labels

def get_next_version():
    if not os.path.exists(META_PATH):
        return 1
    meta = json.load(open(META_PATH))
    return meta.get("version", 0) + 1

def run_training():
    v = get_next_version()
    out_dir = os.path.join(MODELS_DIR, f"run_{v}")  # ultralytics will write results to runs/detect/train by default, but we set project
    # Use ultralytics CLI (must be installed in environment)
    cmd = [
        "yolo", "detect", "train",
        "model="+MODEL_PRETRAINED,
        "data="+DATA_YAML,
        "device=0",
        f"epochs={EPOCHS}",
        f"batch={BATCH}",
        f"imgsz={IMGSZ}",
        "project="+MODELS_DIR,
        "name=" + f"{v}_train"
    ]
    print("Running training:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    # ultralytics saves best.pt under models/<project>/<name>/weights/best.pt
    # Find the produced best.pt
    candidate_dir = os.path.join(MODELS_DIR, f"{v}_train", "weights")
    best_pt = os.path.join(candidate_dir, "best.pt")
    if not os.path.exists(best_pt):
        # fallback to last.pt
        best_pt = os.path.join(candidate_dir, "last.pt")
    if not os.path.exists(best_pt):
        raise FileNotFoundError("No trained .pt found at " + candidate_dir)
    return best_pt, v

def export_to_tflite(pt_path, version):
    # Use ultralytics export
    # This will create a tflite in runs/detect/exp/ or in same folder
    cmd = ["yolo", "export", "model="+pt_path, "format=tflite"]
    print("Exporting to TFLite:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    # ultralytics typically writes exported files in the same weights dir or runs/detect/exp
    # We'll search for .tflite under project dir
    tflite_path = None
    for root,dirs,files in os.walk(os.path.dirname(pt_path)):
        for f in files:
            if f.endswith(".tflite"):
                tflite_path = os.path.join(root, f)
                break
        if tflite_path:
            break
    # fallback search entire MODELS_DIR
    if not tflite_path:
        for root,dirs,files in os.walk(MODELS_DIR):
            for f in files:
                if f.endswith(".tflite"):
                    tflite_path = os.path.join(root, f)
                    break
            if tflite_path:
                break
    if not tflite_path:
        raise FileNotFoundError("TFLite export not found")
    # move to versioned folder
    ver_dir = os.path.join(MODELS_DIR, str(version))
    os.makedirs(ver_dir, exist_ok=True)
    target = os.path.join(ver_dir, "model.tflite")
    shutil.copyfile(tflite_path, target)
    return target

def update_meta(version, model_path):
    meta = {"version": version, "model_path": model_path, "created_at": datetime.utcnow().isoformat()}
    open(META_PATH, "w").write(json.dumps(meta))
    print("Updated meta:", meta)

def main():
    pt, v = run_training()
    tflite = export_to_tflite(pt, v)
    update_meta(v, tflite)
    print("Training+Export complete. Model at:", tflite)

if __name__ == "__main__":
    main()
