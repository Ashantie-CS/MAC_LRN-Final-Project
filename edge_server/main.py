# server/main.py
import os, json, time, sqlite3, subprocess
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
MODELS_DIR = os.path.join(ROOT, "models")
META_PATH = os.path.join(MODELS_DIR, "meta.json")
DB_PATH = os.path.join(ROOT, "server.db")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "dataset", "train"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "dataset", "val"), exist_ok=True)

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
    conn.commit()
    conn.close()
init_db()

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

@app.get("/latest_model")
def latest_model():
    return get_meta()

@app.get("/download_model/{version}")
def download_model(version: int):
    model_file = os.path.join(MODELS_DIR, str(version), "model.tflite")
    if not os.path.exists(model_file):
        return JSONResponse({"error":"not found"}, status_code=404)
    return FileResponse(model_file, filename=f"model_v{version}.tflite", media_type="application/octet-stream")

@app.post("/trigger_train")
def trigger_train():
    # Runs train.py asynchronously; it will save model to models/<version>/model.tflite and update meta.json
    proc = subprocess.Popen(["python", os.path.join(ROOT, "train.py")])
    return {"status":"training_started", "pid": proc.pid}
