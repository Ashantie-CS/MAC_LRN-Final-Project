# MAC_LRN-Final-Project

A continuous learning system for person detection using YOLO, where edge devices collect data and the server automatically retrains models.

## Prerequisites

1. **Python 3.8+** installed
2. **YOLO CLI** working: `yolo --help` (should work after installing ultralytics)
3. **Webcam** connected (for edge device)
4. **Internet connection** (for downloading dependencies)

## Installation

1. **Navigate to the project directory:**
   ```bash
   cd MAC_LRN-Final-Project/edge_server
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify YOLO is working:**
   ```bash
   yolo --help
   ```

## Running the System

### Step 1: Start the Server

Open a terminal and start the FastAPI server:

```bash
cd MAC_LRN-Final-Project/edge_server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

You should see output like:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
```

The server is now running at `http://localhost:8000`

### Step 2: Train an Initial Model (First Time Only)

If you don't have a trained model yet, trigger initial training:

**Option A: Using curl (from another terminal):**
```bash
curl -X POST http://localhost:8000/trigger_train
```

**Option B: Using the API docs:**
- Open `http://localhost:8000/docs` in your browser
- Find the `/trigger_train` endpoint
- Click "Try it out" and then "Execute"

This will:
- Train a model using your existing dataset
- Save it to `edge_server/models/{version}_train/weights/best.pt`
- Export to TFLite format

**Note:** Training may take a while depending on your dataset size and hardware.

### Step 3: Run the Edge Device (Continuous Learning)

Open a **new terminal** and navigate to any directory where you want to run detection:

```bash
# Navigate to your desired directory (or stay in current directory)
cd /path/to/your/directory

# Download model and detection script, then run
curl -X POST http://localhost:8000/trigger_train -o get_and_run.py
python get_and_run.py
```

**Or, if you already have the model and script:**

```bash
# Make sure best.pt and run_detection.py are in your current directory
python run_detection.py
```

**Custom Configuration (Optional):**

You can customize the edge device behavior using environment variables:

```bash
# Windows (PowerShell)
$env:SERVER_URL="http://localhost:8000"
$env:SEND_INTERVAL="10"
$env:MIN_CONFIDENCE="0.6"
python run_detection.py

# Linux/Mac
SERVER_URL=http://localhost:8000 SEND_INTERVAL=10 MIN_CONFIDENCE=0.6 python run_detection.py
```

**Configuration Options:**
- `SERVER_URL`: Server address (default: `http://localhost:8000`)
- `SEND_INTERVAL`: Seconds between sending data (default: `5.0`)
- `MIN_CONFIDENCE`: Minimum detection confidence to send (default: `0.5`)
- `SEND_DATA`: Enable/disable data sending (default: `true`)

### Step 4: Monitor the System

**Check collection status:**
```bash
curl http://localhost:8000/collection_status
```

**Check latest model:**
```bash
curl http://localhost:8000/latest_model
```

**View API documentation:**
- Open `http://localhost:8000/docs` in your browser

## How It Works

### Continuous Learning Flow

1. **Edge Device** runs person detection on webcam feed
2. **Every 5 seconds** (configurable), frames with detections are sent to the server
3. **Server** saves images and annotations in YOLO format:
   - Images → `edge_server/data/dataset/train/images/` or `valid/images/`
   - Labels → `edge_server/data/dataset/train/labels/` or `valid/labels/`
4. **When 100 samples** are collected (configurable), server automatically:
   - Triggers retraining on the combined dataset
   - Trains a new model version
   - The new model is available for download

### File Structure

```
edge_server/
├── main.py                 # FastAPI server
├── train.py                # Training script
├── run_detection.py         # Edge device detection script
├── data/
│   └── dataset/
│       ├── train/          # Training images and labels
│       └── valid/          # Validation images and labels
├── models/                 # Trained models
│   └── {version}_train/
│       └── weights/
│           └── best.pt     # Latest model
└── server.db              # SQLite database for tracking
```

## API Endpoints

- `POST /trigger_train` - Download model and run detection, or start training
- `GET /download_latest_model` - Download the latest `best.pt` model
- `GET /download_detection_script` - Download `run_detection.py`
- `POST /upload_detection_data` - Receive detection data from edge devices
- `GET /collection_status` - Check data collection status
- `GET /latest_model` - Get latest model metadata
- `GET /docs` - Interactive API documentation

## Troubleshooting

### Server won't start
- Check if port 8000 is already in use
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

### Edge device can't connect to server
- Verify server is running: `curl http://localhost:8000/collection_status`
- Check `SERVER_URL` environment variable
- Check firewall settings

### No detections being sent
- Verify webcam is working
- Check `MIN_CONFIDENCE` setting (may be too high)
- Check `SEND_DATA` is not set to `false`
- Look for error messages in the edge device terminal

### Training fails
- Verify dataset YAML exists: `edge_server/data/dataset/data.yaml`
- Check that train/valid directories have images and labels
- Ensure YOLO CLI works: `yolo --help`

## Stopping the System

1. **Stop edge device:** Press `x` in the detection window, or `Ctrl+C` in terminal
2. **Stop server:** Press `Ctrl+C` in the server terminal

## Next Steps

- Adjust `COLLECTION_CONFIG` in `main.py` to change retraining threshold
- Modify training parameters in `train.py` (epochs, batch size, etc.)
- Add more classes or customize detection logic
