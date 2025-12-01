# MAC_LRN-Final-Project

## How to run the server

1. Make sure `yolo` CLI works: `yolo --help`

2. Start API:
``` bash
cd MAC_LRN-Final-Project/edge_server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

3. Prepare dataset YAML: `edge_server/data/dataset/data.yaml` with paths to `train/val` `images/files` according to ultralytics format.

4. Trigger training (from another shell or via the endpoint):
```bash
curl -X POST http://localhost:8000/trigger_train
```

This runs training asynchronously, exports TFLite,0 and writes it into `models/<version>/model.tflite`. `main.py` serves the model via `/latest_model` and `/download_model/<version>`.