# YOLOv8 Training Web UI Plan

## Objective

Build a simple web UI in `vision/ai/web/` for preparing datasets and launching
YOLOv8 object detection training through the existing training script:

```text
vision/ai/train/train_yolov8.py
```

The UI should support:

- Dataset folder upload
- Dataset ZIP upload
- Roboflow dataset download
- Train/validation/test split configuration
- Class ID and class name configuration
- YOLOv8 model size selection
- CUDA/GPU training from a container
- Start, stop, resume, status, and log viewing

## Architecture

Use FastAPI as the backend and a static HTML/CSS/JavaScript frontend.

```text
vision/ai/web/
  app.py
  requirements.txt
  .env
  .env.example
  .gitignore
  Dockerfile.cuda
  docker-compose.train_web.yml
  static/
    index.html
    styles.css
    app.js
```

## Backend Responsibilities

The FastAPI backend will:

1. Serve the web UI.
2. Accept a dataset folder uploaded through the browser.
3. Accept and extract uploaded YOLO dataset ZIP files.
4. Download Roboflow datasets using credentials from `.env`.
5. Generate a YOLO dataset YAML file.
6. Optionally create train/val/test splits.
7. Start training by running `train_yolov8.py` as a subprocess.
8. Stop the current training process.
9. Resume training from `last.pt`.
10. Return training status and logs to the UI.

## Dataset Sources

### Upload Folder

The user selects a dataset folder in the browser and uploads its files to the
FastAPI service.

Supported layouts:

```text
dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Or:

```text
dataset/
  images/
  labels/
```

If the dataset is flat, the backend can create a train/val/test split.

### Upload ZIP

The user uploads a ZIP file containing a YOLO dataset. The backend extracts it
under:

```text
vision/ai/web/datasets/uploads/
```

Then it prepares a generated dataset YAML.

### Roboflow

The backend reads Roboflow credentials from `.env`:

```text
ROBOFLOW_API_KEY=
ROBOFLOW_WORKSPACE=
ROBOFLOW_PROJECT=
ROBOFLOW_VERSION=
```

The backend downloads the selected dataset version using the Roboflow Python
SDK and uses the downloaded YOLOv8 dataset YAML when available.

## Training Flow

The UI sends training settings to:

```text
POST /api/train/start
```

The backend starts:

```bash
python3 vision/ai/train/train_yolov8.py \
  --data <dataset.yaml> \
  --model <yolov8-size>.pt \
  --epochs <epochs> \
  --imgsz <image-size> \
  --batch <batch-size> \
  --patience <patience> \
  --save-period <save-period> \
  --device <device> \
  --project <output-project> \
  --name <run-name>
```

Training runs in the background. Logs are written to:

```text
vision/ai/web/logs/current.log
```

The UI polls:

```text
GET /api/train/status
GET /api/train/logs
```

## CUDA Container

The web app is intended to run inside a CUDA-enabled container. The container
installs FastAPI, Ultralytics, Roboflow support, and starts the FastAPI server.

The host should run it with GPU access, for example:

```bash
docker compose -f vision/ai/web/docker-compose.cuda.yml up --build
```

The compose file uses GPU device reservations so training can use CUDA when the
host supports NVIDIA Container Toolkit.

## Implementation Steps

1. Create FastAPI backend in `app.py`.
2. Add dataset preparation helpers for folder and ZIP uploads.
3. Add Roboflow download endpoint using `.env` credentials.
4. Add training subprocess management.
5. Add status, logs, stop, and resume endpoints.
6. Create a simple single-page frontend.
7. Add `.env`, `.env.example`, `.gitignore`, `requirements.txt`.
8. Add CUDA Dockerfile and Docker Compose file.
9. Run syntax checks.
10. Document how to launch and use the UI.
