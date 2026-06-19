# YOLOv8 Training Web UI

This folder contains a simple FastAPI web UI for preparing a YOLO dataset and
starting training through:

```text
vision/ai/train/train_yolov8.py
```

## Files

```text
app.py                    FastAPI backend
static/                   HTML, CSS, and JavaScript UI
requirements.txt          Python dependencies
.env                      Local Roboflow/runtime settings
.env.example              Example environment settings
Dockerfile.cuda           CUDA container image
docker-compose.cuda.yml   CUDA compose launcher
plan.md                   Implementation plan
```

The `.env` file is ignored by Git.

## Environment

Edit `.env` if you want to use Roboflow or set a default training device:

```text
ROBOFLOW_API_KEY=
ROBOFLOW_WORKSPACE=
ROBOFLOW_PROJECT=
ROBOFLOW_VERSION=
ROBOFLOW_FORMAT=yolov8

WEB_DATA_ROOT=
TRAINING_PYTHON=python3
TRAINING_DEVICE=
```

`WEB_DATA_ROOT` is optional. If empty, prepared datasets are stored under:

```text
vision/ai/web/datasets/
```

`TRAINING_DEVICE` is optional. Use values like:

```text
0      GPU 0
0,1    multiple GPUs
cpu    CPU only
```

## Run Locally

Use this only after the dependencies in `requirements.txt` are available in
your Python environment.

```bash
python3 -m uvicorn vision.ai.web.app:app --host 0.0.0.0 --port 8000
```

Then open:

```text
http://localhost:8000
```

## Run With CUDA Container

Use this when you want the UI and training process to run inside a CUDA-enabled
container.

```bash
docker compose -f vision/ai/web/docker-compose.train_web.yml up --build
```

Then open:

```text
http://localhost:8000
```

The host machine must have NVIDIA drivers and NVIDIA Container Toolkit
configured.

## Dataset Options

### Local Path

Use this when the dataset already exists on the same machine or inside the same
container running FastAPI.

Supported pre-split layout:

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

Supported flat layout:

```text
dataset/
  images/
  labels/
```

If the dataset is flat, enable rebuild split and set the train/val/test
percentages in the UI.

### Upload ZIP

Upload a ZIP file containing a YOLO dataset. The backend extracts it and
generates a dataset YAML.

The ZIP should contain either a pre-split YOLO dataset or a flat YOLO dataset
with `images/` and `labels/` folders.

### Roboflow

Fill in Roboflow credentials in `.env`, or enter workspace/project/version in
the UI. The backend downloads the dataset using Roboflow format `yolov8` by
default.

## Class IDs

Enter one class name per line in the UI. The line order becomes the class ID.

Example:

```text
flat
missing
ok
```

This generates:

```yaml
names:
  0: flat
  1: missing
  2: ok
```

## Training

1. Choose a dataset source.
2. Enter class names.
3. Set train/val/test split percentages if needed.
4. Click `Prepare Dataset`.
5. Choose model size:
   - Nano: `yolov8n.pt`
   - Small: `yolov8s.pt`
   - Medium: `yolov8m.pt`
   - Large: `yolov8l.pt`
   - Extra large: `yolov8x.pt`
6. Set training parameters.
7. Click `Start`.

The backend runs `train_yolov8.py` in the background and writes logs to:

```text
vision/ai/web/logs/current.log
```

## Outputs

Training outputs are saved under the selected project and run name. By default:

```text
runs/detect/train/
```

Useful checkpoint files:

```text
runs/detect/train/weights/best.pt
runs/detect/train/weights/last.pt
```

Use `best.pt` for inference or deployment. Use `last.pt` to resume training.

## Resume Training

To resume from a previous run:

1. Set model/checkpoint path to `runs/detect/train/weights/last.pt`.
2. Enable `Resume from checkpoint`.
3. Click `Start`.

The UI sends `--resume` to the training script.

## Stop Training

Click `Stop` to send a stop signal to the running training process.

After stopping, resume from:

```text
runs/detect/train/weights/last.pt
```

## Notes

- The browser cannot directly browse your whole computer for folders. Local
  path input must point to a path visible to the FastAPI backend.
- Uploaded datasets, prepared datasets, logs, and `.env` are ignored by Git.
- Roboflow download requires network access and valid Roboflow credentials.
