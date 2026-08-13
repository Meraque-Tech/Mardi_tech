# YOLOv8 Training Web UI

This folder contains a FastAPI web UI for preparing YOLO datasets, starting
YOLOv8 training, monitoring metrics/logs, and downloading training artifacts.
Training is launched through:

```text
vision/ai/train/train_yolov8.py
```

## Files

```text
app.py                       FastAPI backend
report_generator.py          Training and combined test PDF reports
static/                      HTML, CSS, and JavaScript UI
requirements.txt             Python dependencies
.env                         Local Roboflow/runtime settings
.env.example                 Example environment settings
Dockerfile.cuda              CUDA container image
docker-compose.train_web.yml CUDA compose launcher
```

The `.env` file is ignored by Git.

## Environment

Create or edit `.env` when you want Roboflow defaults or runtime defaults:

```text
ROBOFLOW_API_KEY=
ROBOFLOW_WORKSPACE=
ROBOFLOW_PROJECT=
ROBOFLOW_VERSION=

WEB_DATA_ROOT=
TRAINING_PYTHON=python3
TRAINING_DEVICE=
SAM_QA_DEVICE=
SAM3_QA_MODEL_PATH=
HOST_UID=1000
HOST_GID=1000
```

Only `ROBOFLOW_API_KEY` and `ROBOFLOW_WORKSPACE` are required when using
Roboflow from environment defaults. `ROBOFLOW_PROJECT` and
`ROBOFLOW_VERSION` can also be entered in the UI.

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

The Dataset, Training, GPU Monitor, Advanced Fine-Tuning, Training Progress &
Logs, and Training Results panels use the same collapsible chevron control.
Their open/closed states are saved in browser storage. Logs and Results open
automatically when training starts, Results opens again when a run completes,
and reopening Results redraws its charts.

Completed training runs provide a **Download Report** action containing the
dataset snapshot, annotated class samples, hyperparameters, training and
validation metrics, per-class results, and available plots. A test run made
with a training run's `best.pt` also provides **Download Report (including
test)** with a separate test section and validation-versus-test comparison.
Tests using standalone uploaded weights are not presented as linked training
runs, so the combined report remains unavailable for those tests.

### Annotation QA

The optional SAM Annotation QA panel compares YOLO detection boxes with
prompted SAM masks. `Box tolerance (%)` keeps the original YOLO box when every
SAM box edge is within that percentage of the YOLO box dimensions, with a
two-pixel minimum allowance for small boxes. Differences up to `Maximum SAM
difference (%)` are reviewable SAM suggestions. Larger differences preserve
YOLO and require manual review; they cannot be queued through the normal safe
SAM action. A reviewer can explicitly choose **Use displayed SAM box anyway**
under **More actions** when a mapped SAM box is available. This requires a
warning confirmation, uses the displayed SAM box, and records the correction
as a human override. SAM replacements inside the review band must also pass
confidence, overlap, coverage, center-shift, area, and prompt-stability checks
unless a reviewer explicitly overrides them. `Shadow` mode
(recommended) calculates strict automatic decisions without queueing them;
`Automatic with audit` queues only strict candidates and requires sampled
audit decisions to be reviewed before a corrected dataset can be created.
`Manual review only` preserves the original review workflow. All corrected
datasets are copy-on-write, and reports created before the current
prompt-mapping, difference-band, and stability safeguards must be rerun before
SAM box corrections can be accepted.

SAM 3 is available as a separate local QA backend when `sam3.pt` exists at
the repository root or at `SAM3_QA_MODEL_PATH`. It requires Ultralytics
8.3.237 or newer. The 16 GB profile caps inference at SAM 3's stride-aligned
1008 pixels, starts with eight prompts per chunk, uses BF16 when supported
(otherwise FP16), reuses the encoded image, and halves the chunk after a CUDA
out-of-memory error. Original
box prompts run first; expanded and jittered stability prompts run only for
reviewable correction candidates. SAM 3 is limited to suggestions or manual
review until its automatic-correction thresholds are calibrated. Runtime
precision, final chunk size, resize count, and peak allocated VRAM are stored
in the QA report.

Datasets fetched from Roboflow also store a strict workspace/project/version
manifest with unique source image IDs. After creating a corrected dataset, use
`Preview Roboflow Changes` to check for upstream annotation conflicts and then
`Publish to Bound Project` to replace only conflict-free annotations in that
same source project. The API key is used for the request but is not stored in
the manifest or audit log. Generated Roboflow versions remain immutable, so a
new version must be generated after publishing before training on Roboflow.
Older downloads without the source manifest must be fetched again before they
can be published.

## Run With CUDA Container

Use this when you want the UI and training process to run inside a CUDA-enabled
container.

From `vision/ai/web`, use the run script:

```bash
./run.sh build     # first run or rebuild dependencies
./run.sh start     # start the existing image
./run.sh restart
./run.sh status
./run.sh logs
./run.sh stop
```

From the repository root:

```bash
docker compose -f vision/ai/web/docker-compose.train_web.yml up --build
```

Or from `vision/ai/web`:

```bash
docker compose -f docker-compose.train_web.yml up --build
```

Then open:

```text
http://localhost:8000
```

The host machine must have NVIDIA drivers and NVIDIA Container Toolkit
configured. The compose file mounts the repository into `/app`, maps
`vision/ai/web/logs`, maps `runs`, and uses host networking. Uvicorn therefore
listens directly on the host's port `8000` without a Compose port mapping. The
service also sets:

```yaml
shm_size: "8gb"
```

The image creates an `appuser` account using `HOST_UID:HOST_GID` and runs the
service as that named user. Files created in bind-mounted datasets, logs, and
run directories therefore remain editable by the host user, while libraries
that require a valid container username continue to work. The defaults are
`1000:1000`; set these values in `.env` when the host user has different IDs.

Because the source tree is mounted into the container, most Python/HTML/CSS/JS
changes only require a container restart:

```bash
docker compose -f docker-compose.train_web.yml restart
```

Rebuild only when dependencies, the Dockerfile, or image-level setup changes.

### Baked Pretrained Weights

The CUDA image bakes a small default set of pretrained weights during build so
new training machines can start common jobs without a first-run model download:

```text
rf-detr-nano.pth
yolo26n.pt
yolo26s.pt
yolov8n.pt
yolov8s.pt
yolo11n.pt
yolo11s.pt
yolov8n-seg.pt
yolo26n-seg.pt
sam2.1_s.pt
sam2.1_t.pt
```

The files are stored outside `/app` because `/app` is bind-mounted from the
host at runtime:

```text
/home/appuser/.roboflow/models/
/home/appuser/.cache/ultralytics/weights/
```

Disable the bake when building a smaller image:

```bash
BAKE_PRETRAINED_WEIGHTS=0 docker compose -f docker-compose.train_web.yml build
```

## Dataset Options

The UI supports three dataset sources.

### Upload ZIP

This is the default option. Upload a ZIP file containing a YOLO dataset. The
backend extracts it and generates a dataset YAML.

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

Existing train/validation/test folders are preserved by default. Flat datasets
use the default 70/15/15 split automatically. Enable `Rebuild existing
train/val/test split` only when you want to replace an existing split or choose
different percentages.

### Upload Folder

Select a dataset folder from your local computer. The browser uploads the files
to the backend, then the backend prepares the dataset in the same way as ZIP
uploads.

This uses browser folder upload, so very large folders can hit browser/server
upload limits. If that happens, ZIP the dataset and use `Upload ZIP`.

ZIP and folder preparation show measured, stage-specific percentages for the
upload, server-side save, ZIP extraction, stratification, split-file copying,
and dataset inspection. Stages that do not apply to the selected dataset are
skipped.

### Roboflow

Fill in Roboflow credentials in `.env`, or enter workspace/project/version in
the UI. The backend always downloads the dataset in YOLOv8 format and uses the
split configured in that Roboflow dataset version.

Roboflow preparation uses hybrid progress reporting. Server-side version and
export generation show Roboflow's reported percentage when available, or an
indeterminate animation otherwise. The ZIP transfer reports downloaded bytes
when `Content-Length` is available, and extraction, optional split rebuilding,
copying, and inspection use measured local progress. Older SDK versions fall
back to the standard Roboflow downloader with indeterminate progress.

After download, the backend validates the exported `data.yaml`. If Roboflow's
relative paths do not resolve but standard `train`, `valid`/`val`, and `test`
folders are present, the backend generates a normalized prepared YAML without
modifying the original export.

Enable `Rebuild downloaded train/val/test split` to combine the downloaded
splits and apply the percentages configured in the UI. This rebuild is local;
it does not modify the Roboflow dataset version. Avoid rebuilding augmented
datasets when related copies could be assigned to different splits.

## Class IDs

Class IDs are one class name per line. The line order becomes the numeric class
ID.

Example:

```text
bee
drone
pollenbee
queen
```

This generates:

```yaml
names:
  0: bee
  1: drone
  2: pollenbee
  3: queen
```

If the dataset already contains `data.yaml` or `dataset.yaml`, you can leave the
class list empty and click `Prepare Dataset`; the backend will read class names
from the YAML. For folder uploads, `Auto Fetch` can read classes from the
selected folder before upload.

## Split Validation

The UI checks that train/val/test percentages total exactly `100%` before
preparing a dataset.

Default split:

```text
train 70%
val   15%
test  15%
```

Validation data is used during training for metrics and early stopping. Test
data is reserved for final evaluation when present in the dataset.

After preparation, click `Download ZIP` in the Dataset panel to download the
exact prepared train/validation/test split. The archive contains the images,
labels, split metadata, and a portable `data.yaml` whose dataset path is `.`.
Large archives are created as temporary server files and removed after the
browser starts the download.

When rebuilding a split, the backend uses deterministic multi-label
stratification with seed `42`. It balances the image-level presence of every
class across train/validation/test, uses object-instance counts as a
tie-breaker, and uses largest-remainder rounding so the split counts add up
exactly. Classes with too few images to appear in every split are prioritized
for train, then validation, then test, and are reported in the dataset summary.

## Training

1. Choose a dataset source.
2. Enter or auto-fetch class names.
3. Set train/val/test percentages if rebuilding the split.
4. Click `Prepare Dataset`.
5. Expand the dataset summary to review split counts, per-class image and instance distribution, and warnings.
6. Choose model size:
   - Nano: `yolov8n.pt`
   - Small: `yolov8s.pt`
   - Medium: `yolov8m.pt`
   - Large: `yolov8l.pt`
   - Extra large: `yolov8x.pt`
7. Set training parameters.
8. Click `Start`.

The backend runs `train_yolov8.py` in the background.

## Training Parameters

Basic controls:

```text
model size   YOLOv8 checkpoint size
device       GPU/CPU selector, for example 0 or cpu
epochs       maximum number of training epochs
image size   training image size
batch        images per training step
patience     early stopping patience
save period  extra checkpoint interval; -1 keeps standard best.pt/last.pt
project      output directory, default runs/detect
run name     base output run name; new runs append a MYT timestamp
resume       continue from last.pt in the selected project/run folder
```

For a new training run, the backend treats `Run name` as a base name and
appends a Malaysia-time timestamp. For example, `train` becomes:

```text
train-YYYYMMDD-HHMMSS
```

If the field already ends with that timestamp pattern, the old suffix is
replaced with a fresh timestamp. After training starts, the UI updates the
`Run name` field to the resolved timestamped name. Resume mode keeps the
selected run name unchanged so the backend can find the existing
`weights/last.pt` checkpoint.

Advanced controls:

```text
workers        dataloader worker count
optimizer      auto, SGD, Adam, AdamW, NAdam, RAdam, RMSProp
seed           reproducibility seed
lr0            initial learning rate
lrf            final learning-rate factor
weight decay   regularization strength
warmup epochs  learning-rate warmup duration
freeze layers  freeze the first N layers
cosine LR      enable cosine learning-rate schedule
exist_ok       reuse the same output folder instead of train-2/train-3
```

Training launched from the web UI always starts from the pretrained weights
for the selected YOLOv8 model size and uses YOLOv8's default SiLU activation.

The defaults shown when the page loads are also the backend defaults: Nano,
100 epochs, image size 640, batch 16, patience 20, 2 workers, Adam, seed 42,
and initial learning rate 0.001.

The UI also includes presets:

```text
Stable
Low VRAM
Quick Test
High Accuracy
```

## GPU Monitor

The GPU Monitor beneath the Training panel reports NVIDIA GPU utilization,
VRAM usage, temperature, and power through `nvidia-smi`. It refreshes with the
existing training-status poll and supports multiple GPUs. When NVIDIA telemetry
is unavailable, the panel shows a diagnostic message without blocking dataset
preparation or training.

## Logs

Current log:

```text
vision/ai/web/logs/current.log
```

Each training session also writes a timestamped log:

```text
vision/ai/web/logs/train-YYYYMMDD-HHMMSS.log
```

The UI can show:

```text
Recent    latest log tail
Full      full current log
Warnings  filtered warnings/errors/tracebacks
```

It can also download the timestamped log for the current or most recent run.

## Metrics

The UI reads Ultralytics `results.csv`, saved web metrics, and validation
artifacts to show:

```text
Macro F1
Weighted F1
Per-class F1
Per-class AP50 and AP50-95
Training loss
Validation loss
mAP50
mAP50-95
Detection performance graph by epoch
Loss graph by epoch
Raw and normalized validation confusion matrices
Validation ROC-AUC curve and per-class AUC summary
Best epoch summary, including lowest training and validation losses
```

`Macro F1` is the equal-weight mean of the final per-class F1 scores. `Weighted F1`
uses the same per-class scores weighted by class instance counts. Both are
saved into `web_metrics.json` after training completes.

For Roboflow exports, generated PDF reports distinguish the original split
before offline augmentation from the exported files used by YOLOv8. The report
records the training-output multiplier and labels reconstructed counts as
estimated when the exported training count is not evenly divisible by it.

When Ultralytics has generated `confusion_matrix.png` and
`confusion_matrix_normalized.png`, the UI displays both plots for the resolved
training run. The x-axis is labeled `Actual` and the y-axis is labeled
`Predicted`. Click either matrix to open its full-resolution image.

The ROC-AUC view is image-level one-vs-rest on the validation split. For each
class, the score is the highest predicted confidence for that class in an image,
and the target is whether that class appears anywhere in the image. This makes
ROC-AUC well-defined for the UI while the AP columns remain box-level detection
metrics from YOLO validation.

RF-DETR runs write the same web-facing `results.csv` and `web_metrics.json`
files, but the values are normalized from RF-DETR/Lightning CSV artifacts when
available and from the RF-DETR validation console table otherwise. RF-DETR does
not generate ROC-AUC or validation confusion-matrix artifacts in this runner, so
the UI and PDF report show those as unavailable instead of pending.

D-FINE-N is exposed as a separate detection backend. The web UI still prepares a
YOLO-format dataset, then `train_dfine.py` converts it to a COCO2017-style
layout before launching the official D-FINE trainer. The CUDA image installs the
official D-FINE checkout under `/opt/D-FINE` by default and sets
`DFINE_REPO_DIR` to that path. For manual setups, set `DFINE_REPO_DIR` to an
official D-FINE checkout, or vendor it under `third_party/D-FINE`, before
starting a D-FINE run. The runner normalizes available outputs back into
`runs/dfine/<run-name>/weights/`, `results.csv`, and `web_metrics.json`.

## Outputs And Downloads

Training outputs are saved under the selected project and resolved run name. By
default, a new detection run is saved as:

```text
runs/detect/train-YYYYMMDD-HHMMSS/
```

Useful checkpoint files:

```text
runs/detect/train-YYYYMMDD-HHMMSS/weights/best.pt
runs/detect/train-YYYYMMDD-HHMMSS/weights/last.pt
```

The UI can download:

```text
prepared-dataset.zip
best.pt
last.pt
results.csv
accuracy_by_epoch.png
loss_by_epoch.png
train-YYYYMMDD-HHMMSS.log
```

Use `best.pt` for inference or deployment. Use `last.pt` to resume training
from the most recent checkpoint.

## Resume Training

To resume from the selected project/run:

1. Keep `Project` and `Run name` pointing at the previous timestamped run.
2. Enable `Resume latest checkpoint`.
3. Click `Start`.

The UI enables Resume only when the resolved run contains `weights/last.pt`.
The backend validates that checkpoint again, loads it as the model, and then
passes `--resume` to restore its dataset, epoch, optimizer, and scheduler
state. Preparing the dataset again is not required for a resume.

## Stop Training

Click `Stop` to send a stop signal to the running training process.

After stopping, resume from the run's latest checkpoint when available:

```text
runs/detect/train-YYYYMMDD-HHMMSS/weights/last.pt
```

## Git-Ignored Runtime Files

Uploaded datasets, prepared datasets, logs, generated runs, and model weights
should not be committed. The repository ignores runtime artifacts such as:

```text
vision/ai/web/datasets/
vision/ai/web/logs/
runs/
*.pt
```

## Notes

- Roboflow download requires network access and valid Roboflow credentials.
- The web browser cannot write directly to arbitrary folders without browser
  support. For weight downloads, supported browsers may prompt for a directory;
  otherwise the normal browser download flow is used.
- If the UI looks stale after a restart, hard-refresh the browser page.
