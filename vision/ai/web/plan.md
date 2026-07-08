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
- Multi-label stratified train/validation/test split configuration
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

## Training Report Improvement Plan

The generated training report should be reorganized for management review while
retaining detailed technical evidence for engineering review. This section is a
living implementation plan for `report_generator.py`.

### Implementation Status

The implementation in `vision/ai/web/report_generator.py` now adds
deterministic report interpretation helpers and reorganizes the generated PDF
around management-facing sections while retaining technical evidence in an
appendix. The Training Results UI also displays overall precision and recall
from the same run metrics used by the report.

Completed so far:

- Added helper functions for numeric parsing, metric selection, weak-class
  detection, result/recommendation text, dataset quality summaries, training
  behaviour summaries, and validation-vs-test comparison text.
- Added an Executive Summary section that can prefer test metrics when a
  combined training-and-test report is generated.
- Added Model and Dataset Overview, Dataset Quality and Risks, Training
  Configuration, Training Behaviour, Validation Performance, Qualitative
  Results, Conclusion and Recommendation, and Technical Appendix sections.
- Added original-vs-exported dataset split reporting when original split
  information is available.
- Added comma formatting for count-style values to improve readability.
- Removed the Operational Performance and Limitations section from the generated
  report.
- Added overall precision and recall cards to the Training Results UI.
- Updated combined reports so the independent test set is treated as the
  primary evidence when test metrics are available.
- Preserved `_add_dataset()` so the existing dataset provenance tests continue
  to pass.

Verified so far:

- `PYTHONDONTWRITEBYTECODE=1 python3 -c "import vision.ai.web.report_generator"`
  passed.
- `python3 -m pytest vision/ai/web/test_dataset_provenance.py` passed.
- `python3 -m py_compile vision/ai/web/report_generator.py` could not complete
  because Python could not write to the existing `vision/ai/web/__pycache__`
  directory.
- Temporary sample training and combined training/test PDFs generated
  successfully with ReportLab.

Remaining follow-up items:

- Review the report visually for section ordering, page breaks, long table
  wrapping, and repeated plots.
- Decide whether the weak-class and recommendation thresholds should remain
  fixed in the generator or become configurable constants.
- Add focused tests for the new helper functions and report section generation.
- Improve qualitative examples if dedicated success, false-positive,
  false-negative, and difficult-case artifacts become available.

### 1. Executive Summary

Add a concise first-page summary containing:

- Dataset size and number of classes
- Best checkpoint and epoch
- mAP50, mAP50-95, precision, recall, and F1
- A one-sentence result and recommendation

### 2. Model and Dataset Overview

Combine the current model and dataset information into one section containing:

- YOLOv8 model variant
- Original and augmented image counts
- Train, validation, and test split counts and percentages
- Augmentation multiplier
- Class names and class distribution
- Dataset source, version, and preparation date

### 3. Dataset Quality and Risks

Add a concise assessment containing:

- Class imbalance summary
- Missing or invalid label counts
- Image and object-instance counts per class
- Underrepresented classes and their associated evaluation risk

## Additional Object Detection Backend Plan

### Objective

Add compatibility for three additional object detection model families while
keeping the current YOLO and RF-DETR workflows stable:

1. YOLO-to-COCO converter
2. D-FINE-N backend
3. RT-DETRv2-S backend
4. LW-DETR-T backend

The web UI should continue to accept the current prepared YOLO detection
dataset layout, then adapt it internally for backends that expect COCO-style
annotations.

```text
Prepared YOLO dataset
  data.yaml
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels

→ backend-specific adapter
→ backend-specific trainer
→ normalized web UI run output
```

The first implementation should keep dataset preparation source-neutral and
format-stable. Dataset preparation always produces the canonical YOLO dataset;
the selected training backend decides at training time whether it can use YOLO
directly or needs a COCO conversion.

```text
Roboflow / ZIP / Folder
→ prepared YOLO dataset
→ data.yaml + train/valid/test labels
→ selected backend needs COCO
→ backend converts YOLO labels to COCO JSON
→ train D-FINE / RT-DETRv2 / LW-DETR
```

Roboflow-native COCO export can be considered later as an optimization, but the
baseline compatibility path should rely on local YOLO-to-COCO conversion so
uploaded ZIP and folder datasets work the same way as Roboflow datasets.

All new backends should expose the same UI-facing run structure:

```text
runs/<family>/<run-name>/
  weights/
    best.pt
    last.pt
  results.csv
  web_metrics.json
  training_report_context.json
```

If a backend produces native `.pth` checkpoints, either copy/symlink the
selected checkpoints into the normalized `weights/` directory or extend the UI
to recognize `.pth` as a first-class trained-weight artifact.

### 1. YOLO-to-COCO Converter

Add a reusable converter, for example:

```text
vision/ai/train/yolo_to_coco.py
```

Responsibilities:

- Read the prepared YOLO `data.yaml`.
- Resolve train, validation, and test image/label directories.
- Read class names from `names`.
- Convert YOLO normalized boxes:

```text
class_id x_center y_center width height
```

to COCO boxes:

```text
[x_min, y_min, width_px, height_px]
```

- Preserve image dimensions, image IDs, annotation IDs, category IDs, and class
  names.
- Skip or report malformed labels without silently corrupting the dataset.
- Emit a conversion summary with image counts, annotation counts, missing label
  counts, malformed rows, and unknown class IDs.

Recommended output layout:

```text
datasets/coco/<dataset-name>/
  train2017/
  val2017/
  test2017/
  annotations/
    instances_train2017.json
    instances_val2017.json
    instances_test2017.json
  conversion_summary.json
```

Validation checks:

- Every COCO image entry points to an existing image.
- Every bbox has positive width and height.
- Category IDs are stable and match the original YOLO class IDs.
- Empty-label images remain valid negative examples.

### 2. D-FINE-N Backend

Use the official D-FINE object detection repository:

```text
https://github.com/Peterande/D-FINE
```

Target model:

```text
D-FINE-N
```

Add a backend runner, for example:

```text
vision/ai/train/train_dfine.py
```

Responsibilities:

- Accept common web UI training arguments:
  - dataset path
  - epochs
  - image size
  - batch size
  - workers
  - device
  - project
  - run name
  - resume flag
- Convert the prepared YOLO dataset to COCO using the shared converter.
- Generate or patch a D-FINE custom config for the number of classes, dataset
  paths, image size, batch size, workers, and output directory.
- Launch D-FINE training with the official command pattern:

```bash
torchrun train.py \
  -c configs/dfine/custom/dfine_hgnetv2_n_custom.yml \
  --use-amp \
  --seed <seed>
```

- Support tuning from pretrained weights when available.
- Capture logs and emit web UI progress markers.
- Normalize checkpoints into:

```text
runs/dfine/<run-name>/weights/best.pt
runs/dfine/<run-name>/weights/last.pt
```

- Convert D-FINE logs/metrics into `results.csv` and `web_metrics.json`.

UI integration:

- Add model selector entry:

```text
Detection - D-FINE
  D-FINE Nano - dfine-n
```

- Add model registry entry:

```text
family: dfine
task: detect
model: dfine-n
runner: train_dfine.py
project_default: runs/dfine
dataset_format: coco
```

Inference/testing:

- Add `vision/ai/web/infer_dfine.py`.
- Route selected `runs/dfine/.../weights/best.pt` weights through the D-FINE
  inference adapter.
- Return the same inference JSON shape as YOLO/RF-DETR.

Docker considerations:

- Install D-FINE dependencies.
- Decide whether the D-FINE repo is vendored under `third_party/`, cloned at
  image build time, or installed as a package if packaging becomes available.

### 3. RT-DETRv2-S Backend

Use the official RT-DETR repository:

```text
https://github.com/lyuwenyu/RT-DETR
```

Target model:

```text
RT-DETRv2-S
```

This is different from Ultralytics RT-DETR-L/X support. Ultralytics support is
not enough for RT-DETRv2-S specifically, so this should be treated as a
separate backend family.

Add a backend runner, for example:

```text
vision/ai/train/train_rtdetrv2.py
```

Responsibilities:

- Accept common web UI training arguments.
- Convert prepared YOLO datasets to COCO.
- Generate or patch the RT-DETRv2-S config for:
  - class count
  - COCO annotation paths
  - image folders
  - batch size
  - epoch count
  - image size
  - output directory
  - pretrained checkpoint
- Launch official RT-DETRv2 PyTorch training.
- Capture logs and emit web UI progress markers.
- Normalize checkpoints into:

```text
runs/rtdetrv2/<run-name>/weights/best.pt
runs/rtdetrv2/<run-name>/weights/last.pt
```

- Convert validation metrics into the shared `results.csv` and
  `web_metrics.json` format.

UI integration:

- Add model selector entry:

```text
Detection - RT-DETRv2
  RT-DETRv2 Small - rtdetrv2-s
```

- Add model registry entry:

```text
family: rtdetrv2
task: detect
model: rtdetrv2-s
runner: train_rtdetrv2.py
project_default: runs/rtdetrv2
dataset_format: coco
```

Inference/testing:

- Add `vision/ai/web/infer_rtdetrv2.py`.
- Support image/video inference and prepared test split evaluation.
- Normalize predictions to the same UI inference result JSON shape.

Docker considerations:

- Install RT-DETR repo dependencies.
- Confirm CUDA/PyTorch compatibility with the existing CUDA base image.
- Decide how pretrained weights are downloaded/cached.

### 4. LW-DETR-T Backend

Use the official LW-DETR repository:

```text
https://github.com/Atten4Vis/LW-DETR
```

Target model:

```text
LW-DETR-tiny
```

Use `lwdetr-tiny` or `lwdetr-t` consistently in the UI, but document that this
maps to the official LW-DETR tiny model.

Add a backend runner, for example:

```text
vision/ai/train/train_lwdetr.py
```

Responsibilities:

- Accept common web UI training arguments.
- Convert the prepared YOLO dataset to COCO2017-style layout.
- Generate or patch LW-DETR tiny training config/script arguments.
- Launch LW-DETR training through the official scripts or equivalent Python
  entrypoint.
- Capture logs and emit web UI progress markers.
- Normalize checkpoints into:

```text
runs/lwdetr/<run-name>/weights/best.pt
runs/lwdetr/<run-name>/weights/last.pt
```

- Convert training/validation metrics into `results.csv` and
  `web_metrics.json`.

UI integration:

- Add model selector entry:

```text
Detection - LW-DETR
  LW-DETR Tiny - lwdetr-tiny
```

- Add model registry entry:

```text
family: lwdetr
task: detect
model: lwdetr-tiny
runner: train_lwdetr.py
project_default: runs/lwdetr
dataset_format: coco
```

Inference/testing:

- Add `vision/ai/web/infer_lwdetr.py`.
- Support image/video inference and prepared test split evaluation.
- Normalize predictions to the common inference JSON shape.

Docker considerations:

- LW-DETR is likely the most Docker-heavy backend.
- Add compiler/build tooling if CUDA operators must be compiled.
- Confirm compatibility with the existing CUDA runtime image. If build tools are
  required, consider changing the training image from a runtime CUDA image to a
  devel CUDA image or using a multi-stage build.
- Cache compiled operators and pretrained weights where practical.

### Shared Backend Integration Tasks

Update backend registry and run discovery:

- Add project defaults:

```text
runs/dfine
runs/rtdetrv2
runs/lwdetr
```

- Include the new roots in training session discovery.
- Include the new roots in inference weight discovery.
- Track `family` in `training_report_context.json`.

Update metrics/reporting:

- Add family-aware metric readers.
- Keep the report generator consuming the normalized metric shape rather than
  backend-specific files.
- Include backend family and native model name in generated reports.

Update inference routing:

- Route selected weights by `family`.
- Keep uploaded weights defaulting to Ultralytics unless a user-facing selector
  is added for uploaded weight family.

Update tests:

- Add converter unit tests with small synthetic YOLO datasets.
- Add model registry tests for each backend entry.
- Add run discovery tests for the new `runs/<family>` roots.
- Add metrics parser tests with sample backend logs/results.
- Add inference routing tests that verify family-specific adapters are selected.

### Recommended Implementation Order

1. Add and test the YOLO-to-COCO converter.
2. Add D-FINE-N backend.
3. Add RT-DETRv2-S backend.
4. Add LW-DETR-T backend.

This order keeps the first milestone focused on reusable dataset conversion,
then adds the clearest custom-dataset backend before moving to the heavier
RT-DETRv2 and LW-DETR integrations.

### 4. Training Configuration

Keep only the settings needed to understand the experiment in the main report:

- Epochs, image size, and batch size
- Optimizer and learning-rate schedule
- Early-stopping patience
- Transfer-learning checkpoint
- Runtime augmentation settings
- Random seed and deterministic mode

Move the complete hyperparameter table to the technical appendix.

### 5. Training Behaviour

Present the training curves before the final evaluation results:

- Training and validation loss
- mAP progression
- Best epoch
- Early-stopping outcome
- A short interpretation of possible overfitting or underfitting

### 6. Validation Performance

Retain and clarify the validation evidence:

- Precision, recall, and F1
- mAP50 and mAP50-95
- Per-class AP and recall
- Confusion matrix
- ROC-AUC only when it supports a meaningful decision

Explicitly highlight weak classes so strong weighted averages do not conceal
class-level failures.

### 7. Test-Set Evaluation

Add a separate evaluation of the untouched test set containing:

- Final overall test metrics
- Per-class test performance
- Test confusion matrix
- Comparison between validation and test results

Use the test-set results as the primary evidence for final model performance.

### 8. Qualitative Results

Add representative annotated examples showing:

- Successful detections
- False positives
- False negatives
- Difficult cases such as overlap, distance, blur, and poor lighting

### 9. Conclusion and Recommendation

End the main report with:

- Summary of overall model performance
- Key strengths observed during evaluation
- Classes or scenarios requiring additional attention
- Overall training and validation behaviour

### 10. Technical Appendix

Move detailed supporting material to an appendix:

- Complete hyperparameter table
- Runtime environment details
- Full class statistics
- Additional plots
- Internal run paths and metadata

### Content to Remove or Consolidate

- Remove repeated metrics and duplicated explanatory text.
- Omit empty or unavailable fields.
- Reduce internal file paths in the main report and retain them in the appendix.
- Omit ROC material when it does not contribute to the evaluation decision.
