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

Work started in `vision/ai/web/report_generator.py` and is currently paused.
The implementation so far adds deterministic report interpretation helpers and
reorganizes the generated PDF around management-facing sections while retaining
technical evidence in an appendix. No frontend changes are planned because the
existing report download endpoints call the same generator functions.

Completed so far:

- Added helper functions for numeric parsing, metric selection, weak-class
  detection, result/recommendation text, dataset quality summaries, training
  behaviour summaries, and validation-vs-test comparison text.
- Added an Executive Summary section that can prefer test metrics when a
  combined training-and-test report is generated.
- Added Model and Dataset Overview, Dataset Quality and Risks, Training
  Configuration, Training Behaviour, Validation Performance, Qualitative
  Results, Operational Performance and Limitations, Conclusion and
  Recommendation, and Technical Appendix sections.
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

Remaining before completion:

- Generate at least one sample training PDF and one combined training/test PDF
  to catch ReportLab layout issues.
- Review the report visually for section ordering, page breaks, long table
  wrapping, and repeated plots.
- Decide whether the weak-class and recommendation thresholds should remain
  fixed in the generator or become configurable constants.
- Add focused tests for the new helper functions and report section generation.
- Consider capturing inference speed explicitly during validation/test runs;
  the current operational section can report artifact size, target hardware,
  thresholds, and limitations, but not measured deployment speed.
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

### 9. Operational Performance and Limitations

Add deployment-relevant information containing:

- Inference speed, model size, and target hardware
- Confidence and NMS thresholds
- Data or operating conditions that were not tested

### 10. Conclusion and Recommendation

End the main report with:

- Summary of overall model performance
- Key strengths observed during evaluation
- Classes or scenarios requiring additional attention
- Overall training and validation behaviour

### 11. Technical Appendix

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
