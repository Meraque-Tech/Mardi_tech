#!/usr/bin/env python3
"""FastAPI web UI backend for YOLOv8 training."""

import csv
import json
import math
import mimetypes
import os
import platform
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Callable, Optional

import yaml
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .dataset_provenance import roboflow_pre_augmentation_summary
from .infer_yolo import InferenceStopped, run_yolo_inference
from .infer_rfdetr import run_rfdetr_inference
from .infer_dfine import run_dfine_inference
from .stratified_split import SPLIT_NAMES, stratified_split
from vision.ai.train.train_rfdetr import RFDETR_RUN_LOG, finalize_rfdetr_artifacts


WEB_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEB_DIR.parents[2]
TRAIN_SCRIPT = REPO_ROOT / "vision" / "ai" / "train" / "train_yolov8.py"
RFDETR_TRAIN_SCRIPT = REPO_ROOT / "vision" / "ai" / "train" / "train_rfdetr.py"
DFINE_TRAIN_SCRIPT = REPO_ROOT / "vision" / "ai" / "train" / "train_dfine.py"
TEST_SCRIPT = REPO_ROOT / "vision" / "ai" / "train" / "test_yolov8.py"
STATIC_DIR = WEB_DIR / "static"
LOG_DIR = WEB_DIR / "logs"
LOG_FILE = LOG_DIR / "current.log"
TEST_LOG_FILE = LOG_DIR / "test-current.log"
RUNS_ROOT = REPO_ROOT / "runs"
ANNOTATION_QA_ROOT = RUNS_ROOT / "annotation_qa"
DETECT_RUNS_ROOT = RUNS_ROOT / "detect"
RFDETR_RUNS_ROOT = RUNS_ROOT / "rfdetr"
DFINE_RUNS_ROOT = RUNS_ROOT / "dfine"
SEGMENT_RUNS_ROOT = RUNS_ROOT / "segment"
SEMANTIC_RUNS_ROOT = RUNS_ROOT / "semantic"
CLASSIFY_RUNS_ROOT = RUNS_ROOT / "classify"
TRAINING_RUNS_ROOTS = (DETECT_RUNS_ROOT, RFDETR_RUNS_ROOT, DFINE_RUNS_ROOT, SEGMENT_RUNS_ROOT, SEMANTIC_RUNS_ROOT, CLASSIFY_RUNS_ROOT)
TEST_RUNS_ROOT = RUNS_ROOT / "test"
INFERENCE_SCRIPT = WEB_DIR / "infer_yolo.py"
MYT = timezone(timedelta(hours=8), name="MYT")

load_dotenv(WEB_DIR / ".env")

DATA_ROOT = Path(os.getenv("WEB_DATA_ROOT") or WEB_DIR / "datasets").expanduser()
INFERENCE_ROOT = DATA_ROOT / "inference"
INFERENCE_UPLOAD_ROOT = INFERENCE_ROOT / "uploads"
INFERENCE_JOB_ROOT = INFERENCE_ROOT / "jobs"
DATASET_UPLOAD_ROOT = DATA_ROOT / "uploads"
DATASET_EXTRACTED_ROOT = DATA_ROOT / "extracted"
DATASET_PREPARED_ROOT = DATA_ROOT / "prepared"
TRAINING_PYTHON = os.getenv("TRAINING_PYTHON", "python3")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
INFERENCE_WEIGHT_EXTENSIONS = {".pt", ".onnx"}
SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val", "test": "test"}
MODEL_MAP = {
    "nano": "yolov8n.pt",
    "small": "yolov8s.pt",
    "medium": "yolov8m.pt",
    "large": "yolov8l.pt",
    "xlarge": "yolov8x.pt",
    "nano-det": "yolov8n.pt",
    "small-det": "yolov8s.pt",
    "medium-det": "yolov8m.pt",
    "large-det": "yolov8l.pt",
    "xlarge-det": "yolov8x.pt",
    "yolo11-nano": "yolo11n.pt",
    "yolo11-small": "yolo11s.pt",
    "yolo11-medium": "yolo11m.pt",
    "yolo11-large": "yolo11l.pt",
    "yolo11-xlarge": "yolo11x.pt",
    "yolo26-nano": "yolo26n.pt",
    "yolo26-small": "yolo26s.pt",
    "yolo26-medium": "yolo26m.pt",
    "yolo26-large": "yolo26l.pt",
    "yolo26-xlarge": "yolo26x.pt",
    "nano-seg": "yolov8n-seg.pt",
    "small-seg": "yolov8s-seg.pt",
    "medium-seg": "yolov8m-seg.pt",
    "large-seg": "yolov8l-seg.pt",
    "xlarge-seg": "yolov8x-seg.pt",
    "yolo11-nano-seg": "yolo11n-seg.pt",
    "yolo11-small-seg": "yolo11s-seg.pt",
    "yolo11-medium-seg": "yolo11m-seg.pt",
    "yolo11-large-seg": "yolo11l-seg.pt",
    "yolo11-xlarge-seg": "yolo11x-seg.pt",
    "yolo26-nano-seg": "yolo26n-seg.pt",
    "yolo26-small-seg": "yolo26s-seg.pt",
    "yolo26-medium-seg": "yolo26m-seg.pt",
    "yolo26-large-seg": "yolo26l-seg.pt",
    "yolo26-xlarge-seg": "yolo26x-seg.pt",
    "yolo26-nano-sem": "yolo26n-sem.pt",
    "yolo26-small-sem": "yolo26s-sem.pt",
    "yolo26-medium-sem": "yolo26m-sem.pt",
    "yolo26-large-sem": "yolo26l-sem.pt",
    "yolo26-xlarge-sem": "yolo26x-sem.pt",
    "nano-cls": "yolov8n-cls.pt",
    "small-cls": "yolov8s-cls.pt",
    "medium-cls": "yolov8m-cls.pt",
    "large-cls": "yolov8l-cls.pt",
    "xlarge-cls": "yolov8x-cls.pt",
    "yolo11-nano-cls": "yolo11n-cls.pt",
    "yolo11-small-cls": "yolo11s-cls.pt",
    "yolo11-medium-cls": "yolo11m-cls.pt",
    "yolo11-large-cls": "yolo11l-cls.pt",
    "yolo11-xlarge-cls": "yolo11x-cls.pt",
    "yolo26-nano-cls": "yolo26n-cls.pt",
    "yolo26-small-cls": "yolo26s-cls.pt",
    "yolo26-medium-cls": "yolo26m-cls.pt",
    "yolo26-large-cls": "yolo26l-cls.pt",
    "yolo26-xlarge-cls": "yolo26x-cls.pt",
}
TRAINING_PROJECT_DEFAULTS = {
    "detect": "runs/detect",
    "rfdetr": "runs/rfdetr",
    "dfine": "runs/dfine",
    "segment": "runs/segment",
    "semantic": "runs/semantic",
    "classify": "runs/classify",
}
RFDETR_DEFAULTS = {
    "model_size": "rfdetr-nano",
    "model": "rfdetr-nano",
    "imgsz": 512,
    "batch": 4,
    "lr0": 1e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 0.0,
    "cos_lr": False,
}
DFINE_DEFAULTS = {
    "model_size": "dfine-n",
    "model": "dfine-n",
    "imgsz": 640,
    "batch": 4,
    "lr0": 4e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 0.0,
    "cos_lr": False,
}
MODEL_REGISTRY = {
    key: {
        "family": "ultralytics",
        "task": training_task if (training_task := (
            "segment" if key.endswith("-seg") else "semantic" if key.endswith("-sem") else "classify" if key.endswith("-cls") else "detect"
        )) else "detect",
        "model": value,
        "runner": TRAIN_SCRIPT,
        "project_default": TRAINING_PROJECT_DEFAULTS[training_task],
    }
    for key, value in MODEL_MAP.items()
}
MODEL_REGISTRY.update({
    "rfdetr-nano": {
        "family": "rfdetr",
        "task": "detect",
        "model": RFDETR_DEFAULTS["model"],
        "runner": RFDETR_TRAIN_SCRIPT,
        "project_default": TRAINING_PROJECT_DEFAULTS["rfdetr"],
    },
    "dfine-n": {
        "family": "dfine",
        "task": "detect",
        "model": DFINE_DEFAULTS["model"],
        "runner": DFINE_TRAIN_SCRIPT,
        "project_default": TRAINING_PROJECT_DEFAULTS["dfine"],
    },
})
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")
PROGRESS_LINE_RE = re.compile(r":\s*\d+%\s+.*\b\d+/\d+\b")
WEB_PROGRESS_RE = re.compile(r"^WEB_TRAINING_PROGRESS\s+epoch=(\d+)\s+total=(\d+)$")
RFDETR_VALIDATION_PROGRESS_RE = re.compile(r"Val\s+\(Epoch\s+(\d+)\s*/\s*(\d+)\)", re.IGNORECASE)
WEB_TEST_PROGRESS_RE = re.compile(
    r"^WEB_TEST_PROGRESS\s+percent=(\d+)\s+stage=([A-Za-z0-9_-]+)(?:\s+detail=(.*))?$"
)
WEB_TEST_RUN_DIR_RE = re.compile(r"^WEB_TEST_RUN_DIR\s+path=(.+)$")
TIMESTAMPED_RUN_SUFFIX_RE = re.compile(r"^(?P<base>.+)-(?P<timestamp>\d{8}-\d{6})$")
RUN_DIRECTORY_PREFIXES = ("Logging results to ", "Results saved to ")
RFDETR_NON_FATAL_WARNING_PATTERNS = (
    "pretrained weights",
    "loaded only partially",
    "different number of positional encodings",
    "patch size 16 instead of 14",
    "litlogger",
    "bf16-mixed is not supported by the model summary",
    "existing log directory",
    "previous log files in this directory will be deleted",
    "detection head will be re-initialized",
)
RFDETR_FATAL_ERROR_PATTERNS = (
    "could not detect dataset format",
    "no module named pytorch_lightning",
    "cuda out of memory",
    "missing labels folder",
)
SPLIT_METADATA_FILE = ".split_metadata.json"
DATASET_SUMMARY_FILE = ".web_dataset_summary.json"
TRAINING_REPORT_CONTEXT_FILE = "training_report_context.json"
TEST_REPORT_CONTEXT_FILE = "test_report_context.json"
ANNOTATION_QA_SUMMARY_FILE = "summary.json"
ANNOTATION_QA_REPORT_JSON_FILE = "qa_report.json"
ANNOTATION_QA_REPORT_CSV_FILE = "qa_report.csv"
ANNOTATION_QA_REVIEW_FILE = "review_state.json"
ANNOTATION_QA_FIX_SUMMARY_FILE = "fix_summary.json"
ANNOTATION_QA_MODEL_DEFAULT = os.getenv("SAM_QA_MODEL", "sam2.1_s.pt")

app = FastAPI(title="YOLOv8 Training UI")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

training_process: Optional[subprocess.Popen] = None
training_started_at: Optional[float] = None
training_log_file: Optional[Path] = None
training_run_info: Optional[dict] = None
test_process: Optional[subprocess.Popen] = None
test_started_at: Optional[float] = None
test_log_file: Optional[Path] = None
test_run_info: Optional[dict] = None
dataset_preparation_jobs: dict[str, dict] = {}
dataset_preparation_lock = threading.Lock()
dataset_downloads: dict[str, dict] = {}
dataset_download_lock = threading.Lock()
inference_jobs: dict[str, dict] = {}
inference_jobs_lock = threading.Lock()
inference_peers: dict[str, set] = {}
inference_peers_lock = threading.Lock()
annotation_qa_jobs: dict[str, dict] = {}
annotation_qa_jobs_lock = threading.Lock()
gpu_status_cache: dict = {"checked_at": 0.0, "payload": None}
gpu_status_lock = threading.Lock()

DatasetProgressCallback = Callable[[str, int, int, str], None]
GPU_STATUS_CACHE_SECONDS = 2.0
DATASET_DOWNLOAD_MAX_AGE_SECONDS = 60 * 60


class SplitConfig(BaseModel):
    train: int = Field(default=70, ge=1, le=100)
    val: int = Field(default=15, ge=0, le=100)
    test: int = Field(default=15, ge=0, le=100)


class RoboflowRequest(BaseModel):
    api_key: Optional[str] = None
    workspace: Optional[str] = None
    project: Optional[str] = None
    version: Optional[str] = None
    classes: list[str] = Field(default_factory=list)
    name: str = "dataset"
    train: int = Field(default=70, ge=1, le=100)
    val: int = Field(default=15, ge=0, le=100)
    test: int = Field(default=15, ge=0, le=100)
    force_split: bool = False
    job_id: str = ""


class DatasetDownloadRequest(BaseModel):
    dataset_yaml: str


class AnnotationQaRequest(BaseModel):
    dataset_yaml: str
    model: str = ANNOTATION_QA_MODEL_DEFAULT
    scope: str = "all"
    preset: str = "balanced"
    max_images: Optional[int] = Field(default=None, ge=1)
    max_side: int = Field(default=1280, ge=320, le=4096)


class AnnotationQaMarkRequest(BaseModel):
    issue_id: str
    status: str


class AnnotationQaFixRequest(BaseModel):
    issue_id: str
    fix: str = "sam_box"
    class_id: Optional[int] = Field(default=None, ge=0)


class TrainRequest(BaseModel):
    dataset_yaml: Optional[str] = None
    model_size: str = "nano"
    epochs: int = Field(default=100, ge=1)
    imgsz: int = Field(default=640, ge=32)
    batch: int = 16
    patience: int = Field(default=20, ge=0)
    save_period: int = -1
    device: Optional[str] = None
    workers: int = Field(default=2, ge=0)
    optimizer: str = "Adam"
    lr0: float = Field(default=0.001, gt=0)
    lrf: float = Field(default=0.01, gt=0)
    weight_decay: float = Field(default=0.0005, ge=0)
    cos_lr: bool = False
    warmup_epochs: float = Field(default=3.0, ge=0)
    freeze: Optional[int] = Field(default=None, ge=0)
    activation: str = "silu"
    exist_ok: bool = False
    seed: int = 42
    project: str = "runs/detect"
    name: str = "train"
    resume: bool = False
    disable_ultralytics_albumentations: bool = True
    mosaic: float = Field(default=1.0, ge=0, le=1)
    close_mosaic: int = Field(default=10, ge=0)
    hsv_h: float = Field(default=0.015, ge=0)
    hsv_s: float = Field(default=0.7, ge=0)
    hsv_v: float = Field(default=0.4, ge=0)
    degrees: float = Field(default=0.0, ge=0)
    translate: float = Field(default=0.1, ge=0)
    scale: float = Field(default=0.5, ge=0)
    shear: float = Field(default=0.0, ge=0)
    perspective: float = Field(default=0.0, ge=0)
    flipud: float = Field(default=0.0, ge=0, le=1)
    fliplr: float = Field(default=0.5, ge=0, le=1)
    bgr: float = Field(default=0.0, ge=0, le=1)
    mixup: float = Field(default=0.0, ge=0, le=1)
    cutmix: float = Field(default=0.0, ge=0, le=1)
    copy_paste: float = Field(default=0.0, ge=0, le=1)
    auto_augment: Optional[str] = None
    erasing: float = Field(default=0.0, ge=0, le=1)


class WeightRequest(BaseModel):
    project: str = "runs/detect"
    name: str = "train"


class ArtifactRequest(BaseModel):
    project: str = "runs/detect"
    name: str = "train"
    artifact: str


class TestArtifactRequest(BaseModel):
    artifact: str


class InferenceWeightsRequest(BaseModel):
    weight_path: str


class WebRTCOffer(BaseModel):
    sdp: str
    type: str


def ensure_dirs():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TEST_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    DATASET_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    DATASET_EXTRACTED_ROOT.mkdir(parents=True, exist_ok=True)
    DATASET_PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
    INFERENCE_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    INFERENCE_JOB_ROOT.mkdir(parents=True, exist_ok=True)


def form_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    return normalized in {"1", "true", "yes", "y", "on"}


def installed_version(package: str) -> str:
    try:
        return package_version(package)
    except PackageNotFoundError:
        return "unknown"


def update_dataset_preparation(
    job_id: str,
    stage: str,
    current: int,
    total: int,
    detail: str,
    status: str = "running",
):
    if not job_id:
        return
    safe_total = max(0, int(total))
    safe_current = max(0, min(int(current), safe_total)) if safe_total else max(0, int(current))
    percent = round((safe_current / safe_total) * 100, 1) if safe_total else 0.0
    with dataset_preparation_lock:
        if job_id not in dataset_preparation_jobs and len(dataset_preparation_jobs) >= 100:
            finished_jobs = [
                item for item in dataset_preparation_jobs.values()
                if item.get("status") != "running"
            ]
            if finished_jobs:
                oldest = min(finished_jobs, key=lambda item: item.get("updated_at", 0))
                dataset_preparation_jobs.pop(oldest["job_id"], None)
        dataset_preparation_jobs[job_id] = {
            "job_id": job_id,
            "status": status,
            "stage": stage,
            "mode": "determinate" if safe_total else "indeterminate",
            "current": safe_current,
            "total": safe_total,
            "percent": percent,
            "detail": detail,
            "updated_at": time.time(),
        }


def dataset_progress_callback(job_id: str) -> DatasetProgressCallback:
    last_stage = None
    last_update = 0.0

    def report(stage: str, current: int, total: int, detail: str):
        nonlocal last_stage, last_update
        now = time.monotonic()
        stage_changed = stage != last_stage
        finished_stage = total > 0 and current >= total
        if not stage_changed and not finished_stage and now - last_update < 0.1:
            return
        update_dataset_preparation(job_id, stage, current, total, detail)
        last_stage = stage
        last_update = now

    return report


def clean_name(value: str, fallback: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def safe_upload_path(filename: str) -> Path:
    normalized = filename.replace("\\", "/")
    if normalized.startswith("/"):
        raise HTTPException(status_code=400, detail=f"Invalid upload path: {filename}")

    parts = [part for part in normalized.split("/") if part and part != "."]
    if not parts or any(part == ".." for part in parts):
        raise HTTPException(status_code=400, detail=f"Invalid upload path: {filename}")

    return Path(*parts)


def resolve_weight_path(project: str, name: str, weight: str) -> Path:
    if weight not in {"best", "last"}:
        raise HTTPException(status_code=404, detail="Unknown weight file.")

    run_dir = resolve_run_dir(project, name)
    candidate = run_dir / "weights" / f"{weight}.pt"
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Weight file not found: {candidate}")

    return candidate


def resolve_project_path(project: str) -> Path:
    project_path = Path(project).expanduser()
    if not project_path.is_absolute():
        project_path = REPO_ROOT / project_path
    return project_path.resolve()


def model_spec_for_size(model_size: str) -> dict:
    return MODEL_REGISTRY.get(str(model_size or ""), {
        "family": "ultralytics",
        "task": training_task_for_model_size(model_size),
        "model": MODEL_MAP.get(model_size, ""),
        "runner": TRAIN_SCRIPT,
        "project_default": TRAINING_PROJECT_DEFAULTS["detect"],
    })


def training_task_for_model_size(model_size: str) -> str:
    spec = MODEL_REGISTRY.get(str(model_size or ""))
    if spec:
        return spec["task"]
    value = str(model_size or "")
    if value.endswith("-seg"):
        return "segment"
    if value.endswith("-sem"):
        return "semantic"
    if value.endswith("-cls"):
        return "classify"
    return "detect"


def training_family_for_model_size(model_size: str) -> str:
    return model_spec_for_size(model_size).get("family", "ultralytics")


def default_project_for_model_size(model_size: str) -> str:
    return model_spec_for_size(model_size).get(
        "project_default",
        default_project_for_training_task(training_task_for_model_size(model_size)),
    )


def rfdetr_training_value(request: TrainRequest, key: str):
    request_value = getattr(request, key)
    default_value = getattr(TrainRequest(), key)
    if request_value == default_value:
        return RFDETR_DEFAULTS[key]
    return request_value


def backend_training_value(request: TrainRequest, key: str, family: str):
    defaults = {
        "rfdetr": RFDETR_DEFAULTS,
        "dfine": DFINE_DEFAULTS,
    }.get(family)
    if not defaults:
        return getattr(request, key)
    request_value = getattr(request, key)
    default_value = getattr(TrainRequest(), key)
    if request_value == default_value:
        return defaults[key]
    return request_value


def default_project_for_training_task(task: str) -> str:
    return TRAINING_PROJECT_DEFAULTS.get(task, TRAINING_PROJECT_DEFAULTS["detect"])


def timestamped_training_run_name(name: str, resume: bool = False, now: Optional[datetime] = None) -> str:
    base_name = (name or "train").strip() or "train"
    if resume:
        return base_name

    timestamp_source = now or datetime.now(MYT)
    timestamp = timestamp_source.astimezone(MYT).strftime("%Y%m%d-%H%M%S")
    match = TIMESTAMPED_RUN_SUFFIX_RE.match(base_name)
    if match is not None:
        base_name = match.group("base")
    return f"{base_name}-{timestamp}"


def training_report_download_filename(run_dir: Path) -> str:
    match = TIMESTAMPED_RUN_SUFFIX_RE.match(run_dir.name)
    if match is None:
        return "training_report.pdf"
    return f"training_report_{match.group('timestamp')}.pdf"


def is_known_training_project_default(project: str) -> bool:
    value = str(project or "").strip().replace("\\", "/").rstrip("/")
    if not value:
        return True

    known_values = set(TRAINING_PROJECT_DEFAULTS.values())
    known_values.update(f"/app/{item}" for item in TRAINING_PROJECT_DEFAULTS.values())
    if value in known_values:
        return True

    try:
        project_path = resolve_project_path(value)
    except (OSError, RuntimeError):
        return False

    known_paths = {resolve_project_path(item) for item in TRAINING_PROJECT_DEFAULTS.values()}
    return project_path in known_paths


def normalize_training_project_path(project: str) -> Path:
    project_path = resolve_project_path(project or "runs/detect")
    ensure_runs_path(project_path)

    runs_root = RUNS_ROOT.resolve()
    relative_parts = list(project_path.relative_to(runs_root).parts)
    while len(relative_parts) >= 3 and relative_parts[1] == "runs" and relative_parts[2] == relative_parts[0]:
        relative_parts = [relative_parts[0], *relative_parts[3:]]

    return runs_root.joinpath(*relative_parts) if relative_parts else runs_root


def ensure_runs_path(path: Path):
    runs_root = RUNS_ROOT.resolve()
    try:
        path.resolve().relative_to(runs_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Training outputs can only be read from the runs directory.",
        ) from exc


def ensure_inference_runs_path(path: Path):
    resolved = path.resolve()
    for root in TRAINING_RUNS_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            return
        except ValueError:
            continue
    raise HTTPException(
        status_code=400,
        detail="Inference weights can only be selected from runs/detect, runs/rfdetr, runs/dfine, runs/segment, runs/semantic, or runs/classify.",
    )


def ensure_detect_runs_path(path: Path):
    ensure_inference_runs_path(path)


def ensure_inference_path(path: Path):
    inference_root = INFERENCE_ROOT.resolve()
    try:
        path.resolve().relative_to(inference_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Inference files can only be read from the inference workspace.",
        ) from exc


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size

    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() or item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def inference_jobs_storage_payload() -> dict:
    ensure_dirs()
    job_dirs = [path for path in INFERENCE_JOB_ROOT.iterdir() if path.is_dir()]
    return {
        "jobs_path": str(INFERENCE_JOB_ROOT),
        "job_count": len(job_dirs),
        "jobs_size": directory_size(INFERENCE_JOB_ROOT),
    }


STORAGE_CLEANUP_TARGETS = {
    "inference_outputs": {
        "label": "Inference output jobs",
        "path": INFERENCE_JOB_ROOT,
        "requires_idle": "inference",
    },
    "inference_uploads": {
        "label": "Inference uploaded weights",
        "path": INFERENCE_UPLOAD_ROOT,
        "requires_idle": "inference",
    },
    "dataset_prepared": {
        "label": "Prepared datasets",
        "path": DATASET_PREPARED_ROOT,
        "requires_idle": "dataset",
    },
    "dataset_extracted": {
        "label": "Extracted datasets",
        "path": DATASET_EXTRACTED_ROOT,
        "requires_idle": "dataset",
    },
    "dataset_uploads": {
        "label": "Uploaded dataset ZIPs",
        "path": DATASET_UPLOAD_ROOT,
        "requires_idle": "dataset",
    },
}

STORAGE_CLEANUP_ALIASES = {
    "inference_output": "inference_outputs",
    "inference_upload": "inference_uploads",
    "uploaded_inference": "inference_uploads",
    "uploaded_inference_weight": "inference_uploads",
    "uploaded_inference_weights": "inference_uploads",
    "dataset_extract": "dataset_extracted",
    "dataset_extracts": "dataset_extracted",
    "extracted_dataset": "dataset_extracted",
    "extracted_datasets": "dataset_extracted",
    "dataset_upload": "dataset_uploads",
    "uploaded_dataset": "dataset_uploads",
    "uploaded_datasets": "dataset_uploads",
    "dataset_preparation": "dataset_prepared",
    "prepared_dataset": "dataset_prepared",
    "prepared_datasets": "dataset_prepared",
}


def normalize_storage_target_key(key: str) -> str:
    return STORAGE_CLEANUP_ALIASES.get(key, key)


def storage_target_payload(key: str) -> dict:
    ensure_dirs()
    key = normalize_storage_target_key(key)
    target = STORAGE_CLEANUP_TARGETS.get(key)
    if target is None:
        raise HTTPException(status_code=404, detail="Storage cleanup target not found.")
    root = target["path"]
    children = list(root.iterdir()) if root.is_dir() else []
    return {
        "key": key,
        "label": target["label"],
        "path": str(root),
        "item_count": len(children),
        "size": directory_size(root),
    }


def storage_payload() -> dict:
    return {
        "targets": [
            storage_target_payload(key)
            for key in STORAGE_CLEANUP_TARGETS
        ],
    }


def has_active_inference_job() -> bool:
    with inference_jobs_lock:
        return any(
            job.get("status") not in {"complete", "failed", "stopped"}
            for job in inference_jobs.values()
        )


def has_active_dataset_preparation() -> bool:
    with dataset_preparation_lock:
        return any(
            job.get("status") == "running"
            for job in dataset_preparation_jobs.values()
        )


def ensure_storage_target_idle(target: dict):
    if target.get("requires_idle") == "inference" and has_active_inference_job():
        raise HTTPException(status_code=409, detail="Stop the active inference job before clearing outputs.")
    if target.get("requires_idle") == "dataset":
        if current_status()["running"]:
            raise HTTPException(status_code=409, detail="Stop training before clearing dataset storage.")
        if test_status()["running"]:
            raise HTTPException(status_code=409, detail="Stop testing before clearing dataset storage.")
        if has_active_dataset_preparation():
            raise HTTPException(status_code=409, detail="Wait for dataset preparation to finish before clearing dataset storage.")


def clear_storage_target(key: str) -> dict:
    key = normalize_storage_target_key(key)
    target = STORAGE_CLEANUP_TARGETS.get(key)
    if target is None:
        raise HTTPException(status_code=404, detail="Storage cleanup target not found.")
    ensure_storage_target_idle(target)

    before = storage_target_payload(key)
    removed_items = 0
    root = target["path"].resolve()
    for child in list(target["path"].iterdir()):
        target = child.resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=500, detail="Refusing to delete outside storage target.") from exc
        if child.is_dir():
            shutil.rmtree(child)
            removed_items += 1
        elif child.is_file() or child.is_symlink():
            child.unlink()
            removed_items += 1

    if key == "inference_outputs":
        with inference_jobs_lock:
            inference_jobs.clear()

    after = storage_target_payload(key)
    return {
        "key": key,
        "label": before["label"],
        "path": before["path"],
        "removed_items": removed_items,
        "removed_jobs": removed_items if key == "inference_outputs" else 0,
        "freed_bytes": max(0, before["size"] - after["size"]),
        "before": before,
        "after": after,
    }


def clear_inference_outputs() -> dict:
    return clear_storage_target("inference_outputs")


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def resolve_inference_weight_path(weight_path: str) -> Path:
    value = Path(str(weight_path or "")).expanduser()
    if value.is_absolute():
        candidate = value.resolve()
    else:
        candidate = (REPO_ROOT / value).resolve()
    ensure_inference_runs_path(candidate)
    if candidate.suffix.lower() not in INFERENCE_WEIGHT_EXTENSIONS or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Selected inference weights were not found.")
    return candidate


def family_for_weight_path(weights_path: Path) -> str:
    parts = {part.lower() for part in weights_path.parts}
    if "rfdetr" in parts:
        return "rfdetr"
    if "dfine" in parts:
        return "dfine"
    context = read_json_object(weights_path.parent.parent / TRAINING_REPORT_CONTEXT_FILE)
    family = str(context.get("family") or context.get("hyperparameters", {}).get("family") or "").lower()
    return family if family in {"dfine", "rfdetr", "ultralytics"} else "ultralytics"


def task_for_runs_root(root: Path) -> str:
    if root in {RFDETR_RUNS_ROOT, DFINE_RUNS_ROOT}:
        return "detect"
    if root == SEGMENT_RUNS_ROOT:
        return "segment"
    if root == SEMANTIC_RUNS_ROOT:
        return "semantic"
    if root == CLASSIFY_RUNS_ROOT:
        return "classify"
    return "detect"


def family_for_runs_root(root: Path) -> str:
    if root == RFDETR_RUNS_ROOT:
        return "rfdetr"
    if root == DFINE_RUNS_ROOT:
        return "dfine"
    return "ultralytics"


def available_inference_weights() -> list[dict]:
    weights = []
    for root in TRAINING_RUNS_ROOTS:
        if not root.is_dir():
            continue
        task = task_for_runs_root(root)
        family = family_for_runs_root(root)
        candidates = [
            path
            for path in root.rglob("weights/*")
            if path.suffix.lower() in INFERENCE_WEIGHT_EXTENSIONS
        ]
        for path in sorted(candidates):
            try:
                ensure_inference_runs_path(path)
            except HTTPException:
                continue
            run_dir = path.parent.parent
            stat = path.stat()
            try:
                run_label = run_dir.relative_to(root).as_posix()
            except ValueError:
                run_label = run_dir.name
            weights.append({
                "label": f"{task}/{run_label}/{path.name}",
                "path": relative_to_repo(path),
                "family": family,
                "task": task,
                "run": run_label,
                "weight": path.stem,
                "format": path.suffix.lower().lstrip("."),
                "size": stat.st_size,
                "modified_at": stat.st_mtime,
            })
    weights.sort(key=lambda item: item["modified_at"], reverse=True)
    return weights


def available_detection_weights() -> list[dict]:
    return available_inference_weights()


def normalize_exported_model_path(exported, fallback: Path) -> Path:
    if isinstance(exported, (list, tuple)):
        exported = exported[0] if exported else fallback
    value = Path(str(exported or fallback)).expanduser()
    if not value.is_absolute():
        value = (fallback.parent / value).resolve()
    return value.resolve()


def export_inference_pt_to_onnx(weights_path: Path, imgsz: int) -> Path:
    if weights_path.suffix.lower() != ".pt":
        return weights_path

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("ONNX export requires ultralytics in the web container.") from exc

    fallback_path = weights_path.with_suffix(".onnx")
    try:
        exported = YOLO(str(weights_path)).export(
            format="onnx",
            imgsz=int(imgsz),
            opset=12,
            simplify=False,
            dynamic=False,
        )
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        raise RuntimeError(
            f"ONNX export requires {missing}. Rebuild the container after updating requirements.txt."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"ONNX export failed: {exc}") from exc

    exported_path = normalize_exported_model_path(exported, fallback_path)
    if not exported_path.is_file() and fallback_path.is_file():
        exported_path = fallback_path.resolve()
    if not exported_path.is_file():
        raise RuntimeError("ONNX export did not produce an .onnx file.")
    if exported_path.suffix.lower() != ".onnx":
        raise RuntimeError(f"ONNX export produced an unexpected file: {exported_path.name}")
    return exported_path


def available_training_sessions() -> list[dict]:
    sessions = []
    for root in TRAINING_RUNS_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_dir() or not is_run_dir(path):
                continue
            try:
                ensure_runs_path(path)
            except HTTPException:
                continue
            file_times = [item.stat().st_mtime for item in path.rglob("*") if item.is_file()]
            modified_at = max(file_times, default=path.stat().st_mtime)
            project_path = path.parent
            task = task_for_runs_root(root)
            family = family_for_runs_root(root)
            sessions.append({
                "label": relative_to_repo(path),
                "name": path.name,
                "project": relative_to_repo(project_path),
                "run_dir": relative_to_repo(path),
                "family": family,
                "task": task,
                "modified_at": modified_at,
                "has_results": (path / "results.csv").is_file(),
                "has_best": (path / "weights" / "best.pt").is_file(),
                "has_last": (path / "weights" / "last.pt").is_file(),
            })

    sessions.sort(key=lambda item: item["modified_at"], reverse=True)
    return sessions


def ensure_inference_script() -> Path:
    if not INFERENCE_SCRIPT.is_file():
        raise HTTPException(status_code=500, detail="Python inference script is missing.")
    return INFERENCE_SCRIPT


def inference_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    raise HTTPException(status_code=400, detail="Upload an image or video file for inference.")


def inference_job_payload(job: dict) -> dict:
    public_keys = {
        "job_id",
        "status",
        "stage",
        "percent",
        "detail",
        "frames",
        "total_frames",
        "detections",
        "classifications",
        "task",
        "elapsed_ms",
        "media_type",
        "weights",
        "preview_available",
        "preview_fps",
        "result",
        "error",
        "stoppable",
    }
    return {key: value for key, value in job.items() if key in public_keys}


def update_inference_job(job_id: str, **updates):
    condition = None
    with inference_jobs_lock:
        job = inference_jobs.get(job_id)
        if job is None:
            return
        job.update(updates)
        job["updated_at"] = time.time()
        condition = job.get("preview_condition")
    if isinstance(condition, threading.Condition):
        with condition:
            condition.notify_all()


def inference_has_live_preview(job_id: str, preview_path: Path) -> bool:
    with inference_jobs_lock:
        job = inference_jobs.get(job_id) or {}
        if job.get("preview_frame"):
            return True
    return preview_path.is_file()


def update_inference_preview_frame(job_id: str, frame: bytes):
    condition = None
    now = time.time()
    with inference_jobs_lock:
        job = inference_jobs.get(job_id)
        if job is None:
            return
        timestamps = [
            timestamp for timestamp in job.get("preview_frame_times", [])
            if now - float(timestamp) <= 2.0
        ]
        timestamps.append(now)
        elapsed = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        job["preview_frame"] = frame
        job["preview_available"] = True
        job["preview_updated_at"] = now
        job["preview_frame_times"] = timestamps
        job["preview_fps"] = round((len(timestamps) - 1) / elapsed, 1) if elapsed > 0 else 0.0
        job["updated_at"] = now
        condition = job.get("preview_condition")
    if isinstance(condition, threading.Condition):
        with condition:
            condition.notify_all()


def update_inference_webrtc_frame(job_id: str, frame):
    with inference_jobs_lock:
        job = inference_jobs.get(job_id)
        if job is None:
            return
        job["webrtc_frame"] = frame.copy()
        job["webrtc_updated_at"] = time.time()
        job["webrtc_frame_seq"] = int(job.get("webrtc_frame_seq") or 0) + 1


def latest_inference_webrtc_frame(job_id: str):
    with inference_jobs_lock:
        job = inference_jobs.get(job_id)
        if job is None:
            return None, "missing"
        frame = job.get("webrtc_frame")
        status = job.get("status", "")
    return (frame.copy() if frame is not None else None), status


def inference_mjpeg_stream(job_id: str):
    last_updated_at = 0.0
    while True:
        with inference_jobs_lock:
            job = inference_jobs.get(job_id)
            if job is None:
                break
            frame = job.get("preview_frame")
            updated_at = float(job.get("preview_updated_at") or 0)
            status = job.get("status", "")
            condition = job.get("preview_condition")

        if frame and updated_at != last_updated_at:
            last_updated_at = updated_at
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-store\r\n"
                + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                + frame
                + b"\r\n"
            )

        if status in {"complete", "failed", "stopped"}:
            break

        if isinstance(condition, threading.Condition):
            with condition:
                condition.wait(timeout=1.0)
        else:
            time.sleep(0.2)


def read_inference_progress(progress_path: Path) -> dict:
    if not progress_path.is_file():
        return {}
    return read_json_object(progress_path)


def inference_result_payload_from_disk(job_id: str, job_dir: Path) -> dict | None:
    result_path = job_dir / "result.json"
    if not result_path.is_file():
        return None
    payload = read_json_object(result_path)
    if not payload:
        return None
    return {
        **payload,
        "job_id": job_id,
        "result_url": f"/api/inference/result/{job_id}",
        "download_url": f"/api/inference/result/{job_id}?download=1",
    }


def inference_job_from_disk(job_id: str) -> dict | None:
    job_dir = (INFERENCE_JOB_ROOT / job_id).resolve()
    try:
        ensure_inference_path(job_dir)
    except HTTPException:
        return None
    if not job_dir.is_dir():
        return None

    preview_path = job_dir / "preview.jpg"
    output_candidates = [job_dir / "annotated.jpg", job_dir / "annotated.mp4"]
    output_path = next((candidate for candidate in output_candidates if candidate.is_file()), None)
    result = inference_result_payload_from_disk(job_id, job_dir)
    if result and output_path:
        return {
            "job_id": job_id,
            "status": "complete",
            "stage": "complete",
            "percent": 100,
            "detail": "Inference complete.",
            "frames": result.get("frames", 0),
            "total_frames": result.get("frames", 0),
            "detections": result.get("detections", 0),
            "classifications": result.get("classifications", 0),
            "task": result.get("task", ""),
            "elapsed_ms": result.get("elapsed_ms", 0),
            "media_type": result.get("media_type", "video" if output_path.suffix == ".mp4" else "image"),
            "preview_available": preview_path.is_file(),
            "preview_fps": 0,
            "result": result,
        }

    progress = read_inference_progress(job_dir / "progress.json")
    if progress:
        return {
            "job_id": job_id,
            "status": "running",
            "stage": progress.get("stage", "processing"),
            "percent": progress.get("percent", 0),
            "detail": progress.get("detail", "Processing inference."),
            "frames": progress.get("frames", 0),
            "total_frames": progress.get("total_frames", 0),
            "detections": progress.get("detections", 0),
            "classifications": progress.get("classifications", 0),
            "task": progress.get("task", ""),
            "elapsed_ms": 0,
            "media_type": "video" if (job_dir / "annotated.mp4").exists() else "image",
            "preview_available": preview_path.is_file(),
            "preview_fps": 0,
        }

    log_path = job_dir / "inference.log"
    if log_path.is_file():
        return {
            "job_id": job_id,
            "status": "running",
            "stage": "starting",
            "percent": 0,
            "detail": "Inference process has started.",
            "frames": 0,
            "total_frames": 0,
            "detections": 0,
            "classifications": 0,
            "task": "",
            "elapsed_ms": 0,
            "preview_available": preview_path.is_file(),
            "preview_fps": 0,
        }
    return None


def run_inference_job(
    job_id: str,
    inference_request: dict,
    result_path: Path,
    output_path: Path,
    preview_path: Path,
    stop_event: threading.Event,
):
    log_path = result_path.with_name("inference.log")
    started = time.time()
    inference_family = str(inference_request.pop("family", "ultralytics") or "ultralytics")
    log_path.write_text(f"Starting in-process Python {inference_family} inference.\n", encoding="utf-8")

    def handle_progress(progress: dict):
        update_inference_job(
            job_id,
            status="running",
            stage=progress.get("stage", "processing"),
            percent=progress.get("percent", 0),
            detail=progress.get("detail", "Processing inference."),
            frames=progress.get("frames", 0),
            total_frames=progress.get("total_frames", 0),
            detections=progress.get("detections", 0),
            classifications=progress.get("classifications", 0),
            task=progress.get("task", ""),
            elapsed_ms=round((time.time() - started) * 1000, 2),
            preview_available=inference_has_live_preview(job_id, preview_path),
            stoppable=True,
        )

    def handle_preview(frame: bytes):
        update_inference_preview_frame(job_id, frame)

    def handle_live_frame(frame):
        update_inference_webrtc_frame(job_id, frame)

    update_inference_job(
        job_id,
        status="running",
        stage="starting",
        percent=0,
        detail="Starting inference process.",
        elapsed_ms=0,
        stoppable=True,
    )
    try:
        if inference_request.pop("convert_to_onnx", False):
            original_weights = Path(inference_request["weights_path"])
            update_inference_job(
                job_id,
                status="running",
                stage="converting",
                percent=3,
                detail="Converting uploaded .pt weights to ONNX.",
                elapsed_ms=round((time.time() - started) * 1000, 2),
                stoppable=True,
            )
            onnx_path = export_inference_pt_to_onnx(
                original_weights,
                int(inference_request.get("imgsz") or 640),
            )
            inference_request["weights_path"] = onnx_path
            if onnx_path != original_weights:
                update_inference_job(
                    job_id,
                    weights=f"{original_weights.name} -> {onnx_path.name}",
                    detail="ONNX conversion complete.",
                    elapsed_ms=round((time.time() - started) * 1000, 2),
                )

        inference_runners = {
            "dfine": run_dfine_inference,
            "rfdetr": run_rfdetr_inference,
            "ultralytics": run_yolo_inference,
        }
        inference_runner = inference_runners.get(inference_family, run_yolo_inference)
        payload = inference_runner(
            **inference_request,
            progress_callback=handle_progress,
            preview_callback=handle_preview,
            live_frame_callback=handle_live_frame,
            stop_event=stop_event,
            use_model_cache=True,
        )
    except InferenceStopped:
        update_inference_job(
            job_id,
            status="stopped",
            stage="stopped",
            percent=100,
            detail="Inference stopped by user.",
            elapsed_ms=round((time.time() - started) * 1000, 2),
            preview_available=inference_has_live_preview(job_id, preview_path),
            stoppable=False,
            stop_event=None,
        )
        return
    except Exception as exc:
        detail = f"Inference failed: {exc}"
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{detail}\n")
        update_inference_job(
            job_id,
            status="failed",
            stage="failed",
            percent=100,
            detail=detail,
            error=detail,
            elapsed_ms=round((time.time() - started) * 1000, 2),
            preview_available=inference_has_live_preview(job_id, preview_path),
            stoppable=False,
            stop_event=None,
        )
        return

    if not output_path.is_file():
        detail = "Inference completed without producing an output file."
        update_inference_job(
            job_id,
            status="failed",
            stage="failed",
            percent=100,
            detail=detail,
            error=detail,
            elapsed_ms=round((time.time() - started) * 1000, 2),
            preview_available=inference_has_live_preview(job_id, preview_path),
            stoppable=False,
            stop_event=None,
        )
        return

    with inference_jobs_lock:
        existing_job = dict(inference_jobs.get(job_id) or {})
    result_payload = {
        **payload,
        "job_id": job_id,
        "media_type": existing_job.get("media_type", payload.get("media_type")),
        "weights": existing_job.get("weights", ""),
        "stdout": read_log_file(log_path).strip(),
        "result_url": f"/api/inference/result/{job_id}",
        "download_url": f"/api/inference/result/{job_id}?download=1",
    }
    update_inference_job(
        job_id,
        status="complete",
        stage="complete",
        percent=100,
        detail="Inference complete.",
        frames=payload.get("frames", 0),
        total_frames=payload.get("frames", 0),
        detections=payload.get("detections", 0),
        classifications=payload.get("classifications", 0),
        task=payload.get("task", ""),
        elapsed_ms=payload.get("elapsed_ms", round((time.time() - started) * 1000, 2)),
        preview_available=inference_has_live_preview(job_id, preview_path),
        result=result_payload,
        stoppable=False,
        stop_event=None,
    )


def stop_inference_process(job_id: str) -> dict:
    with inference_jobs_lock:
        job = inference_jobs.get(job_id)
        if job is None:
            disk_job = inference_job_from_disk(job_id)
            if disk_job is None:
                raise HTTPException(status_code=404, detail="Inference job not found.")
            return disk_job
        process = job.get("process")
        stop_event = job.get("stop_event")
        status = job.get("status")
        if status in {"complete", "failed", "stopped"}:
            return inference_job_payload(dict(job))
        job["stop_requested"] = True
        job["status"] = "stopping"
        job["stage"] = "stopping"
        job["detail"] = "Stopping inference process."
        job["stoppable"] = False
        job["updated_at"] = time.time()

    if isinstance(stop_event, threading.Event):
        stop_event.set()

    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    with inference_jobs_lock:
        return inference_job_payload(dict(inference_jobs[job_id]))


def is_run_dir(path: Path) -> bool:
    return (
        (path / "weights" / "best.pt").is_file()
        or (path / "weights" / "last.pt").is_file()
        or (path / "results.csv").is_file()
        or (path / TRAINING_REPORT_CONTEXT_FILE).is_file()
        or any(path.glob("checkpoint*.pth"))
    )


def find_named_run_dir(search_root: Path, name: str) -> Optional[Path]:
    if not search_root.exists() or not name:
        return None

    candidates = [
        path
        for path in search_root.rglob("*")
        if path.is_dir()
        and is_run_dir(path)
        and (path.name == name or path.parent.name == name)
    ]
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: max((item.stat().st_mtime for item in path.rglob("*") if item.is_file()), default=0),
    )


def resolve_run_dir_details(project: str, name: str) -> tuple[Path, str]:
    project_path = resolve_project_path(project)
    ensure_runs_path(project_path)

    current_info = training_run_info or {}
    current_run_dir = current_info.get("run_dir")
    current_project = current_info.get("requested_project") or current_info.get("project")
    if current_run_dir and current_info.get("name") == name and current_project:
        try:
            same_project = resolve_project_path(current_project) == project_path
        except (OSError, RuntimeError):
            same_project = False
        actual = Path(current_run_dir).expanduser().resolve()
        if same_project and actual.is_dir():
            ensure_runs_path(actual)
            return actual, "actual"

    exact = (project_path / name).resolve()
    ensure_runs_path(exact)
    if is_run_dir(exact):
        return exact, "exact"

    named = find_named_run_dir(project_path, name)
    if named:
        return named, "legacy"

    raise HTTPException(
        status_code=404,
        detail=f"No completed run matching project '{project}' and name '{name}' was found.",
    )


def resolve_run_dir(project: str, name: str) -> Path:
    run_dir, _ = resolve_run_dir_details(project, name)
    return run_dir


def validate_classes(classes: list[str]) -> dict[int, str]:
    names = [item.strip() for item in classes if item.strip()]
    if not names:
        raise HTTPException(status_code=400, detail="At least one class name is required.")
    return {index: name for index, name in enumerate(names)}


def normalize_yaml_names(raw_names) -> list[str]:
    if isinstance(raw_names, list):
        return [str(name).strip() for name in raw_names if str(name).strip()]

    if isinstance(raw_names, dict):
        normalized = []
        def sort_key(item):
            try:
                return (0, int(item[0]))
            except (TypeError, ValueError):
                return (1, str(item[0]))

        for key, value in sorted(raw_names.items(), key=sort_key):
            name = str(value).strip()
            if name:
                normalized.append(name)
        return normalized

    return []


def read_yaml_class_names(yaml_path: Path) -> list[str]:
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not read dataset YAML: {exc}") from exc

    names = normalize_yaml_names(payload.get("names"))
    if not names:
        raise HTTPException(status_code=400, detail=f"No class names found in {yaml_path}")
    return names


def detect_dataset_classes(source: Path) -> list[str]:
    yaml_path = source if source.is_file() and source.suffix.lower() in {".yaml", ".yml"} else find_dataset_yaml(find_dataset_root(source))
    if not yaml_path:
        raise HTTPException(status_code=404, detail="No data.yaml or dataset.yaml file found.")
    return read_yaml_class_names(yaml_path)


def resolve_dataset_classes(root: Path, classes: list[str]) -> dict[int, str]:
    names = [item.strip() for item in classes if item.strip()]
    if not names:
        names = detect_dataset_classes(root)
    return validate_classes(names)


def dataset_response(
    yaml_path: Path,
    message: str,
    progress_callback: Optional[DatasetProgressCallback] = None,
    source_metadata: Optional[dict] = None,
) -> dict:
    try:
        classes = read_yaml_class_names(yaml_path)
    except HTTPException:
        classes = []
    summary = inspect_dataset_yaml(yaml_path, classes, progress_callback, source_metadata)
    try:
        (yaml_path.parent / DATASET_SUMMARY_FILE).write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    except OSError:
        pass
    return {
        "dataset_yaml": str(yaml_path),
        "classes": classes,
        "summary": summary,
        "message": message,
    }


def resolve_yaml_dataset_root(yaml_path: Path, payload: dict) -> Path:
    root = payload.get("path") or yaml_path.parent
    root_path = Path(root).expanduser()
    if not root_path.is_absolute():
        root_path = yaml_path.parent / root_path
    return root_path.resolve()


def split_image_folder(dataset_root: Path, value) -> Optional[Path]:
    if not value:
        return None
    candidates = value if isinstance(value, list) else [value]
    for item in candidates:
        path = Path(str(item)).expanduser()
        if not path.is_absolute():
            path = dataset_root / path
        if path.exists():
            return path
    return None


def prepared_dataset_yaml(dataset_yaml: str) -> tuple[Path, Path, dict]:
    yaml_path = Path(dataset_yaml).expanduser()
    if not yaml_path.is_absolute():
        yaml_path = DATA_ROOT / yaml_path
    yaml_path = yaml_path.resolve()
    data_root = DATA_ROOT.resolve()

    try:
        yaml_path.relative_to(data_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=403,
            detail="Dataset downloads are limited to the web dataset workspace.",
        ) from exc

    if not yaml_path.is_file() or yaml_path.suffix.lower() not in {".yaml", ".yml"}:
        raise HTTPException(status_code=404, detail="Prepared dataset YAML was not found.")

    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not read dataset YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Dataset YAML must contain a mapping.")

    dataset_root = resolve_yaml_dataset_root(yaml_path, payload)
    try:
        dataset_root.relative_to(data_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=403,
            detail="The prepared dataset points outside the web dataset workspace.",
        ) from exc
    if not dataset_root.is_dir():
        raise HTTPException(status_code=404, detail="Prepared dataset directory was not found.")

    portable_payload = dict(payload)
    portable_payload["path"] = "."
    for split in SPLIT_NAMES:
        split_value = payload.get(split)
        if not split_value:
            portable_payload.pop(split, None)
            continue

        split_values = split_value if isinstance(split_value, list) else [split_value]
        portable_values = []
        for value in split_values:
            split_path = Path(str(value)).expanduser()
            if not split_path.is_absolute():
                split_path = dataset_root / split_path
            split_path = split_path.resolve()
            try:
                relative_path = split_path.relative_to(dataset_root)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dataset {split} path points outside the dataset directory.",
                ) from exc
            if not split_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset {split} path was not found: {relative_path}",
                )
            portable_values.append(relative_path.as_posix())

        portable_payload[split] = portable_values if isinstance(split_value, list) else portable_values[0]

    if not portable_payload.get("train") or not portable_payload.get("val"):
        raise HTTPException(status_code=400, detail="Prepared dataset must contain train and val paths.")

    return yaml_path, dataset_root, portable_payload


def build_dataset_archive(dataset_yaml: str) -> tuple[Path, str]:
    yaml_path, dataset_root, portable_payload = prepared_dataset_yaml(dataset_yaml)
    archive_name = f"{clean_name(dataset_root.name, 'prepared_dataset')}.zip"
    source_yaml_relative = (
        yaml_path.relative_to(dataset_root)
        if yaml_path.is_relative_to(dataset_root)
        else None
    )
    descriptor, archive_value = tempfile.mkstemp(prefix="yolov8-dataset-", suffix=".zip")
    os.close(descriptor)
    archive_path = Path(archive_value)

    try:
        with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
            archive.writestr(
                "data.yaml",
                yaml.safe_dump(portable_payload, sort_keys=False),
                compress_type=zipfile.ZIP_DEFLATED,
            )
            files = sorted(
                (item for item in dataset_root.rglob("*") if item.is_file()),
                key=lambda item: item.relative_to(dataset_root).as_posix(),
            )
            for item in files:
                relative_path = item.relative_to(dataset_root)
                if (
                    relative_path.as_posix() == "data.yaml"
                    or relative_path == source_yaml_relative
                    or item.is_symlink()
                ):
                    continue
                compression = (
                    zipfile.ZIP_STORED
                    if item.suffix.lower() in IMAGE_EXTENSIONS
                    else zipfile.ZIP_DEFLATED
                )
                archive.write(item, relative_path.as_posix(), compress_type=compression)
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise

    return archive_path, archive_name


def build_annotated_dataset_archive(dataset_yaml: str) -> tuple[Path, str]:
    _yaml_path, dataset_root, portable_payload = prepared_dataset_yaml(dataset_yaml)
    try:
        from vision.image_annotation import annotate_dataset
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Annotated dataset export requires Pillow and the image_annotation module.",
        ) from exc

    output_root = Path(tempfile.mkdtemp(prefix="annotated-dataset-"))
    descriptor, archive_value = tempfile.mkstemp(prefix="annotated-yolov8-dataset-", suffix=".zip")
    os.close(descriptor)
    archive_path = Path(archive_value)
    archive_name = f"{clean_name(dataset_root.name, 'prepared_dataset')}_annotated.zip"

    try:
        annotated, missing_labels = annotate_dataset(dataset_root, output_root, max_images=0, line_width=0)
        if annotated == 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No annotated images were created. "
                    f"Images without matching labels: {missing_labels}."
                ),
            )

        with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
            archive.writestr(
                "data.yaml",
                yaml.safe_dump(portable_payload, sort_keys=False),
                compress_type=zipfile.ZIP_DEFLATED,
            )
            for item in sorted(output_root.rglob("*"), key=lambda path: path.relative_to(output_root).as_posix()):
                if not item.is_file() or item.is_symlink():
                    continue
                relative_path = item.relative_to(output_root)
                compression = (
                    zipfile.ZIP_STORED
                    if item.suffix.lower() in IMAGE_EXTENSIONS
                    else zipfile.ZIP_DEFLATED
                )
                archive.write(item, relative_path.as_posix(), compress_type=compression)
    except HTTPException:
        archive_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        archive_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Annotated dataset export failed: {exc}") from exc
    finally:
        shutil.rmtree(output_root, ignore_errors=True)

    return archive_path, archive_name


def cleanup_expired_dataset_downloads():
    cutoff = time.time() - DATASET_DOWNLOAD_MAX_AGE_SECONDS
    expired_paths = []
    with dataset_download_lock:
        for download_id, download in list(dataset_downloads.items()):
            if download["created_at"] < cutoff:
                expired_paths.append(Path(download["path"]))
                dataset_downloads.pop(download_id, None)
    for path in expired_paths:
        path.unlink(missing_ok=True)


def label_folder_for_images(dataset_root: Path, images_path: Optional[Path]) -> Optional[Path]:
    if images_path is None:
        return None
    parts = list(images_path.parts)
    if "images" in parts:
        index = parts.index("images")
        parts[index] = "labels"
        return Path(*parts)
    relative = images_path.relative_to(dataset_root) if images_path.is_relative_to(dataset_root) else images_path.name
    return dataset_root / "labels" / relative


def update_class_distribution(
    images: list[Path],
    labels_path: Optional[Path],
    distribution: dict[int, dict],
    progress_callback: Optional[Callable[[int], None]] = None,
) -> tuple[int, int, int]:
    missing_labels = 0
    malformed_rows = 0
    unknown_class_rows = 0
    for index, image in enumerate(images, start=1):
        label_path = labels_path / f"{image.stem}.txt" if labels_path else None
        if label_path is None or not label_path.is_file():
            missing_labels += 1
            if progress_callback:
                progress_callback(index)
            continue

        classes_in_image: set[int] = set()
        for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.strip().split()
            if not fields:
                continue
            try:
                class_id = int(fields[0])
            except ValueError:
                malformed_rows += 1
                continue
            if class_id not in distribution:
                unknown_class_rows += 1
                continue
            distribution[class_id]["instances"] += 1
            classes_in_image.add(class_id)

        for class_id in classes_in_image:
            distribution[class_id]["images"] += 1
        if progress_callback:
            progress_callback(index)

    return missing_labels, malformed_rows, unknown_class_rows


def read_split_metadata(dataset_root: Path, yaml_path: Path) -> dict:
    candidates = [dataset_root / SPLIT_METADATA_FILE, yaml_path.parent / SPLIT_METADATA_FILE]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def inspect_dataset_yaml(
    yaml_path: Path,
    classes: list[str],
    progress_callback: Optional[DatasetProgressCallback] = None,
    source_metadata: Optional[dict] = None,
) -> dict:
    warnings = []
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return {"warnings": [f"Could not inspect dataset YAML: {exc}"]}

    dataset_root = resolve_yaml_dataset_root(yaml_path, payload)
    splits = {}
    total_images = 0
    total_missing = 0
    malformed_rows = 0
    unknown_class_rows = 0
    class_distribution = {
        class_id: {
            "class_id": class_id,
            "class_name": class_name,
            "images": 0,
            "instances": 0,
        }
        for class_id, class_name in enumerate(classes)
    }
    split_distributions = {}
    split_contexts = []
    for split in SPLIT_NAMES:
        images_path = split_image_folder(dataset_root, payload.get(split))
        labels_path = label_folder_for_images(dataset_root, images_path)
        images = image_files(images_path) if images_path and images_path.is_dir() else []
        split_distribution = {
            class_id: {
                "class_id": class_id,
                "class_name": class_name,
                "images": 0,
                "instances": 0,
            }
            for class_id, class_name in enumerate(classes)
        }
        total_images += len(images)
        split_contexts.append((split, images_path, labels_path, images, split_distribution))

    inspected_images = 0
    if progress_callback:
        progress_callback("inspecting", 0, total_images, f"Inspecting 0 of {total_images} images")

    for split, images_path, labels_path, images, split_distribution in split_contexts:
        split_start = inspected_images
        inspection_progress = None
        if progress_callback:
            def inspection_progress(count: int, offset: int = split_start):
                current = offset + count
                progress_callback(
                    "inspecting",
                    current,
                    total_images,
                    f"Inspecting {current} of {total_images} images",
                )
        missing_labels, split_malformed, split_unknown = update_class_distribution(
            images,
            labels_path,
            split_distribution,
            inspection_progress,
        )
        inspected_images += len(images)
        total_missing += missing_labels
        malformed_rows += split_malformed
        unknown_class_rows += split_unknown
        split_distributions[split] = split_distribution
        for class_id, row in split_distribution.items():
            class_distribution[class_id]["images"] += row["images"]
            class_distribution[class_id]["instances"] += row["instances"]
        splits[split] = {
            "images": len(images),
            "missing_labels": missing_labels,
            "image_path": str(images_path) if images_path else "",
            "label_path": str(labels_path) if labels_path else "",
            "class_distribution": list(split_distribution.values()),
        }

    split_metadata = read_split_metadata(dataset_root, yaml_path)
    split_strategy = split_metadata.get("strategy") or "existing"
    configured_ratios = split_metadata.get("ratios") or {}
    ratios = {
        split: float(configured_ratios.get(split, 0))
        for split in SPLIT_NAMES
    }
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-6):
        ratios = {
            split: (splits[split]["images"] / total_images if total_images else 0.0)
            for split in SPLIT_NAMES
        }

    active_splits = [split for split in SPLIT_NAMES if ratios[split] > 0]
    class_balance = []
    for class_id, class_name in enumerate(classes):
        total_class_images = class_distribution[class_id]["images"]
        total_class_instances = class_distribution[class_id]["instances"]
        balance_splits = {}
        max_image_deviation = 0.0
        for split in SPLIT_NAMES:
            row = split_distributions[split][class_id]
            image_share = (
                row["images"] / total_class_images
                if total_class_images else 0.0
            )
            instance_share = (
                row["instances"] / total_class_instances
                if total_class_instances else 0.0
            )
            image_deviation = (image_share - ratios[split]) * 100
            max_image_deviation = max(max_image_deviation, abs(image_deviation))
            balance_splits[split] = {
                "images": row["images"],
                "instances": row["instances"],
                "image_share": round(image_share * 100, 2),
                "instance_share": round(instance_share * 100, 2),
                "target_share": round(ratios[split] * 100, 2),
                "image_deviation": round(image_deviation, 2),
            }

        class_balance.append({
            "class_id": class_id,
            "class_name": class_name,
            "images": total_class_images,
            "instances": total_class_instances,
            "splits": balance_splits,
            "max_image_deviation": round(max_image_deviation, 2),
        })

        if total_class_images == 0:
            warnings.append(f"Class '{class_name}' has no labeled images.")
        elif total_class_images < len(active_splits):
            warnings.append(
                f"Class '{class_name}' appears in only {total_class_images} "
                f"image{'s' if total_class_images != 1 else ''}; representation in all "
                f"{len(active_splits)} splits is not possible."
            )
        elif split_strategy == "multi_label_stratified":
            missing_splits = [
                split for split in active_splits
                if balance_splits[split]["images"] == 0
            ]
            if missing_splits:
                warnings.append(
                    f"Class '{class_name}' could not be represented in: "
                    f"{', '.join(missing_splits)}."
                )
            elif total_class_images >= 10 and max_image_deviation > 10:
                warnings.append(
                    f"Class '{class_name}' differs from the requested split ratio by up "
                    f"to {max_image_deviation:.1f} percentage points because of "
                    "multi-label constraints."
                )

    if not classes:
        warnings.append("No class names found.")
    if splits["train"]["images"] == 0:
        warnings.append("No training images found.")
    if splits["val"]["images"] == 0:
        warnings.append("No validation images found.")
    if total_missing:
        warnings.append(f"{total_missing} images are missing label files.")
    if malformed_rows:
        warnings.append(f"{malformed_rows} malformed annotation rows were ignored.")
    if unknown_class_rows:
        warnings.append(f"{unknown_class_rows} annotations reference unknown class IDs.")

    summary = {
        "dataset_root": str(dataset_root),
        "class_count": len(classes),
        "classes": classes,
        "class_distribution": list(class_distribution.values()),
        "class_balance": class_balance,
        "splits": splits,
        "split_strategy": split_strategy,
        "split_ratios": {split: round(ratios[split] * 100, 2) for split in SPLIT_NAMES},
        "split_seed": split_metadata.get("seed"),
        "total_images": total_images,
        "missing_labels": total_missing,
        "warnings": warnings,
    }
    pre_augmentation = roboflow_pre_augmentation_summary(dataset_root, splits, source_metadata)
    if pre_augmentation:
        summary["pre_augmentation"] = pre_augmentation
    return summary


def float_value(row: dict, key: str) -> Optional[float]:
    value = row.get(key)
    if value is None:
        value = row.get(f" {key}")
    if value is None:
        return None
    try:
        parsed = float(str(value).strip())
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def sum_values(row: dict, keys: list[str]) -> Optional[float]:
    values = [float_value(row, key) for key in keys]
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def loss_components(row: dict, prefix: str) -> dict[str, float]:
    components = {}
    for key in row:
        normalized = str(key).strip()
        if not normalized.startswith(f"{prefix}/"):
            continue
        suffix = normalized.removeprefix(f"{prefix}/")
        if suffix == "loss":
            component = "loss"
        elif suffix.endswith("_loss"):
            component = suffix.removesuffix("_loss")
        else:
            continue
        parsed = float_value(row, normalized)
        if parsed is not None:
            components[component] = parsed
    return components


def raw_loss_sum(row: dict, prefix: str) -> Optional[float]:
    components = loss_components(row, prefix)
    if components:
        return sum(components.values())
    return sum_values(row, [f"{prefix}/box_loss", f"{prefix}/cls_loss", f"{prefix}/dfl_loss"])


def comparable_loss_component_names(task: str, row: dict) -> list[str]:
    train = loss_components(row, "train")
    val = loss_components(row, "val")
    if task == "classify":
        candidates = ["loss"] if "loss" in train and "loss" in val else ["cls"]
    elif task == "segment":
        candidates = ["box", "seg", "cls", "dfl"]
    else:
        candidates = ["box", "cls", "dfl"]
    return [component for component in candidates if component in train and component in val]


def loss_summary(row: dict, task: str) -> dict:
    train = loss_components(row, "train")
    val = loss_components(row, "val")
    comparable_components = comparable_loss_component_names(task, row)
    train_display_components = list(comparable_components)
    val_display_components = list(comparable_components)
    if not comparable_components and "loss" in train:
        train_display_components = ["loss"]
        val_display_components = ["loss"] if "loss" in val else []

    def component_sum(components: dict[str, float], names: list[str]) -> Optional[float]:
        if not names:
            return None
        return sum(components[name] for name in names)

    auxiliary_train = {
        key: value for key, value in train.items() if key not in train_display_components
    }
    auxiliary_val = {
        key: value for key, value in val.items() if key not in val_display_components
    }
    return {
        "training_loss": component_sum(train, train_display_components),
        "testing_loss": component_sum(val, val_display_components),
        "raw_training_loss": sum(train.values()) if train else None,
        "raw_testing_loss": sum(val.values()) if val else None,
        "auxiliary_training_loss": sum(auxiliary_train.values()) if auxiliary_train else None,
        "auxiliary_testing_loss": sum(auxiliary_val.values()) if auxiliary_val else None,
        "loss_components": {
            "train": train,
            "val": val,
            "comparable": comparable_components,
            "auxiliary": {
                "train": auxiliary_train,
                "val": auxiliary_val,
            },
        },
    }


def loss_note(summary: dict) -> str:
    components = summary.get("loss_components") or {}
    auxiliary = components.get("auxiliary") or {}
    auxiliary_names = sorted({
        f"{prefix}/{name}_loss" if name != "loss" else f"{prefix}/loss"
        for prefix in ("train", "val")
        for name in (auxiliary.get(prefix) or {})
    })
    if auxiliary_names:
        return (
            "Train and val loss use matching loss parts only; extra model-specific losses are excluded: "
            f"{', '.join(auxiliary_names)}."
        )
    return "Train and val loss use matching loss parts."


def format_metric(value: Optional[float], digits: int = 4):
    return round(value, digits) if value is not None and math.isfinite(value) else None


def f1_from_precision_recall(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None or precision + recall <= 0:
        return None
    return 2 * precision * recall / (precision + recall)


def parse_metric_row(line: str) -> Optional[dict]:
    parts = clean_log_line(line).split()
    if len(parts) < 7:
        return None
    if parts[0].lower() == "all":
        return None

    numeric_tail = []
    for part in reversed(parts):
        value = float_value({"value": part}, "value")
        if value is None:
            break
        numeric_tail.append(value)
    numeric_tail.reverse()

    if len(numeric_tail) >= 10:
        metric_values = numeric_tail[-10:]
        class_parts = parts[: -10]
        images, instances = metric_values[0], metric_values[1]
        precision, recall, map50, map50_95 = metric_values[-4:]
    elif len(numeric_tail) >= 6:
        metric_values = numeric_tail[-6:]
        class_parts = parts[: -6]
        images, instances, precision, recall, map50, map50_95 = metric_values
    else:
        return None

    class_name = " ".join(class_parts).strip()
    if not class_name or class_name.lower() in {"all", "class", "epoch"}:
        return None

    return {
        "class_name": class_name,
        "images": int(images),
        "instances": int(instances),
        "precision": format_metric(precision),
        "recall": format_metric(recall),
        "f1": format_metric(f1_from_precision_recall(precision, recall)),
        "map50": format_metric(map50),
        "map50_95": format_metric(map50_95),
    }


def parse_class_metrics_from_log(log_path: Path) -> dict:
    if not log_path.is_file():
        return {"classes": [], "macro_f1": None, "weighted_f1": None}

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    validating_indexes = [index for index, line in enumerate(lines) if "Validating " in clean_log_line(line)]
    search_lines = lines[validating_indexes[-1] + 1:] if validating_indexes else lines

    classes_by_name = {}
    for line in search_lines:
        row = parse_metric_row(line)
        if not row:
            continue
        if row["class_name"] == "all":
            continue
        classes_by_name[row["class_name"]] = row

    classes = list(classes_by_name.values())
    macro_f1 = None
    if classes:
        macro_f1 = sum(row["f1"] or 0 for row in classes) / len(classes)

    total_instances = sum(row["instances"] for row in classes)
    weighted_f1 = None
    if total_instances:
        weighted_f1 = sum((row["f1"] or 0) * row["instances"] for row in classes) / total_instances

    return {
        "classes": classes,
        "macro_f1": format_metric(macro_f1),
        "weighted_f1": format_metric(weighted_f1),
    }


def infer_task_from_results_columns(row: dict) -> str:
    keys = {str(key).strip() for key in row}
    if "metrics/mAP50(M)" in keys or "metrics/mAP50-95(M)" in keys:
        return "segment"
    if "metrics/accuracy_top1" in keys or "metrics/accuracy_top5" in keys:
        return "classify"
    return "detect"


def infer_run_task(run_dir: Path, row: Optional[dict] = None) -> str:
    context = load_training_report_context(run_dir)
    task = str(context.get("task") or context.get("hyperparameters", {}).get("task") or "").lower()
    if task in {"detect", "segment", "semantic", "classify"}:
        return task
    parts = {part.lower() for part in run_dir.parts}
    if "segment" in parts:
        return "segment"
    if "semantic" in parts:
        return "semantic"
    if "classify" in parts:
        return "classify"
    if row:
        return infer_task_from_results_columns(row)
    return "detect"


def metric_profile(task: str, row: dict) -> dict:
    if task == "segment":
        use_mask = float_value(row, "metrics/mAP50(M)") is not None or float_value(row, "metrics/mAP50-95(M)") is not None
        suffix = "M" if use_mask else "B"
        metric_type = "mask" if use_mask else "box"
        prefix = "Mask " if use_mask else "Box "
        return {
            "task": "segment",
            "metric_type": metric_type,
            "metric_label": "Segmentation Mask" if use_mask else "Segmentation Box",
            "chart_title": "Segmentation Mask Performance by Epoch" if use_mask else "Segmentation Box Performance by Epoch",
            "precision_key": f"metrics/precision({suffix})",
            "recall_key": f"metrics/recall({suffix})",
            "map50_key": f"metrics/mAP50({suffix})",
            "map50_95_key": f"metrics/mAP50-95({suffix})",
            "labels": {
                "precision": f"{prefix}Precision",
                "recall": f"{prefix}Recall",
                "map50": f"{prefix}mAP50",
                "map50_95": f"{prefix}mAP50-95",
            },
        }
    if task == "classify":
        return {
            "task": "classify",
            "metric_type": "classification",
            "metric_label": "Classification",
            "chart_title": "Classification Performance by Epoch",
            "precision_key": None,
            "recall_key": None,
            "map50_key": "metrics/accuracy_top1",
            "map50_95_key": "metrics/accuracy_top5",
            "labels": {
                "precision": "Precision",
                "recall": "Recall",
                "map50": "Top-1 Accuracy",
                "map50_95": "Top-5 Accuracy",
            },
        }
    return {
        "task": "detect",
        "metric_type": "box",
        "metric_label": "Detection",
        "chart_title": "Detection Performance by Epoch",
        "precision_key": "metrics/precision(B)",
        "recall_key": "metrics/recall(B)",
        "map50_key": "metrics/mAP50(B)",
        "map50_95_key": "metrics/mAP50-95(B)",
        "labels": {
            "precision": "Precision",
            "recall": "Recall",
            "map50": "mAP50",
            "map50_95": "mAP50-95",
        },
    }


def build_metric_history(rows: list[dict], profile: dict) -> list[dict]:
    history = []
    for row in rows:
        losses = loss_summary(row, profile["task"])
        history.append({
            "epoch": int(float_value(row, "epoch") or 0),
            "map50": format_metric(float_value(row, profile["map50_key"])) if profile.get("map50_key") else None,
            "map50_95": format_metric(float_value(row, profile["map50_95_key"])) if profile.get("map50_95_key") else None,
            "training_loss": format_metric(losses["training_loss"]),
            "testing_loss": format_metric(losses["testing_loss"]),
            "raw_training_loss": format_metric(losses["raw_training_loss"]),
            "raw_testing_loss": format_metric(losses["raw_testing_loss"]),
            "auxiliary_training_loss": format_metric(losses["auxiliary_training_loss"]),
            "auxiliary_testing_loss": format_metric(losses["auxiliary_testing_loss"]),
            "loss_components": losses["loss_components"],
        })
    return history


def best_metric_summary(history: list[dict]) -> dict:
    def best_by(key: str, higher_is_better: bool = True):
        candidates = [row for row in history if row.get(key) is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda row: row[key]) if higher_is_better else min(candidates, key=lambda row: row[key])

    best_map95 = best_by("map50_95")
    best_map50 = best_by("map50")
    best_train_loss = best_by("training_loss", higher_is_better=False)
    best_val_loss = best_by("testing_loss", higher_is_better=False)
    return {
        "best_map50_95": best_map95,
        "best_map50": best_map50,
        "lowest_training_loss": best_train_loss,
        "lowest_validation_loss": best_val_loss,
    }


def artifact_status(path: Path) -> dict:
    available = path.is_file()
    stat = path.stat() if available else None
    return {
        "available": available,
        "path": str(path),
        "size": stat.st_size if stat else 0,
        "modified_at": str(stat.st_mtime_ns) if stat else "0",
    }


def run_artifact_statuses(run_dir: Path) -> dict:
    return {
        "results_csv": artifact_status(run_dir / "results.csv"),
        "accuracy_graph": artifact_status(run_dir / "accuracy_by_epoch.png"),
        "loss_graph": artifact_status(run_dir / "loss_by_epoch.png"),
        "confusion_matrix": artifact_status(run_dir / "confusion_matrix.png"),
        "confusion_matrix_normalized": artifact_status(
            run_dir / "confusion_matrix_normalized.png"
        ),
        "roc_auc_curve": artifact_status(run_dir / "roc_auc_curve.png"),
    }


def read_web_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "web_metrics.json"
    if not metrics_path.is_file():
        return {}
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def is_rfdetr_run(run_dir: Path) -> bool:
    parts = {part.lower() for part in run_dir.parts}
    if "rfdetr" in parts:
        return True
    context = read_json_object(run_dir / TRAINING_REPORT_CONTEXT_FILE)
    family = str(context.get("family") or context.get("hyperparameters", {}).get("family") or "").lower()
    return family == "rfdetr"


def rfdetr_class_names_from_context(context: dict) -> list[str]:
    summary = context.get("dataset_summary") if isinstance(context.get("dataset_summary"), dict) else {}
    classes = summary.get("classes")
    if isinstance(classes, list):
        return [str(name) for name in classes]
    distribution = summary.get("class_distribution")
    if isinstance(distribution, list):
        return [str(row.get("class_name")) for row in distribution if row.get("class_name")]
    return []


def rfdetr_dataset_audit_from_context(context: dict) -> dict:
    class_names = rfdetr_class_names_from_context(context)
    summary = context.get("dataset_summary") if isinstance(context.get("dataset_summary"), dict) else {}
    summary_splits = summary.get("splits") if isinstance(summary.get("splits"), dict) else {}
    splits = {}
    for source_name, target_name in (("train", "train"), ("val", "valid"), ("valid", "valid"), ("test", "test")):
        split = summary_splits.get(source_name)
        if not isinstance(split, dict) or target_name in splits:
            continue
        rows = []
        for row in split.get("class_distribution") or []:
            rows.append({
                "class_id": row.get("class_id"),
                "class_name": row.get("class_name"),
                "images": int(row.get("images") or 0),
                "instances": int(row.get("instances") or 0),
            })
        missing_ids = [
            index
            for index, row in enumerate(rows)
            if int(row.get("instances") or 0) == 0
        ]
        splits[target_name] = {
            "split": target_name,
            "label_path": split.get("label_path", ""),
            "classes": rows,
            "present_class_ids": [row.get("class_id") for row in rows if int(row.get("instances") or 0) > 0],
            "missing_class_ids": missing_ids,
            "malformed_labels": 0,
            "warnings": split.get("warnings") or [],
        }
    return {
        "dataset_dir": summary.get("dataset_root", ""),
        "classes": class_names,
        "splits": splits,
    }


def rfdetr_model_id_from_context(context: dict) -> str:
    hyperparameters = context.get("hyperparameters") if isinstance(context.get("hyperparameters"), dict) else {}
    value = context.get("model") or hyperparameters.get("model") or hyperparameters.get("model_size")
    return str(value or RFDETR_DEFAULTS["model"])


def rfdetr_epochs_from_context(context: dict) -> int:
    hyperparameters = context.get("hyperparameters") if isinstance(context.get("hyperparameters"), dict) else {}
    try:
        return max(1, int(hyperparameters.get("epochs") or 1))
    except (TypeError, ValueError):
        return 1


def find_rfdetr_log_for_run(run_dir: Path) -> Optional[Path]:
    candidates = [run_dir / RFDETR_RUN_LOG]
    if training_log_file is not None:
        candidates.append(training_log_file)
    candidates.append(LOG_FILE)
    candidates.extend(sorted(LOG_DIR.glob("train-*.log"), key=lambda path: path.stat().st_mtime, reverse=True))

    seen = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        data = read_log_file(path)
        if run_dir.name in data or str(run_dir) in data:
            return path
    return None


def ensure_rfdetr_web_artifacts(run_dir: Path, force: bool = False):
    if not is_rfdetr_run(run_dir):
        return
    if not force and (run_dir / "results.csv").is_file() and (run_dir / "web_metrics.json").is_file():
        return
    log_path = find_rfdetr_log_for_run(run_dir)
    if log_path is None:
        return
    context = read_json_object(run_dir / TRAINING_REPORT_CONTEXT_FILE)
    try:
        finalize_rfdetr_artifacts(
            run_dir,
            model_id=rfdetr_model_id_from_context(context),
            class_names=rfdetr_class_names_from_context(context),
            epochs=rfdetr_epochs_from_context(context),
            log_path=log_path,
            dataset_audit=rfdetr_dataset_audit_from_context(context),
            training_completed=None,
            quiet=True,
        )
    except Exception:
        return


def read_run_metrics(run_dir: Path) -> dict:
    ensure_rfdetr_web_artifacts(run_dir, force=is_rfdetr_run(run_dir))
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        return {
            "available": False,
            "run_dir": str(run_dir),
            "results_csv": "",
            "artifacts": run_artifact_statuses(run_dir),
        }

    with results_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {
            "available": False,
            "run_dir": str(run_dir),
            "results_csv": str(results_path),
            "artifacts": run_artifact_statuses(run_dir),
        }

    row = rows[-1]
    task = infer_run_task(run_dir, row)
    profile = metric_profile(task, row)
    precision = float_value(row, profile["precision_key"]) if profile.get("precision_key") else None
    recall = float_value(row, profile["recall_key"]) if profile.get("recall_key") else None
    losses = loss_summary(row, profile["task"])
    map50 = float_value(row, profile["map50_key"]) if profile.get("map50_key") else None
    map50_95 = float_value(row, profile["map50_95_key"]) if profile.get("map50_95_key") else None
    web_metrics = read_web_metrics(run_dir)
    backend = web_metrics.get("backend") or family_for_runs_root(run_dir.parent)
    class_metrics = {
        "macro_f1": web_metrics.get("macro_f1"),
        "weighted_f1": web_metrics.get("weighted_f1"),
        "classes": web_metrics.get("per_class"),
    }
    if not isinstance(class_metrics["classes"], list) or not class_metrics["classes"]:
        class_metrics = parse_class_metrics_from_log(LOG_FILE)
    roc_auc = web_metrics.get("roc_auc")
    if not isinstance(roc_auc, dict):
        roc_auc = {
            "mode": "image_presence",
            "split": "val",
            "classes": [],
            "note": "ROC-AUC will appear after web metrics are generated for this run.",
        }
    history = build_metric_history(rows, profile)
    metrics_note = (
        web_metrics.get("per_class_note")
        if backend in {"dfine", "rfdetr"} and web_metrics.get("per_class_note")
        else "Macro and weighted F1 are calculated from final per-class validation rows when available. "
        + loss_note(losses)
    )

    return {
        "available": True,
        "run_dir": str(run_dir),
        "results_csv": str(results_path),
        "backend": backend,
        "per_class_source": web_metrics.get("per_class_source"),
        "per_class_note": web_metrics.get("per_class_note"),
        "epoch": int(float_value(row, "epoch") or 0),
        "task": profile["task"],
        "metric_type": profile["metric_type"],
        "metric_label": profile["metric_label"],
        "metric_labels": profile["labels"],
        "chart_title": profile["chart_title"],
        "macro_f1": class_metrics["macro_f1"],
        "weighted_f1": class_metrics["weighted_f1"],
        "per_class": class_metrics["classes"],
        "roc_auc": roc_auc,
        "training_loss": format_metric(losses["training_loss"]),
        "testing_loss": format_metric(losses["testing_loss"]),
        "raw_training_loss": format_metric(losses["raw_training_loss"]),
        "raw_testing_loss": format_metric(losses["raw_testing_loss"]),
        "auxiliary_training_loss": format_metric(losses["auxiliary_training_loss"]),
        "auxiliary_testing_loss": format_metric(losses["auxiliary_testing_loss"]),
        "loss_components": losses["loss_components"],
        "precision": format_metric(precision),
        "recall": format_metric(recall),
        "map50": format_metric(map50),
        "map50_95": format_metric(map50_95),
        "history": history,
        "best": best_metric_summary(history),
        "artifacts": run_artifact_statuses(run_dir),
        "note": metrics_note,
    }


def validate_split(split: SplitConfig) -> tuple[float, float, float]:
    total = split.train + split.val + split.test
    if total != 100:
        raise HTTPException(status_code=400, detail="Train, val, and test split values must total 100%.")
    return split.train / total, split.val / total, split.test / total


def find_dataset_yaml(root: Path) -> Optional[Path]:
    for name in ("data.yaml", "dataset.yaml"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    matches = list(root.rglob("data.yaml")) + list(root.rglob("dataset.yaml"))
    return matches[0] if matches else None


def find_dataset_root(root: Path) -> Path:
    if (root / "images").is_dir() or (root / "train").is_dir() or find_dataset_yaml(root):
        return root

    for candidate in root.rglob("*"):
        if not candidate.is_dir():
            continue
        if (candidate / "images").is_dir() or (candidate / "train").is_dir() or find_dataset_yaml(candidate):
            return candidate

    return root


def split_dirs(root: Path) -> dict[str, dict[str, Path]]:
    layouts: dict[str, dict[str, Path]] = {}

    for alias, split in SPLIT_ALIASES.items():
        images_a = root / "images" / alias
        labels_a = root / "labels" / alias
        if images_a.is_dir():
            layouts[split] = {"images": images_a, "labels": labels_a}

        images_b = root / alias / "images"
        labels_b = root / alias / "labels"
        if images_b.is_dir():
            layouts[split] = {"images": images_b, "labels": labels_b}

    return layouts


def flat_dirs(root: Path) -> Optional[dict[str, Path]]:
    images = root / "images"
    labels = root / "labels"
    if images.is_dir():
        return {"images": images, "labels": labels}
    return None


def image_files(folder: Path) -> list[Path]:
    return sorted(
        item for item in folder.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def has_image_files(folder: Optional[Path]) -> bool:
    return bool(
        folder
        and folder.is_dir()
        and any(
            item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            for item in folder.rglob("*")
        )
    )


def copy_pair(image_path: Path, label_dir: Path, output_images: Path, output_labels: Path):
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    target_image = output_images / image_path.name
    target_label = output_labels / f"{image_path.stem}.txt"
    source_label = label_dir / f"{image_path.stem}.txt"

    shutil.copy2(image_path, target_image)
    if source_label.is_file():
        shutil.copy2(source_label, target_label)
    else:
        target_label.write_text("", encoding="utf-8")


def write_dataset_yaml(output_dir: Path, dataset_root: Path, names: dict[int, str], layout: dict[str, str]):
    yaml_path = output_dir / "data.yaml"
    payload = {
        "path": str(dataset_root),
        "train": layout["train"],
        "val": layout["val"],
        "names": names,
    }
    if layout.get("test"):
        payload["test"] = layout["test"]

    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)

    return yaml_path


def prepare_existing_split(root: Path, name: str, names: dict[int, str]) -> Path:
    layouts = split_dirs(root)
    if "train" not in layouts or "val" not in layouts:
        raise HTTPException(
            status_code=400,
            detail="Dataset must contain train and val/valid image folders.",
        )

    output_dir = DATA_ROOT / "prepared" / clean_name(name, "dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SPLIT_METADATA_FILE).unlink(missing_ok=True)

    train_path = layouts["train"]["images"].relative_to(root)
    val_path = layouts["val"]["images"].relative_to(root)
    test_path = layouts.get("test", {}).get("images")

    layout = {
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path.relative_to(root)) if test_path else "",
    }
    return write_dataset_yaml(output_dir, root, names, layout)


def collect_source_images(root: Path) -> list[tuple[Path, Path]]:
    collected: list[tuple[Path, Path]] = []

    flat = flat_dirs(root)
    if flat:
        collected.extend((image, flat["labels"]) for image in image_files(flat["images"]))

    for layout in split_dirs(root).values():
        collected.extend((image, layout["labels"]) for image in image_files(layout["images"]))

    seen: set[Path] = set()
    unique: list[tuple[Path, Path]] = []
    for image, labels in collected:
        if image in seen:
            continue
        seen.add(image)
        unique.append((image, labels))

    return unique


def prepare_split_dataset(
    root: Path,
    name: str,
    names: dict[int, str],
    split: SplitConfig,
    progress_callback: Optional[DatasetProgressCallback] = None,
) -> Path:
    source_images = collect_source_images(root)
    if not source_images:
        raise HTTPException(status_code=400, detail="No images found in the dataset path.")

    train_ratio, val_ratio, test_ratio = validate_split(split)
    split_progress = None
    if progress_callback:
        stage_details = {
            "reading_labels": lambda current, total: f"Reading labels: {current} of {total} images",
            "calculating_targets": lambda current, total: "Calculating per-class split targets",
            "assigning": lambda current, total: f"Assigning images: {current} of {total}",
            "finalizing_split": lambda current, total: "Finalizing split assignments",
        }

        def split_progress(stage: str, current: int, total: int):
            progress_callback(
                stage,
                current,
                total,
                stage_details[stage](current, total),
            )
    groups, diagnostics = stratified_split(
        source_images,
        {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        set(names),
        seed=42,
        progress_callback=split_progress,
    )

    output_root = DATA_ROOT / "prepared" / clean_name(name, "dataset")
    if output_root.exists():
        shutil.rmtree(output_root)

    copied = 0
    copy_total = len(source_images)
    if progress_callback:
        progress_callback("copying", 0, copy_total, f"Copying 0 of {copy_total} images")
    for split_name, pairs in groups.items():
        for image_path, label_dir in pairs:
            copy_pair(
                image_path,
                label_dir,
                output_root / "images" / split_name,
                output_root / "labels" / split_name,
            )
            copied += 1
            if progress_callback:
                progress_callback(
                    "copying",
                    copied,
                    copy_total,
                    f"Copying {copied} of {copy_total} images",
                )

    yaml_path = write_dataset_yaml(
        output_root,
        output_root,
        names,
        {"train": "images/train", "val": "images/val", "test": "images/test"},
    )
    (output_root / SPLIT_METADATA_FILE).write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return yaml_path


def prepare_dataset(
    source: Path,
    name: str,
    classes: list[str],
    split: SplitConfig,
    force_split: bool,
    progress_callback: Optional[DatasetProgressCallback] = None,
) -> Path:
    if source.is_file() and source.suffix.lower() in {".yaml", ".yml"}:
        return source

    root = find_dataset_root(source)
    names = resolve_dataset_classes(root, classes)

    if not force_split and "train" in split_dirs(root) and "val" in split_dirs(root):
        return prepare_existing_split(root, name, names)

    return prepare_split_dataset(root, name, names, split, progress_callback)


def prepare_roboflow_download(
    dataset_root: Path,
    name: str,
    classes: list[str],
    split: Optional[SplitConfig] = None,
    progress_callback: Optional[DatasetProgressCallback] = None,
) -> tuple[Path, str]:
    root = find_dataset_root(dataset_root)
    source_yaml = find_dataset_yaml(root)

    if source_yaml:
        try:
            payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Roboflow data.yaml could not be read: {exc}",
            ) from exc

        yaml_root = resolve_yaml_dataset_root(source_yaml, payload)
        train_path = split_image_folder(yaml_root, payload.get("train"))
        val_path = split_image_folder(yaml_root, payload.get("val"))
        test_value = payload.get("test")
        test_path = split_image_folder(yaml_root, test_value)
        test_is_valid = not test_value or (test_path and test_path.is_dir())
        if has_image_files(train_path) and has_image_files(val_path) and test_is_valid:
            return source_yaml, "valid"

    layouts = split_dirs(root)
    if (
        "train" in layouts
        and "val" in layouts
        and has_image_files(layouts["train"]["images"])
        and has_image_files(layouts["val"]["images"])
    ):
        names = resolve_dataset_classes(root, classes)
        return prepare_existing_split(root, name, names), "normalized"

    train_layout = layouts.get("train")
    if train_layout and has_image_files(train_layout["images"]):
        yaml_path = prepare_dataset(
            source=root,
            name=name,
            classes=classes,
            split=split or SplitConfig(),
            force_split=True,
            progress_callback=progress_callback,
        )
        return yaml_path, "rebuilt"

    if source_yaml:
        train_value = payload.get("train") or "<missing>"
        val_value = payload.get("val") or "<missing>"
        test_value = payload.get("test") or "<not configured>"
        raise HTTPException(
            status_code=400,
            detail=(
                "Roboflow dataset paths are invalid and no usable train/validation "
                f"folders were detected under {root}. data.yaml specifies "
                f"train={train_value!r}, val={val_value!r}, test={test_value!r}."
            ),
        )

    yaml_path = prepare_dataset(
        source=root,
        name=name,
        classes=classes,
        split=SplitConfig(),
        force_split=False,
    )
    return yaml_path, "valid"


def resolve_test_split_source(
    dataset_root: Path,
    payload: Optional[dict] = None,
) -> tuple[str, Optional[Path]]:
    if isinstance(payload, dict):
        for split_name in ("test", "val", "train"):
            images_path = split_image_folder(dataset_root, payload.get(split_name))
            if has_image_files(images_path):
                return split_name, images_path

    layouts = split_dirs(dataset_root)
    for split_name in ("test", "val", "train"):
        layout = layouts.get(split_name)
        if layout and has_image_files(layout["images"]):
            return split_name, layout["images"]

    flat = flat_dirs(dataset_root)
    if flat and has_image_files(flat["images"]):
        return "flat", flat["images"]

    return "", None


def build_test_dataset_yaml(
    dataset_root: Path,
    images_path: Path,
    classes: list[str],
    output_name: str,
) -> Path:
    dataset_root = dataset_root.resolve()
    images_path = images_path.resolve()
    labels_path = label_folder_for_images(dataset_root, images_path)
    if labels_path is None or not labels_path.exists():
        raise HTTPException(
            status_code=400,
            detail="The uploaded test dataset does not contain a matching labels folder.",
        )

    try:
        relative_images = images_path.relative_to(dataset_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="The detected test images are outside the dataset root.",
        ) from exc

    output_dir = DATA_ROOT / "test_prepared" / clean_name(output_name, "test_dataset")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "data.yaml"
    payload = {
        "path": str(dataset_root),
        "test": relative_images.as_posix(),
        "names": {index: name for index, name in enumerate(classes)},
    }
    yaml_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return yaml_path


def resolve_reference_classes(reference_dataset_yaml: str) -> list[str]:
    if not reference_dataset_yaml:
        return []
    reference_path, _, _ = prepared_dataset_yaml(reference_dataset_yaml)
    return read_yaml_class_names(reference_path)


def prepare_custom_test_dataset(
    source: Path,
    output_name: str,
    reference_dataset_yaml: str = "",
) -> tuple[Path, dict]:
    dataset_root = find_dataset_root(source)
    source_yaml = find_dataset_yaml(dataset_root)
    payload = None
    classes: list[str] = []
    yaml_root = dataset_root

    if source_yaml:
        try:
            payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read uploaded dataset YAML: {exc}",
            ) from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Uploaded dataset YAML must contain a mapping.")
        yaml_root = resolve_yaml_dataset_root(source_yaml, payload)
        classes = normalize_yaml_names(payload.get("names"))

    if not classes:
        classes = resolve_reference_classes(reference_dataset_yaml)
    if not classes:
        raise HTTPException(
            status_code=400,
            detail=(
                "No class names were found in the uploaded dataset. Include a data.yaml "
                "or prepare a dataset in the UI first so its class names can be reused."
            ),
        )

    selected_split, images_path = resolve_test_split_source(yaml_root, payload)
    if images_path is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "No usable labeled images were found in the uploaded test dataset. "
                "Include a YOLO images/labels layout or a data.yaml with a test split."
            ),
        )

    yaml_path = build_test_dataset_yaml(yaml_root, images_path, classes, output_name)
    return yaml_path, {
        "dataset_root": str(yaml_root),
        "source_split": selected_split,
        "classes": classes,
    }


def resolve_prepared_test_dataset_yaml(dataset_yaml: str) -> tuple[Path, dict]:
    yaml_path, dataset_root, portable_payload = prepared_dataset_yaml(dataset_yaml)
    if not portable_payload.get("test"):
        raise HTTPException(
            status_code=400,
            detail="The prepared dataset does not contain a test split to evaluate.",
        )
    return yaml_path, {
        "dataset_root": str(dataset_root),
        "source_split": "test",
        "classes": normalize_yaml_names(portable_payload.get("names")),
    }


def clean_log_line(raw_line: str) -> str:
    cleaned = ANSI_ESCAPE_RE.sub("", raw_line)
    cleaned = cleaned.replace("\r", "")
    cleaned = CONTROL_CHAR_RE.sub("", cleaned)
    return cleaned.strip()


def should_write_log_line(line: str) -> bool:
    if not line:
        return False

    if "it/s" in line and "%" in line:
        return False

    if PROGRESS_LINE_RE.search(line):
        return False

    return True


def read_json_object(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json_object(path: Path, payload: dict):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def cached_dataset_summary(yaml_path: Path) -> dict:
    summary = read_json_object(yaml_path.parent / DATASET_SUMMARY_FILE)
    if summary:
        return summary
    try:
        classes = read_yaml_class_names(yaml_path)
        summary = inspect_dataset_yaml(yaml_path, classes)
    except (HTTPException, OSError, yaml.YAMLError):
        return {}
    write_json_object(yaml_path.parent / DATASET_SUMMARY_FILE, summary)
    return summary


def annotation_qa_job_id() -> str:
    return f"{datetime.now(MYT).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def normalize_annotation_qa_model_name(model: str) -> str:
    value = str(model or "").strip() or ANNOTATION_QA_MODEL_DEFAULT
    aliases = {
        "sam2.1_hiera_tiny": "sam2.1_t.pt",
        "sam2.1_hiera_tiny.pt": "sam2.1_t.pt",
        "sam2.1_hiera_small": "sam2.1_s.pt",
        "sam2.1_hiera_small.pt": "sam2.1_s.pt",
        "sam2.1_hiera_base_plus": "sam2.1_b.pt",
        "sam2.1_hiera_base_plus.pt": "sam2.1_b.pt",
        "sam2.1_hiera_large": "sam2.1_l.pt",
        "sam2.1_hiera_large.pt": "sam2.1_l.pt",
    }
    return aliases.get(value, value)


def ensure_annotation_qa_path(path: Path):
    qa_root = ANNOTATION_QA_ROOT.resolve()
    try:
        path.resolve().relative_to(qa_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Annotation QA files can only be read from runs/annotation_qa.",
        ) from exc


def annotation_qa_payload(job: dict) -> dict:
    hidden = {"stop_event", "thread"}
    return {key: value for key, value in job.items() if key not in hidden}


def update_annotation_qa_job(job_id: str, **updates):
    with annotation_qa_jobs_lock:
        job = annotation_qa_jobs.get(job_id)
        if job is None:
            return
        job.update(updates)
        job["updated_at"] = time.time()


def annotation_qa_thresholds(preset: str) -> dict:
    presets = {
        "lenient": {
            "bbox_iou": 0.4,
            "center_shift": 0.3,
            "mask_area_ratio_min": 0.08,
            "loose_area_ratio": 0.35,
            "tight_edge_count": 3,
            "duplicate_iou": 0.9,
        },
        "strict": {
            "bbox_iou": 0.65,
            "center_shift": 0.15,
            "mask_area_ratio_min": 0.2,
            "loose_area_ratio": 0.6,
            "tight_edge_count": 2,
            "duplicate_iou": 0.8,
        },
    }
    return presets.get(str(preset or "").lower(), {
        "bbox_iou": 0.55,
        "center_shift": 0.2,
        "mask_area_ratio_min": 0.15,
        "loose_area_ratio": 0.5,
        "tight_edge_count": 2,
        "duplicate_iou": 0.85,
    })


def annotation_qa_split_names(scope: str, payload: dict) -> list[str]:
    normalized = str(scope or "val").lower()
    available = [split for split in SPLIT_NAMES if payload.get(split)]
    if normalized == "all":
        return available
    if normalized in SPLIT_NAMES and payload.get(normalized):
        return [normalized]
    if payload.get("val"):
        return ["val"]
    return available[:1]


def yolo_label_path(image_path: Path, labels_path: Optional[Path]) -> Optional[Path]:
    return labels_path / f"{image_path.stem}.txt" if labels_path else None


def pixel_bbox_from_yolo(fields: list[str], width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    if len(fields) != 5:
        return None
    try:
        _class_id = int(float(fields[0]))
        x_center, y_center, box_width, box_height = [float(value) for value in fields[1:5]]
    except ValueError:
        return None
    if box_width <= 0 or box_height <= 0:
        return None
    x1 = int(round((x_center - box_width / 2) * width))
    y1 = int(round((y_center - box_height / 2) * height))
    x2 = int(round((x_center + box_width / 2) * width))
    y2 = int(round((y_center + box_height / 2) * height))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def yolo_bbox_from_pixels(class_id, box: list[int] | tuple[int, int, int, int], width: int, height: int) -> str:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Corrected bounding box is empty.")
    x_center = ((x1 + x2) / 2) / width
    y_center = ((y1 + y2) / 2) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return " ".join([
        str(int(float(class_id))),
        f"{x_center:.6f}",
        f"{y_center:.6f}",
        f"{box_width:.6f}",
        f"{box_height:.6f}",
    ])


def bbox_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = bbox_area(box_a) + bbox_area(box_b) - intersection
    return intersection / union if union > 0 else 0.0


def bbox_center_shift(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax = (box_a[0] + box_a[2]) / 2
    ay = (box_a[1] + box_a[3]) / 2
    bx = (box_b[0] + box_b[2]) / 2
    by = (box_b[1] + box_b[3]) / 2
    scale = max(1.0, box_a[2] - box_a[0], box_a[3] - box_a[1])
    return math.hypot(ax - bx, ay - by) / scale


def mask_bbox(mask) -> Optional[tuple[int, int, int, int]]:
    import numpy as np

    rows, cols = np.where(mask > 0)
    if rows.size == 0 or cols.size == 0:
        return None
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def mask_to_uint8(mask, width: int, height: int):
    import cv2
    import numpy as np

    array = np.asarray(mask)
    if array.shape[:2] != (height, width):
        array = cv2.resize(array.astype("float32"), (width, height), interpolation=cv2.INTER_NEAREST)
    return (array > 0.5).astype("uint8")


def annotation_issue(
    job_id: str,
    index: int,
    image_path: Path,
    split: str,
    class_id: Optional[int],
    class_name: str,
    issue_type: str,
    severity: str,
    score: float,
    message: str,
    original_bbox: Optional[tuple[int, int, int, int]] = None,
    sam_bbox: Optional[tuple[int, int, int, int]] = None,
    metrics: Optional[dict] = None,
    preview: str = "",
    label_row: Optional[int] = None,
    recommended_bbox: Optional[tuple[int, int, int, int]] = None,
) -> dict:
    return {
        "issue_id": f"{job_id}-{index:06d}",
        "image": str(image_path),
        "image_name": image_path.name,
        "split": split,
        "class_id": class_id,
        "class_name": class_name,
        "issue_type": issue_type,
        "severity": severity,
        "score": round(float(score), 4),
        "message": message,
        "original_bbox": list(original_bbox) if original_bbox else None,
        "sam_bbox": list(sam_bbox) if sam_bbox else None,
        "recommended_bbox": list(recommended_bbox) if recommended_bbox else None,
        "fix_type": "replace_box" if recommended_bbox else "",
        "accepted_fix": "",
        "accepted_class_id": None,
        "accepted_class_name": "",
        "label_row": label_row,
        "metrics": metrics or {},
        "preview": preview,
        "review_status": "unreviewed",
    }


def draw_annotation_qa_preview(
    image_path: Path,
    output_path: Path,
    issue: dict,
    mask=None,
):
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        return
    height, width = image.shape[:2]
    if mask is not None:
        mask_array = mask_to_uint8(mask, width, height)
        overlay = image.copy()
        overlay[mask_array > 0] = (255, 220, 70)
        image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)
    original = issue.get("original_bbox")
    if original:
        x1, y1, x2, y2 = [int(value) for value in original]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 210, 255), 2)
    sam_box = issue.get("sam_bbox")
    if sam_box:
        x1, y1, x2, y2 = [int(value) for value in sam_box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 150, 0), 2)
    caption = f"{issue['severity'].upper()} {issue['issue_type']} {issue['score']:.2f}"
    cv2.rectangle(image, (8, 8), (min(width - 8, 16 + len(caption) * 9), 38), (20, 32, 40), -1)
    cv2.putText(image, caption, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 84])


def annotation_qa_summary(issues: list[dict], images_scanned: int, labels_checked: int, model: str, scope: str, preset: str) -> dict:
    severity_counts = {"high": 0, "medium": 0, "low": 0}
    type_counts: dict[str, int] = {}
    for issue in issues:
        severity = issue.get("severity", "low")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        issue_type = issue.get("issue_type", "unknown")
        type_counts[issue_type] = type_counts.get(issue_type, 0) + 1
    return {
        "model": model,
        "scope": scope,
        "preset": preset,
        "images_scanned": images_scanned,
        "labels_checked": labels_checked,
        "issues": len(issues),
        "high": severity_counts.get("high", 0),
        "medium": severity_counts.get("medium", 0),
        "low": severity_counts.get("low", 0),
        "issue_types": type_counts,
        "completed_at": datetime.now(MYT).isoformat(),
    }


def write_annotation_qa_report(run_dir: Path, issues: list[dict], summary: dict):
    write_json_object(run_dir / ANNOTATION_QA_SUMMARY_FILE, summary)
    write_json_object(run_dir / ANNOTATION_QA_REPORT_JSON_FILE, {"summary": summary, "issues": issues})
    write_json_object(run_dir / ANNOTATION_QA_REVIEW_FILE, {
        issue["issue_id"]: issue.get("review_status", "unreviewed")
        for issue in issues
    })
    csv_path = run_dir / ANNOTATION_QA_REPORT_CSV_FILE
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "issue_id", "image_name", "split", "class_id", "class_name", "issue_type",
        "severity", "score", "message", "preview", "review_status", "accepted_fix",
        "accepted_class_id", "accepted_class_name", "applied", "corrected_label_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for issue in issues:
            writer.writerow({field: issue.get(field, "") for field in fields})


def annotation_qa_run_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
        raise HTTPException(status_code=404, detail="Annotation QA job not found.")
    run_dir = (ANNOTATION_QA_ROOT / job_id).resolve()
    ensure_annotation_qa_path(run_dir)
    return run_dir


def load_annotation_qa_report(run_dir: Path) -> dict:
    report = read_json_object(run_dir / ANNOTATION_QA_REPORT_JSON_FILE)
    if not isinstance(report.get("issues"), list):
        raise HTTPException(status_code=404, detail="Annotation QA results are not available.")
    if not isinstance(report.get("summary"), dict):
        report["summary"] = read_json_object(run_dir / ANNOTATION_QA_SUMMARY_FILE)
    return report


def set_annotation_qa_issue_fix(run_dir: Path, issue_id: str, fix: str, class_id: Optional[int] = None) -> dict:
    report = load_annotation_qa_report(run_dir)
    issue = next(
        (item for item in report["issues"] if isinstance(item, dict) and item.get("issue_id") == issue_id),
        None,
    )
    if issue is None:
        raise HTTPException(status_code=404, detail="Annotation QA issue not found.")

    fix_value = str(fix or "").strip()
    if fix_value in {"", "none", "clear"}:
        issue["accepted_fix"] = ""
        issue["accepted_class_id"] = None
        issue["accepted_class_name"] = ""
    elif fix_value == "sam_box":
        if issue.get("fix_type") != "replace_box" or not issue.get("recommended_bbox"):
            raise HTTPException(status_code=400, detail="This issue does not have a SAM box recommendation.")
        issue["accepted_fix"] = "sam_box"
        issue["review_status"] = "fix_accepted"
    elif fix_value == "class":
        if class_id is None:
            raise HTTPException(status_code=400, detail="Class fix requires a class id.")
        summary = report.get("summary") or {}
        dataset_yaml = summary.get("dataset_yaml")
        if not dataset_yaml:
            raise HTTPException(status_code=400, detail="Annotation QA report is missing dataset class metadata.")
        class_names = read_yaml_class_names(Path(dataset_yaml))
        if class_id < 0 or class_id >= len(class_names):
            raise HTTPException(status_code=400, detail="Selected class is not in the dataset class list.")
        issue["accepted_class_id"] = class_id
        issue["accepted_class_name"] = class_names[class_id]
        issue["review_status"] = "fix_accepted"
    else:
        raise HTTPException(status_code=400, detail="Unknown annotation QA fix.")

    write_annotation_qa_report(run_dir, report["issues"], report.get("summary") or {})
    return issue


def issue_label_path_in_copy(issue: dict, dataset_root: Path, corrected_root: Path) -> Path:
    image_path = Path(str(issue.get("image", ""))).expanduser().resolve()
    try:
        relative_image = image_path.relative_to(dataset_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="QA issue image is outside the source dataset.") from exc

    parts = list(relative_image.parts)
    try:
        images_index = parts.index("images")
    except ValueError:
        split = str(issue.get("split") or "").strip() or "train"
        label_relative = Path("labels") / split / f"{image_path.stem}.txt"
    else:
        parts[images_index] = "labels"
        label_relative = Path(*parts).with_suffix(".txt")
    return corrected_root / label_relative


def apply_annotation_qa_fix(issue: dict, dataset_root: Path, corrected_root: Path) -> dict:
    has_box_fix = issue.get("accepted_fix") == "sam_box"
    has_class_fix = issue.get("accepted_class_id") is not None
    if not has_box_fix and not has_class_fix:
        return {"applied": False, "reason": "No accepted fix."}
    if has_box_fix and (issue.get("fix_type") != "replace_box" or not issue.get("recommended_bbox")):
        return {"applied": False, "reason": "Issue has no replaceable SAM box."}
    label_row = issue.get("label_row")
    if not isinstance(label_row, int) or label_row < 1:
        return {"applied": False, "reason": "Issue has no label row reference."}

    label_path = issue_label_path_in_copy(issue, dataset_root, corrected_root)
    if not label_path.is_file():
        return {"applied": False, "reason": f"Copied label file not found: {label_path}"}
    try:
        relative_image = Path(str(issue.get("image"))).expanduser().resolve().relative_to(dataset_root)
    except ValueError:
        return {"applied": False, "reason": "Issue image is outside the source dataset."}
    image_path = corrected_root / relative_image
    try:
        import cv2
        image = cv2.imread(str(image_path))
    except Exception:
        image = None
    if image is None:
        return {"applied": False, "reason": f"Copied image could not be read: {image_path}"}
    height, width = image.shape[:2]

    lines = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
    line_index = label_row - 1
    if line_index >= len(lines):
        return {"applied": False, "reason": "Referenced label row no longer exists."}

    original_row = lines[line_index]
    original_fields = original_row.strip().split()
    if len(original_fields) != 5:
        return {"applied": False, "reason": "Referenced label row is not a YOLO detection box."}
    corrected_class_id = issue.get("accepted_class_id") if has_class_fix else issue.get("class_id")
    try:
        if has_box_fix:
            corrected_row = yolo_bbox_from_pixels(corrected_class_id, issue["recommended_bbox"], width, height)
        else:
            corrected_row = " ".join([str(int(float(corrected_class_id))), *original_fields[1:5]])
    except (TypeError, ValueError) as exc:
        return {"applied": False, "reason": str(exc)}
    lines[line_index] = corrected_row
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    issue["applied"] = True
    issue["corrected_label_path"] = str(label_path)
    issue["original_yolo_row"] = original_row
    issue["corrected_yolo_row"] = corrected_row
    if has_class_fix:
        issue["class_id"] = issue.get("accepted_class_id")
        issue["class_name"] = issue.get("accepted_class_name") or issue.get("class_name", "")
    return {"applied": True, "label_path": str(label_path)}


def apply_annotation_qa_fixes(job_id: str) -> dict:
    run_dir = annotation_qa_run_dir(job_id)
    report = load_annotation_qa_report(run_dir)
    summary = report.get("summary") or {}
    dataset_yaml = summary.get("dataset_yaml")
    if not dataset_yaml:
        raise HTTPException(status_code=400, detail="Annotation QA report is missing the source dataset YAML.")
    yaml_path, dataset_root, portable_payload = prepared_dataset_yaml(dataset_yaml)

    corrected_name = clean_name(f"{dataset_root.name}_qa_corrected_{job_id}", "qa_corrected_dataset")
    corrected_root = (DATASET_PREPARED_ROOT / corrected_name).resolve()
    try:
        corrected_root.relative_to(DATA_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="Corrected dataset path escaped the data workspace.") from exc
    if corrected_root.exists():
        shutil.rmtree(corrected_root)
    shutil.copytree(dataset_root, corrected_root, ignore=shutil.ignore_patterns(ANNOTATION_QA_FIX_SUMMARY_FILE))

    corrected_yaml = corrected_root / "data.yaml"
    corrected_payload = dict(portable_payload)
    corrected_payload["path"] = "."
    corrected_yaml.write_text(yaml.safe_dump(corrected_payload, sort_keys=False), encoding="utf-8")

    def issue_has_accepted_fix(item: dict) -> bool:
        return item.get("accepted_fix") == "sam_box" or item.get("accepted_class_id") is not None

    applied = 0
    skipped = []
    for issue in report["issues"]:
        if not isinstance(issue, dict) or not issue_has_accepted_fix(issue):
            continue
        result = apply_annotation_qa_fix(issue, dataset_root, corrected_root)
        if result.get("applied"):
            applied += 1
        else:
            issue["applied"] = False
            issue["apply_error"] = result.get("reason", "Fix was not applied.")
            skipped.append({"issue_id": issue.get("issue_id"), "reason": issue["apply_error"]})

    fix_summary = {
        "job_id": job_id,
        "source_dataset_yaml": str(yaml_path),
        "source_dataset_root": str(dataset_root),
        "corrected_dataset_yaml": str(corrected_yaml),
        "corrected_dataset_root": str(corrected_root),
        "accepted_fixes": sum(1 for item in report["issues"] if isinstance(item, dict) and issue_has_accepted_fix(item)),
        "applied_fixes": applied,
        "skipped_fixes": skipped,
        "applied_at": datetime.now(MYT).isoformat(),
    }
    write_json_object(corrected_root / ANNOTATION_QA_FIX_SUMMARY_FILE, fix_summary)
    write_json_object(run_dir / ANNOTATION_QA_FIX_SUMMARY_FILE, fix_summary)
    report["summary"]["corrected_dataset_yaml"] = str(corrected_yaml)
    report["summary"]["corrected_dataset_root"] = str(corrected_root)
    report["summary"]["applied_fixes"] = applied
    write_annotation_qa_report(run_dir, report["issues"], report["summary"])

    response = dataset_response(
        corrected_yaml,
        f"Applied {applied} accepted SAM fix{'es' if applied != 1 else ''}. Training will use the corrected dataset copy.",
    )
    response.update(fix_summary)
    response["download_available"] = True
    return response


def load_sam_model(model_name: str):
    try:
        from ultralytics import SAM
    except Exception as exc:
        raise RuntimeError("Ultralytics SAM is not available in this environment.") from exc
    return SAM(model_name)


def sam_masks_for_image(model, image_path: Path, bboxes: list[tuple[int, int, int, int]], device: str):
    if not bboxes:
        return []
    kwargs = {"source": str(image_path), "bboxes": [list(box) for box in bboxes], "verbose": False}
    if device:
        kwargs["device"] = device
    results = model.predict(**kwargs)
    if not results:
        return []
    masks = getattr(results[0], "masks", None)
    data = getattr(masks, "data", None)
    if data is None:
        return []
    try:
        return [item.detach().cpu().numpy() for item in data]
    except AttributeError:
        return list(data)


def run_annotation_qa_job(job_id: str, request_payload: dict, stop_event: threading.Event):
    run_dir = (ANNOTATION_QA_ROOT / job_id).resolve()
    ensure_annotation_qa_path(run_dir)
    preview_dir = run_dir / "previews"
    issues: list[dict] = []
    issue_index = 0
    labels_checked = 0
    images_scanned = 0
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        yaml_path, dataset_root, payload = prepared_dataset_yaml(request_payload["dataset_yaml"])
        class_names = normalize_yaml_names(payload.get("names"))
        split_names = annotation_qa_split_names(request_payload.get("scope", "val"), payload)
        if not split_names:
            raise RuntimeError("No dataset split was available for annotation QA.")
        split_contexts = []
        for split in split_names:
            images_path = split_image_folder(dataset_root, payload.get(split))
            labels_path = label_folder_for_images(dataset_root, images_path)
            images = image_files(images_path) if images_path and images_path.is_dir() else []
            split_contexts.append((split, images_path, labels_path, images))
        max_images = request_payload.get("max_images")
        total_images = sum(len(images) for _split, _images_path, _labels_path, images in split_contexts)
        if max_images:
            total_images = min(total_images, int(max_images))
        if total_images <= 0:
            raise RuntimeError("No images were found for the selected QA scope.")

        update_annotation_qa_job(
            job_id,
            status="running",
            stage="loading_model",
            percent=2,
            detail=f"Loading {request_payload['model']}.",
            total_images=total_images,
            run_dir=str(run_dir),
            report_available=False,
        )
        model = load_sam_model(request_payload["model"])
        device = os.getenv("SAM_QA_DEVICE") or os.getenv("TRAINING_DEVICE") or ""
        thresholds = annotation_qa_thresholds(request_payload.get("preset", "balanced"))
        processed_limit = int(max_images) if max_images else None

        for split, _images_path, labels_path, images in split_contexts:
            for image_path in images:
                if stop_event.is_set():
                    raise InferenceStopped("Annotation QA was stopped.")
                if processed_limit is not None and images_scanned >= processed_limit:
                    break
                images_scanned += 1
                percent = 5 + int((images_scanned / max(1, total_images)) * 90)
                update_annotation_qa_job(
                    job_id,
                    stage="scanning",
                    percent=min(95, percent),
                    detail=f"Scanning {images_scanned} of {total_images}: {image_path.name}",
                    images_scanned=images_scanned,
                    labels_checked=labels_checked,
                    issues=len(issues),
                )
                import cv2

                image = cv2.imread(str(image_path))
                if image is None:
                    issue_index += 1
                    issues.append(annotation_issue(
                        job_id, issue_index, image_path, split, None, "",
                        "image_read_error", "high", 1.0, "Image could not be read.",
                    ))
                    continue
                height, width = image.shape[:2]
                label_path = yolo_label_path(image_path, labels_path)
                if label_path is None or not label_path.is_file():
                    continue
                labels = []
                for row_index, line in enumerate(label_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                    fields = line.strip().split()
                    if not fields:
                        continue
                    try:
                        class_id = int(float(fields[0]))
                    except ValueError:
                        issue_index += 1
                        issues.append(annotation_issue(
                            job_id, issue_index, image_path, split, None, "",
                            "invalid_label", "high", 1.0, f"Invalid class id on row {row_index}.",
                        ))
                        continue
                    class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
                    if len(fields) != 5:
                        issue_index += 1
                        issues.append(annotation_issue(
                            job_id, issue_index, image_path, split, class_id, class_name,
                            "unsupported_annotation", "low", 0.2,
                            "V1 SAM QA checks YOLO detection boxes only.",
                        ))
                        continue
                    box = pixel_bbox_from_yolo(fields, width, height)
                    if box is None:
                        issue_index += 1
                        issues.append(annotation_issue(
                            job_id, issue_index, image_path, split, class_id, class_name,
                            "invalid_label", "high", 1.0, f"Invalid YOLO bbox on row {row_index}.",
                        ))
                        continue
                    labels.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "bbox": box,
                        "row_index": row_index,
                    })

                for left in range(len(labels)):
                    for right in range(left + 1, len(labels)):
                        if labels[left]["class_id"] != labels[right]["class_id"]:
                            continue
                        overlap = bbox_iou(labels[left]["bbox"], labels[right]["bbox"])
                        if overlap >= thresholds["duplicate_iou"]:
                            issue_index += 1
                            issue = annotation_issue(
                                job_id, issue_index, image_path, split,
                                labels[left]["class_id"], labels[left]["class_name"],
                                "duplicate_box", "medium", overlap,
                                "Two same-class boxes overlap heavily.",
                                labels[left]["bbox"], labels[right]["bbox"], {"bbox_iou": overlap},
                                label_row=labels[left].get("row_index"),
                            )
                            issues.append(issue)

                boxes = [label["bbox"] for label in labels]
                masks = sam_masks_for_image(model, image_path, boxes, device)
                for label_index, label in enumerate(labels):
                    labels_checked += 1
                    box = label["bbox"]
                    box_area = max(1, bbox_area(box))
                    mask = masks[label_index] if label_index < len(masks) else None
                    if mask is None:
                        issue_index += 1
                        issue = annotation_issue(
                            job_id, issue_index, image_path, split,
                            label["class_id"], label["class_name"],
                            "empty_mask", "high", 1.0,
                            "SAM did not return a mask for this box.",
                            box,
                            label_row=label.get("row_index"),
                        )
                        issues.append(issue)
                        continue
                    mask_array = mask_to_uint8(mask, width, height)
                    sam_box = mask_bbox(mask_array)
                    if sam_box is None:
                        issue_index += 1
                        issue = annotation_issue(
                            job_id, issue_index, image_path, split,
                            label["class_id"], label["class_name"],
                            "empty_mask", "high", 1.0,
                            "SAM returned an empty mask for this box.",
                            box,
                            label_row=label.get("row_index"),
                        )
                        issue["preview"] = f"previews/{issue['issue_id']}.jpg"
                        draw_annotation_qa_preview(image_path, run_dir / issue["preview"], issue, None)
                        issues.append(issue)
                        continue
                    overlap = bbox_iou(box, sam_box)
                    center_shift = bbox_center_shift(box, sam_box)
                    mask_area = int(mask_array.sum())
                    mask_area_ratio = mask_area / box_area
                    sam_box_area_ratio = bbox_area(sam_box) / box_area
                    metrics = {
                        "bbox_iou": round(overlap, 4),
                        "center_shift": round(center_shift, 4),
                        "mask_area_ratio": round(mask_area_ratio, 4),
                        "sam_bbox_area_ratio": round(sam_box_area_ratio, 4),
                    }
                    edge_margin_x = max(2, int((box[2] - box[0]) * 0.03))
                    edge_margin_y = max(2, int((box[3] - box[1]) * 0.03))
                    edge_hits = sum([
                        abs(sam_box[0] - box[0]) <= edge_margin_x,
                        abs(sam_box[2] - box[2]) <= edge_margin_x,
                        abs(sam_box[1] - box[1]) <= edge_margin_y,
                        abs(sam_box[3] - box[3]) <= edge_margin_y,
                    ])
                    candidates = []
                    if mask_area_ratio < thresholds["mask_area_ratio_min"]:
                        candidates.append(("low_mask_coverage", "high", 1 - mask_area_ratio, "SAM mask covers very little of the labeled box."))
                    if overlap < thresholds["bbox_iou"]:
                        severity = "high" if overlap < thresholds["bbox_iou"] * 0.65 else "medium"
                        candidates.append(("low_box_agreement", severity, 1 - overlap, "SAM mask bbox disagrees with the YOLO box."))
                    if center_shift > thresholds["center_shift"]:
                        candidates.append(("shifted_box", "medium", center_shift, "SAM mask center is far from the YOLO box center."))
                    if sam_box_area_ratio < thresholds["loose_area_ratio"]:
                        candidates.append(("loose_box", "medium", 1 - sam_box_area_ratio, "YOLO box is much larger than the SAM mask bbox."))
                    if edge_hits >= thresholds["tight_edge_count"] and mask_area_ratio > 0.35:
                        candidates.append(("possibly_tight_box", "low", edge_hits / 4, "SAM mask touches multiple edges of the YOLO box."))

                    if candidates:
                        issue_type, severity, score, message = sorted(
                            candidates,
                            key=lambda item: {"high": 3, "medium": 2, "low": 1}[item[1]],
                            reverse=True,
                        )[0]
                        issue_index += 1
                        issue = annotation_issue(
                            job_id, issue_index, image_path, split,
                            label["class_id"], label["class_name"], issue_type, severity,
                            score, message, box, sam_box, metrics,
                            label_row=label.get("row_index"),
                            recommended_bbox=sam_box,
                        )
                        issue["preview"] = f"previews/{issue['issue_id']}.jpg"
                        draw_annotation_qa_preview(image_path, run_dir / issue["preview"], issue, mask_array)
                        issues.append(issue)
            if processed_limit is not None and images_scanned >= processed_limit:
                break

        summary = annotation_qa_summary(
            issues,
            images_scanned,
            labels_checked,
            request_payload["model"],
            request_payload.get("scope", "val"),
            request_payload.get("preset", "balanced"),
        )
        summary.update({
            "dataset_yaml": str(yaml_path),
            "dataset_root": str(dataset_root),
            "splits": split_names,
        })
        write_annotation_qa_report(run_dir, issues, summary)
        update_annotation_qa_job(
            job_id,
            status="completed",
            stage="completed",
            percent=100,
            detail=f"QA complete: {len(issues)} issues flagged.",
            images_scanned=images_scanned,
            labels_checked=labels_checked,
            issues=len(issues),
            high=summary["high"],
            medium=summary["medium"],
            low=summary["low"],
            summary=summary,
            report_available=True,
        )
    except InferenceStopped:
        summary = annotation_qa_summary(
            issues,
            images_scanned,
            labels_checked,
            request_payload.get("model", ANNOTATION_QA_MODEL_DEFAULT),
            request_payload.get("scope", "val"),
            request_payload.get("preset", "balanced"),
        )
        write_annotation_qa_report(run_dir, issues, summary)
        update_annotation_qa_job(
            job_id,
            status="stopped",
            stage="stopped",
            percent=0,
            detail="Annotation QA stopped.",
            images_scanned=images_scanned,
            labels_checked=labels_checked,
            issues=len(issues),
            summary=summary,
            report_available=True,
        )
    except Exception as exc:
        update_annotation_qa_job(
            job_id,
            status="failed",
            stage="failed",
            percent=0,
            detail=str(exc),
            error=str(exc),
            images_scanned=images_scanned,
            labels_checked=labels_checked,
            issues=len(issues),
            report_available=False,
        )
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def annotation_qa_job_from_disk(job_id: str) -> Optional[dict]:
    run_dir = (ANNOTATION_QA_ROOT / job_id).resolve()
    try:
        ensure_annotation_qa_path(run_dir)
    except HTTPException:
        return None
    report = read_json_object(run_dir / ANNOTATION_QA_REPORT_JSON_FILE)
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else read_json_object(run_dir / ANNOTATION_QA_SUMMARY_FILE)
    if not summary:
        return None
    return {
        "job_id": job_id,
        "status": "completed",
        "stage": "completed",
        "percent": 100,
        "detail": f"QA complete: {summary.get('issues', 0)} issues flagged.",
        "run_dir": str(run_dir),
        "images_scanned": summary.get("images_scanned", 0),
        "labels_checked": summary.get("labels_checked", 0),
        "issues": summary.get("issues", 0),
        "high": summary.get("high", 0),
        "medium": summary.get("medium", 0),
        "low": summary.get("low", 0),
        "summary": summary,
        "report_available": True,
        "created_at": run_dir.stat().st_mtime,
        "updated_at": run_dir.stat().st_mtime,
    }


def persist_training_report_context(run_dir: Path):
    if training_run_info is None:
        return
    context = training_run_info.get("report_context")
    if isinstance(context, dict):
        write_json_object(run_dir / TRAINING_REPORT_CONTEXT_FILE, context)


def persist_test_report_context(run_dir: Path):
    if test_run_info is None:
        return
    context = test_run_info.get("report_context")
    if isinstance(context, dict):
        write_json_object(run_dir / TEST_REPORT_CONTEXT_FILE, context)


def capture_training_run_dir(line: str):
    global training_run_info
    prefix = next((item for item in RUN_DIRECTORY_PREFIXES if line.startswith(item)), None)
    if prefix is None or training_run_info is None:
        return

    run_dir = Path(line[len(prefix):].strip()).expanduser().resolve()
    try:
        ensure_runs_path(run_dir)
    except HTTPException:
        return

    training_run_info["run_dir"] = str(run_dir)
    training_run_info["resolution_type"] = "actual"
    persist_training_report_context(run_dir)


def capture_epoch_progress(line: str) -> bool:
    global training_run_info
    match = WEB_PROGRESS_RE.match(line)
    if match is not None:
        current_epoch = int(match.group(1))
        total_epochs = int(match.group(2))
        if training_run_info is not None and current_epoch > 0 and total_epochs > 0:
            training_run_info["current_epoch"] = min(current_epoch, total_epochs)
            training_run_info["total_epochs"] = total_epochs
            training_run_info["progress_detail"] = ""
        return True

    match = RFDETR_VALIDATION_PROGRESS_RE.search(line)
    if match is None:
        return False

    current_epoch = int(match.group(1))
    total_epochs = int(match.group(2))
    if training_run_info is not None and current_epoch > 0 and total_epochs > 0:
        training_run_info["current_epoch"] = min(current_epoch, total_epochs)
        training_run_info["total_epochs"] = total_epochs
        training_run_info["progress_detail"] = "RF-DETR validation metrics were updated from the latest validation block."
    return False


def capture_test_run_dir(line: str):
    global test_run_info
    match = WEB_TEST_RUN_DIR_RE.match(line)
    if match is not None and test_run_info is not None:
        run_dir = Path(match.group(1).strip()).expanduser().resolve()
        try:
            ensure_runs_path(run_dir)
        except HTTPException:
            return
        test_run_info["run_dir"] = str(run_dir)
        persist_test_report_context(run_dir)
        return

    prefix = next((item for item in RUN_DIRECTORY_PREFIXES if line.startswith(item)), None)
    if prefix is None or test_run_info is None:
        return

    run_dir = Path(line[len(prefix):].strip()).expanduser().resolve()
    try:
        ensure_runs_path(run_dir)
    except HTTPException:
        return
    test_run_info["run_dir"] = str(run_dir)
    persist_test_report_context(run_dir)


def capture_test_progress(line: str) -> bool:
    global test_run_info
    match = WEB_TEST_PROGRESS_RE.match(line)
    if match is None:
        return False

    if test_run_info is not None:
        test_run_info["progress_percent"] = max(0, min(100, int(match.group(1))))
        test_run_info["progress_stage"] = match.group(2)
        test_run_info["progress_detail"] = (match.group(3) or "").strip()
    return True


def stream_training_logs(process: subprocess.Popen, log_paths: list[Path]):
    handles = [path.open("a", encoding="utf-8") for path in log_paths]
    try:
        if process.stdout is None:
            return

        for raw_line in process.stdout:
            line = clean_log_line(raw_line)
            capture_training_run_dir(line)
            if capture_epoch_progress(line):
                continue
            if not should_write_log_line(line):
                continue
            for handle in handles:
                handle.write(line + "\n")
                handle.flush()
    finally:
        for handle in handles:
            handle.close()


def stream_test_logs(process: subprocess.Popen, log_paths: list[Path]):
    handles = [path.open("a", encoding="utf-8") for path in log_paths]
    try:
        if process.stdout is None:
            return

        for raw_line in process.stdout:
            line = clean_log_line(raw_line)
            capture_test_run_dir(line)
            if capture_test_progress(line):
                continue
            if not should_write_log_line(line):
                continue
            for handle in handles:
                handle.write(line + "\n")
                handle.flush()
    finally:
        for handle in handles:
            handle.close()


def read_log_tail(max_chars: int = 20000) -> str:
    if not LOG_FILE.is_file():
        return ""
    data = LOG_FILE.read_text(encoding="utf-8", errors="replace")
    return data[-max_chars:]


def read_log_tail_from(path: Path, max_chars: int = 20000) -> str:
    if not path.is_file():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[-max_chars:]


def read_log_file(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def classify_rfdetr_log_line(line: str) -> str:
    lower = line.lower()
    if any(pattern in lower for pattern in RFDETR_FATAL_ERROR_PATTERNS):
        return "fatal"
    if "rf-detr" not in lower and "rfdetr" not in lower:
        return ""
    if any(pattern in lower for pattern in RFDETR_NON_FATAL_WARNING_PATTERNS):
        return "non_fatal"
    return ""


def read_error_log(path: Path) -> str:
    keywords = ("error", "warning", "traceback", "exception", "failed", "no space", "not found")
    lines = read_log_file(path).splitlines()
    fatal_lines = []
    regular_lines = []
    expected_rfdetr = []
    for line in lines:
        lower = line.lower()
        if not any(keyword in lower for keyword in keywords):
            continue
        classification = classify_rfdetr_log_line(line)
        if classification == "fatal":
            fatal_lines.append(line)
        elif classification == "non_fatal":
            expected_rfdetr.append(line)
        else:
            regular_lines.append(line)

    output = []
    if expected_rfdetr:
        output.append("RF-DETR note: training can continue with expected model-loading warnings.")
        output.extend(f"Expected RF-DETR warning: {line}" for line in expected_rfdetr)
    if fatal_lines:
        if output:
            output.append("")
        output.append("Fatal or blocking RF-DETR issues:")
        output.extend(fatal_lines)
    if regular_lines:
        if output:
            output.append("")
        output.extend(regular_lines)
    return "\n".join(output)


def latest_timestamped_log() -> Optional[Path]:
    logs = [path for path in LOG_DIR.glob("train-*.log") if path.is_file()]
    if not logs:
        return None
    return max(logs, key=lambda path: path.stat().st_mtime)


def latest_timestamped_test_log() -> Optional[Path]:
    logs = [path for path in LOG_DIR.glob("test-*.log") if path.is_file()]
    if not logs:
        return None
    return max(logs, key=lambda path: path.stat().st_mtime)


def latest_artifact_run_dir(search_root: Path, marker_paths: tuple[str, ...]) -> Optional[Path]:
    if not search_root.exists():
        return None

    candidates = []
    for path in search_root.rglob("*"):
        if not path.is_dir():
            continue
        if any((path / marker).is_file() for marker in marker_paths):
            candidates.append(path)

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: max(
            (item.stat().st_mtime for item in path.rglob("*") if item.is_file()),
            default=0,
        ),
    )


def completed_epoch_from_results(run_info: dict) -> int:
    run_dir_value = run_info.get("run_dir")
    if not run_dir_value:
        return 0

    results_path = Path(run_dir_value) / "results.csv"
    if not results_path.is_file():
        return 0

    try:
        with results_path.open("r", encoding="utf-8", newline="") as file:
            rows = csv.DictReader(file)
            last_row = None
            for last_row in rows:
                pass
        return int(float_value(last_row or {}, "epoch") or 0)
    except (OSError, ValueError):
        return 0


def epoch_progress(run_info: dict, running: bool) -> dict:
    total = max(0, int(run_info.get("total_epochs") or 0))
    completed = completed_epoch_from_results(run_info)
    current = max(0, int(run_info.get("current_epoch") or 0))
    detail = str(run_info.get("progress_detail") or "").strip()

    if running and current == 0 and total:
        current = min(completed + 1, total)
    elif not running and completed:
        current = max(current, completed)

    if running and not detail and str(run_info.get("family") or "").lower() == "rfdetr":
        detail = "RF-DETR is training. Progress updates when validation metrics are logged."
    if running and not detail and str(run_info.get("family") or "").lower() == "dfine":
        detail = "D-FINE is training. Progress updates when the backend logs epoch markers."

    if total:
        current = min(current, total)
        completed = min(completed, total)
        percent = round((current / total) * 100, 1)
    else:
        percent = 0.0

    return {
        "current": current,
        "completed": completed,
        "total": total,
        "percent": percent,
        "detail": detail,
    }


def resolve_artifact_path(request: ArtifactRequest) -> Path:
    run_dir = resolve_run_dir(request.project, request.name)
    artifact_map = {
        "results_csv": run_dir / "results.csv",
        "accuracy_graph": run_dir / "accuracy_by_epoch.png",
        "loss_graph": run_dir / "loss_by_epoch.png",
        "confusion_matrix": run_dir / "confusion_matrix.png",
        "confusion_matrix_normalized": run_dir / "confusion_matrix_normalized.png",
        "roc_auc_curve": run_dir / "roc_auc_curve.png",
        "best": run_dir / "weights" / "best.pt",
        "last": run_dir / "weights" / "last.pt",
    }
    path = artifact_map.get(request.artifact)
    if path is None:
        raise HTTPException(status_code=404, detail="Unknown artifact.")
    ensure_runs_path(path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {path}")
    return path


def gpu_float(value: str) -> Optional[float]:
    cleaned = str(value).strip()
    if not cleaned or "N/A" in cleaned.upper():
        return None
    try:
        parsed = float(cleaned)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def query_gpu_status() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
    except FileNotFoundError:
        return {
            "available": False,
            "gpus": [],
            "message": "GPU information unavailable because nvidia-smi is not installed.",
        }
    except subprocess.TimeoutExpired:
        return {
            "available": False,
            "gpus": [],
            "message": "GPU information unavailable because nvidia-smi timed out.",
        }
    except subprocess.CalledProcessError:
        return {
            "available": False,
            "gpus": [],
            "message": "GPU information unavailable. Check Docker NVIDIA runtime access.",
        }

    gpus = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) < 8:
            continue
        memory_used = gpu_float(row[3])
        memory_total = gpu_float(row[4])
        memory_percent = (
            (memory_used / memory_total) * 100
            if memory_used is not None and memory_total
            else None
        )
        try:
            index = int(row[0].strip())
        except ValueError:
            index = len(gpus)
        gpus.append({
            "index": index,
            "name": row[1].strip(),
            "utilization_percent": format_metric(gpu_float(row[2]), 1),
            "memory_used_mb": format_metric(memory_used, 1),
            "memory_total_mb": format_metric(memory_total, 1),
            "memory_percent": format_metric(memory_percent, 1),
            "temperature_c": format_metric(gpu_float(row[5]), 1),
            "power_draw_w": format_metric(gpu_float(row[6]), 1),
            "power_limit_w": format_metric(gpu_float(row[7]), 1),
        })

    if not gpus:
        return {
            "available": False,
            "gpus": [],
            "message": "No NVIDIA GPUs were reported by nvidia-smi.",
        }
    return {"available": True, "gpus": gpus, "message": ""}


def gpu_status() -> dict:
    now = time.monotonic()
    with gpu_status_lock:
        cached = gpu_status_cache.get("payload")
        if cached is not None and now - gpu_status_cache["checked_at"] < GPU_STATUS_CACHE_SECONDS:
            return cached
        payload = query_gpu_status()
        gpu_status_cache["checked_at"] = time.monotonic()
        gpu_status_cache["payload"] = payload
        return payload


def current_status() -> dict:
    global training_process
    returncode = None
    running = False
    if training_process is not None:
        returncode = training_process.poll()
        if returncode is None:
            running = True
        else:
            training_process = None

    run_info = dict(training_run_info or {})
    if not running and not run_info.get("run_dir") and run_info.get("project") and run_info.get("name"):
        try:
            run_dir, resolution_type = resolve_run_dir_details(run_info["project"], run_info["name"])
            run_info["run_dir"] = str(run_dir)
            run_info["resolution_type"] = resolution_type
        except HTTPException:
            run_info.setdefault("run_dir", "")
            run_info["resolution_type"] = "not_found"

    progress = epoch_progress(run_info, running)
    run_info.pop("report_context", None)
    return {
        "running": running,
        "returncode": returncode,
        "started_at": training_started_at,
        "log_file": str(LOG_FILE),
        "history_log_file": str(training_log_file) if training_log_file else "",
        "training_run": run_info,
        "epoch_progress": progress,
    }


def current_test_run_dir() -> Optional[Path]:
    run_info = test_run_info or {}
    run_dir_value = run_info.get("run_dir")
    if run_dir_value:
        run_dir = Path(run_dir_value).expanduser().resolve()
        if run_dir.is_dir():
            try:
                ensure_runs_path(run_dir)
            except HTTPException:
                pass
            else:
                return run_dir

    latest = latest_artifact_run_dir(
        TEST_RUNS_ROOT,
        ("test_metrics.json", "confusion_matrix.png", "confusion_matrix_normalized.png"),
    )
    if latest:
        ensure_runs_path(latest)
    return latest


def run_test_artifact_statuses(run_dir: Path) -> dict:
    return {
        "metrics_json": artifact_status(run_dir / "test_metrics.json"),
        "confusion_matrix": artifact_status(run_dir / "confusion_matrix.png"),
        "confusion_matrix_normalized": artifact_status(
            run_dir / "confusion_matrix_normalized.png"
        ),
        "roc_auc_curve": artifact_status(run_dir / "roc_auc_curve.png"),
    }


def read_test_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "test_metrics.json"
    if not metrics_path.is_file():
        return {
            "available": False,
            "run_dir": str(run_dir),
            "artifacts": run_test_artifact_statuses(run_dir),
        }

    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {
            "available": False,
            "run_dir": str(run_dir),
            "artifacts": run_test_artifact_statuses(run_dir),
        }

    if not isinstance(payload, dict):
        payload = {}

    payload.update(
        {
            "available": True,
            "run_dir": str(run_dir),
            "artifacts": run_test_artifact_statuses(run_dir),
            "combined_report_available": bool(
                read_json_object(run_dir / TEST_REPORT_CONTEXT_FILE).get("training_run_dir")
            ),
        }
    )
    return payload


def load_training_report_context(run_dir: Path) -> dict:
    context = read_json_object(run_dir / TRAINING_REPORT_CONTEXT_FILE)
    args = read_json_object(run_dir / "args.json")
    if not args:
        try:
            args_payload = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            args_payload = {}
        args = args_payload if isinstance(args_payload, dict) else {}

    dataset_value = context.get("dataset_yaml") or args.get("data") or ""
    dataset_path = Path(str(dataset_value)).expanduser() if dataset_value else None
    if dataset_path is not None and not dataset_path.is_absolute():
        dataset_path = (REPO_ROOT / dataset_path).resolve()
    if dataset_path is not None and dataset_path.is_file():
        context["dataset_yaml"] = str(dataset_path)
        if not context.get("dataset_summary"):
            context["dataset_summary"] = cached_dataset_summary(dataset_path)
    stored_hyperparameters = context.get("hyperparameters")
    if not isinstance(stored_hyperparameters, dict):
        stored_hyperparameters = {}
    context["hyperparameters"] = {**args, **stored_hyperparameters}
    context.setdefault("model", args.get("model", "Unknown"))
    context.setdefault("pretrained", args.get("pretrained", True))
    context.setdefault("device", args.get("device", "auto"))
    return context


def resolve_test_artifact_path(artifact: str) -> Path:
    run_dir = current_test_run_dir()
    if run_dir is None:
        raise HTTPException(status_code=404, detail="No test run artifacts were found.")

    artifact_map = {
        "metrics_json": run_dir / "test_metrics.json",
        "confusion_matrix": run_dir / "confusion_matrix.png",
        "confusion_matrix_normalized": run_dir / "confusion_matrix_normalized.png",
        "roc_auc_curve": run_dir / "roc_auc_curve.png",
    }
    path = artifact_map.get(artifact)
    if path is None:
        raise HTTPException(status_code=404, detail="Unknown test artifact.")
    ensure_runs_path(path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Test artifact not found: {path}")
    return path


def current_test_status() -> dict:
    global test_process
    returncode = None
    running = False
    if test_process is not None:
        returncode = test_process.poll()
        if returncode is None:
            running = True
        else:
            test_process = None

    run_info = dict(test_run_info or {})
    if not run_info.get("run_dir"):
        run_dir = current_test_run_dir()
        if run_dir is not None:
            run_info["run_dir"] = str(run_dir)

    progress_percent = max(0, min(100, int(run_info.get("progress_percent") or 0)))
    progress_stage = run_info.get("progress_stage") or ("running" if running else "idle")
    progress_detail = run_info.get("progress_detail") or ""
    if not running and returncode == 0:
        progress_percent = 100
        progress_stage = "complete"
        progress_detail = progress_detail or "Model testing complete."
    elif not running and returncode not in {None, 0}:
        progress_stage = "failed"
        progress_detail = progress_detail or "Model testing failed."

    run_info.pop("report_context", None)
    return {
        "running": running,
        "returncode": returncode,
        "started_at": test_started_at,
        "log_file": str(TEST_LOG_FILE),
        "history_log_file": str(test_log_file) if test_log_file else "",
        "test_run": run_info,
        "progress": {
            "percent": progress_percent,
            "stage": progress_stage,
            "detail": progress_detail,
        },
    }


@app.on_event("startup")
def startup():
    ensure_dirs()


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def config():
    return {
        "data_root": str(DATA_ROOT),
        "roboflow": {
            "has_api_key": bool(os.getenv("ROBOFLOW_API_KEY")),
            "workspace": os.getenv("ROBOFLOW_WORKSPACE", ""),
            "project": os.getenv("ROBOFLOW_PROJECT", ""),
            "version": os.getenv("ROBOFLOW_VERSION", ""),
        },
        "default_device": os.getenv("TRAINING_DEVICE", ""),
    }


@app.get("/api/inference/weights")
def inference_weights():
    return {"weights": available_inference_weights()}


@app.get("/api/inference/storage")
def inference_storage():
    return inference_jobs_storage_payload()


@app.delete("/api/inference/outputs")
def delete_inference_outputs():
    return clear_inference_outputs()


@app.get("/api/storage")
def storage():
    return storage_payload()


@app.get("/api/storage/{target_key}")
def storage_target(target_key: str):
    return storage_target_payload(target_key)


@app.delete("/api/storage/{target_key}")
def delete_storage_target(target_key: str):
    return clear_storage_target(target_key)


@app.get("/api/train/sessions")
def train_sessions():
    return {"sessions": available_training_sessions()}


@app.post("/api/inference/run")
def run_inference(
    weight_source: str = Form("selected"),
    weight_path: str = Form(""),
    weight_family: str = Form("auto"),
    convert_to_onnx: bool = Form(False),
    imgsz: int = Form(512),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    vid_stride: int = Form(1),
    show_boxes: bool = Form(True),
    show_labels: bool = Form(True),
    show_conf: bool = Form(True),
    show_masks: bool = Form(True),
    weight_file: Optional[UploadFile] = File(None),
    media_file: UploadFile = File(...),
):
    ensure_dirs()
    if current_status()["running"]:
        raise HTTPException(status_code=409, detail="Training is running. Stop training before inference.")
    if imgsz < 32:
        raise HTTPException(status_code=400, detail="Image size must be at least 32.")
    if not 0 <= conf <= 1 or not 0 <= iou <= 1:
        raise HTTPException(status_code=400, detail="Confidence and IoU thresholds must be between 0 and 1.")
    if vid_stride < 1:
        raise HTTPException(status_code=400, detail="Video frame stride must be at least 1.")

    job_id = datetime.now(MYT).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    job_dir = (INFERENCE_JOB_ROOT / job_id).resolve()
    ensure_inference_path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    if weight_source == "selected":
        weights_path = resolve_inference_weight_path(weight_path)
        inference_family = family_for_weight_path(weights_path)
        weights_label = relative_to_repo(weights_path)
        convert_to_onnx = False
    elif weight_source == "upload":
        if weight_file is None or not weight_file.filename:
            raise HTTPException(status_code=400, detail="Choose a .pt or .onnx weights file to upload.")
        weight_suffix = Path(weight_file.filename).suffix.lower()
        if weight_suffix not in INFERENCE_WEIGHT_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Uploaded weights must be a .pt or .onnx file.")
        normalized_family = str(weight_family or "ultralytics").strip().lower()
        if normalized_family == "auto":
            normalized_family = "ultralytics"
        if normalized_family not in {"ultralytics", "rfdetr", "dfine"}:
            raise HTTPException(status_code=400, detail="Unknown inference weights backend.")
        if normalized_family in {"dfine", "rfdetr"} and weight_suffix != ".pt":
            raise HTTPException(status_code=400, detail="Uploaded DETR-family weights must be a .pt file.")
        if convert_to_onnx and weight_suffix != ".pt":
            convert_to_onnx = False
        if normalized_family in {"dfine", "rfdetr"}:
            convert_to_onnx = False
        weights_path = (INFERENCE_UPLOAD_ROOT / job_id / Path(weight_file.filename).name).resolve()
        ensure_inference_path(weights_path)
        save_upload(weight_file, weights_path, lambda *_args: None, "saving", "Saving inference weights")
        weights_label = f"uploaded {weights_path.name}{' -> ONNX' if convert_to_onnx else ''}"
        inference_family = normalized_family
    else:
        raise HTTPException(status_code=400, detail="Unknown inference weights source.")

    if not media_file.filename:
        raise HTTPException(status_code=400, detail="Choose an image or video file for inference.")
    media_name = safe_upload_path(media_file.filename).name
    input_path = (job_dir / media_name).resolve()
    ensure_inference_path(input_path)
    save_upload(media_file, input_path, lambda *_args: None, "saving", "Saving inference media")
    media_kind = inference_media_type(input_path)
    output_path = job_dir / ("annotated.jpg" if media_kind == "image" else "annotated.mp4")
    result_path = job_dir / "result.json"
    progress_path = job_dir / "progress.json"
    preview_path = job_dir / "preview.jpg"
    inference_device = os.getenv("INFERENCE_DEVICE") or os.getenv("TRAINING_DEVICE") or ""
    inference_request = {
        "weights_path": weights_path,
        "input_path": input_path,
        "output_path": output_path,
        "json_path": result_path,
        "progress_path": progress_path,
        "preview_path": preview_path,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "device": inference_device,
        "vid_stride": vid_stride,
        "show_boxes": show_boxes,
        "show_labels": show_labels,
        "show_conf": show_conf,
        "show_masks": show_masks,
        "convert_to_onnx": convert_to_onnx,
        "family": inference_family,
    }
    stop_event = threading.Event()
    preview_condition = threading.Condition()

    job_payload = {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "percent": 0,
        "detail": "Inference job queued.",
        "frames": 0,
        "total_frames": 0,
        "detections": 0,
        "classifications": 0,
        "task": "",
        "elapsed_ms": 0,
        "media_type": media_kind,
        "weights": weights_label,
        "preview_available": False,
        "preview_fps": 0,
        "stoppable": False,
        "stop_event": stop_event,
        "preview_condition": preview_condition,
        "preview_frame": None,
        "preview_frame_times": [],
        "preview_updated_at": 0,
        "webrtc_frame": None,
        "webrtc_frame_seq": 0,
        "webrtc_updated_at": 0,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    with inference_jobs_lock:
        inference_jobs[job_id] = job_payload

    thread = threading.Thread(
        target=run_inference_job,
        args=(job_id, inference_request, result_path, output_path, preview_path, stop_event),
        daemon=True,
    )
    thread.start()
    return inference_job_payload(job_payload)


@app.get("/api/inference/status/{job_id}")
def inference_status(job_id: str):
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
        raise HTTPException(status_code=404, detail="Inference job not found.")
    with inference_jobs_lock:
        job = inference_jobs.get(job_id)
        if job is not None:
            return inference_job_payload(dict(job))
    job = inference_job_from_disk(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Inference job not found.")
    with inference_jobs_lock:
        inference_jobs[job_id] = dict(job)
    return inference_job_payload(job)


@app.post("/api/inference/stop/{job_id}")
def stop_inference(job_id: str):
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
        raise HTTPException(status_code=404, detail="Inference job not found.")
    return stop_inference_process(job_id)


@app.get("/api/inference/preview/{job_id}")
def inference_preview(job_id: str):
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
        raise HTTPException(status_code=404, detail="Inference preview not found.")
    job_dir = (INFERENCE_JOB_ROOT / job_id).resolve()
    ensure_inference_path(job_dir)
    path = job_dir / "preview.jpg"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Inference preview not available yet.")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/inference/stream/{job_id}")
def inference_stream(job_id: str):
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
        raise HTTPException(status_code=404, detail="Inference stream not found.")
    with inference_jobs_lock:
        if job_id not in inference_jobs:
            raise HTTPException(status_code=404, detail="Inference stream not available.")
    return StreamingResponse(
        inference_mjpeg_stream(job_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/inference/webrtc/{job_id}/offer")
async def inference_webrtc_offer(job_id: str, offer: WebRTCOffer):
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
        raise HTTPException(status_code=404, detail="Inference WebRTC stream not found.")
    with inference_jobs_lock:
        if job_id not in inference_jobs:
            raise HTTPException(status_code=404, detail="Inference WebRTC stream not available.")

    try:
        import numpy as np
        from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
        from aiortc.mediastreams import MediaStreamError
        from av import VideoFrame
    except ModuleNotFoundError as exc:
        missing = exc.name or "aiortc"
        raise HTTPException(
            status_code=501,
            detail=f"WebRTC live preview requires {missing}. Rebuild the container after updating requirements.txt.",
        ) from exc

    class InferenceVideoTrack(VideoStreamTrack):
        kind = "video"

        def __init__(self, stream_job_id: str):
            super().__init__()
            self.stream_job_id = stream_job_id
            self.last_frame = None

        async def recv(self):
            pts, time_base = await self.next_timestamp()
            frame, status = latest_inference_webrtc_frame(self.stream_job_id)
            if status in {"complete", "failed", "stopped", "missing"}:
                raise MediaStreamError
            if frame is not None:
                self.last_frame = frame
            elif self.last_frame is not None:
                frame = self.last_frame
            else:
                frame = np.zeros((360, 640, 3), dtype=np.uint8)

            video_frame = VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="bgr24")
            video_frame.pts = pts
            video_frame.time_base = time_base
            return video_frame

    peer = RTCPeerConnection()
    peer.addTrack(InferenceVideoTrack(job_id))
    with inference_peers_lock:
        inference_peers.setdefault(job_id, set()).add(peer)

    @peer.on("connectionstatechange")
    async def on_connectionstatechange():
        if peer.connectionState in {"failed", "disconnected", "closed"}:
            with inference_peers_lock:
                peers = inference_peers.get(job_id)
                if peers is not None:
                    peers.discard(peer)
                    if not peers:
                        inference_peers.pop(job_id, None)
            if peer.connectionState != "closed":
                await peer.close()

    try:
        remote_offer = RTCSessionDescription(sdp=offer.sdp, type=offer.type)
        await peer.setRemoteDescription(remote_offer)
        answer = await peer.createAnswer()
        await peer.setLocalDescription(answer)
    except Exception as exc:
        with inference_peers_lock:
            peers = inference_peers.get(job_id)
            if peers is not None:
                peers.discard(peer)
        await peer.close()
        raise HTTPException(status_code=400, detail=f"Could not create WebRTC stream: {exc}") from exc

    return {
        "sdp": peer.localDescription.sdp,
        "type": peer.localDescription.type,
    }


@app.get("/api/inference/result/{job_id}")
def inference_result(job_id: str, download: bool = False):
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
        raise HTTPException(status_code=404, detail="Inference result not found.")
    job_dir = (INFERENCE_JOB_ROOT / job_id).resolve()
    ensure_inference_path(job_dir)
    candidates = [job_dir / "annotated.jpg", job_dir / "annotated.mp4"]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise HTTPException(status_code=404, detail="Inference result not found.")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    filename = f"inference_{job_id}{path.suffix}" if download else path.name
    return FileResponse(path, media_type=media_type, filename=filename)


@app.post("/api/annotation-qa/start")
def start_annotation_qa(request: AnnotationQaRequest):
    if training_process is not None and training_process.poll() is None:
        raise HTTPException(status_code=409, detail="Training is running. Stop training before running annotation QA.")
    if test_process is not None and test_process.poll() is None:
        raise HTTPException(status_code=409, detail="Model testing is running. Stop testing before running annotation QA.")
    with annotation_qa_jobs_lock:
        active = next(
            (
                job
                for job in annotation_qa_jobs.values()
                if job.get("status") in {"queued", "running", "stopping"}
            ),
            None,
        )
    if active is not None:
        raise HTTPException(status_code=409, detail="Annotation QA is already running.")

    yaml_path, _dataset_root, _payload = prepared_dataset_yaml(request.dataset_yaml)
    model = normalize_annotation_qa_model_name(request.model)
    job_id = annotation_qa_job_id()
    run_dir = (ANNOTATION_QA_ROOT / job_id).resolve()
    ensure_annotation_qa_path(run_dir)
    stop_event = threading.Event()
    request_payload = {
        "dataset_yaml": str(yaml_path),
        "model": model,
        "scope": request.scope,
        "preset": request.preset,
        "max_images": request.max_images,
        "max_side": request.max_side,
    }
    job = {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "percent": 0,
        "detail": "Annotation QA queued.",
        "run_dir": str(run_dir),
        "dataset_yaml": str(yaml_path),
        "model": model,
        "scope": request.scope,
        "preset": request.preset,
        "images_scanned": 0,
        "total_images": 0,
        "labels_checked": 0,
        "issues": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
        "summary": {},
        "report_available": False,
        "stop_event": stop_event,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    with annotation_qa_jobs_lock:
        annotation_qa_jobs[job_id] = job
    thread = threading.Thread(
        target=run_annotation_qa_job,
        args=(job_id, request_payload, stop_event),
        daemon=True,
    )
    job["thread"] = thread
    thread.start()
    return annotation_qa_payload(job)


@app.get("/api/annotation-qa/status/{job_id}")
def annotation_qa_status(job_id: str):
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
        raise HTTPException(status_code=404, detail="Annotation QA job not found.")
    with annotation_qa_jobs_lock:
        job = annotation_qa_jobs.get(job_id)
        if job is not None:
            return annotation_qa_payload(dict(job))
    job = annotation_qa_job_from_disk(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Annotation QA job not found.")
    with annotation_qa_jobs_lock:
        annotation_qa_jobs[job_id] = dict(job)
    return annotation_qa_payload(job)


@app.post("/api/annotation-qa/stop/{job_id}")
def stop_annotation_qa(job_id: str):
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
        raise HTTPException(status_code=404, detail="Annotation QA job not found.")
    with annotation_qa_jobs_lock:
        job = annotation_qa_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Annotation QA job not found.")
        event = job.get("stop_event")
        if isinstance(event, threading.Event):
            event.set()
        job["status"] = "stopping"
        job["stage"] = "stopping"
        job["detail"] = "Stopping annotation QA..."
        job["updated_at"] = time.time()
        return annotation_qa_payload(dict(job))


@app.get("/api/annotation-qa/results/{job_id}")
def annotation_qa_results(job_id: str):
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
        raise HTTPException(status_code=404, detail="Annotation QA results not found.")
    run_dir = (ANNOTATION_QA_ROOT / job_id).resolve()
    ensure_annotation_qa_path(run_dir)
    report_path = run_dir / ANNOTATION_QA_REPORT_JSON_FILE
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail="Annotation QA results are not available yet.")
    return FileResponse(report_path, media_type="application/json")


@app.get("/api/annotation-qa/preview/{job_id}/{preview_name}")
def annotation_qa_preview(job_id: str, preview_name: str):
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
        raise HTTPException(status_code=404, detail="Annotation QA preview not found.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.jpg", preview_name):
        raise HTTPException(status_code=404, detail="Annotation QA preview not found.")
    preview_path = (ANNOTATION_QA_ROOT / job_id / "previews" / preview_name).resolve()
    ensure_annotation_qa_path(preview_path)
    if not preview_path.is_file():
        raise HTTPException(status_code=404, detail="Annotation QA preview not found.")
    return FileResponse(preview_path, media_type="image/jpeg")


@app.get("/api/annotation-qa/download/{job_id}/{artifact}")
def annotation_qa_download(job_id: str, artifact: str):
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
        raise HTTPException(status_code=404, detail="Annotation QA report not found.")
    filenames = {
        "json": ANNOTATION_QA_REPORT_JSON_FILE,
        "csv": ANNOTATION_QA_REPORT_CSV_FILE,
        "summary": ANNOTATION_QA_SUMMARY_FILE,
    }
    filename = filenames.get(artifact)
    if filename is None:
        raise HTTPException(status_code=404, detail="Annotation QA report not found.")
    report_path = (ANNOTATION_QA_ROOT / job_id / filename).resolve()
    ensure_annotation_qa_path(report_path)
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail="Annotation QA report not found.")
    media_type = "text/csv" if artifact == "csv" else "application/json"
    return FileResponse(report_path, media_type=media_type, filename=filename)


@app.post("/api/annotation-qa/mark/{job_id}")
def mark_annotation_qa_issue(job_id: str, request: AnnotationQaMarkRequest):
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
        raise HTTPException(status_code=404, detail="Annotation QA job not found.")
    allowed = {"unreviewed", "fix_accepted", "accepted", "false_positive", "needs_fix", "ignored"}
    status = str(request.status or "").strip()
    if status not in allowed:
        raise HTTPException(status_code=400, detail="Unknown annotation QA review status.")
    run_dir = (ANNOTATION_QA_ROOT / job_id).resolve()
    ensure_annotation_qa_path(run_dir)
    report_path = run_dir / ANNOTATION_QA_REPORT_JSON_FILE
    report = read_json_object(report_path)
    issues = report.get("issues")
    if not isinstance(issues, list):
        raise HTTPException(status_code=404, detail="Annotation QA results are not available.")
    matched = False
    for issue in issues:
        if isinstance(issue, dict) and issue.get("issue_id") == request.issue_id:
            issue["review_status"] = status
            if status != "fix_accepted":
                issue["accepted_fix"] = ""
                issue["accepted_class_id"] = None
                issue["accepted_class_name"] = ""
            matched = True
            break
    if not matched:
        raise HTTPException(status_code=404, detail="Annotation QA issue not found.")
    write_annotation_qa_report(run_dir, issues, report.get("summary") or read_json_object(run_dir / ANNOTATION_QA_SUMMARY_FILE))
    return {"issue_id": request.issue_id, "status": status}


@app.post("/api/annotation-qa/fix/{job_id}")
def accept_annotation_qa_fix(job_id: str, request: AnnotationQaFixRequest):
    run_dir = annotation_qa_run_dir(job_id)
    issue = set_annotation_qa_issue_fix(run_dir, request.issue_id, request.fix, request.class_id)
    return {
        "issue_id": request.issue_id,
        "accepted_fix": issue.get("accepted_fix", ""),
        "accepted_class_id": issue.get("accepted_class_id"),
        "accepted_class_name": issue.get("accepted_class_name", ""),
        "review_status": issue.get("review_status", "unreviewed"),
    }


@app.post("/api/annotation-qa/apply/{job_id}")
def apply_annotation_qa_corrections(job_id: str):
    if training_process is not None and training_process.poll() is None:
        raise HTTPException(status_code=409, detail="Stop training before applying annotation fixes.")
    if test_process is not None and test_process.poll() is None:
        raise HTTPException(status_code=409, detail="Stop model testing before applying annotation fixes.")
    return apply_annotation_qa_fixes(job_id)


@app.get("/api/dataset/preparation/status")
def dataset_preparation_status(job_id: str):
    with dataset_preparation_lock:
        progress = dataset_preparation_jobs.get(job_id)
        if progress is None:
            raise HTTPException(status_code=404, detail="Dataset preparation job not found.")
        return dict(progress)


@app.post("/api/dataset/download/prepare")
def prepare_dataset_download(request: DatasetDownloadRequest):
    cleanup_expired_dataset_downloads()
    archive_path, filename = build_dataset_archive(request.dataset_yaml)
    download_id = uuid.uuid4().hex
    with dataset_download_lock:
        dataset_downloads[download_id] = {
            "path": str(archive_path),
            "filename": filename,
            "created_at": time.time(),
        }
    return {
        "download_url": f"/api/dataset/download/{download_id}",
        "filename": filename,
        "size": archive_path.stat().st_size,
    }


@app.post("/api/dataset/download/annotated/prepare")
def prepare_annotated_dataset_download(request: DatasetDownloadRequest):
    cleanup_expired_dataset_downloads()
    archive_path, filename = build_annotated_dataset_archive(request.dataset_yaml)
    download_id = uuid.uuid4().hex
    with dataset_download_lock:
        dataset_downloads[download_id] = {
            "path": str(archive_path),
            "filename": filename,
            "created_at": time.time(),
        }
    return {
        "download_url": f"/api/dataset/download/{download_id}",
        "filename": filename,
        "size": archive_path.stat().st_size,
    }


@app.get("/api/dataset/download/{download_id}")
def download_prepared_dataset(download_id: str, background_tasks: BackgroundTasks):
    with dataset_download_lock:
        download = dataset_downloads.pop(download_id, None)
    if download is None:
        raise HTTPException(status_code=404, detail="Dataset download is unavailable or has expired.")

    archive_path = Path(download["path"])
    if not archive_path.is_file():
        raise HTTPException(status_code=404, detail="Prepared dataset ZIP was not found.")
    background_tasks.add_task(archive_path.unlink, missing_ok=True)
    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename=download["filename"],
    )


def upload_size(upload: UploadFile) -> int:
    position = upload.file.tell()
    upload.file.seek(0, os.SEEK_END)
    size = upload.file.tell()
    upload.file.seek(position)
    return max(0, int(size))


def save_upload(
    upload: UploadFile,
    target: Path,
    progress_callback: DatasetProgressCallback,
    stage: str,
    detail_prefix: str,
    offset: int = 0,
    total: Optional[int] = None,
) -> int:
    size = upload_size(upload)
    work_total = total if total is not None else size
    copied = 0
    upload.file.seek(0)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as output:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            copied += len(chunk)
            progress_callback(
                stage,
                offset + copied,
                work_total,
                f"{detail_prefix}: {offset + copied} of {work_total} bytes",
            )
    return copied


def extract_zip_with_progress(
    zip_path: Path,
    extract_dir: Path,
    progress_callback: DatasetProgressCallback,
    stage: str = "extracting",
):
    with zipfile.ZipFile(zip_path) as archive:
        entries = archive.infolist()
        total_bytes = sum(entry.file_size for entry in entries)
        work_total = total_bytes or len(entries)
        extracted_bytes = 0
        progress_callback(stage, 0, work_total, f"Extracting 0 of {len(entries)} ZIP entries")
        for index, entry in enumerate(entries, start=1):
            archive.extract(entry, extract_dir)
            extracted_bytes += entry.file_size
            current = extracted_bytes if total_bytes else index
            progress_callback(
                stage,
                current,
                work_total,
                f"Extracting {index} of {len(entries)} ZIP entries",
            )


class RoboflowDirectDownloadUnavailable(Exception):
    """Raised when the installed SDK cannot expose its export download flow."""


def normalized_roboflow_percent(value) -> Optional[float]:
    if value is None:
        return None
    try:
        percent = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(percent):
        return None
    if 0 <= percent <= 1:
        percent *= 100
    return min(100.0, max(0.0, percent))


def report_roboflow_server_progress(
    progress_callback: DatasetProgressCallback,
    stage: str,
    progress,
    indeterminate_detail: str,
    determinate_prefix: str,
):
    percent = normalized_roboflow_percent(progress)
    if percent is None:
        progress_callback(stage, 0, 0, indeterminate_detail)
        return
    progress_callback(
        stage,
        round(percent * 10),
        1000,
        f"{determinate_prefix}: {percent:.1f}%",
    )


def wait_for_roboflow_export(
    rfapi,
    api_key: str,
    workspace: str,
    project_name: str,
    version: str,
    dataset_format: str,
    progress_callback: DatasetProgressCallback,
    timeout_seconds: int = 1800,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("Roboflow version preparation timed out.")
        try:
            version_response = rfapi.get_version(
                api_key=api_key,
                workspace_url=workspace,
                project_url=project_name,
                version=version,
                nocache=True,
            )
        except TypeError as exc:
            raise RoboflowDirectDownloadUnavailable from exc
        version_info = version_response.get("version", {})
        generating = bool(
            version_info.get("generating")
            or version_info.get("images", 0) == 0
        )
        if not generating:
            break
        report_roboflow_server_progress(
            progress_callback,
            "preparing_roboflow_version",
            version_info.get("progress"),
            "Roboflow is preparing the dataset version.",
            "Preparing Roboflow dataset version",
        )
        time.sleep(5)

    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("Roboflow export preparation timed out.")
        try:
            export_info = rfapi.get_version_export(
                api_key=api_key,
                workspace_url=workspace,
                project_url=project_name,
                version=version,
                format=dataset_format,
            )
        except TypeError as exc:
            raise RoboflowDirectDownloadUnavailable from exc

        export = export_info.get("export")
        if isinstance(export, dict) and export.get("link"):
            progress_callback(
                "preparing_roboflow_export",
                1000,
                1000,
                "Roboflow export is ready.",
            )
            return str(export["link"])
        if export_info.get("ready") is False:
            report_roboflow_server_progress(
                progress_callback,
                "preparing_roboflow_export",
                export_info.get("progress"),
                "Roboflow is generating the YOLOv8 export.",
                "Generating YOLOv8 export",
            )
            time.sleep(1)
            continue
        response_keys = ", ".join(sorted(str(key) for key in export_info)) or "none"
        raise RuntimeError(
            f"Unexpected Roboflow export response fields: {response_keys}."
        )


def readable_byte_count(value: int) -> str:
    amount = float(max(0, value))
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            digits = 0 if unit == "B" else 1
            return f"{amount:.{digits}f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TB"


def stream_roboflow_export(
    link: str,
    zip_path: Path,
    progress_callback: DatasetProgressCallback,
):
    try:
        import requests
    except ImportError as exc:
        raise RoboflowDirectDownloadUnavailable from exc

    try:
        response_context = requests.get(link, stream=True, timeout=(30, 300))
    except TypeError as exc:
        raise RoboflowDirectDownloadUnavailable from exc

    with response_context as response:
        response.raise_for_status()
        try:
            total_bytes = max(0, int(response.headers.get("content-length", 0)))
        except (TypeError, ValueError):
            total_bytes = 0
        downloaded = 0
        progress_callback(
            "downloading_roboflow",
            0,
            total_bytes,
            (
                f"Downloading Roboflow dataset: 0 of {readable_byte_count(total_bytes)}"
                if total_bytes
                else "Downloading Roboflow dataset; total size is unavailable."
            ),
        )
        with zip_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                output.write(chunk)
                downloaded += len(chunk)
                detail = f"Downloaded {readable_byte_count(downloaded)}"
                if total_bytes:
                    detail += f" of {readable_byte_count(total_bytes)}"
                progress_callback(
                    "downloading_roboflow",
                    downloaded,
                    total_bytes,
                    detail,
                )


def download_roboflow_dataset_with_progress(
    version_obj,
    api_key: str,
    workspace: str,
    project_name: str,
    version: str,
    dataset_format: str,
    download_dir: Path,
    progress_callback: DatasetProgressCallback,
) -> Path:
    try:
        from roboflow.adapters import rfapi
    except ImportError as exc:
        raise RoboflowDirectDownloadUnavailable from exc

    required_attributes = ("get_version", "get_version_export")
    if any(not hasattr(rfapi, attribute) for attribute in required_attributes):
        raise RoboflowDirectDownloadUnavailable

    resolved_workspace = str(getattr(version_obj, "workspace", workspace) or workspace)
    resolved_project = str(getattr(version_obj, "project", project_name) or project_name)
    link = wait_for_roboflow_export(
        rfapi,
        api_key,
        resolved_workspace,
        resolved_project,
        str(version_obj.version if hasattr(version_obj, "version") else version),
        dataset_format,
        progress_callback,
    )
    zip_path = download_dir / "roboflow.zip"
    stream_roboflow_export(link, zip_path, progress_callback)
    extract_zip_with_progress(
        zip_path,
        download_dir,
        progress_callback,
        stage="extracting_roboflow",
    )
    zip_path.unlink(missing_ok=True)
    return download_dir


def mark_dataset_preparation_failed(job_id: str, exc: Exception):
    detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
    update_dataset_preparation(job_id, "failed", 0, 0, str(detail), status="failed")


@app.post("/api/dataset/upload")
def upload_dataset(
    file: UploadFile = File(...),
    classes: str = Form(...),
    train: int = Form(70),
    val: int = Form(15),
    test: int = Form(15),
    name: str = Form("dataset"),
    force_split: bool = Form(False),
    job_id: str = Form(""),
):
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a ZIP file.")

    try:
        class_names = json.loads(classes)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Classes must be valid JSON.") from exc
    split = SplitConfig(train=train, val=val, test=test)
    clean = clean_name(name, "uploaded_dataset")
    upload_dir = DATA_ROOT / "uploads" / clean
    extract_dir = DATA_ROOT / "extracted" / clean
    progress_callback = dataset_progress_callback(job_id)
    try:
        zip_size = upload_size(file)
        progress_callback("saving", 0, zip_size, "Preparing the upload destination")
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        extract_dir.mkdir(parents=True, exist_ok=True)

        zip_path = upload_dir / file.filename
        progress_callback("saving", 0, zip_size, f"Saving uploaded ZIP: 0 of {zip_size} bytes")
        save_upload(file, zip_path, progress_callback, "saving", "Saving uploaded ZIP")
        extract_zip_with_progress(zip_path, extract_dir, progress_callback)

        yaml_path = prepare_dataset(
            source=extract_dir,
            name=clean,
            classes=class_names,
            split=split,
            force_split=force_split,
            progress_callback=progress_callback,
        )
        response = dataset_response(yaml_path, "Uploaded dataset is ready.", progress_callback)
        update_dataset_preparation(job_id, "complete", 1, 1, "Dataset preparation complete.", status="complete")
        return response
    except zipfile.BadZipFile as exc:
        mark_dataset_preparation_failed(job_id, exc)
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP.") from exc
    except Exception as exc:
        mark_dataset_preparation_failed(job_id, exc)
        raise


@app.post("/api/dataset/folder")
def upload_folder_dataset(
    files: list[UploadFile] = File(...),
    classes: str = Form(...),
    train: int = Form(70),
    val: int = Form(15),
    test: int = Form(15),
    name: str = Form("dataset"),
    force_split: bool = Form(False),
    job_id: str = Form(""),
):
    if not files:
        raise HTTPException(status_code=400, detail="Choose a dataset folder first.")

    try:
        class_names = json.loads(classes)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Classes must be valid JSON.") from exc

    split = SplitConfig(train=train, val=val, test=test)
    clean = clean_name(name, "uploaded_folder_dataset")
    upload_dir = DATA_ROOT / "folder_uploads" / clean
    progress_callback = dataset_progress_callback(job_id)
    try:
        total_bytes = sum(upload_size(upload) for upload in files)
        progress_callback("saving", 0, total_bytes, "Preparing the upload destination")
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_root = upload_dir.resolve()
        saved_bytes = 0
        saved_count = 0
        progress_callback("saving", 0, total_bytes, f"Saving 0 of {len(files)} files")

        for upload in files:
            relative_path = safe_upload_path(upload.filename or "")
            target = (upload_dir / relative_path).resolve()
            try:
                target.relative_to(upload_root)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid upload path: {upload.filename}") from exc

            saved_bytes += save_upload(
                upload,
                target,
                progress_callback,
                "saving",
                f"Saving file {saved_count + 1} of {len(files)}",
                offset=saved_bytes,
                total=total_bytes,
            )
            saved_count += 1

        if saved_count == 0:
            raise HTTPException(status_code=400, detail="No files were uploaded.")

        yaml_path = prepare_dataset(
            source=upload_dir,
            name=clean,
            classes=class_names,
            split=split,
            force_split=force_split,
            progress_callback=progress_callback,
        )
        response = dataset_response(yaml_path, "Uploaded folder dataset is ready.", progress_callback)
        update_dataset_preparation(job_id, "complete", 1, 1, "Dataset preparation complete.", status="complete")
        return response
    except Exception as exc:
        mark_dataset_preparation_failed(job_id, exc)
        raise


@app.post("/api/dataset/roboflow")
def roboflow_dataset(request: RoboflowRequest):
    api_key = request.api_key or os.getenv("ROBOFLOW_API_KEY")
    workspace = request.workspace or os.getenv("ROBOFLOW_WORKSPACE")
    project_name = request.project or os.getenv("ROBOFLOW_PROJECT")
    version = request.version or os.getenv("ROBOFLOW_VERSION")
    dataset_format = "yolov8"

    missing = [
        name for name, value in {
            "api_key": api_key,
            "workspace": workspace,
            "project": project_name,
            "version": version,
        }.items()
        if not value
    ]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing Roboflow fields: {', '.join(missing)}")

    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="The roboflow package is not installed. Run: pip install roboflow",
        ) from exc

    clean = clean_name(request.name, "dataset")
    download_dir = DATA_ROOT / "roboflow" / clean
    progress_callback = dataset_progress_callback(request.job_id)
    progress_callback(
        "preparing_roboflow_version",
        0,
        0,
        "Connecting to Roboflow and checking the dataset version.",
    )

    try:
        if download_dir.exists():
            shutil.rmtree(download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)
        version_obj = project.version(int(version))
        try:
            dataset_root = download_roboflow_dataset_with_progress(
                version_obj,
                str(api_key),
                str(workspace),
                str(project_name),
                str(version),
                dataset_format,
                download_dir,
                progress_callback,
            )
        except RoboflowDirectDownloadUnavailable:
            progress_callback(
                "fetching_roboflow",
                0,
                0,
                "The installed Roboflow SDK does not expose download progress; using its standard downloader.",
            )
            try:
                dataset = version_obj.download(
                    dataset_format,
                    location=str(download_dir),
                    overwrite=True,
                )
            except TypeError:
                dataset = version_obj.download(dataset_format, location=str(download_dir))
            dataset_root = Path(getattr(dataset, "location", download_dir))
    except Exception as exc:
        shutil.rmtree(download_dir, ignore_errors=True)
        error = HTTPException(status_code=502, detail=f"Roboflow download failed: {exc}")
        mark_dataset_preparation_failed(request.job_id, error)
        raise error from exc

    try:
        if request.force_split:
            split = SplitConfig(train=request.train, val=request.val, test=request.test)
            yaml_path = prepare_dataset(
                source=dataset_root,
                name=clean,
                classes=request.classes,
                split=split,
                force_split=True,
                progress_callback=progress_callback,
            )
            message = (
                "Roboflow dataset is ready. The downloaded split was rebuilt locally "
                f"to {request.train}/{request.val}/{request.test}; the Roboflow version "
                "was not modified."
            )
        else:
            progress_callback(
                "validating_dataset",
                0,
                0,
                "Validating the downloaded dataset paths and split folders.",
            )
            split = SplitConfig(train=request.train, val=request.val, test=request.test)
            yaml_path, preparation_action = prepare_roboflow_download(
                dataset_root,
                clean,
                request.classes,
                split,
                progress_callback,
            )
            if preparation_action == "normalized":
                progress_callback(
                    "normalizing_paths",
                    1,
                    1,
                    "Normalized invalid export paths to the detected split folders.",
                )
            elif preparation_action == "rebuilt":
                progress_callback(
                    "rebuilding_split",
                    1,
                    1,
                    "Rebuilt a local train/validation/test split from the usable downloaded images.",
                )
            if preparation_action == "normalized":
                message = (
                    "Roboflow dataset is ready. Invalid export paths were normalized "
                    "to the detected train/validation/test folders."
                )
            elif preparation_action == "rebuilt":
                message = (
                    "Roboflow dataset is ready. The downloaded export had invalid split paths, "
                    f"so a local {request.train}/{request.val}/{request.test} split was rebuilt "
                    "from the usable images."
                )
            else:
                message = "Roboflow dataset is ready."

        response = dataset_response(
            yaml_path,
            message,
            progress_callback,
            source_metadata=(
                {
                    "augmentation": getattr(version_obj, "augmentation", None),
                    "images": getattr(version_obj, "images", None),
                    "splits": getattr(version_obj, "splits", None),
                }
                if not request.force_split
                else None
            ),
        )
        update_dataset_preparation(
            request.job_id,
            "complete",
            1,
            1,
            "Dataset preparation complete.",
            status="complete",
        )
        return response
    except Exception as exc:
        mark_dataset_preparation_failed(request.job_id, exc)
        raise


@app.post("/api/test/start")
def start_test(
    weight_source: str = Form("trained"),
    project: str = Form("runs/detect"),
    name: str = Form("train"),
    prepared_dataset_yaml: str = Form(""),
    reference_dataset_yaml: str = Form(""),
    dataset_source: str = Form("prepared"),
    imgsz: int = Form(640),
    batch: int = Form(16),
    workers: int = Form(2),
    device: str = Form(""),
    weight_file: Optional[UploadFile] = File(None),
    dataset_zip: Optional[UploadFile] = File(None),
    dataset_files: Optional[list[UploadFile]] = File(None),
):
    global test_process, test_started_at, test_log_file, test_run_info

    if current_status()["running"]:
        raise HTTPException(status_code=409, detail="Training is running. Stop training before testing.")
    if current_test_status()["running"]:
        raise HTTPException(status_code=409, detail="Model testing is already running.")

    ensure_dirs()
    test_name = f"test-{datetime.now(MYT).strftime('%Y%m%d-%H%M%S')}"
    test_project_path = normalize_training_project_path("runs/test")
    test_input_root = DATA_ROOT / "test_inputs" / test_name
    test_input_root.mkdir(parents=True, exist_ok=True)

    if weight_source == "trained":
        weights_path = resolve_weight_path(project, name, "best")
        weights_label = f"best.pt from {project}/{name}"
        source_training_run = weights_path.parent.parent.resolve()
    elif weight_source == "upload":
        if weight_file is None or not weight_file.filename:
            raise HTTPException(status_code=400, detail="Choose a .pt weights file to upload.")
        if Path(weight_file.filename).suffix.lower() != ".pt":
            raise HTTPException(status_code=400, detail="Uploaded weights must be a .pt file.")
        weights_dir = test_input_root / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / Path(weight_file.filename).name
        save_upload(weight_file, weights_path, lambda *_args: None, "saving", "Saving weights")
        weights_label = f"uploaded weights {weights_path.name}"
        source_training_run = None
    else:
        raise HTTPException(status_code=400, detail="Unknown weight source.")

    if dataset_source == "prepared":
        if not prepared_dataset_yaml:
            raise HTTPException(status_code=400, detail="Prepare a dataset before using the prepared test split.")
        dataset_yaml, dataset_info = resolve_prepared_test_dataset_yaml(prepared_dataset_yaml)
    elif dataset_source == "upload_zip":
        if dataset_zip is None or not dataset_zip.filename:
            raise HTTPException(status_code=400, detail="Choose a labeled YOLO ZIP file for testing.")
        if not dataset_zip.filename.lower().endswith(".zip"):
            raise HTTPException(status_code=400, detail="Uploaded test dataset must be a ZIP file.")
        upload_dir = test_input_root / "zip"
        extract_dir = test_input_root / "zip_extracted"
        upload_dir.mkdir(parents=True, exist_ok=True)
        extract_dir.mkdir(parents=True, exist_ok=True)
        zip_path = upload_dir / Path(dataset_zip.filename).name
        save_upload(dataset_zip, zip_path, lambda *_args: None, "saving", "Saving test ZIP")
        try:
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(extract_dir)
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=400, detail="Uploaded test dataset is not a valid ZIP.") from exc
        dataset_yaml, dataset_info = prepare_custom_test_dataset(
            extract_dir,
            f"{test_name}-dataset",
            reference_dataset_yaml,
        )
    elif dataset_source == "upload_folder":
        files = dataset_files or []
        if not files:
            raise HTTPException(status_code=400, detail="Choose a labeled YOLO folder for testing.")
        upload_dir = test_input_root / "folder"
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_root = upload_dir.resolve()
        for upload in files:
            relative_path = safe_upload_path(upload.filename or "")
            target = (upload_dir / relative_path).resolve()
            try:
                target.relative_to(upload_root)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid upload path: {upload.filename}") from exc
            save_upload(upload, target, lambda *_args: None, "saving", "Saving test file")
        dataset_yaml, dataset_info = prepare_custom_test_dataset(
            upload_dir,
            f"{test_name}-dataset",
            reference_dataset_yaml,
        )
    else:
        raise HTTPException(status_code=400, detail="Unknown dataset source.")

    test_run_info = {
        "project": str(test_project_path),
        "name": test_name,
        "expected_run_dir": str(test_project_path / test_name),
        "run_dir": "",
        "weights_source": weight_source,
        "weights_label": weights_label,
        "dataset_source": dataset_source,
        "dataset_yaml": str(dataset_yaml),
        "dataset_root": dataset_info.get("dataset_root", ""),
        "dataset_split": dataset_info.get("source_split", ""),
        "progress_percent": 0,
        "progress_stage": "starting",
        "progress_detail": "Launching the test job.",
        "report_context": {
            "created_at": datetime.now(MYT).isoformat(),
            "weight_source": weight_source,
            "weights_label": weights_label,
            "weights_path": str(weights_path),
            "training_run_dir": str(source_training_run) if source_training_run else "",
            "dataset_source": dataset_source,
            "dataset_yaml": str(dataset_yaml),
            "dataset_root": dataset_info.get("dataset_root", ""),
            "dataset_split": dataset_info.get("source_split", "test"),
            "parameters": {
                "imgsz": imgsz,
                "batch": batch,
                "workers": workers,
                "device": device or "auto",
            },
        },
    }

    TEST_LOG_FILE.write_text("", encoding="utf-8")
    timestamp = datetime.now(MYT).strftime("%Y%m%d-%H%M%S")
    test_log_file = LOG_DIR / f"test-{timestamp}.log"
    cmd = [
        TRAINING_PYTHON,
        str(TEST_SCRIPT),
        "--weights",
        str(weights_path),
        "--data",
        str(dataset_yaml),
        "--imgsz",
        str(imgsz),
        "--batch",
        str(batch),
        "--workers",
        str(workers),
        "--split",
        "test",
        "--project",
        str(test_project_path),
        "--name",
        test_name,
    ]
    if device.strip():
        cmd.extend(["--device", device.strip()])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    header = (
        "Model testing started.\n"
        f"Command: {' '.join(cmd)}\n"
        f"Weights: {weights_label}\n"
        f"Dataset: {dataset_info.get('source_split', 'test')} split from {dataset_info.get('dataset_root', dataset_yaml)}\n\n"
    )
    TEST_LOG_FILE.write_text(header, encoding="utf-8")
    test_log_file.write_text(header, encoding="utf-8")
    test_process = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
        text=True,
        bufsize=1,
    )
    threading.Thread(
        target=stream_test_logs,
        args=(test_process, [TEST_LOG_FILE, test_log_file]),
        daemon=True,
    ).start()
    test_started_at = time.time()

    return {
        "message": "Model testing started.",
        "pid": test_process.pid,
        "command": cmd,
        "log_file": str(TEST_LOG_FILE),
        "history_log_file": str(test_log_file),
        "test_run": {key: value for key, value in test_run_info.items() if key != "report_context"},
    }


@app.post("/api/test/stop")
def stop_test():
    global test_process

    if test_process is None or test_process.poll() is not None:
        test_process = None
        return {"message": "No model-testing process is running."}

    os.killpg(os.getpgid(test_process.pid), signal.SIGTERM)
    return {"message": "Stop signal sent to the test job."}


@app.get("/api/test/status")
def test_status():
    status = current_test_status()
    status["log_tail"] = read_log_tail_from(TEST_LOG_FILE, 4000)
    status["gpu"] = gpu_status()
    return status


@app.get("/api/test/logs", response_class=PlainTextResponse)
def test_logs():
    return read_log_tail_from(TEST_LOG_FILE)


@app.get("/api/test/logs/download")
def download_test_log():
    path = test_log_file if test_log_file and test_log_file.is_file() else latest_timestamped_test_log()
    if path is None and TEST_LOG_FILE.is_file():
        path = TEST_LOG_FILE
    if not path:
        raise HTTPException(status_code=404, detail="No test log found.")
    return FileResponse(path, media_type="text/plain", filename=path.name)


@app.get("/api/test/results")
def test_results():
    run_dir = current_test_run_dir()
    if run_dir is None:
        return {"available": False, "run_dir": "", "artifacts": {}}
    return read_test_metrics(run_dir)


@app.post("/api/test/artifacts/download")
def download_test_artifact(request: TestArtifactRequest):
    path = resolve_test_artifact_path(request.artifact)
    media_type = "application/json" if path.suffix == ".json" else "image/png"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.post("/api/test/report/download")
def download_combined_test_report():
    if current_test_status()["running"]:
        raise HTTPException(status_code=409, detail="Wait for model testing to finish before generating the report.")
    test_dir = current_test_run_dir()
    if test_dir is None:
        raise HTTPException(status_code=404, detail="No completed test run was found.")
    test_metrics_payload = read_test_metrics(test_dir)
    if not test_metrics_payload.get("available"):
        raise HTTPException(status_code=409, detail="The selected test run has no completed metrics.")
    test_context = read_json_object(test_dir / TEST_REPORT_CONTEXT_FILE)
    test_dataset_value = test_context.get("dataset_yaml") or test_metrics_payload.get("dataset_yaml")
    if test_dataset_value and not test_context.get("dataset_summary"):
        test_dataset_path = Path(str(test_dataset_value)).expanduser()
        if test_dataset_path.is_file():
            test_context["dataset_summary"] = cached_dataset_summary(test_dataset_path)
    training_dir_value = test_context.get("training_run_dir")
    if not training_dir_value:
        raise HTTPException(
            status_code=409,
            detail="A combined report is unavailable because this test used uploaded weights with no linked training run.",
        )
    training_dir = Path(training_dir_value).expanduser().resolve()
    ensure_runs_path(training_dir)
    training_metrics_payload = read_run_metrics(training_dir)
    if not training_metrics_payload.get("available"):
        raise HTTPException(status_code=409, detail="The linked training run has no completed metrics.")
    try:
        from .report_generator import generate_combined_report

        report_path = generate_combined_report(
            training_dir,
            load_training_report_context(training_dir),
            training_metrics_payload,
            test_dir,
            test_context,
            test_metrics_payload,
        )
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="PDF reporting requires the reportlab dependency. Rebuild the container image.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not generate the combined report: {exc}") from exc
    return FileResponse(report_path, media_type="application/pdf", filename="training_and_test_report.pdf")


@app.get("/api/test/artifacts/view/{artifact}")
def view_test_artifact(artifact: str):
    path = resolve_test_artifact_path(artifact)
    if path.suffix.lower() != ".png":
        raise HTTPException(status_code=400, detail="Only image artifacts can be viewed.")
    return FileResponse(path, media_type="image/png")


@app.post("/api/train/start")
def start_training(request: TrainRequest):
    global training_process, training_started_at, training_log_file, training_run_info

    status = current_status()
    if status["running"]:
        raise HTTPException(status_code=409, detail="Training is already running.")
    if current_test_status()["running"]:
        raise HTTPException(status_code=409, detail="Model testing is running. Stop testing before training.")
    with annotation_qa_jobs_lock:
        annotation_qa_running = any(
            job.get("status") in {"queued", "running", "stopping"}
            for job in annotation_qa_jobs.values()
        )
    if annotation_qa_running:
        raise HTTPException(status_code=409, detail="Annotation QA is running. Stop QA or wait before training.")

    dataset_yaml = Path(request.dataset_yaml).expanduser() if request.dataset_yaml else None
    if not request.resume and (dataset_yaml is None or not dataset_yaml.is_file()):
        raise HTTPException(status_code=400, detail="Prepare a valid dataset before starting a new training run.")

    model_spec = MODEL_REGISTRY.get(request.model_size)
    if not model_spec:
        raise HTTPException(status_code=400, detail=f"Unknown model size: {request.model_size}")
    model = model_spec["model"]
    model_family = model_spec["family"]
    model_task = model_spec["task"]
    requested_project = request.project
    if is_known_training_project_default(requested_project):
        requested_project = default_project_for_model_size(request.model_size)
    training_project_path = normalize_training_project_path(requested_project)
    requested_name = (request.name or "train").strip() or "train"
    run_name = timestamped_training_run_name(requested_name, resume=request.resume)
    resume_checkpoint = None
    resume_run_dir = None
    if request.resume:
        try:
            resume_run_dir, _ = resolve_run_dir_details(requested_project, run_name)
        except HTTPException as exc:
            raise HTTPException(
                status_code=400,
                detail="Cannot resume: no existing run was found for the selected project and run name.",
            ) from exc
        resume_checkpoint = resume_run_dir / "weights" / "last.pt"
        ensure_runs_path(resume_checkpoint)
        if not resume_checkpoint.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"Cannot resume: last.pt was not found in {resume_run_dir}.",
            )
        if model_family == "ultralytics":
            model = str(resume_checkpoint)

    training_values = {
        "imgsz": request.imgsz,
        "batch": request.batch,
        "lr0": request.lr0,
        "weight_decay": request.weight_decay,
        "warmup_epochs": request.warmup_epochs,
        "cos_lr": request.cos_lr,
    }
    if model_family in {"dfine", "rfdetr"}:
        for key in training_values:
            training_values[key] = backend_training_value(request, key, model_family)

    request_payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    request_payload.update(training_values)
    request_payload["family"] = model_family
    request_payload["task"] = model_task
    request_payload["model"] = model
    request_payload["project"] = str(training_project_path)
    request_payload["requested_name"] = requested_name
    request_payload["name"] = run_name
    report_context = {}
    if resume_run_dir:
        report_context = read_json_object(resume_run_dir / TRAINING_REPORT_CONTEXT_FILE)
        if model_family in {"dfine", "rfdetr"} and dataset_yaml is None and report_context.get("dataset_yaml"):
            dataset_yaml = Path(str(report_context["dataset_yaml"])).expanduser()
    if not report_context:
        report_dataset_yaml = dataset_yaml
        if report_dataset_yaml is not None:
            report_dataset_yaml = report_dataset_yaml.resolve()
        report_context = {
            "created_at": datetime.now(MYT).isoformat(),
            "dataset_yaml": str(report_dataset_yaml) if report_dataset_yaml else "",
            "dataset_summary": cached_dataset_summary(report_dataset_yaml) if report_dataset_yaml else {},
            "hyperparameters": request_payload,
            "task": model_task,
            "family": model_family,
            "model": model,
            "pretrained": True,
            "device": request.device or os.getenv("TRAINING_DEVICE") or "auto",
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "ultralytics": installed_version("ultralytics"),
                "rfdetr": installed_version("rfdetr"),
                "dfine_repo": os.getenv("DFINE_REPO_DIR") or str(REPO_ROOT / "third_party" / "D-FINE"),
                "torch": installed_version("torch"),
                "gpus": gpu_status().get("gpus", []),
            },
        }
    report_context["last_started_at"] = datetime.now(MYT).isoformat()
    report_context["task"] = model_task
    report_context["family"] = model_family
    report_context["model"] = model
    report_context["hyperparameters"] = request_payload
    report_context["resume"] = bool(request.resume)

    training_run_info = {
        "requested_project": requested_project,
        "requested_name": requested_name,
        "project": str(training_project_path),
        "name": run_name,
        "family": model_family,
        "task": model_task,
        "model": model,
        "expected_run_dir": str(training_project_path / run_name),
        "run_dir": str(resume_run_dir) if resume_run_dir else "",
        "resolution_type": "actual" if resume_run_dir else "pending",
        "current_epoch": 0,
        "total_epochs": request.epochs,
        "report_context": report_context,
    }

    if resume_run_dir:
        persist_training_report_context(resume_run_dir)

    ensure_dirs()
    LOG_FILE.write_text("", encoding="utf-8")
    timestamp = datetime.now(MYT).strftime("%Y%m%d-%H%M%S")
    training_log_file = LOG_DIR / f"train-{timestamp}.log"

    train_script = Path(model_spec["runner"])
    if not train_script.is_file():
        raise HTTPException(status_code=500, detail=f"Training script is missing: {train_script}")

    cmd = [
        TRAINING_PYTHON,
        str(train_script),
        "--model", model,
        "--epochs", str(request.epochs),
        "--imgsz", str(training_values["imgsz"]),
        "--batch", str(training_values["batch"]),
        "--patience", str(request.patience),
        "--save-period", str(request.save_period),
        "--workers", str(request.workers),
        "--lr0", str(training_values["lr0"]),
        "--weight-decay", str(training_values["weight_decay"]),
        "--warmup-epochs", str(training_values["warmup_epochs"]),
        "--seed", str(request.seed),
        "--project", str(training_project_path),
        "--name", run_name,
    ]
    if model_family == "ultralytics":
        cmd.extend([
            "--optimizer", request.optimizer,
            "--lrf", str(request.lrf),
            "--pretrained", "true",
            "--activation", request.activation,
            "--disable-ultralytics-albumentations", str(request.disable_ultralytics_albumentations).lower(),
            "--mosaic", str(request.mosaic),
            "--close-mosaic", str(request.close_mosaic),
            "--hsv-h", str(request.hsv_h),
            "--hsv-s", str(request.hsv_s),
            "--hsv-v", str(request.hsv_v),
            "--degrees", str(request.degrees),
            "--translate", str(request.translate),
            "--scale", str(request.scale),
            "--shear", str(request.shear),
            "--perspective", str(request.perspective),
            "--flipud", str(request.flipud),
            "--fliplr", str(request.fliplr),
            "--bgr", str(request.bgr),
            "--mixup", str(request.mixup),
            "--cutmix", str(request.cutmix),
            "--copy-paste", str(request.copy_paste),
            "--erasing", str(request.erasing),
        ])
        if request.auto_augment:
            cmd.extend(["--auto-augment", request.auto_augment])

    if dataset_yaml is not None and (not request.resume or model_family in {"dfine", "rfdetr"}):
        cmd.extend(["--data", str(dataset_yaml)])

    device = request.device or os.getenv("TRAINING_DEVICE")
    if device:
        cmd.extend(["--device", device])
    if training_values["cos_lr"]:
        cmd.append("--cos-lr")
    if model_family == "ultralytics" and request.freeze is not None:
        cmd.extend(["--freeze", str(request.freeze)])
    if model_family == "ultralytics" and request.exist_ok:
        cmd.append("--exist-ok")
    if request.resume:
        cmd.append("--resume")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    header = (
        "Training started.\n"
        f"Command: {' '.join(cmd)}\n\n"
    )
    LOG_FILE.write_text(header, encoding="utf-8")
    training_log_file.write_text(header, encoding="utf-8")
    training_process = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
        text=True,
        bufsize=1,
    )
    threading.Thread(
        target=stream_training_logs,
        args=(training_process, [LOG_FILE, training_log_file]),
        daemon=True,
    ).start()
    training_started_at = time.time()

    return {
        "message": "Training started.",
        "pid": training_process.pid,
        "command": cmd,
        "log_file": str(LOG_FILE),
        "history_log_file": str(training_log_file),
        "training_run": {key: value for key, value in training_run_info.items() if key != "report_context"},
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else "",
    }


@app.post("/api/train/stop")
def stop_training():
    global training_process

    if training_process is None or training_process.poll() is not None:
        training_process = None
        return {"message": "No training process is running."}

    os.killpg(os.getpgid(training_process.pid), signal.SIGTERM)
    return {"message": "Stop signal sent."}


@app.get("/api/train/status")
def train_status():
    status = current_status()
    status["log_tail"] = read_log_tail(4000)
    status["gpu"] = gpu_status()
    return status


@app.get("/api/train/logs", response_class=PlainTextResponse)
def train_logs():
    return read_log_tail()


@app.get("/api/train/logs/full", response_class=PlainTextResponse)
def train_logs_full():
    return read_log_file(LOG_FILE)


@app.get("/api/train/logs/errors", response_class=PlainTextResponse)
def train_logs_errors():
    return read_error_log(LOG_FILE)


@app.get("/api/train/logs/download")
def download_run_log():
    path = training_log_file if training_log_file and training_log_file.is_file() else latest_timestamped_log()
    if path is None and LOG_FILE.is_file():
        path = LOG_FILE
    if not path:
        raise HTTPException(status_code=404, detail="No training log found.")
    return FileResponse(path, media_type="text/plain", filename=path.name)


@app.post("/api/train/weights/status")
def weights_status(request: WeightRequest):
    try:
        run_dir, resolution_type = resolve_run_dir_details(request.project, request.name)
    except HTTPException:
        run_dir = None
        resolution_type = "not_found"

    result = {
        "run_dir": str(run_dir) if run_dir else "",
        "resolution_type": resolution_type,
    }
    for weight in ("best", "last"):
        path = run_dir / "weights" / f"{weight}.pt" if run_dir else None
        if path and path.is_file():
            result[weight] = {"available": True, "path": str(path), "size": path.stat().st_size}
        else:
            result[weight] = {"available": False, "path": "", "size": 0}
    return result


@app.post("/api/train/metrics")
def train_metrics(request: WeightRequest):
    try:
        run_dir, resolution_type = resolve_run_dir_details(request.project, request.name)
    except HTTPException:
        return {
            "available": False,
            "run_dir": "",
            "resolution_type": "not_found",
            "artifacts": {},
        }

    result = read_run_metrics(run_dir)
    result["resolution_type"] = resolution_type
    return result


@app.post("/api/train/artifacts/download")
def download_artifact(request: ArtifactRequest):
    path = resolve_artifact_path(request)
    media_type = "text/csv" if path.suffix == ".csv" else "image/png" if path.suffix == ".png" else "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.post("/api/train/report/download")
def download_training_report(request: WeightRequest):
    if current_status()["running"]:
        raise HTTPException(status_code=409, detail="Wait for training to finish before generating the report.")
    run_dir = resolve_run_dir(request.project, request.name)
    metrics = read_run_metrics(run_dir)
    if not metrics.get("available"):
        raise HTTPException(status_code=409, detail="The selected training run has no completed metrics.")
    try:
        from .report_generator import generate_training_report

        report_path = generate_training_report(
            run_dir,
            load_training_report_context(run_dir),
            metrics,
        )
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="PDF reporting requires the reportlab dependency. Rebuild the container image.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not generate the training report: {exc}") from exc
    return FileResponse(report_path, media_type="application/pdf", filename=training_report_download_filename(run_dir))


@app.get("/api/train/artifacts/view/{artifact}")
def view_artifact(artifact: str, project: str = "runs/detect", name: str = "train"):
    path = resolve_artifact_path(
        ArtifactRequest(project=project, name=name, artifact=artifact)
    )
    if path.suffix.lower() != ".png":
        raise HTTPException(status_code=400, detail="Only image artifacts can be viewed.")
    return FileResponse(path, media_type="image/png")


@app.get("/api/train/weights/{weight}")
def download_weight(weight: str, project: str = "runs/detect", name: str = "train"):
    path = resolve_weight_path(project, name, weight)
    timestamp = datetime.now(MYT).strftime("%Y%m%d_%H%M%S")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"{weight}_{timestamp}.pt",
    )
