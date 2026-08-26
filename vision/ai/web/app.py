#!/usr/bin/env python3
"""FastAPI web UI backend for YOLOv8 training."""

import csv
import functools
import hashlib
import json
import math
import mimetypes
import os
import platform
import random
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Callable, Optional

import yaml
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .dataset_provenance import roboflow_pre_augmentation_summary
from .common.cache import SignatureCache, TimedCache
from .common.files import file_signature, read_text_tail
from .common.uploads import (
    UploadLimits,
    UploadValidationError,
    extract_zip_safely,
    replace_directory,
    safe_leaf_filename,
    validate_upload_totals,
)
from .roboflow_sync import (
    ROBOFLOW_SYNC_LOG_FILE,
    ROBOFLOW_SYNC_PREVIEW_FILE,
    annotation_digest,
    build_roboflow_provenance,
    canonical_local_annotation,
    canonical_remote_annotation,
    load_roboflow_provenance,
    redact_secret,
    roboflow_image_details,
    roboflow_search_images,
    roboflow_upload_annotation,
    sync_preview_digest,
    write_roboflow_provenance,
)
from .sam_qa_runtime import SamQaRuntime
from .infer_yolo import InferenceStopped, run_yolo_inference
from .infer_rfdetr import run_rfdetr_inference
from .infer_dfine import run_dfine_inference
from .stratified_split import SPLIT_NAMES, stratified_split
from .runtime.job_manager import PersistentJobStore, ResourceBusyError, ResourceCoordinator
from .schemas import (
    AnnotationQaFixRequest,
    AnnotationQaMarkRequest,
    AnnotationQaRequest,
    AnnotationQaRoboflowRequest,
    DatasetDownloadRequest,
    RoboflowRequest,
    SplitConfig,
    TestArtifactRequest,
    WebRTCOffer,
)
from vision.ai.train.train_dfine import generate_dfine_report_artifacts
from vision.ai.train.train_rfdetr import RFDETR_RUN_LOG, finalize_rfdetr_artifacts, generate_rfdetr_report_artifacts


WEB_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEB_DIR.parents[2]
TRAIN_SCRIPT = REPO_ROOT / "vision" / "ai" / "train" / "train_yolov8.py"
RFDETR_TRAIN_SCRIPT = REPO_ROOT / "vision" / "ai" / "train" / "train_rfdetr.py"
DFINE_TRAIN_SCRIPT = REPO_ROOT / "vision" / "ai" / "train" / "train_dfine.py"
TEST_SCRIPT = REPO_ROOT / "vision" / "ai" / "train" / "test_yolov8.py"
RFDETR_TEST_SCRIPT = REPO_ROOT / "vision" / "ai" / "train" / "test_rfdetr.py"
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
MAGIC_METRICS_FILE = "magic_metrics.json"

load_dotenv(WEB_DIR / ".env")

DATA_ROOT = Path(os.getenv("WEB_DATA_ROOT") or WEB_DIR / "datasets").expanduser()
INFERENCE_ROOT = DATA_ROOT / "inference"
INFERENCE_UPLOAD_ROOT = INFERENCE_ROOT / "uploads"
INFERENCE_JOB_ROOT = INFERENCE_ROOT / "jobs"
DATASET_UPLOAD_ROOT = DATA_ROOT / "uploads"
DATASET_EXTRACTED_ROOT = DATA_ROOT / "extracted"
DATASET_PREPARED_ROOT = DATA_ROOT / "prepared"
RUNTIME_ROOT = DATA_ROOT / "runtime"
TRAINING_PYTHON = os.getenv("TRAINING_PYTHON", "python3")
UPLOAD_LIMITS = UploadLimits.from_environment()

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
    "warmup_epochs": 500.0,
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
WEB_PROGRESS_RE = re.compile(r"^WEB_TRAINING_PROGRESS\s+epoch=(\d+)\s+total=(\d+)(?:\s+(?P<fields>.*))?$")
DFINE_PROGRESS_LOG_RE = re.compile(
    r"^Epoch:\s*\[\s*(?P<epoch>\d+)\s*/\s*(?P<total>\d+)\s*\]\s*"
    r"\[\s*(?P<step>\d+)\s*/\s*(?P<steps>\d+)\s*\].*?"
    r"(?:^|\s)eta:\s*(?P<eta>\S+).*?"
    r"\blr:\s*(?P<lr>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?).*?"
    r"\bloss:\s*(?P<loss>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*"
    r"\((?P<loss_avg>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\)",
    re.IGNORECASE,
)
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
ANNOTATION_QA_REPORT_VERSION = 4
ANNOTATION_QA_SAFE_MAPPING_VERSION = 2
SAM3_QA_MODEL_PATH = Path(os.getenv("SAM3_QA_MODEL_PATH") or REPO_ROOT / "sam3.pt").expanduser().resolve()
SAM3_MIN_ULTRALYTICS_VERSION = (8, 3, 237)
SAM_QA_MODEL_REGISTRY = {
    "sam2.1_s.pt": {
        "label": "SAM2.1 Hiera Small",
        "path": "sam2.1_s.pt",
        "backend": "sam2",
        "max_side": 1280,
        "prompt_chunk": 1024,
        "automatic_allowed": True,
    },
    "sam2.1_t.pt": {
        "label": "SAM2.1 Hiera Tiny",
        "path": "sam2.1_t.pt",
        "backend": "sam2",
        "max_side": 1280,
        "prompt_chunk": 1024,
        "automatic_allowed": True,
    },
    "sam2.1_b.pt": {
        "label": "SAM2.1 Hiera Base+",
        "path": "sam2.1_b.pt",
        "backend": "sam2",
        "max_side": 1280,
        "prompt_chunk": 1024,
        "automatic_allowed": True,
    },
    "sam2.1_l.pt": {
        "label": "SAM2.1 Hiera Large",
        "path": "sam2.1_l.pt",
        "backend": "sam2",
        "max_side": 1280,
        "prompt_chunk": 1024,
        "automatic_allowed": True,
    },
    "sam3": {
        "label": "SAM 3",
        "path": str(SAM3_QA_MODEL_PATH),
        "backend": "sam3",
        "max_side": 1008,
        "prompt_chunk": 8,
        "automatic_allowed": False,
        "recommended_vram_gb": 16,
    },
}

@asynccontextmanager
async def application_lifespan(_app: FastAPI):
    ensure_dirs()
    job_store.recover_interrupted()
    yield


app = FastAPI(title="YOLOv8 Training UI", lifespan=application_lifespan)
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
resource_coordinator = ResourceCoordinator()
job_store = PersistentJobStore(RUNTIME_ROOT / "jobs.json")
run_metrics_cache = SignatureCache(max_items=128)
training_sessions_cache = TimedCache(ttl_seconds=5.0)
compute_start_lock = threading.RLock()

DatasetProgressCallback = Callable[[str, int, int, str], None]
GPU_STATUS_CACHE_SECONDS = 2.0
DATASET_DOWNLOAD_MAX_AGE_SECONDS = 60 * 60


def serialized_compute_start(function):
    """Make the check-and-start portion of GPU workflows atomic."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with compute_start_lock:
            return function(*args, **kwargs)

    return wrapper


def active_compute_operations() -> list[str]:
    active = []
    if training_process is not None and training_process.poll() is None:
        active.append("training")
    if test_process is not None and test_process.poll() is None:
        active.append("model testing")
    with annotation_qa_jobs_lock:
        if any(job.get("status") in {"queued", "running", "stopping"} for job in annotation_qa_jobs.values()):
            active.append("annotation QA")
    with inference_jobs_lock:
        if any(job.get("status") not in {"complete", "failed", "stopped"} for job in inference_jobs.values()):
            active.append("inference")
    return active


def ensure_compute_available(operation: str):
    active = active_compute_operations()
    if active:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot start {operation} while {', '.join(active)} is active.",
        )


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
    optimizer: str = "auto"
    lr0: float = Field(default=0.001, gt=0)
    lrf: float = Field(default=0.01, gt=0)
    weight_decay: float = Field(default=0.0005, ge=0)
    cls_pw: float = Field(default=0.0, ge=0, le=1)
    cos_lr: bool = False
    warmup_epochs: float = Field(default=3.0, ge=0)
    freeze: Optional[int] = Field(default=None, ge=0)
    activation: str = "silu"
    exist_ok: bool = False
    seed: int = 42
    project: str = "runs/detect"
    name: str = "train"
    resume: bool = False
    augmentation_enabled: bool = False
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


class MagicMetricAdjustment(BaseModel):
    scope: str = "overall"
    metric_key: str = "map50_95"
    target: float = Field(ge=0, le=1)
    class_name: str = ""


class MagicMetricsRequest(WeightRequest):
    adjustments: list[MagicMetricAdjustment] = Field(default_factory=list)
    scope: str = "overall"
    metric_key: str = "map50_95"
    target: Optional[float] = Field(default=None, ge=0, le=1)
    class_name: str = ""


class ArtifactRequest(BaseModel):
    project: str = "runs/detect"
    name: str = "train"
    artifact: str


def ensure_dirs():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TEST_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    DATASET_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    DATASET_EXTRACTED_ROOT.mkdir(parents=True, exist_ok=True)
    DATASET_PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
    INFERENCE_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    INFERENCE_JOB_ROOT.mkdir(parents=True, exist_ok=True)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)


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


def upload_validation_http_error(exc: UploadValidationError) -> HTTPException:
    return HTTPException(status_code=413, detail=str(exc))


def acquire_resource(resource: str, owner: str, operation: str):
    try:
        return resource_coordinator.acquire(resource, owner, operation)
    except ResourceBusyError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot start {operation}; {exc.active_operation} is already using {resource}.",
        ) from exc


def persist_job(*args, **kwargs):
    """Keep observability persistence from interrupting an active ML job."""
    try:
        return job_store.update(*args, **kwargs)
    except (OSError, TypeError, ValueError):
        return None


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


def model_registry_payload() -> list[dict]:
    models = []
    for model_id, spec in MODEL_REGISTRY.items():
        family = str(spec.get("family") or "ultralytics")
        models.append({
            "id": model_id,
            "family": family,
            "task": spec.get("task") or "detect",
            "checkpoint": spec.get("model") or "",
            "project_default": spec.get("project_default") or "runs/detect",
            "capabilities": {
                "training": True,
                "testing": family != "dfine",
                "inference": family != "dfine",
                "onnx_export": family == "ultralytics",
                "augmentation": family == "ultralytics",
            },
        })
    return models


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


def combined_report_download_filename(test_dir: Path) -> str:
    match = TIMESTAMPED_RUN_SUFFIX_RE.match(test_dir.name)
    if match is None:
        return "training_and_test_report.pdf"
    return f"training_and_test_report_{match.group('timestamp')}.pdf"


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


from .services import inference as inference_service
from .services.inference import (
    InferenceDependencies,
    STORAGE_CLEANUP_TARGETS,
    configure_inference,
    directory_size,
    inference_jobs_storage_payload,
    normalize_storage_target_key,
    storage_target_payload,
    storage_payload,
    has_active_inference_job,
    has_active_dataset_preparation,
    ensure_storage_target_idle,
    clear_storage_target,
    clear_inference_outputs,
    relative_to_repo,
    resolve_inference_weight_path,
    family_for_weight_path,
    task_for_runs_root,
    family_for_runs_root,
    available_inference_weights,
    available_detection_weights,
    normalize_exported_model_path,
    export_inference_pt_to_onnx,
    available_training_sessions,
    ensure_inference_script,
    inference_media_type,
    inference_job_payload,
    update_inference_job,
    inference_has_live_preview,
    update_inference_preview_frame,
    update_inference_webrtc_frame,
    latest_inference_webrtc_frame,
    inference_mjpeg_stream,
    read_inference_progress,
    inference_result_payload_from_disk,
    inference_job_from_disk,
    run_inference_job,
    stop_inference_process,
    is_run_dir,
    find_named_run_dir,
    resolve_run_dir_details,
    resolve_run_dir,
)

configure_inference(
    InferenceDependencies(
        repo_root=REPO_ROOT,
        inference_script=INFERENCE_SCRIPT,
        inference_upload_root=INFERENCE_UPLOAD_ROOT,
        inference_job_root=INFERENCE_JOB_ROOT,
        dataset_upload_root=DATASET_UPLOAD_ROOT,
        dataset_extracted_root=DATASET_EXTRACTED_ROOT,
        dataset_prepared_root=DATASET_PREPARED_ROOT,
        training_runs_roots=TRAINING_RUNS_ROOTS,
        rfdetr_runs_root=RFDETR_RUNS_ROOT,
        dfine_runs_root=DFINE_RUNS_ROOT,
        segment_runs_root=SEGMENT_RUNS_ROOT,
        semantic_runs_root=SEMANTIC_RUNS_ROOT,
        classify_runs_root=CLASSIFY_RUNS_ROOT,
        image_extensions=IMAGE_EXTENSIONS,
        video_extensions=VIDEO_EXTENSIONS,
        weight_extensions=INFERENCE_WEIGHT_EXTENSIONS,
        training_context_filename=TRAINING_REPORT_CONTEXT_FILE,
        jobs=inference_jobs,
        jobs_lock=inference_jobs_lock,
        dataset_jobs=dataset_preparation_jobs,
        dataset_jobs_lock=dataset_preparation_lock,
        current_status=lambda: current_status(),
        test_status=lambda: current_test_status(),
        ensure_dirs=ensure_dirs,
        ensure_inference_path=ensure_inference_path,
        ensure_inference_runs_path=ensure_inference_runs_path,
        ensure_runs_path=ensure_runs_path,
        persist_job=persist_job,
        read_json_object=lambda path: read_json_object(path),
        read_log_file=lambda path: read_log_file(path),
        resolve_project_path=resolve_project_path,
        training_run_info=lambda: training_run_info,
    )
)

_available_inference_weights = available_inference_weights


def available_inference_weights() -> list[dict]:
    """Compatibility facade that honors runtime-overridden run roots."""
    inference_service.TRAINING_RUNS_ROOTS = TRAINING_RUNS_ROOTS
    inference_service.RFDETR_RUNS_ROOT = RFDETR_RUNS_ROOT
    inference_service.DFINE_RUNS_ROOT = DFINE_RUNS_ROOT
    inference_service.SEGMENT_RUNS_ROOT = SEGMENT_RUNS_ROOT
    inference_service.SEMANTIC_RUNS_ROOT = SEMANTIC_RUNS_ROOT
    inference_service.CLASSIFY_RUNS_ROOT = CLASSIFY_RUNS_ROOT
    return _available_inference_weights()


from .services.datasets import (
    DatasetDependencies,
    configure_datasets,
    validate_classes,
    normalize_yaml_names,
    read_yaml_class_names,
    detect_dataset_classes,
    resolve_dataset_classes,
    dataset_response,
    resolve_yaml_dataset_root,
    split_image_folder,
    prepared_dataset_yaml,
    build_dataset_archive,
    build_annotated_dataset_archive,
    cleanup_expired_dataset_downloads,
    label_folder_for_images,
    update_class_distribution,
    read_split_metadata,
    inspect_dataset_yaml,
    validate_split,
    find_dataset_yaml,
    find_dataset_root,
    split_dirs,
    flat_dirs,
    image_files,
    has_image_files,
    copy_pair,
    write_dataset_yaml,
    prepare_existing_split,
    collect_source_images,
    prepare_split_dataset,
    prepare_dataset,
    prepare_roboflow_download,
    resolve_test_split_source,
    build_test_dataset_yaml,
    resolve_reference_classes,
    prepare_custom_test_dataset,
    resolve_prepared_test_dataset_yaml,
)

configure_datasets(
    DatasetDependencies(
        data_root=DATA_ROOT,
        image_extensions=IMAGE_EXTENSIONS,
        split_aliases=SPLIT_ALIASES,
        split_metadata_filename=SPLIT_METADATA_FILE,
        summary_filename=DATASET_SUMMARY_FILE,
        download_max_age_seconds=DATASET_DOWNLOAD_MAX_AGE_SECONDS,
        downloads=dataset_downloads,
        downloads_lock=dataset_download_lock,
        clean_name=clean_name,
    )
)


from .services.metrics import (
    MetricsDependencies,
    configure_metrics,
    float_value,
    sum_values,
    loss_components,
    raw_loss_sum,
    comparable_loss_component_names,
    loss_summary,
    loss_note,
    format_metric,
    f1_from_precision_recall,
    parse_metric_row,
    parse_class_metrics_from_log,
    find_training_log_for_run,
    infer_task_from_results_columns,
    infer_run_task,
    metric_profile,
    build_metric_history,
    best_metric_summary,
    artifact_status,
    run_artifact_statuses,
    read_web_metrics,
    is_rfdetr_run,
    is_dfine_run,
    rfdetr_class_names_from_context,
    rfdetr_dataset_audit_from_context,
    rfdetr_model_id_from_context,
    rfdetr_epochs_from_context,
    find_rfdetr_log_for_run,
    ensure_rfdetr_web_artifacts,
    rfdetr_report_artifacts_ready,
    rfdetr_report_dataset_yaml,
    rfdetr_report_weights_available,
    dfine_report_artifacts_ready,
    dfine_report_dataset_yaml,
    dfine_report_weights_available,
    ensure_rfdetr_report_artifacts_for_report,
    ensure_dfine_report_artifacts_for_report,
    ensure_model_report_artifacts_for_report,
    read_run_metrics,
    run_metrics_signature,
    cached_run_metrics,
)


def clamp_metric(value: float) -> float:
    return min(1.0, max(0.0, value))


def magic_metric_value(row: dict, key: str, default: float) -> float:
    value = float_value(row, key) if isinstance(row, dict) else None
    return clamp_metric(value if value is not None else default)


def rebalanced_metric_values(values: list[float], target: float, digits: int = 4) -> list[float]:
    return rebalanced_metric_total(values, target * len(values), digits)


def rebalanced_metric_total(values: list[float], target_total: float, digits: int = 4) -> list[float]:
    if not values:
        return []
    scale = 10 ** digits
    target_integer_total = int(round(target_total * scale))
    integers = [min(scale, max(0, int(round(value * scale)))) for value in values]
    difference = target_integer_total - sum(integers)
    while difference:
        step = 1 if difference > 0 else -1
        progressed = False
        for index, value in enumerate(integers):
            if step > 0 and value >= scale:
                continue
            if step < 0 and value <= 0:
                continue
            integers[index] += step
            difference -= step
            progressed = True
            if difference == 0:
                break
        if not progressed:
            break
    return [value / scale for value in integers]


def adjusted_f1_summary(classes: list[dict]) -> tuple[Optional[float], Optional[float]]:
    if not classes:
        return None, None
    macro_f1 = sum(magic_metric_value(row, "f1", 0.0) for row in classes) / len(classes)
    total_instances = sum(max(0, int(row.get("instances") or 0)) for row in classes)
    if total_instances:
        weighted_f1 = sum(
            magic_metric_value(row, "f1", 0.0) * max(0, int(row.get("instances") or 0))
            for row in classes
        ) / total_instances
    else:
        weighted_f1 = macro_f1
    return format_metric(macro_f1), format_metric(weighted_f1)


MAGIC_OVERALL_METRICS = {
    "map50_95": "mAP50-95",
    "map50": "mAP50",
    "precision": "Precision",
    "recall": "Recall",
    "macro_f1": "Macro F1",
    "weighted_f1": "Weighted F1",
}
MAGIC_PER_CLASS_METRICS = {
    "map50_95": "AP50-95",
    "map50": "AP50",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}


def magic_adjustment_label(scope: str, metric_key: str, class_name: str = "") -> str:
    if scope == "per_class":
        metric_label = MAGIC_PER_CLASS_METRICS.get(metric_key, metric_key)
        return f"{class_name} {metric_label}".strip()
    return MAGIC_OVERALL_METRICS.get(metric_key, metric_key)


def precision_recall_for_f1(target_f1: float, rng: random.Random) -> tuple[float, float]:
    target = clamp_metric(target_f1)
    if target in {0.0, 1.0}:
        return target, target
    for _ in range(12):
        precision = clamp_metric(target + rng.uniform(-0.04, 0.04))
        if precision <= target / 2:
            continue
        recall = (target * precision) / ((2 * precision) - target)
        if math.isfinite(recall) and 0 <= recall <= 1:
            return precision, recall
    return target, target


def row_with_f1(row: dict, target_f1: float, rng: random.Random) -> dict:
    precision, recall = precision_recall_for_f1(target_f1, rng)
    adjusted = dict(row)
    adjusted.update({
        "precision": format_metric(precision),
        "recall": format_metric(recall),
        "f1": format_metric(f1_from_precision_recall(precision, recall)),
    })
    return adjusted


def summarize_adjusted_classes(classes: list[dict], fallback: dict) -> dict:
    if not classes:
        return {
            "precision": fallback.get("precision"),
            "recall": fallback.get("recall"),
            "map50": fallback.get("map50"),
            "map50_95": fallback.get("map50_95"),
            "macro_f1": fallback.get("macro_f1"),
            "weighted_f1": fallback.get("weighted_f1"),
        }
    macro_f1, weighted_f1 = adjusted_f1_summary(classes)
    return {
        "precision": format_metric(sum(magic_metric_value(row, "precision", 0.0) for row in classes) / len(classes)),
        "recall": format_metric(sum(magic_metric_value(row, "recall", 0.0) for row in classes) / len(classes)),
        "map50": format_metric(sum(magic_metric_value(row, "map50", 0.0) for row in classes) / len(classes)),
        "map50_95": format_metric(sum(magic_metric_value(row, "map50_95", 0.0) for row in classes) / len(classes)),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def base_magic_classes(metrics: dict, rng: random.Random) -> list[dict]:
    source_classes = metrics.get("per_class") if isinstance(metrics.get("per_class"), list) else []
    classes = []
    for row in source_classes:
        precision = magic_metric_value(row, "precision", magic_metric_value(metrics, "precision", 0.0))
        recall = magic_metric_value(row, "recall", magic_metric_value(metrics, "recall", 0.0))
        map50_95 = magic_metric_value(row, "map50_95", magic_metric_value(metrics, "map50_95", 0.0))
        map50 = max(map50_95, magic_metric_value(row, "map50", magic_metric_value(metrics, "map50", map50_95)))
        adjusted = dict(row)
        adjusted.update({
            "precision": format_metric(precision),
            "recall": format_metric(recall),
            "f1": format_metric(f1_from_precision_recall(precision, recall)),
            "map50": format_metric(map50),
            "map50_95": format_metric(map50_95),
        })
        classes.append(adjusted)
    return classes


def adjust_overall_magic_metric(
    metrics: dict,
    metric_key: str,
    target: float,
    rng: random.Random,
    locked_metrics: Optional[set[tuple[str, str]]] = None,
) -> tuple[dict, list[dict]]:
    classes = base_magic_classes(metrics, rng)
    summary = summarize_adjusted_classes(classes, metrics)
    locked_metrics = locked_metrics or set()
    if not classes:
        summary[metric_key] = format_metric(target)
        if metric_key in {"precision", "recall"}:
            precision = magic_metric_value(summary, "precision", target)
            recall = magic_metric_value(summary, "recall", target)
            summary["macro_f1"] = format_metric(f1_from_precision_recall(precision, recall))
            summary["weighted_f1"] = summary["macro_f1"]
        return summary, classes

    if metric_key in {"map50_95", "map50", "precision", "recall"}:
        current = magic_metric_value(summary, metric_key, target)
        free_indexes = [
            index
            for index, row in enumerate(classes)
            if (str(row.get("class_name") or ""), metric_key) not in locked_metrics
        ]
        locked_total = sum(
            magic_metric_value(row, metric_key, current)
            for index, row in enumerate(classes)
            if index not in free_indexes
        )
        remaining_total = (target * len(classes)) - locked_total
        if not free_indexes and abs(remaining_total) <= 0.00005:
            summary[metric_key] = format_metric(target)
            return summary, classes
        if not free_indexes or remaining_total < -0.00005 or remaining_total > len(free_indexes) + 0.00005:
            raise HTTPException(
                status_code=409,
                detail=f"The fixed per-class values cannot satisfy the requested {MAGIC_OVERALL_METRICS[metric_key]} target.",
            )
        free_target = remaining_total / len(free_indexes)
        raw_values = []
        free_current = sum(magic_metric_value(classes[index], metric_key, current) for index in free_indexes) / len(free_indexes)
        for index in free_indexes:
            value = magic_metric_value(classes[index], metric_key, current)
            raw_values.append(clamp_metric(value + (free_target - free_current)))
        balanced = rebalanced_metric_total(raw_values, remaining_total)
        for index, value in zip(free_indexes, balanced):
            row = classes[index]
            row[metric_key] = format_metric(value)
            if metric_key == "map50_95":
                row["map50"] = format_metric(max(value, magic_metric_value(row, "map50", value)))
            elif metric_key == "map50":
                row["map50_95"] = format_metric(min(value, magic_metric_value(row, "map50_95", value)))
            elif metric_key in {"precision", "recall"}:
                precision = magic_metric_value(row, "precision", target)
                recall = magic_metric_value(row, "recall", target)
                row["f1"] = format_metric(f1_from_precision_recall(precision, recall))
    elif metric_key in {"macro_f1", "weighted_f1"}:
        free_indexes = [
            index
            for index, row in enumerate(classes)
            if (str(row.get("class_name") or ""), "f1") not in locked_metrics
        ]
        if not free_indexes:
            raise HTTPException(status_code=409, detail="The fixed per-class F1 values leave no classes available to rebalance.")
        if metric_key == "macro_f1":
            locked_total = sum(
                magic_metric_value(row, "f1", 0.0)
                for index, row in enumerate(classes)
                if index not in free_indexes
            )
            remaining_total = (target * len(classes)) - locked_total
            if remaining_total < -0.00005 or remaining_total > len(free_indexes) + 0.00005:
                raise HTTPException(status_code=409, detail="The fixed per-class F1 values cannot satisfy the requested Macro F1 target.")
            free_current = sum(magic_metric_value(classes[index], "f1", target) for index in free_indexes) / len(free_indexes)
            free_target = remaining_total / len(free_indexes)
            raw_values = [
                clamp_metric(magic_metric_value(classes[index], "f1", free_current) + (free_target - free_current))
                for index in free_indexes
            ]
            f1_values = rebalanced_metric_total(raw_values, remaining_total)
        else:
            instance_weights = [max(0, int(row.get("instances") or 0)) for row in classes]
            if not sum(instance_weights):
                instance_weights = [1 for _ in classes]
            locked_total = sum(
                magic_metric_value(row, "f1", 0.0) * instance_weights[index]
                for index, row in enumerate(classes)
                if index not in free_indexes
            )
            free_weight = sum(instance_weights[index] for index in free_indexes)
            remaining_weighted_total = (target * sum(instance_weights)) - locked_total
            if not free_weight or remaining_weighted_total < -0.00005 or remaining_weighted_total > free_weight + 0.00005:
                raise HTTPException(status_code=409, detail="The fixed per-class F1 values cannot satisfy the requested Weighted F1 target.")
            f1_values = [clamp_metric(remaining_weighted_total / free_weight) for _ in free_indexes]
        for index, value in zip(free_indexes, f1_values):
            classes[index] = row_with_f1(classes[index], value, rng)

    summary = summarize_adjusted_classes(classes, metrics)
    summary[metric_key] = format_metric(target)
    if metric_key == "map50":
        summary["map50_95"] = format_metric(min(magic_metric_value(summary, "map50_95", target), target))
    elif metric_key == "map50_95":
        summary["map50"] = format_metric(max(magic_metric_value(summary, "map50", target), target))
    return summary, classes


def adjust_per_class_magic_metric(metrics: dict, metric_key: str, target: float, class_name: str, rng: random.Random) -> tuple[dict, list[dict]]:
    classes = base_magic_classes(metrics, rng)
    if not classes:
        raise HTTPException(status_code=409, detail="The selected run has no per-class metrics to adjust.")
    matched = False
    for index, row in enumerate(classes):
        if str(row.get("class_name") or "") != class_name:
            continue
        matched = True
        if metric_key == "f1":
            classes[index] = row_with_f1(row, target, rng)
        else:
            row[metric_key] = format_metric(target)
            if metric_key == "map50_95":
                row["map50"] = format_metric(max(target, magic_metric_value(row, "map50", target)))
            elif metric_key == "map50":
                row["map50_95"] = format_metric(min(target, magic_metric_value(row, "map50_95", target)))
            elif metric_key in {"precision", "recall"}:
                precision = magic_metric_value(row, "precision", target)
                recall = magic_metric_value(row, "recall", target)
                row["f1"] = format_metric(f1_from_precision_recall(precision, recall))
        break
    if not matched:
        raise HTTPException(status_code=404, detail="Selected class was not found in the run metrics.")
    return summarize_adjusted_classes(classes, metrics), classes


def normalize_magic_adjustment(adjustment: dict) -> dict:
    if not isinstance(adjustment, dict):
        raise HTTPException(status_code=400, detail="Each Magic Button adjustment must be a JSON object.")
    scope = str(adjustment.get("scope") or "overall").strip().lower()
    metric_key = str(adjustment.get("metric_key") or "map50_95").strip().lower()
    class_name = str(adjustment.get("class_name") or "").strip()
    try:
        target = float(adjustment.get("target"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Every Magic Button target must be a number between 0 and 1.") from None
    if not math.isfinite(target) or target < 0 or target > 1:
        raise HTTPException(status_code=400, detail="Every Magic Button target must be a number between 0 and 1.")
    if scope == "overall" and metric_key not in MAGIC_OVERALL_METRICS:
        raise HTTPException(status_code=400, detail="Choose a supported overall metric to adjust.")
    if scope == "per_class" and metric_key not in MAGIC_PER_CLASS_METRICS:
        raise HTTPException(status_code=400, detail="Choose a supported per-class metric to adjust.")
    if scope not in {"overall", "per_class"}:
        raise HTTPException(status_code=400, detail="Magic Button scope must be overall or per_class.")
    if scope == "per_class" and not class_name:
        raise HTTPException(status_code=400, detail="Choose a class for every per-class adjustment.")
    return {
        "scope": scope,
        "metric_key": metric_key,
        "target": format_metric(target),
        "class_name": class_name if scope == "per_class" else "",
    }


def validate_magic_adjustments(adjustments: list[dict]):
    if not adjustments:
        raise HTTPException(status_code=400, detail="Add at least one score before applying Magic Button adjustments.")
    if len(adjustments) > 50:
        raise HTTPException(status_code=400, detail="Magic Button supports at most 50 adjustments at once.")

    identities = set()
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    for adjustment in adjustments:
        identity = (adjustment["scope"], adjustment["class_name"], adjustment["metric_key"])
        if identity in identities:
            raise HTTPException(status_code=400, detail=f"Duplicate Magic Button adjustment: {magic_adjustment_label(adjustment['scope'], adjustment['metric_key'], adjustment['class_name'])}.")
        identities.add(identity)
        group_key = (adjustment["scope"], adjustment["class_name"])
        grouped.setdefault(group_key, {})[adjustment["metric_key"]] = adjustment["target"]

    for (scope, class_name), targets in grouped.items():
        if "map50" in targets and "map50_95" in targets and targets["map50"] < targets["map50_95"]:
            label = class_name or "Overall"
            raise HTTPException(status_code=409, detail=f"{label} AP50 cannot be lower than AP50-95.")
        if scope == "per_class" and "f1" in targets and ({"precision", "recall"} & targets.keys()):
            raise HTTPException(
                status_code=409,
                detail=f"Choose either F1 or Precision/Recall for {class_name}; F1 is derived automatically when Precision or Recall is adjusted.",
            )

    overall = grouped.get(("overall", ""), {})
    if ({"macro_f1", "weighted_f1"} & overall.keys()) and ({"precision", "recall"} & overall.keys()):
        raise HTTPException(
            status_code=409,
            detail="Overall F1 targets cannot be combined with overall Precision or Recall targets because they control the same values.",
        )
    if "macro_f1" in overall and "weighted_f1" in overall:
        raise HTTPException(status_code=409, detail="Choose either Macro F1 or Weighted F1 in one adjustment set.")


def magic_locked_metrics(adjustments: list[dict]) -> set[tuple[str, str]]:
    locked = set()
    for adjustment in adjustments:
        if adjustment["scope"] != "per_class":
            continue
        class_name = adjustment["class_name"]
        metric_key = adjustment["metric_key"]
        locked.add((class_name, metric_key))
        if metric_key == "f1":
            locked.update({(class_name, "precision"), (class_name, "recall")})
        if metric_key in {"precision", "recall"}:
            locked.add((class_name, "f1"))
    return locked


def verify_magic_adjustment_targets(summary: dict, classes: list[dict], adjustments: list[dict]):
    class_rows = {str(row.get("class_name") or ""): row for row in classes}
    for adjustment in adjustments:
        source = summary if adjustment["scope"] == "overall" else class_rows.get(adjustment["class_name"])
        if source is None:
            raise HTTPException(status_code=404, detail=f"Selected class {adjustment['class_name']} was not found in the run metrics.")
        actual = float_value(source, adjustment["metric_key"])
        if actual is None or abs(actual - adjustment["target"]) > 0.0002:
            label = magic_adjustment_label(adjustment["scope"], adjustment["metric_key"], adjustment["class_name"])
            raise HTTPException(
                status_code=409,
                detail=f"The selected combination cannot preserve the requested {label} target. Remove a related target and try again.",
            )


def build_magic_metrics_overlay(
    metrics: dict,
    scope,
    metric_key: Optional[str] = None,
    target_value: Optional[float] = None,
    class_name: str = "",
) -> dict:
    if isinstance(scope, list):
        adjustments = [normalize_magic_adjustment(item) for item in scope]
    else:
        adjustments = [normalize_magic_adjustment({
            "scope": scope,
            "metric_key": metric_key,
            "target": target_value,
            "class_name": class_name,
        })]
    validate_magic_adjustments(adjustments)

    rng = random.Random(0)
    working = dict(metrics)
    working["per_class"] = base_magic_classes(metrics, rng)
    summary_keys = ("precision", "recall", "map50", "map50_95", "macro_f1", "weighted_f1")

    per_class_adjustments = [item for item in adjustments if item["scope"] == "per_class"]
    per_class_order = {"map50_95": 0, "map50": 1, "precision": 2, "recall": 3, "f1": 4}
    for adjustment in sorted(per_class_adjustments, key=lambda item: per_class_order[item["metric_key"]]):
        summary, adjusted_classes = adjust_per_class_magic_metric(
            working,
            adjustment["metric_key"],
            adjustment["target"],
            adjustment["class_name"],
            rng,
        )
        working.update(summary)
        working["per_class"] = adjusted_classes

    locked_metrics = magic_locked_metrics(adjustments)
    overall_adjustments = [item for item in adjustments if item["scope"] == "overall"]
    overall_order = {"map50_95": 0, "map50": 1, "precision": 2, "recall": 3, "macro_f1": 4, "weighted_f1": 5}
    for adjustment in sorted(overall_adjustments, key=lambda item: overall_order[item["metric_key"]]):
        summary, adjusted_classes = adjust_overall_magic_metric(
            working,
            adjustment["metric_key"],
            adjustment["target"],
            rng,
            locked_metrics,
        )
        working.update(summary)
        working["per_class"] = adjusted_classes

    summary = {key: working.get(key) for key in summary_keys}
    adjusted_classes = working.get("per_class") or []
    verify_magic_adjustment_targets(summary, adjusted_classes, adjustments)

    labels = [magic_adjustment_label(item["scope"], item["metric_key"], item["class_name"]) for item in adjustments]
    first = adjustments[0]
    note = (
        "Magic Button adjusted metrics are active. These values are generated from the original run metrics "
        "for report preview and are not raw validation results."
    )

    return {
        "created_at": datetime.now(MYT).isoformat(),
        "adjustments": adjustments,
        "target": first["target"],
        "scope": first["scope"],
        "metric_key": first["metric_key"],
        "class_name": first["class_name"],
        "original": {
            **{key: metrics.get(key) for key in summary_keys},
            "per_class": [dict(row) for row in metrics.get("per_class") or []],
        },
        "metrics": {
            "precision": summary.get("precision"),
            "recall": summary.get("recall"),
            "map50": summary.get("map50"),
            "map50_95": summary.get("map50_95"),
            "macro_f1": summary.get("macro_f1"),
            "weighted_f1": summary.get("weighted_f1"),
            "per_class": adjusted_classes,
            "magic_adjusted": True,
            "magic_adjustments": adjustments,
            "magic_target": first["target"],
            "magic_metric_key": first["metric_key"],
            "magic_scope": first["scope"],
            "magic_class_name": first["class_name"],
            "magic_adjustment_label": ", ".join(labels),
            "overall_metric_source": "magic_button",
            "per_class_source": "magic_button",
            "note": note,
        },
    }


def parse_magic_metrics_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Magic Button request body must be a JSON object.")
    raw_adjustments = payload.get("adjustments")
    if raw_adjustments is None:
        raw_adjustments = [{
            "scope": payload.get("scope") or "overall",
            "metric_key": payload.get("metric_key") or "map50_95",
            "target": payload.get("target", payload.get("map50_95")),
            "class_name": payload.get("class_name") or "",
        }]
    if not isinstance(raw_adjustments, list):
        raise HTTPException(status_code=400, detail="Magic Button adjustments must be a JSON array.")
    adjustments = [normalize_magic_adjustment(item) for item in raw_adjustments]
    validate_magic_adjustments(adjustments)
    return {
        "project": str(payload.get("project") or "runs/detect"),
        "name": str(payload.get("name") or "train"),
        "adjustments": adjustments,
    }


def apply_magic_metrics_overlay(run_dir: Path, metrics: dict) -> dict:
    overlay = read_json_object(run_dir / MAGIC_METRICS_FILE)
    adjusted = overlay.get("metrics") if isinstance(overlay.get("metrics"), dict) else None
    if not adjusted:
        return metrics
    merged = dict(metrics)
    merged.update(adjusted)
    primary_type = str(merged.get("primary_metric_type") or merged.get("metric_type") or "").lower()
    if primary_type in {"box", "mask"} and isinstance(merged.get("per_class"), list):
        merged[f"per_class_{primary_type}"] = merged["per_class"]
        families = dict(merged.get("metric_families") or {})
        primary_family = dict(families.get(primary_type) or {})
        primary_family.update({
            "per_class": merged["per_class"],
            "macro_f1": merged.get("macro_f1"),
            "weighted_f1": merged.get("weighted_f1"),
        })
        families[primary_type] = primary_family
        merged["metric_families"] = families
        primary_overall = {
            key: merged.get(key)
            for key in ("precision", "recall", "map50", "map50_95")
        }
        merged["overall"] = primary_overall
        overall_by_type = dict(merged.get("overall_by_type") or {})
        overall_by_type[primary_type] = primary_overall
        merged["overall_by_type"] = overall_by_type
    merged["magic_adjusted"] = True
    merged["magic_created_at"] = overlay.get("created_at")
    merged["magic_original"] = overlay.get("original") if isinstance(overlay.get("original"), dict) else {}
    merged["magic_adjustments"] = overlay.get("adjustments") if isinstance(overlay.get("adjustments"), list) else merged.get("magic_adjustments", [])
    return merged


configure_metrics(
    MetricsDependencies(
        repo_root=REPO_ROOT,
        log_dir=LOG_DIR,
        log_file=LOG_FILE,
        magic_metrics_filename=MAGIC_METRICS_FILE,
        training_context_filename=TRAINING_REPORT_CONTEXT_FILE,
        rfdetr_defaults=RFDETR_DEFAULTS,
        run_metrics_cache=run_metrics_cache,
        clean_log_line=lambda value: clean_log_line(value),
        current_status=lambda: current_status(),
        family_for_runs_root=family_for_runs_root,
        load_training_report_context=lambda path: load_training_report_context(path),
        magic_overlay=apply_magic_metrics_overlay,
        read_json_object=lambda path: read_json_object(path),
        read_log_file=lambda path: read_log_file(path),
        training_log_file=lambda: training_log_file,
    )
)




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


from .services.annotation_qa import (
    AnnotationQaDependencies,
    configure_annotation_qa,
    annotation_qa_job_id,
    normalize_annotation_qa_model_name,
    annotation_qa_version_tuple,
    annotation_qa_model_status,
    annotation_qa_model_config,
    ensure_annotation_qa_path,
    annotation_qa_payload,
    update_annotation_qa_job,
    annotation_qa_thresholds,
    annotation_qa_split_names,
    yolo_label_path,
    pixel_bbox_from_yolo,
    yolo_bbox_from_pixels,
    bbox_area,
    bbox_iou,
    bbox_center_shift,
    bbox_edge_differences,
    annotation_qa_difference_band,
    annotation_qa_prompt_box,
    annotation_qa_prompt_plan,
    annotation_qa_mask_iou,
    annotation_qa_prompt_stability,
    annotation_qa_select_candidate,
    annotation_qa_max_neighbor_iou,
    annotation_qa_auto_gate,
    annotation_qa_audit_required,
    issue_is_safe_sam_replacement,
    mask_bbox,
    mask_to_uint8,
    annotation_issue,
    draw_annotation_qa_preview,
    annotation_qa_summary,
    write_annotation_qa_report,
    annotation_qa_run_dir,
    load_annotation_qa_report,
    set_annotation_qa_issue_fix,
    issue_label_path_in_copy,
    apply_annotation_qa_fix,
    apply_annotation_qa_fixes,
    annotation_qa_roboflow_context,
    build_annotation_qa_roboflow_preview,
    publish_annotation_qa_to_roboflow,
    load_sam_model,
    sam_masks_for_image,
    annotation_qa_candidates_for_image,
    run_annotation_qa_job,
    annotation_qa_job_from_disk,
)

configure_annotation_qa(
    AnnotationQaDependencies(
        annotation_qa_root=ANNOTATION_QA_ROOT,
        data_root=DATA_ROOT,
        dataset_prepared_root=DATASET_PREPARED_ROOT,
        timezone=MYT,
        model_default=ANNOTATION_QA_MODEL_DEFAULT,
        report_version=ANNOTATION_QA_REPORT_VERSION,
        safe_mapping_version=ANNOTATION_QA_SAFE_MAPPING_VERSION,
        sam3_model_path=SAM3_QA_MODEL_PATH,
        sam3_min_ultralytics_version=SAM3_MIN_ULTRALYTICS_VERSION,
        model_registry=SAM_QA_MODEL_REGISTRY,
        summary_filename=ANNOTATION_QA_SUMMARY_FILE,
        report_json_filename=ANNOTATION_QA_REPORT_JSON_FILE,
        report_csv_filename=ANNOTATION_QA_REPORT_CSV_FILE,
        review_filename=ANNOTATION_QA_REVIEW_FILE,
        fix_summary_filename=ANNOTATION_QA_FIX_SUMMARY_FILE,
        jobs=annotation_qa_jobs,
        jobs_lock=annotation_qa_jobs_lock,
        clean_name=clean_name,
        dataset_response=dataset_response,
        image_files=image_files,
        label_folder_for_images=label_folder_for_images,
        normalize_yaml_names=normalize_yaml_names,
        persist_job=persist_job,
        prepared_dataset_yaml=prepared_dataset_yaml,
        read_json_object=read_json_object,
        read_yaml_class_names=read_yaml_class_names,
        split_image_folder=split_image_folder,
        write_json_object=write_json_object,
    )
)


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


def parse_progress_fields(text: str | None) -> dict[str, str]:
    fields = {}
    for item in str(text or "").split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key:
            fields[key] = value.strip()
    return fields


def progress_number(value):
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def progress_int(value):
    number = progress_number(value)
    return int(number) if number is not None else None


def dfine_progress_detail(
    current_epoch: int,
    total_epochs: int,
    step: int | None,
    total_steps: int | None,
    fields: dict[str, str],
) -> str:
    parts = [f"D-FINE epoch {current_epoch}/{total_epochs}"]
    if step is not None and total_steps:
        parts.append(f"step {step}/{total_steps}")
    loss = fields.get("loss")
    loss_avg = fields.get("loss_avg")
    if loss:
        loss_text = f"loss {loss}"
        if loss_avg:
            loss_text += f" (avg {loss_avg})"
        parts.append(loss_text)
    eta = fields.get("eta")
    if eta:
        parts.append(f"ETA {eta}")
    return " · ".join(parts) + "."


def update_training_progress(
    current_epoch: int,
    total_epochs: int,
    fields: dict[str, str] | None = None,
    *,
    detail: str = "",
):
    if training_run_info is None or current_epoch <= 0 or total_epochs <= 0:
        return

    fields = fields or {}
    current_epoch = min(current_epoch, total_epochs)
    training_run_info["current_epoch"] = current_epoch
    training_run_info["total_epochs"] = total_epochs

    step = progress_int(fields.get("step"))
    total_steps = progress_int(fields.get("steps"))
    if step is not None and total_steps and total_steps > 0:
        step = max(0, min(step, total_steps))
        training_run_info["current_step"] = step
        training_run_info["total_steps"] = total_steps
        epoch_fraction = step / total_steps
        progress_percent = ((current_epoch - 1 + epoch_fraction) / total_epochs) * 100
        training_run_info["progress_percent"] = round(max(0, min(100, progress_percent)), 1)
        training_run_info["progress_detail"] = detail or dfine_progress_detail(
            current_epoch,
            total_epochs,
            step,
            total_steps,
            fields,
        )
    else:
        training_run_info.pop("current_step", None)
        training_run_info.pop("total_steps", None)
        training_run_info.pop("progress_percent", None)
        training_run_info["progress_detail"] = detail


def capture_epoch_progress(line: str) -> bool:
    global training_run_info
    match = WEB_PROGRESS_RE.match(line)
    if match is not None:
        current_epoch = int(match.group(1))
        total_epochs = int(match.group(2))
        update_training_progress(current_epoch, total_epochs, parse_progress_fields(match.group("fields")))
        return True

    match = DFINE_PROGRESS_LOG_RE.match(line)
    if match is not None:
        raw_epoch = int(match.group("epoch"))
        total_epochs = int(match.group("total"))
        fields = {
            "step": match.group("step"),
            "steps": match.group("steps"),
            "eta": match.group("eta"),
            "lr": match.group("lr"),
            "loss": match.group("loss"),
            "loss_avg": match.group("loss_avg"),
        }
        update_training_progress(raw_epoch + 1, total_epochs, fields)
        return True

    match = RFDETR_VALIDATION_PROGRESS_RE.search(line)
    if match is None:
        return False

    current_epoch = int(match.group(1))
    total_epochs = int(match.group(2))
    update_training_progress(
        current_epoch,
        total_epochs,
        detail="RF-DETR validation metrics were updated from the latest validation block.",
    )
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
        info = dict(training_run_info or {})
        job_id = f"training:{info.get('name') or process.pid}"
        returncode = process.wait()
        persist_job(
            job_id,
            "training",
            "complete" if returncode == 0 else "failed",
            {"returncode": returncode, "run": {key: value for key, value in info.items() if key != "report_context"}},
            force=True,
        )


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
        info = dict(test_run_info or {})
        job_id = f"testing:{info.get('name') or process.pid}"
        returncode = process.wait()
        persist_job(
            job_id,
            "testing",
            "complete" if returncode == 0 else "failed",
            {"returncode": returncode, "run": {key: value for key, value in info.items() if key != "report_context"}},
            force=True,
        )


def read_log_tail(max_chars: int = 20000) -> str:
    return read_text_tail(LOG_FILE, max_chars=max_chars)


def read_log_tail_from(path: Path, max_chars: int = 20000) -> str:
    return read_text_tail(path, max_chars=max_chars)


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
    step = max(0, int(run_info.get("current_step") or 0))
    total_steps = max(0, int(run_info.get("total_steps") or 0))
    detail = str(run_info.get("progress_detail") or "").strip()
    progress_percent = progress_number(run_info.get("progress_percent"))

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
        if running and progress_percent is not None:
            percent = round(max(0, min(100, progress_percent)), 1)
        else:
            percent = round((current / total) * 100, 1)
    else:
        percent = 0.0

    return {
        "current": current,
        "completed": completed,
        "total": total,
        "step": step,
        "steps": total_steps,
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
        "annotation_qa_models": [
            annotation_qa_model_status(model_name)
            for model_name in SAM_QA_MODEL_REGISTRY
        ],
        "models": model_registry_payload(),
        "default_device": os.getenv("TRAINING_DEVICE", ""),
    }


@app.get("/api/jobs")
def jobs(limit: int = 100):
    return {
        "jobs": job_store.list(limit=max(1, min(int(limit), 500))),
        "active_resources": resource_coordinator.snapshot(),
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
    cached = training_sessions_cache.get()
    if cached is not None:
        return cached
    return training_sessions_cache.put({"sessions": available_training_sessions()})


from .routers.inference import InferenceRouterDependencies, create_inference_router

app.include_router(
    create_inference_router(
        InferenceRouterDependencies(
            job_root=INFERENCE_JOB_ROOT,
            upload_root=INFERENCE_UPLOAD_ROOT,
            weight_extensions=INFERENCE_WEIGHT_EXTENSIONS,
            timezone=MYT,
            upload_limits=UPLOAD_LIMITS,
            jobs=inference_jobs,
            jobs_lock=inference_jobs_lock,
            peers=inference_peers,
            peers_lock=inference_peers_lock,
            compute_start_lock=compute_start_lock,
            ensure_compute_available=ensure_compute_available,
            ensure_dirs=ensure_dirs,
            ensure_inference_path=ensure_inference_path,
            family_for_weight_path=family_for_weight_path,
            inference_job_from_disk=inference_job_from_disk,
            inference_job_payload=inference_job_payload,
            inference_media_type=inference_media_type,
            inference_mjpeg_stream=inference_mjpeg_stream,
            latest_inference_webrtc_frame=latest_inference_webrtc_frame,
            persist_job=persist_job,
            relative_to_repo=relative_to_repo,
            resolve_inference_weight_path=resolve_inference_weight_path,
            run_inference_job=run_inference_job,
            safe_upload_path=safe_upload_path,
            stop_inference_process=stop_inference_process,
            upload_validation_http_error=upload_validation_http_error,
        )
    )
)


from .routers.annotation_qa import AnnotationQaRouterDependencies, create_annotation_qa_router

app.include_router(
    create_annotation_qa_router(
        AnnotationQaRouterDependencies(
            annotation_qa_root=ANNOTATION_QA_ROOT,
            report_json_filename=ANNOTATION_QA_REPORT_JSON_FILE,
            report_csv_filename=ANNOTATION_QA_REPORT_CSV_FILE,
            summary_filename=ANNOTATION_QA_SUMMARY_FILE,
            jobs=annotation_qa_jobs,
            jobs_lock=annotation_qa_jobs_lock,
            compute_start_lock=compute_start_lock,
            ensure_compute_available=ensure_compute_available,
            persist_job=persist_job,
            prepared_dataset_yaml=prepared_dataset_yaml,
            read_json_object=read_json_object,
            training_active=lambda: training_process is not None and training_process.poll() is None,
            test_active=lambda: test_process is not None and test_process.poll() is None,
        )
    )
)


from .routers.datasets import DatasetRouterDependencies, create_dataset_router

app.include_router(
    create_dataset_router(
        DatasetRouterDependencies(
            data_root=DATA_ROOT,
            summary_filename=DATASET_SUMMARY_FILE,
            upload_limits=UPLOAD_LIMITS,
            preparation_jobs=dataset_preparation_jobs,
            preparation_lock=dataset_preparation_lock,
            downloads=dataset_downloads,
            downloads_lock=dataset_download_lock,
            resource_coordinator=resource_coordinator,
            acquire_resource=acquire_resource,
            clean_name=clean_name,
            dataset_progress_callback=dataset_progress_callback,
            safe_upload_path=safe_upload_path,
            update_dataset_preparation=update_dataset_preparation,
            upload_validation_http_error=upload_validation_http_error,
            write_json_object=write_json_object,
        )
    )
)


from .routers.testing import TestingRouterDependencies, create_testing_router

def _get_test_runtime_state():
    return test_process, test_started_at, test_log_file, test_run_info


def _set_test_runtime_state(process, started_at, log_file, run_info):
    global test_process, test_started_at, test_log_file, test_run_info
    test_process = process
    test_started_at = started_at
    test_log_file = log_file
    test_run_info = run_info


app.include_router(
    create_testing_router(
        TestingRouterDependencies(
            data_root=DATA_ROOT,
            log_dir=LOG_DIR,
            repo_root=REPO_ROOT,
            test_script=TEST_SCRIPT,
            rfdetr_test_script=RFDETR_TEST_SCRIPT,
            test_log_file=TEST_LOG_FILE,
            test_context_filename=TEST_REPORT_CONTEXT_FILE,
            training_python=TRAINING_PYTHON,
            timezone=MYT,
            upload_limits=UPLOAD_LIMITS,
            compute_start_lock=compute_start_lock,
            get_state=_get_test_runtime_state,
            set_state=_set_test_runtime_state,
            callbacks={
                "cached_dataset_summary": cached_dataset_summary,
                "combined_report_download_filename": combined_report_download_filename,
                "current_test_run_dir": current_test_run_dir,
                "current_test_status": current_test_status,
                "ensure_compute_available": ensure_compute_available,
                "ensure_dirs": ensure_dirs,
                "ensure_model_report_artifacts_for_report": ensure_model_report_artifacts_for_report,
                "ensure_runs_path": ensure_runs_path,
                "family_for_weight_path": family_for_weight_path,
                "gpu_status": gpu_status,
                "latest_timestamped_test_log": latest_timestamped_test_log,
                "load_training_report_context": load_training_report_context,
                "normalize_training_project_path": normalize_training_project_path,
                "persist_job": persist_job,
                "prepare_custom_test_dataset": prepare_custom_test_dataset,
                "prepared_dataset_yaml": prepared_dataset_yaml,
                "read_json_object": read_json_object,
                "read_log_tail_from": read_log_tail_from,
                "read_run_metrics": read_run_metrics,
                "read_test_metrics": read_test_metrics,
                "read_web_metrics": read_web_metrics,
                "resolve_prepared_test_dataset_yaml": resolve_prepared_test_dataset_yaml,
                "resolve_test_artifact_path": resolve_test_artifact_path,
                "resolve_weight_path": resolve_weight_path,
                "safe_upload_path": safe_upload_path,
                "stream_test_logs": stream_test_logs,
                "upload_validation_http_error": upload_validation_http_error,
            },
        )
    )
)


from .routers.training import TrainingRouterDependencies, create_training_router

def _get_training_runtime_state():
    return training_process, training_started_at, training_log_file, training_run_info


def _set_training_runtime_state(process, started_at, log_file, run_info):
    global training_process, training_started_at, training_log_file, training_run_info
    training_process = process
    training_started_at = started_at
    training_log_file = log_file
    training_run_info = run_info


app.include_router(
    create_training_router(
        TrainingRouterDependencies(
            repo_root=REPO_ROOT,
            log_dir=LOG_DIR,
            log_file=LOG_FILE,
            training_python=TRAINING_PYTHON,
            timezone=MYT,
            model_registry=MODEL_REGISTRY,
            training_context_filename=TRAINING_REPORT_CONTEXT_FILE,
            compute_start_lock=compute_start_lock,
            training_sessions_cache=training_sessions_cache,
            run_metrics_cache=run_metrics_cache,
            train_request_type=TrainRequest,
            weight_request_type=WeightRequest,
            artifact_request_type=ArtifactRequest,
            get_state=_get_training_runtime_state,
            set_state=_set_training_runtime_state,
            callbacks={
                "backend_training_value": backend_training_value,
                "cached_dataset_summary": cached_dataset_summary,
                "cached_run_metrics": cached_run_metrics,
                "current_status": current_status,
                "default_project_for_model_size": default_project_for_model_size,
                "ensure_compute_available": ensure_compute_available,
                "ensure_dirs": ensure_dirs,
                "ensure_model_report_artifacts_for_report": ensure_model_report_artifacts_for_report,
                "ensure_runs_path": ensure_runs_path,
                "gpu_status": gpu_status,
                "installed_version": installed_version,
                "is_known_training_project_default": is_known_training_project_default,
                "latest_timestamped_log": latest_timestamped_log,
                "load_training_report_context": load_training_report_context,
                "normalize_training_project_path": normalize_training_project_path,
                "persist_job": persist_job,
                "persist_training_report_context": persist_training_report_context,
                "read_error_log": read_error_log,
                "read_json_object": read_json_object,
                "read_log_file": read_log_file,
                "read_log_tail": read_log_tail,
                "read_run_metrics": read_run_metrics,
                "resolve_artifact_path": resolve_artifact_path,
                "resolve_run_dir": resolve_run_dir,
                "resolve_run_dir_details": resolve_run_dir_details,
                "resolve_weight_path": resolve_weight_path,
                "stream_training_logs": stream_training_logs,
                "timestamped_training_run_name": timestamped_training_run_name,
                "training_report_download_filename": training_report_download_filename,
            },
        )
    )
)


@app.post("/api/train/metrics/magic")
def magic_train_metrics(payload: Optional[dict] = Body(default=None)):
    request = parse_magic_metrics_payload(payload or {})
    if current_status()["running"]:
        raise HTTPException(status_code=409, detail="Wait for training to finish before adjusting metrics.")
    run_dir, resolution_type = resolve_run_dir_details(request["project"], request["name"])
    metrics = read_run_metrics(run_dir, include_magic=False)
    if not metrics.get("available"):
        raise HTTPException(status_code=409, detail="The selected training run has no completed metrics to adjust.")
    overlay = build_magic_metrics_overlay(
        metrics,
        request["adjustments"],
    )
    write_json_object(run_dir / MAGIC_METRICS_FILE, overlay)
    result = apply_magic_metrics_overlay(run_dir, metrics)
    result["resolution_type"] = resolution_type
    return result


@app.post("/api/train/metrics/magic/reset")
def reset_magic_train_metrics(request: WeightRequest):
    if current_status()["running"]:
        raise HTTPException(status_code=409, detail="Wait for training to finish before resetting adjusted metrics.")
    run_dir, resolution_type = resolve_run_dir_details(request.project, request.name)
    try:
        (run_dir / MAGIC_METRICS_FILE).unlink(missing_ok=True)
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Could not remove the Magic Button adjustment file.") from exc
    result = read_run_metrics(run_dir, include_magic=False)
    result["resolution_type"] = resolution_type
    return result
