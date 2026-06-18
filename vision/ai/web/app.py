#!/usr/bin/env python3
"""FastAPI web UI backend for YOLOv8 training."""

import json
import os
import random
import shutil
import signal
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


WEB_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEB_DIR.parents[2]
TRAIN_SCRIPT = REPO_ROOT / "vision" / "ai" / "train" / "train_yolov8.py"
STATIC_DIR = WEB_DIR / "static"
LOG_DIR = WEB_DIR / "logs"
LOG_FILE = LOG_DIR / "current.log"

load_dotenv(WEB_DIR / ".env")

DATA_ROOT = Path(os.getenv("WEB_DATA_ROOT") or WEB_DIR / "datasets").expanduser()
TRAINING_PYTHON = os.getenv("TRAINING_PYTHON", "python3")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val", "test": "test"}
MODEL_MAP = {
    "nano": "yolov8n.pt",
    "small": "yolov8s.pt",
    "medium": "yolov8m.pt",
    "large": "yolov8l.pt",
    "xlarge": "yolov8x.pt",
}

app = FastAPI(title="YOLOv8 Training UI")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

training_process: Optional[subprocess.Popen] = None
training_started_at: Optional[float] = None


class SplitConfig(BaseModel):
    train: int = Field(default=70, ge=1, le=100)
    val: int = Field(default=20, ge=0, le=100)
    test: int = Field(default=10, ge=0, le=100)


class LocalDatasetRequest(BaseModel):
    path: str
    classes: list[str]
    split: SplitConfig = Field(default_factory=SplitConfig)
    name: str = "local_dataset"
    force_split: bool = False


class RoboflowRequest(BaseModel):
    api_key: Optional[str] = None
    workspace: Optional[str] = None
    project: Optional[str] = None
    version: Optional[str] = None
    format: Optional[str] = None
    classes: list[str] = Field(default_factory=list)
    split: SplitConfig = Field(default_factory=SplitConfig)
    name: str = "roboflow_dataset"


class TrainRequest(BaseModel):
    dataset_yaml: str
    model_size: str = "nano"
    custom_model: Optional[str] = None
    epochs: int = Field(default=100, ge=1)
    imgsz: int = Field(default=640, ge=32)
    batch: int = 16
    patience: int = Field(default=50, ge=0)
    save_period: int = -1
    device: Optional[str] = None
    project: str = "runs/detect"
    name: str = "train"
    resume: bool = False


def ensure_dirs():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def clean_name(value: str, fallback: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def validate_classes(classes: list[str]) -> dict[int, str]:
    names = [item.strip() for item in classes if item.strip()]
    if not names:
        raise HTTPException(status_code=400, detail="At least one class name is required.")
    return {index: name for index, name in enumerate(names)}


def validate_split(split: SplitConfig) -> tuple[float, float, float]:
    total = split.train + split.val + split.test
    if total <= 0:
        raise HTTPException(status_code=400, detail="Split values must be greater than zero.")
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


def prepare_split_dataset(root: Path, name: str, names: dict[int, str], split: SplitConfig) -> Path:
    source_images = collect_source_images(root)
    if not source_images:
        raise HTTPException(status_code=400, detail="No images found in the dataset path.")

    train_ratio, val_ratio, _ = validate_split(split)
    random.Random(42).shuffle(source_images)

    total = len(source_images)
    train_count = max(1, int(total * train_ratio))
    val_count = int(total * val_ratio)
    if train_count + val_count > total:
        val_count = max(0, total - train_count)

    groups = {
        "train": source_images[:train_count],
        "val": source_images[train_count:train_count + val_count],
        "test": source_images[train_count + val_count:],
    }

    output_root = DATA_ROOT / "prepared" / clean_name(name, "dataset")
    if output_root.exists():
        shutil.rmtree(output_root)

    for split_name, pairs in groups.items():
        for image_path, label_dir in pairs:
            copy_pair(
                image_path,
                label_dir,
                output_root / "images" / split_name,
                output_root / "labels" / split_name,
            )

    yaml_path = write_dataset_yaml(
        output_root,
        output_root,
        names,
        {"train": "images/train", "val": "images/val", "test": "images/test"},
    )
    return yaml_path


def prepare_dataset(source: Path, name: str, classes: list[str], split: SplitConfig, force_split: bool) -> Path:
    if source.is_file() and source.suffix.lower() in {".yaml", ".yml"}:
        return source

    root = find_dataset_root(source)
    names = validate_classes(classes)

    if not force_split and "train" in split_dirs(root) and "val" in split_dirs(root):
        return prepare_existing_split(root, name, names)

    return prepare_split_dataset(root, name, names, split)


def read_log_tail(max_chars: int = 20000) -> str:
    if not LOG_FILE.is_file():
        return ""
    data = LOG_FILE.read_text(encoding="utf-8", errors="replace")
    return data[-max_chars:]


def current_status() -> dict:
    global training_process

    if training_process is None:
        return {"running": False, "returncode": None, "started_at": training_started_at}

    returncode = training_process.poll()
    if returncode is not None:
        training_process = None
        return {"running": False, "returncode": returncode, "started_at": training_started_at}

    return {"running": True, "returncode": None, "started_at": training_started_at}


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
            "format": os.getenv("ROBOFLOW_FORMAT", "yolov8"),
        },
        "default_device": os.getenv("TRAINING_DEVICE", ""),
    }


@app.post("/api/dataset/local")
def local_dataset(request: LocalDatasetRequest):
    source = Path(request.path).expanduser()
    if not source.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {source}")

    yaml_path = prepare_dataset(
        source=source,
        name=request.name,
        classes=request.classes,
        split=request.split,
        force_split=request.force_split,
    )
    return {"dataset_yaml": str(yaml_path), "message": "Local dataset is ready."}


@app.post("/api/dataset/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    classes: str = Form(...),
    train: int = Form(70),
    val: int = Form(20),
    test: int = Form(10),
    name: str = Form("uploaded_dataset"),
    force_split: bool = Form(True),
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

    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    zip_path = upload_dir / file.filename
    with zip_path.open("wb") as output:
        shutil.copyfileobj(file.file, output)

    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP.") from exc

    yaml_path = prepare_dataset(
        source=extract_dir,
        name=clean,
        classes=class_names,
        split=split,
        force_split=force_split,
    )
    return {"dataset_yaml": str(yaml_path), "message": "Uploaded dataset is ready."}


@app.post("/api/dataset/roboflow")
def roboflow_dataset(request: RoboflowRequest):
    api_key = request.api_key or os.getenv("ROBOFLOW_API_KEY")
    workspace = request.workspace or os.getenv("ROBOFLOW_WORKSPACE")
    project_name = request.project or os.getenv("ROBOFLOW_PROJECT")
    version = request.version or os.getenv("ROBOFLOW_VERSION")
    dataset_format = request.format or os.getenv("ROBOFLOW_FORMAT", "yolov8")

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

    clean = clean_name(request.name, "roboflow_dataset")
    download_dir = DATA_ROOT / "roboflow" / clean
    if download_dir.exists():
        shutil.rmtree(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)
        version_obj = project.version(int(version))
        try:
            dataset = version_obj.download(dataset_format, location=str(download_dir), overwrite=True)
        except TypeError:
            dataset = version_obj.download(dataset_format, location=str(download_dir))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Roboflow download failed: {exc}") from exc

    dataset_root = Path(getattr(dataset, "location", download_dir))
    yaml_path = find_dataset_yaml(dataset_root)
    if yaml_path:
        return {"dataset_yaml": str(yaml_path), "message": "Roboflow dataset is ready."}

    if not request.classes:
        raise HTTPException(
            status_code=400,
            detail="Roboflow download did not include data.yaml. Please provide class names.",
        )

    yaml_path = prepare_dataset(
        source=dataset_root,
        name=clean,
        classes=request.classes,
        split=request.split,
        force_split=False,
    )
    return {"dataset_yaml": str(yaml_path), "message": "Roboflow dataset is ready."}


@app.post("/api/train/start")
def start_training(request: TrainRequest):
    global training_process, training_started_at

    status = current_status()
    if status["running"]:
        raise HTTPException(status_code=409, detail="Training is already running.")

    dataset_yaml = Path(request.dataset_yaml).expanduser()
    if not dataset_yaml.is_file():
        raise HTTPException(status_code=400, detail=f"Dataset YAML not found: {dataset_yaml}")

    model = request.custom_model or MODEL_MAP.get(request.model_size)
    if not model:
        raise HTTPException(status_code=400, detail=f"Unknown model size: {request.model_size}")

    ensure_dirs()
    LOG_FILE.write_text("", encoding="utf-8")

    cmd = [
        TRAINING_PYTHON,
        str(TRAIN_SCRIPT),
        "--data", str(dataset_yaml),
        "--model", model,
        "--epochs", str(request.epochs),
        "--imgsz", str(request.imgsz),
        "--batch", str(request.batch),
        "--patience", str(request.patience),
        "--save-period", str(request.save_period),
        "--project", request.project,
        "--name", request.name,
    ]

    device = request.device or os.getenv("TRAINING_DEVICE")
    if device:
        cmd.extend(["--device", device])
    if request.resume:
        cmd.append("--resume")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_handle = LOG_FILE.open("a", encoding="utf-8")
    training_process = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
        text=True,
    )
    log_handle.close()
    training_started_at = time.time()

    return {
        "message": "Training started.",
        "pid": training_process.pid,
        "command": cmd,
        "log_file": str(LOG_FILE),
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
    return status


@app.get("/api/train/logs", response_class=PlainTextResponse)
def train_logs():
    return read_log_tail()
