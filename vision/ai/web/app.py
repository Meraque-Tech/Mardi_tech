#!/usr/bin/env python3
"""FastAPI web UI backend for YOLOv8 training."""

import csv
import json
import os
import random
import re
import shutil
import signal
import subprocess
import threading
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
RUNS_ROOT = REPO_ROOT / "runs"

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
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")
PROGRESS_LINE_RE = re.compile(r":\s*\d+%\s+.*\b\d+/\d+\b")

app = FastAPI(title="YOLOv8 Training UI")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

training_process: Optional[subprocess.Popen] = None
training_started_at: Optional[float] = None
training_log_file: Optional[Path] = None


class SplitConfig(BaseModel):
    train: int = Field(default=70, ge=1, le=100)
    val: int = Field(default=20, ge=0, le=100)
    test: int = Field(default=10, ge=0, le=100)


class LocalDatasetRequest(BaseModel):
    path: str
    classes: list[str] = Field(default_factory=list)
    split: SplitConfig = Field(default_factory=SplitConfig)
    name: str = "local_dataset"
    force_split: bool = False


class ClassDetectRequest(BaseModel):
    path: str


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
    workers: int = Field(default=2, ge=0)
    optimizer: str = "auto"
    lr0: float = Field(default=0.01, gt=0)
    lrf: float = Field(default=0.01, gt=0)
    weight_decay: float = Field(default=0.0005, ge=0)
    cos_lr: bool = False
    warmup_epochs: float = Field(default=3.0, ge=0)
    freeze: Optional[int] = Field(default=None, ge=0)
    pretrained: bool = True
    activation: str = "silu"
    exist_ok: bool = False
    seed: int = 0
    project: str = "runs/detect"
    name: str = "train"
    resume: bool = False


class WeightRequest(BaseModel):
    project: str = "runs/detect"
    name: str = "train"


def ensure_dirs():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


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


def ensure_runs_path(path: Path):
    runs_root = RUNS_ROOT.resolve()
    try:
        path.resolve().relative_to(runs_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Training outputs can only be read from the runs directory.",
        ) from exc


def is_run_dir(path: Path) -> bool:
    return (
        (path / "weights" / "best.pt").is_file()
        or (path / "weights" / "last.pt").is_file()
        or (path / "results.csv").is_file()
    )


def latest_run_dir(search_root: Path) -> Optional[Path]:
    if not search_root.exists():
        return None

    candidates = [path for path in search_root.rglob("*") if path.is_dir() and is_run_dir(path)]
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: max((item.stat().st_mtime for item in path.rglob("*") if item.is_file()), default=0),
    )


def resolve_run_dir(project: str, name: str) -> Path:
    project_path = resolve_project_path(project)
    ensure_runs_path(project_path)

    exact = (project_path / name).resolve()
    ensure_runs_path(exact)
    if is_run_dir(exact):
        return exact

    scoped_latest = latest_run_dir(project_path)
    if scoped_latest:
        return scoped_latest

    global_latest = latest_run_dir(RUNS_ROOT.resolve())
    if global_latest:
        return global_latest

    raise HTTPException(status_code=404, detail="No completed training run found under the runs directory.")


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


def dataset_response(yaml_path: Path, message: str) -> dict:
    try:
        classes = read_yaml_class_names(yaml_path)
    except HTTPException:
        classes = []
    return {"dataset_yaml": str(yaml_path), "classes": classes, "message": message}


def float_value(row: dict, key: str) -> Optional[float]:
    value = row.get(key)
    if value is None:
        value = row.get(f" {key}")
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def sum_values(row: dict, keys: list[str]) -> Optional[float]:
    values = [float_value(row, key) for key in keys]
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def format_metric(value: Optional[float], digits: int = 4):
    return round(value, digits) if value is not None else None


def f1_from_precision_recall(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None or precision + recall <= 0:
        return None
    return 2 * precision * recall / (precision + recall)


def parse_metric_row(line: str) -> Optional[dict]:
    parts = clean_log_line(line).split()
    if len(parts) < 7:
        return None

    try:
        images = int(float(parts[-6]))
        instances = int(float(parts[-5]))
        precision = float(parts[-4])
        recall = float(parts[-3])
        map50 = float(parts[-2])
        map50_95 = float(parts[-1])
    except ValueError:
        return None

    class_name = " ".join(parts[:-6]).strip()
    if not class_name or class_name.lower() in {"class", "epoch"}:
        return None

    return {
        "class_name": class_name,
        "images": images,
        "instances": instances,
        "precision": format_metric(precision),
        "recall": format_metric(recall),
        "f1": format_metric(f1_from_precision_recall(precision, recall)),
        "map50": format_metric(map50),
        "map50_95": format_metric(map50_95),
    }


def parse_class_metrics_from_log(log_path: Path) -> dict:
    if not log_path.is_file():
        return {"overall": None, "classes": [], "weighted_f1": None}

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    validating_indexes = [index for index, line in enumerate(lines) if "Validating " in clean_log_line(line)]
    search_lines = lines[validating_indexes[-1] + 1:] if validating_indexes else lines

    overall = None
    classes_by_name = {}
    for line in search_lines:
        row = parse_metric_row(line)
        if not row:
            continue
        if row["class_name"] == "all":
            overall = row
        else:
            classes_by_name[row["class_name"]] = row

    classes = list(classes_by_name.values())
    total_instances = sum(row["instances"] for row in classes)
    weighted_f1 = None
    if total_instances:
        weighted_f1 = sum((row["f1"] or 0) * row["instances"] for row in classes) / total_instances

    return {
        "overall": overall,
        "classes": classes,
        "weighted_f1": format_metric(weighted_f1),
    }


def build_metric_history(rows: list[dict]) -> list[dict]:
    history = []
    for row in rows:
        precision = float_value(row, "metrics/precision(B)")
        recall = float_value(row, "metrics/recall(B)")
        history.append({
            "epoch": int(float_value(row, "epoch") or 0),
            "overall_f1": format_metric(f1_from_precision_recall(precision, recall)),
            "map50": format_metric(float_value(row, "metrics/mAP50(B)")),
            "map50_95": format_metric(float_value(row, "metrics/mAP50-95(B)")),
            "training_loss": format_metric(sum_values(row, ["train/box_loss", "train/cls_loss", "train/dfl_loss"])),
            "testing_loss": format_metric(sum_values(row, ["val/box_loss", "val/cls_loss", "val/dfl_loss"])),
        })
    return history


def read_run_metrics(run_dir: Path) -> dict:
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        return {"available": False, "run_dir": str(run_dir), "results_csv": ""}

    with results_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {"available": False, "run_dir": str(run_dir), "results_csv": str(results_path)}

    row = rows[-1]
    precision = float_value(row, "metrics/precision(B)")
    recall = float_value(row, "metrics/recall(B)")
    f1_score = f1_from_precision_recall(precision, recall)
    training_loss = sum_values(row, ["train/box_loss", "train/cls_loss", "train/dfl_loss"])
    testing_loss = sum_values(row, ["val/box_loss", "val/cls_loss", "val/dfl_loss"])
    map50 = float_value(row, "metrics/mAP50(B)")
    map50_95 = float_value(row, "metrics/mAP50-95(B)")
    class_metrics = parse_class_metrics_from_log(LOG_FILE)

    return {
        "available": True,
        "run_dir": str(run_dir),
        "results_csv": str(results_path),
        "epoch": int(float_value(row, "epoch") or 0),
        "overall_f1": format_metric(f1_score),
        "weighted_f1": class_metrics["weighted_f1"],
        "per_class": class_metrics["classes"],
        "training_loss": format_metric(training_loss),
        "testing_loss": format_metric(testing_loss),
        "precision": format_metric(precision),
        "recall": format_metric(recall),
        "map50": format_metric(map50),
        "map50_95": format_metric(map50_95),
        "history": build_metric_history(rows),
        "note": "F1 is derived from validation precision and recall. Weighted F1 is calculated from final per-class validation rows when available.",
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
    names = resolve_dataset_classes(root, classes)

    if not force_split and "train" in split_dirs(root) and "val" in split_dirs(root):
        return prepare_existing_split(root, name, names)

    return prepare_split_dataset(root, name, names, split)


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


def stream_training_logs(process: subprocess.Popen, log_paths: list[Path]):
    handles = [path.open("a", encoding="utf-8") for path in log_paths]
    try:
        if process.stdout is None:
            return

        for raw_line in process.stdout:
            line = clean_log_line(raw_line)
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


def current_status() -> dict:
    global training_process
    log_info = {
        "log_file": str(LOG_FILE),
        "history_log_file": str(training_log_file) if training_log_file else "",
    }

    if training_process is None:
        return {"running": False, "returncode": None, "started_at": training_started_at, **log_info}

    returncode = training_process.poll()
    if returncode is not None:
        training_process = None
        return {"running": False, "returncode": returncode, "started_at": training_started_at, **log_info}

    return {"running": True, "returncode": None, "started_at": training_started_at, **log_info}


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
    return dataset_response(yaml_path, "Local dataset is ready.")


@app.post("/api/dataset/classes")
def dataset_classes(request: ClassDetectRequest):
    source = Path(request.path).expanduser()
    if not source.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {source}")

    classes = detect_dataset_classes(source)
    return {"classes": classes, "message": f"Detected {len(classes)} class names."}


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
    return dataset_response(yaml_path, "Uploaded dataset is ready.")


@app.post("/api/dataset/folder")
async def upload_folder_dataset(
    files: list[UploadFile] = File(...),
    classes: str = Form(...),
    train: int = Form(70),
    val: int = Form(20),
    test: int = Form(10),
    name: str = Form("uploaded_folder_dataset"),
    force_split: bool = Form(False),
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
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_root = upload_dir.resolve()

    saved_count = 0
    for upload in files:
        relative_path = safe_upload_path(upload.filename or "")
        target = (upload_dir / relative_path).resolve()
        try:
            target.relative_to(upload_root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid upload path: {upload.filename}") from exc

        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as output:
            shutil.copyfileobj(upload.file, output)
        saved_count += 1

    if saved_count == 0:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    yaml_path = prepare_dataset(
        source=upload_dir,
        name=clean,
        classes=class_names,
        split=split,
        force_split=force_split,
    )
    return dataset_response(yaml_path, "Uploaded folder dataset is ready.")


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
        return dataset_response(yaml_path, "Roboflow dataset is ready.")

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
    return dataset_response(yaml_path, "Roboflow dataset is ready.")


@app.post("/api/train/start")
def start_training(request: TrainRequest):
    global training_process, training_started_at, training_log_file

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
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    training_log_file = LOG_DIR / f"train-{timestamp}.log"

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
        "--workers", str(request.workers),
        "--optimizer", request.optimizer,
        "--lr0", str(request.lr0),
        "--lrf", str(request.lrf),
        "--weight-decay", str(request.weight_decay),
        "--warmup-epochs", str(request.warmup_epochs),
        "--pretrained", str(request.pretrained).lower(),
        "--activation", request.activation,
        "--seed", str(request.seed),
        "--project", request.project,
        "--name", request.name,
    ]

    device = request.device or os.getenv("TRAINING_DEVICE")
    if device:
        cmd.extend(["--device", device])
    if request.cos_lr:
        cmd.append("--cos-lr")
    if request.freeze is not None:
        cmd.extend(["--freeze", str(request.freeze)])
    if request.exist_ok:
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


@app.post("/api/train/weights/status")
def weights_status(request: WeightRequest):
    try:
        run_dir = resolve_run_dir(request.project, request.name)
    except HTTPException:
        run_dir = None

    result = {"run_dir": str(run_dir) if run_dir else ""}
    for weight in ("best", "last"):
        try:
            path = resolve_weight_path(request.project, request.name, weight)
            result[weight] = {"available": True, "path": str(path), "size": path.stat().st_size}
        except HTTPException:
            result[weight] = {"available": False, "path": "", "size": 0}
    return result


@app.post("/api/train/metrics")
def train_metrics(request: WeightRequest):
    run_dir = resolve_run_dir(request.project, request.name)
    return read_run_metrics(run_dir)


@app.get("/api/train/weights/{weight}")
def download_weight(weight: str, project: str = "runs/detect", name: str = "train"):
    path = resolve_weight_path(project, name, weight)
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=path.name,
    )
