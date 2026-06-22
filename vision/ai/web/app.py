#!/usr/bin/env python3
"""FastAPI web UI backend for YOLOv8 training."""

import csv
import json
import math
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .stratified_split import SPLIT_NAMES, stratified_split


WEB_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEB_DIR.parents[2]
TRAIN_SCRIPT = REPO_ROOT / "vision" / "ai" / "train" / "train_yolov8.py"
STATIC_DIR = WEB_DIR / "static"
LOG_DIR = WEB_DIR / "logs"
LOG_FILE = LOG_DIR / "current.log"
RUNS_ROOT = REPO_ROOT / "runs"
MYT = timezone(timedelta(hours=8), name="MYT")

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
WEB_PROGRESS_RE = re.compile(r"^WEB_TRAINING_PROGRESS\s+epoch=(\d+)\s+total=(\d+)$")
RUN_DIRECTORY_PREFIXES = ("Logging results to ", "Results saved to ")
SPLIT_METADATA_FILE = ".split_metadata.json"

app = FastAPI(title="YOLOv8 Training UI")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

training_process: Optional[subprocess.Popen] = None
training_started_at: Optional[float] = None
training_log_file: Optional[Path] = None
training_run_info: Optional[dict] = None
dataset_preparation_jobs: dict[str, dict] = {}
dataset_preparation_lock = threading.Lock()

DatasetProgressCallback = Callable[[str, int, int, str], None]


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


class WeightRequest(BaseModel):
    project: str = "runs/detect"
    name: str = "train"


class ArtifactRequest(BaseModel):
    project: str = "runs/detect"
    name: str = "train"
    artifact: str


def ensure_dirs():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


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


def is_run_dir(path: Path) -> bool:
    return (
        (path / "weights" / "best.pt").is_file()
        or (path / "weights" / "last.pt").is_file()
        or (path / "results.csv").is_file()
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
) -> dict:
    try:
        classes = read_yaml_class_names(yaml_path)
    except HTTPException:
        classes = []
    return {
        "dataset_yaml": str(yaml_path),
        "classes": classes,
        "summary": inspect_dataset_yaml(yaml_path, classes, progress_callback),
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

    return {
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


def build_metric_history(rows: list[dict]) -> list[dict]:
    history = []
    for row in rows:
        history.append({
            "epoch": int(float_value(row, "epoch") or 0),
            "map50": format_metric(float_value(row, "metrics/mAP50(B)")),
            "map50_95": format_metric(float_value(row, "metrics/mAP50-95(B)")),
            "training_loss": format_metric(sum_values(row, ["train/box_loss", "train/cls_loss", "train/dfl_loss"])),
            "testing_loss": format_metric(sum_values(row, ["val/box_loss", "val/cls_loss", "val/dfl_loss"])),
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
    return {
        "available": path.is_file(),
        "path": str(path),
        "size": path.stat().st_size if path.is_file() else 0,
    }


def read_run_metrics(run_dir: Path) -> dict:
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        return {
            "available": False,
            "run_dir": str(run_dir),
            "results_csv": "",
            "artifacts": {
                "results_csv": artifact_status(results_path),
                "accuracy_graph": artifact_status(run_dir / "accuracy_by_epoch.png"),
                "loss_graph": artifact_status(run_dir / "loss_by_epoch.png"),
            },
        }

    with results_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {"available": False, "run_dir": str(run_dir), "results_csv": str(results_path)}

    row = rows[-1]
    precision = float_value(row, "metrics/precision(B)")
    recall = float_value(row, "metrics/recall(B)")
    training_loss = sum_values(row, ["train/box_loss", "train/cls_loss", "train/dfl_loss"])
    testing_loss = sum_values(row, ["val/box_loss", "val/cls_loss", "val/dfl_loss"])
    map50 = float_value(row, "metrics/mAP50(B)")
    map50_95 = float_value(row, "metrics/mAP50-95(B)")
    class_metrics = parse_class_metrics_from_log(LOG_FILE)
    history = build_metric_history(rows)

    return {
        "available": True,
        "run_dir": str(run_dir),
        "results_csv": str(results_path),
        "epoch": int(float_value(row, "epoch") or 0),
        "macro_f1": class_metrics["macro_f1"],
        "weighted_f1": class_metrics["weighted_f1"],
        "per_class": class_metrics["classes"],
        "training_loss": format_metric(training_loss),
        "testing_loss": format_metric(testing_loss),
        "precision": format_metric(precision),
        "recall": format_metric(recall),
        "map50": format_metric(map50),
        "map50_95": format_metric(map50_95),
        "history": history,
        "best": best_metric_summary(history),
        "artifacts": {
            "results_csv": artifact_status(results_path),
            "accuracy_graph": artifact_status(run_dir / "accuracy_by_epoch.png"),
            "loss_graph": artifact_status(run_dir / "loss_by_epoch.png"),
        },
        "note": "Macro and weighted F1 are calculated from final per-class validation rows when available.",
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
) -> tuple[Path, bool]:
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
            return source_yaml, False

    layouts = split_dirs(root)
    if (
        "train" in layouts
        and "val" in layouts
        and has_image_files(layouts["train"]["images"])
        and has_image_files(layouts["val"]["images"])
    ):
        names = resolve_dataset_classes(root, classes)
        return prepare_existing_split(root, name, names), True

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
    return yaml_path, False


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


def capture_epoch_progress(line: str) -> bool:
    global training_run_info
    match = WEB_PROGRESS_RE.match(line)
    if match is None:
        return False

    current_epoch = int(match.group(1))
    total_epochs = int(match.group(2))
    if training_run_info is not None and current_epoch > 0 and total_epochs > 0:
        training_run_info["current_epoch"] = min(current_epoch, total_epochs)
        training_run_info["total_epochs"] = total_epochs
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


def read_log_tail(max_chars: int = 20000) -> str:
    if not LOG_FILE.is_file():
        return ""
    data = LOG_FILE.read_text(encoding="utf-8", errors="replace")
    return data[-max_chars:]


def read_log_file(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_error_log(path: Path) -> str:
    keywords = ("error", "warning", "traceback", "exception", "failed", "no space", "not found")
    lines = read_log_file(path).splitlines()
    return "\n".join(line for line in lines if any(keyword in line.lower() for keyword in keywords))


def latest_timestamped_log() -> Optional[Path]:
    logs = [path for path in LOG_DIR.glob("train-*.log") if path.is_file()]
    if not logs:
        return None
    return max(logs, key=lambda path: path.stat().st_mtime)


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

    if running and current == 0 and total:
        current = min(completed + 1, total)
    elif not running and completed:
        current = max(current, completed)

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
    }


def resolve_artifact_path(request: ArtifactRequest) -> Path:
    run_dir = resolve_run_dir(request.project, request.name)
    artifact_map = {
        "results_csv": run_dir / "results.csv",
        "accuracy_graph": run_dir / "accuracy_by_epoch.png",
        "loss_graph": run_dir / "loss_by_epoch.png",
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

    return {
        "running": running,
        "returncode": returncode,
        "started_at": training_started_at,
        "log_file": str(LOG_FILE),
        "history_log_file": str(training_log_file) if training_log_file else "",
        "training_run": run_info,
        "epoch_progress": epoch_progress(run_info, running),
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


@app.get("/api/dataset/preparation/status")
def dataset_preparation_status(job_id: str):
    with dataset_preparation_lock:
        progress = dataset_preparation_jobs.get(job_id)
        if progress is None:
            raise HTTPException(status_code=404, detail="Dataset preparation job not found.")
        return dict(progress)


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
):
    with zipfile.ZipFile(zip_path) as archive:
        entries = archive.infolist()
        total_bytes = sum(entry.file_size for entry in entries)
        work_total = total_bytes or len(entries)
        extracted_bytes = 0
        progress_callback("extracting", 0, work_total, f"Extracting 0 of {len(entries)} ZIP entries")
        for index, entry in enumerate(entries, start=1):
            archive.extract(entry, extract_dir)
            extracted_bytes += entry.file_size
            current = extracted_bytes if total_bytes else index
            progress_callback(
                "extracting",
                current,
                work_total,
                f"Extracting {index} of {len(entries)} ZIP entries",
            )


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
        "fetching_roboflow",
        0,
        0,
        "Roboflow is exporting and downloading the dataset.",
    )

    try:
        if download_dir.exists():
            shutil.rmtree(download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)
        version_obj = project.version(int(version))
        try:
            dataset = version_obj.download(dataset_format, location=str(download_dir), overwrite=True)
        except TypeError:
            dataset = version_obj.download(dataset_format, location=str(download_dir))
    except Exception as exc:
        error = HTTPException(status_code=502, detail=f"Roboflow download failed: {exc}")
        mark_dataset_preparation_failed(request.job_id, error)
        raise error from exc

    try:
        dataset_root = Path(getattr(dataset, "location", download_dir))
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
            yaml_path, normalized = prepare_roboflow_download(
                dataset_root,
                clean,
                request.classes,
            )
            if normalized:
                progress_callback(
                    "normalizing_paths",
                    1,
                    1,
                    "Normalized invalid export paths to the detected split folders.",
                )
            message = (
                "Roboflow dataset is ready. Invalid export paths were normalized "
                "to the detected train/validation/test folders."
                if normalized
                else "Roboflow dataset is ready."
            )

        response = dataset_response(yaml_path, message, progress_callback)
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


@app.post("/api/train/start")
def start_training(request: TrainRequest):
    global training_process, training_started_at, training_log_file, training_run_info

    status = current_status()
    if status["running"]:
        raise HTTPException(status_code=409, detail="Training is already running.")

    dataset_yaml = Path(request.dataset_yaml).expanduser() if request.dataset_yaml else None
    if not request.resume and (dataset_yaml is None or not dataset_yaml.is_file()):
        raise HTTPException(status_code=400, detail="Prepare a valid dataset before starting a new training run.")

    model = MODEL_MAP.get(request.model_size)
    if not model:
        raise HTTPException(status_code=400, detail=f"Unknown model size: {request.model_size}")
    training_project_path = normalize_training_project_path(request.project)
    resume_checkpoint = None
    resume_run_dir = None
    if request.resume:
        try:
            resume_run_dir, _ = resolve_run_dir_details(request.project, request.name)
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
        model = str(resume_checkpoint)

    training_run_info = {
        "requested_project": request.project,
        "requested_name": request.name,
        "project": str(training_project_path),
        "name": request.name,
        "expected_run_dir": str(training_project_path / request.name),
        "run_dir": str(resume_run_dir) if resume_run_dir else "",
        "resolution_type": "actual" if resume_run_dir else "pending",
        "current_epoch": 0,
        "total_epochs": request.epochs,
    }

    ensure_dirs()
    LOG_FILE.write_text("", encoding="utf-8")
    timestamp = datetime.now(MYT).strftime("%Y%m%d-%H%M%S")
    training_log_file = LOG_DIR / f"train-{timestamp}.log"

    cmd = [
        TRAINING_PYTHON,
        str(TRAIN_SCRIPT),
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
        "--pretrained", "true",
        "--activation", request.activation,
        "--seed", str(request.seed),
        "--project", str(training_project_path),
        "--name", request.name,
    ]

    if dataset_yaml is not None and not request.resume:
        cmd.extend(["--data", str(dataset_yaml)])

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
        "training_run": training_run_info,
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


@app.get("/api/train/weights/{weight}")
def download_weight(weight: str, project: str = "runs/detect", name: str = "train"):
    path = resolve_weight_path(project, name, weight)
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=path.name,
    )
