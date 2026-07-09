#!/usr/bin/env python3
"""Train an RF-DETR object detection model for the web UI."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import re
import shutil
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import yaml


WEB_PROGRESS_PREFIX = "WEB_TRAINING_PROGRESS"
RFDETR_DATASET_VIEW_DIR = ".rfdetr"
RFDETR_RUN_LOG = "rfdetr_training.log"
MODEL_CLASSES = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-large": "RFDETRLarge",
}
RFDETR_VAL_RE = re.compile(r"Val\s+\(Epoch\s+(\d+)\s*/\s*(\d+)\)", re.IGNORECASE)
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RESULT_FIELDNAMES = [
    "epoch",
    "train/loss",
    "val/loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]
LOSS_FIELDNAMES = ("train/loss", "val/loss")
DETECTION_FIELDNAMES = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)
METRIC_ALIASES = {
    "epoch": ["epoch", "trainer/global_step", "global_step", "step"],
    "train_loss": [
        "train/loss",
        "train/loss_epoch",
        "train/loss_step",
        "train_loss",
        "train_loss_epoch",
        "loss",
        "loss_epoch",
        "train/total_loss",
        "train/total_loss_epoch",
    ],
    "val_loss": [
        "val/loss",
        "val/loss_epoch",
        "val_loss",
        "val_loss_epoch",
        "validation/loss",
        "validation_loss",
        "val/total_loss",
        "val/total_loss_epoch",
    ],
    "precision": ["val/precision", "precision", "prec", "metrics/precision(B)", "val/prec"],
    "recall": ["val/recall", "recall", "metrics/recall(B)"],
    "f1": ["val/f1", "f1", "F1"],
    "map50": ["val/mAP_50", "val/map_50", "val/mAP50", "map_50", "mAP_50", "mAP50", "metrics/mAP50(B)"],
    "map50_95": [
        "val/mAP_50_95",
        "val/map_50_95",
        "val/mAP50_95",
        "val/mAP50-95",
        "map",
        "mAP",
        "map_50_95",
        "mAP_50_95",
        "mAP50_95",
        "metrics/mAP50-95(B)",
    ],
    "map75": ["val/mAP_75", "val/map_75", "map_75", "mAP_75", "mAP75"],
    "mar": ["val/mAR", "mAR", "mar", "AR"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train an RF-DETR detection model.")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml or dataset directory.")
    parser.add_argument("--model", default="rfdetr-nano", choices=sorted(MODEL_CLASSES))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cos-lr", action="store_true", default=False)
    parser.add_argument("--warmup-epochs", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="runs/rfdetr")
    parser.add_argument("--name", default="train")
    parser.add_argument("--resume", action="store_true", default=False)
    return parser.parse_args()


class TeeStream:
    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary

    def write(self, data):
        self.primary.write(data)
        self.secondary.write(data)
        return len(data)

    def flush(self):
        self.primary.flush()
        self.secondary.flush()

    def __getattr__(self, name):
        return getattr(self.primary, name)


@contextmanager
def tee_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("a", encoding="utf-8") as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def normalize_device(device: str | None) -> str:
    requested = str(device or "").strip()
    if not requested:
        return "cuda"
    normalized = requested.lower()
    if normalized in {"cpu", "cuda", "mps"}:
        return normalized
    if normalized.startswith("cuda"):
        return "cuda"
    if all(part.strip().isdigit() for part in requested.split(",") if part.strip()):
        os.environ["CUDA_VISIBLE_DEVICES"] = requested
        return "cuda"
    return requested


def resolve_dataset_dir(data: str) -> Path:
    path = Path(data).expanduser().resolve()
    if path.is_file():
        return path.parent
    if path.is_dir():
        return path
    raise FileNotFoundError(f"Dataset path not found: {path}")


def find_dataset_yaml(dataset_dir: Path) -> Path:
    yaml_path = next(
        (dataset_dir / name for name in ("data.yaml", "data.yml", "dataset.yaml") if (dataset_dir / name).is_file()),
        None,
    )
    if yaml_path is None:
        raise FileNotFoundError(f"Dataset YAML not found in {dataset_dir}.")
    return yaml_path


def read_dataset_yaml(dataset_dir: Path) -> tuple[Path, dict]:
    yaml_path = find_dataset_yaml(dataset_dir)
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Could not read dataset YAML at {yaml_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Dataset YAML must contain a mapping: {yaml_path}")
    return yaml_path, payload


def resolve_yaml_dataset_root(yaml_path: Path, payload: dict) -> Path:
    root = payload.get("path") or yaml_path.parent
    root_path = Path(str(root)).expanduser()
    if not root_path.is_absolute():
        root_path = yaml_path.parent / root_path
    return root_path.resolve()


def resolve_split_image_dir(dataset_root: Path, value) -> Path | None:
    if not value:
        return None
    values = value if isinstance(value, list) else [value]
    for item in values:
        candidate = Path(str(item)).expanduser()
        if not candidate.is_absolute():
            candidate = dataset_root / candidate
        candidate = candidate.resolve()
        if candidate.is_dir():
            return candidate
    return None


def infer_label_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        candidate = Path(*parts[:index], "labels", *parts[index + 1:])
        if candidate.is_dir():
            return candidate

    candidates = [
        image_dir.parent / "labels",
        image_dir.parent.parent / "labels" / image_dir.name,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find YOLO labels folder for image folder: {image_dir}")


def link_or_copy_dir(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    try:
        target.symlink_to(source.resolve(), target_is_directory=True)
    except OSError:
        shutil.copytree(source, target)


def normalize_names(names):
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        normalized = {}
        for key, value in names.items():
            try:
                normalized[int(key)] = str(value)
            except (TypeError, ValueError):
                normalized[str(key)] = str(value)
        return normalized
    return {}


def prepare_rfdetr_dataset(dataset_dir: Path) -> Path:
    yaml_path, payload = read_dataset_yaml(dataset_dir)
    source_root = resolve_yaml_dataset_root(yaml_path, payload)
    split_values = {
        "train": payload.get("train"),
        "valid": payload.get("val") or payload.get("valid"),
        "test": payload.get("test"),
    }
    split_dirs = {}
    for split_name, split_value in split_values.items():
        image_dir = resolve_split_image_dir(source_root, split_value)
        if image_dir is None:
            if split_name == "test":
                continue
            raise FileNotFoundError(
                f"RF-DETR could not adapt the dataset. Missing {split_name} images from {yaml_path}."
            )
        split_dirs[split_name] = {
            "images": image_dir,
            "labels": infer_label_dir(image_dir),
        }

    adapter_dir = dataset_dir / RFDETR_DATASET_VIEW_DIR
    if adapter_dir.exists() or adapter_dir.is_symlink():
        if adapter_dir.is_symlink() or adapter_dir.is_file():
            adapter_dir.unlink()
        else:
            shutil.rmtree(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    for split_name, paths in split_dirs.items():
        link_or_copy_dir(paths["images"], adapter_dir / split_name / "images")
        link_or_copy_dir(paths["labels"], adapter_dir / split_name / "labels")

    adapter_payload = {
        "path": ".",
        "train": "train/images",
        "val": "valid/images",
        "names": normalize_names(payload.get("names")),
    }
    if "test" in split_dirs:
        adapter_payload["test"] = "test/images"
    if payload.get("nc") is not None:
        adapter_payload["nc"] = payload.get("nc")

    (adapter_dir / "data.yaml").write_text(yaml.safe_dump(adapter_payload, sort_keys=False), encoding="utf-8")
    print(f"RF-DETR dataset view created at {adapter_dir}", flush=True)
    for split_name, paths in split_dirs.items():
        print(
            f"RF-DETR dataset adapter: {paths['images']} -> {adapter_dir / split_name / 'images'}",
            flush=True,
        )
    return adapter_dir


def read_class_names(dataset_dir: Path) -> list[str]:
    try:
        _yaml_path, payload = read_dataset_yaml(dataset_dir)
    except (FileNotFoundError, RuntimeError):
        return []
    names = payload.get("names")
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        def sort_key(item):
            try:
                return (0, int(item[0]))
            except (TypeError, ValueError):
                return (1, str(item[0]))

        return [str(value) for _, value in sorted(names.items(), key=sort_key)]
    return []


def expected_class_ids(class_names: list[str], observed_ids: set[int] | None = None) -> list[int]:
    ids = set(range(len(class_names)))
    if observed_ids:
        ids.update(observed_ids)
    return sorted(ids)


def class_name_for_id(class_id: int, class_names: list[str]) -> str:
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return f"class_{class_id}"


def parse_label_class_id(line: str):
    parts = line.strip().split()
    if not parts:
        return None
    try:
        value = float(parts[0])
    except ValueError:
        return None
    class_id = int(value)
    return class_id if class_id == value else None


def audit_label_split(adapter_dir: Path, split_name: str, class_names: list[str]) -> dict:
    label_dir = adapter_dir / split_name / "labels"
    instance_counts: Counter[int] = Counter()
    image_counts: Counter[int] = Counter()
    warnings = []
    malformed = 0

    if not label_dir.is_dir():
        warnings.append(f"Missing labels folder: {label_dir}")
        return {
            "split": split_name,
            "label_path": str(label_dir),
            "classes": [],
            "present_class_ids": [],
            "missing_class_ids": list(range(len(class_names))),
            "malformed_labels": 0,
            "warnings": warnings,
        }

    for label_file in sorted(label_dir.glob("*.txt")):
        seen_in_file = set()
        try:
            lines = label_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            warnings.append(f"Could not read label file {label_file}: {exc}")
            continue
        for line in lines:
            if not line.strip():
                continue
            class_id = parse_label_class_id(line)
            if class_id is None:
                malformed += 1
                continue
            instance_counts[class_id] += 1
            seen_in_file.add(class_id)
        for class_id in seen_in_file:
            image_counts[class_id] += 1

    observed_ids = set(instance_counts)
    class_ids = expected_class_ids(class_names, observed_ids)
    classes = [
        {
            "class_id": class_id,
            "class_name": class_name_for_id(class_id, class_names),
            "images": int(image_counts.get(class_id, 0)),
            "instances": int(instance_counts.get(class_id, 0)),
        }
        for class_id in class_ids
    ]
    missing_ids = [class_id for class_id in range(len(class_names)) if instance_counts.get(class_id, 0) == 0]
    if malformed:
        warnings.append(f"{malformed} malformed label row(s) were ignored in {label_dir}.")
    for class_id in sorted(observed_ids):
        if class_names and class_id >= len(class_names):
            warnings.append(f"Unknown class id {class_id} appears in {label_dir}.")

    return {
        "split": split_name,
        "label_path": str(label_dir),
        "classes": classes,
        "present_class_ids": sorted(observed_ids),
        "missing_class_ids": missing_ids,
        "malformed_labels": malformed,
        "warnings": warnings,
    }


def audit_rfdetr_dataset(adapter_dir: Path, class_names: list[str]) -> dict:
    splits = {}
    for split_name in ("train", "valid", "test"):
        if (adapter_dir / split_name).exists():
            splits[split_name] = audit_label_split(adapter_dir, split_name, class_names)
    return {
        "dataset_dir": str(adapter_dir),
        "classes": class_names,
        "splits": splits,
    }


def print_dataset_audit(audit: dict):
    print("RF-DETR dataset audit:", flush=True)
    for split_name in ("train", "valid", "test"):
        split = (audit.get("splits") or {}).get(split_name)
        if not split:
            continue
        classes = split.get("classes") or []
        present = [row for row in classes if row.get("instances", 0) > 0]
        counts = ", ".join(f"{row['class_name']}={row.get('instances', 0)}" for row in classes) or "no labels found"
        print(f"{split_name}: {len(present)} classes present: {counts}", flush=True)
        for class_id in split.get("missing_class_ids") or []:
            print(
                f"WARNING: RF-DETR {split_name} split has no labels for class {class_name_for_id(class_id, audit.get('classes') or [])}.",
                flush=True,
            )
        for warning in split.get("warnings") or []:
            print(f"WARNING: RF-DETR dataset audit: {warning}", flush=True)


def import_model_class(model_id: str):
    try:
        import rfdetr
    except ImportError as exc:
        raise SystemExit(
            "RF-DETR is not installed. Rebuild the Docker image after adding rfdetr to requirements.txt."
        ) from exc

    class_name = MODEL_CLASSES[model_id]
    try:
        return getattr(rfdetr, class_name)
    except AttributeError as exc:
        raise SystemExit(f"The installed rfdetr package does not provide {class_name}.") from exc


def copy_if_exists(source: Path, destination: Path):
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def normalize_checkpoints(run_dir: Path):
    weights_dir = run_dir / "weights"
    copy_if_exists(run_dir / "checkpoint_best_total.pth", weights_dir / "best.pt")
    copy_if_exists(run_dir / "checkpoint.pth", weights_dir / "last.pt")
    if not (weights_dir / "best.pt").is_file():
        copy_if_exists(run_dir / "checkpoint_best_ema.pth", weights_dir / "best.pt")
    if not (weights_dir / "best.pt").is_file():
        copy_if_exists(run_dir / "checkpoint_best_regular.pth", weights_dir / "best.pt")
    if not (weights_dir / "last.pt").is_file():
        latest = max(run_dir.glob("checkpoint_*.pth"), key=lambda path: path.stat().st_mtime, default=None)
        if latest is not None:
            copy_if_exists(latest, weights_dir / "last.pt")


def float_value(row: dict, key: str):
    value = row.get(key)
    if value is None:
        return None
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def normalized_row_lookup(row: dict) -> dict:
    lookup = {}
    for key, value in row.items():
        text = str(key).strip()
        lookup[text] = value
        lookup[text.lower()] = value
    return lookup


def find_rfdetr_metric_rows(run_dir: Path) -> list[dict]:
    candidates = [
        path for path in run_dir.rglob("*.csv")
        if path.name != "results.csv" and path.is_file()
    ]
    rows: list[dict] = []
    for path in candidates:
        try:
            with path.open("r", encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if any(normalized_metric(row, keys) is not None for keys in METRIC_ALIASES.values()):
                        rows.append(row)
        except (OSError, csv.Error):
            continue
    return rows


def normalized_metric(row: dict, keys: list[str]):
    lookup = normalized_row_lookup(row)
    for key in keys:
        value = float_value({"value": lookup.get(key) if key in lookup else lookup.get(key.lower())}, "value")
        if value is not None:
            return value
    return None


def parse_float_cell(value: str):
    text = str(value).strip().replace(",", "")
    if text.endswith("%"):
        text = text[:-1]
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def table_cells(line: str) -> list[str]:
    clean = ANSI_ESCAPE_RE.sub("", line)
    if "│" not in clean:
        return []
    return [cell.strip() for cell in clean.split("│")[1:-1]]


def parse_rfdetr_log_metrics(log_path: Path | None) -> dict:
    if log_path is None or not log_path.is_file():
        return {"overall": [], "per_class": [], "source": None}

    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {"overall": [], "per_class": [], "source": None}

    overall_rows = []
    per_class_by_name = {}
    last_overall = None
    current_epoch = None
    total_epochs = None
    in_per_class_table = False

    for line in lines:
        clean = ANSI_ESCAPE_RE.sub("", line)
        match = RFDETR_VAL_RE.search(clean)
        if match is not None:
            current_epoch = int(match.group(1))
            total_epochs = int(match.group(2))

        cells = table_cells(clean)
        numbers = [parse_float_cell(cell) for cell in cells]
        if len(numbers) >= 7 and all(value is not None for value in numbers[:7]):
            last_overall = {
                "epoch": current_epoch,
                "total_epochs": total_epochs,
                "map50_95": numbers[0],
                "map50": numbers[1],
                "map75": numbers[2],
                "mar": numbers[3],
                "f1": numbers[4],
                "precision": numbers[5],
                "recall": numbers[6],
                "source": "rfdetr_log",
            }
            continue

        if "Per-class Metrics" in clean:
            in_per_class_table = True
            if last_overall is not None:
                row = dict(last_overall)
                row["epoch"] = current_epoch or row.get("epoch") or len(overall_rows) + 1
                row["total_epochs"] = total_epochs or row.get("total_epochs")
                overall_rows.append(row)
            continue

        if not in_per_class_table:
            continue

        if "└" in clean:
            in_per_class_table = False
            continue

        if len(cells) < 6 or not cells[0] or cells[0].lower() == "class":
            continue
        metric_values = [parse_float_cell(cell) for cell in cells[1:6]]
        if any(value is None for value in metric_values):
            continue
        class_name = cells[0]
        per_class_by_name[class_name] = {
            "epoch": current_epoch,
            "class_name": class_name,
            "map50": None,
            "map50_95": metric_values[0],
            "ar": metric_values[1],
            "f1": metric_values[2],
            "precision": metric_values[3],
            "recall": metric_values[4],
            "source": "rfdetr_log",
        }

    return {
        "overall": overall_rows,
        "per_class": list(per_class_by_name.values()),
        "source": "rfdetr_log" if overall_rows or per_class_by_name else None,
    }


def normalize_csv_metric_rows(rows: list[dict]) -> list[dict]:
    normalized_rows = []
    for index, row in enumerate(rows, start=1):
        epoch = normalized_metric(row, METRIC_ALIASES["epoch"]) or index
        normalized_rows.append({
            "epoch": int(epoch),
            "train/loss": normalized_metric(row, METRIC_ALIASES["train_loss"]),
            "val/loss": normalized_metric(row, METRIC_ALIASES["val_loss"]),
            "metrics/precision(B)": normalized_metric(row, METRIC_ALIASES["precision"]),
            "metrics/recall(B)": normalized_metric(row, METRIC_ALIASES["recall"]),
            "metrics/mAP50(B)": normalized_metric(row, METRIC_ALIASES["map50"]),
            "metrics/mAP50-95(B)": normalized_metric(row, METRIC_ALIASES["map50_95"]),
        })
    return normalized_rows


def normalize_log_metric_rows(log_metrics: dict) -> list[dict]:
    normalized_rows = []
    for index, row in enumerate(log_metrics.get("overall") or [], start=1):
        normalized_rows.append({
            "epoch": int(row.get("epoch") or index),
            "train/loss": None,
            "val/loss": None,
            "metrics/precision(B)": row.get("precision"),
            "metrics/recall(B)": row.get("recall"),
            "metrics/mAP50(B)": row.get("map50"),
            "metrics/mAP50-95(B)": row.get("map50_95"),
        })
    return normalized_rows


def row_has_value(row: dict, keys: tuple[str, ...]) -> bool:
    return any(row.get(key) is not None for key in keys)


def merge_loss_metrics(log_rows: list[dict], csv_rows: list[dict]) -> list[dict]:
    if not log_rows or not csv_rows:
        return log_rows

    loss_rows = [row for row in csv_rows if row_has_value(row, LOSS_FIELDNAMES)]
    if not loss_rows:
        return log_rows

    loss_by_epoch = {int(row.get("epoch") or 0): row for row in loss_rows}
    merged_rows = []
    for index, log_row in enumerate(log_rows, start=1):
        merged = dict(log_row)
        epoch = int(merged.get("epoch") or index)
        loss_row = loss_by_epoch.get(epoch)
        if loss_row is None and len(loss_rows) == len(log_rows):
            loss_row = loss_rows[index - 1]
        if loss_row is None:
            prior_rows = [row for row in loss_rows if int(row.get("epoch") or 0) <= epoch]
            loss_row = prior_rows[-1] if prior_rows else None
        if loss_row is not None:
            for key in LOSS_FIELDNAMES:
                if merged.get(key) is None and loss_row.get(key) is not None:
                    merged[key] = loss_row[key]
        merged_rows.append(merged)
    return merged_rows


def placeholder_result_row(epochs: int) -> dict:
    return {
        "epoch": max(1, int(epochs)),
        "train/loss": None,
        "val/loss": None,
        "metrics/precision(B)": None,
        "metrics/recall(B)": None,
        "metrics/mAP50(B)": None,
        "metrics/mAP50-95(B)": None,
    }


def build_results_rows(run_dir: Path, epochs: int, log_path: Path | None = None) -> tuple[list[dict], str]:
    rows = normalize_csv_metric_rows(find_rfdetr_metric_rows(run_dir))
    log_rows = normalize_log_metric_rows(parse_rfdetr_log_metrics(log_path))
    if log_rows:
        merged_log_rows = merge_loss_metrics(log_rows, rows)
        if not rows:
            return merged_log_rows, "rfdetr_log"
        latest_csv = rows[-1]
        latest_log = log_rows[-1]
        csv_has_metrics = row_has_value(latest_csv, DETECTION_FIELDNAMES)
        if not csv_has_metrics or latest_log.get("epoch", 0) >= latest_csv.get("epoch", 0):
            return merged_log_rows, "rfdetr_log"
    if rows:
        return rows, "csv"
    return [placeholder_result_row(epochs)], "placeholder"


def write_results_csv(run_dir: Path, epochs: int, log_path: Path | None = None) -> str:
    normalized_rows, source = build_results_rows(run_dir, epochs, log_path)
    results_path = run_dir / "results.csv"

    with results_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        for row in normalized_rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in RESULT_FIELDNAMES})
    return source


def rounded_metric(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 4) if math.isfinite(number) else None


def validation_counts_by_name(dataset_audit: dict | None) -> dict:
    split = ((dataset_audit or {}).get("splits") or {}).get("valid") or {}
    return {row.get("class_name"): row for row in split.get("classes") or []}


def metric_average(rows: list[dict], key: str):
    if not rows:
        return None
    values = [row.get(key) for row in rows]
    if any(value is None for value in values):
        return None
    return rounded_metric(sum(values) / len(values))


def weighted_metric_average(rows: list[dict], key: str):
    if not rows:
        return None
    if any(row.get(key) is None for row in rows):
        return None
    total = sum(int(row.get("instances") or 0) for row in rows)
    if total <= 0:
        return None
    return rounded_metric(sum((row.get(key) or 0) * int(row.get("instances") or 0) for row in rows) / total)


def latest_overall_metrics(run_dir: Path, log_metrics: dict) -> dict:
    if log_metrics.get("overall"):
        return dict(log_metrics["overall"][-1])
    rows = normalize_csv_metric_rows(find_rfdetr_metric_rows(run_dir))
    if not rows:
        return {}
    row = rows[-1]
    return {
        "precision": row.get("metrics/precision(B)"),
        "recall": row.get("metrics/recall(B)"),
        "map50": row.get("metrics/mAP50(B)"),
        "map50_95": row.get("metrics/mAP50-95(B)"),
        "source": "csv",
    }


def write_web_metrics(
    run_dir: Path,
    model_id: str,
    class_names: list[str],
    log_path: Path | None = None,
    dataset_audit: dict | None = None,
    training_completed: bool | None = None,
):
    log_metrics = parse_rfdetr_log_metrics(log_path)
    per_class_by_name = {row["class_name"]: row for row in log_metrics.get("per_class") or []}
    counts_by_name = validation_counts_by_name(dataset_audit)
    ordered_names = list(class_names)
    for class_name in per_class_by_name:
        if class_name not in ordered_names:
            ordered_names.append(class_name)

    per_class_rows = []
    missing_metric_names = []
    for class_name in ordered_names:
        metrics = per_class_by_name.get(class_name) or {}
        counts = counts_by_name.get(class_name) or {}
        if not metrics:
            missing_metric_names.append(class_name)
        per_class_rows.append({
            "class_name": class_name,
            "images": int(counts.get("images") or 0),
            "instances": int(counts.get("instances") or 0),
            "precision": rounded_metric(metrics.get("precision")),
            "recall": rounded_metric(metrics.get("recall")),
            "f1": rounded_metric(metrics.get("f1")),
            "map50": rounded_metric(metrics.get("map50")),
            "map50_95": rounded_metric(metrics.get("map50_95")),
        })

    overall = latest_overall_metrics(run_dir, log_metrics)
    per_class_source = log_metrics.get("source") if per_class_by_name else "not_available_from_rfdetr_csv"
    note = "RF-DETR metrics are parsed from Lightning/RF-DETR validation logs. ROC-AUC is not generated by this backend."
    if missing_metric_names:
        note += " Per-class metrics were not reported for: " + ", ".join(missing_metric_names) + "."

    payload = {
        "backend": "rfdetr",
        "model": model_id,
        "overall": {
            "precision": rounded_metric(overall.get("precision")),
            "recall": rounded_metric(overall.get("recall")),
            "f1": rounded_metric(overall.get("f1")),
            "map50": rounded_metric(overall.get("map50")),
            "map50_95": rounded_metric(overall.get("map50_95")),
            "map75": rounded_metric(overall.get("map75")),
            "mar": rounded_metric(overall.get("mar")),
            "source": overall.get("source"),
        },
        "per_class": per_class_rows,
        "per_class_source": per_class_source,
        "per_class_note": note,
        "macro_f1": metric_average(per_class_rows, "f1"),
        "weighted_f1": weighted_metric_average(per_class_rows, "f1"),
        "training_completed": training_completed,
        "roc_auc": {
            "mode": "not_available",
            "split": "val",
            "classes": [],
            "note": "ROC-AUC is not generated by the RF-DETR training runner.",
        },
        "dataset_audit": dataset_audit,
    }
    (run_dir / "web_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def finalize_rfdetr_artifacts(
    run_dir: Path,
    model_id: str = "rfdetr-nano",
    class_names: list[str] | None = None,
    epochs: int = 1,
    log_path: Path | None = None,
    dataset_audit: dict | None = None,
    training_completed: bool | None = None,
    quiet: bool = False,
):
    class_names = class_names or []
    normalize_checkpoints(run_dir)
    results_source = write_results_csv(run_dir, epochs, log_path)
    write_web_metrics(run_dir, model_id, class_names, log_path, dataset_audit, training_completed)
    if not quiet:
        print(f"RF-DETR web results saved to {run_dir / 'results.csv'} from {results_source}.", flush=True)
        print(f"RF-DETR web metrics saved to {run_dir / 'web_metrics.json'}.", flush=True)


def supported_train_kwargs(model, kwargs: dict) -> dict:
    try:
        signature = inspect.signature(model.train)
    except (TypeError, ValueError):
        return kwargs
    parameters = signature.parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return kwargs
    supported = {parameter.name for parameter in parameters}
    return {key: value for key, value in kwargs.items() if key in supported}


def main():
    args = parse_args()
    dataset_dir = resolve_dataset_dir(args.data)
    run_dir = (Path(args.project).expanduser() / args.name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = run_dir / RFDETR_RUN_LOG
    with tee_output(run_log_path):
        class_names = read_class_names(dataset_dir)
        training_dataset_dir = prepare_rfdetr_dataset(dataset_dir)
        dataset_audit = audit_rfdetr_dataset(training_dataset_dir, class_names)
        print_dataset_audit(dataset_audit)
        device = normalize_device(args.device)
        batch_size = max(1, int(args.batch))
        grad_accum_steps = max(1, math.ceil(16 / batch_size))
        checkpoint_interval = args.save_period if args.save_period and args.save_period > 0 else 10

        print(f"Logging results to {run_dir}", flush=True)
        print(f"{WEB_PROGRESS_PREFIX} epoch=1 total={args.epochs}", flush=True)

        model_class = import_model_class(args.model)
        resume_checkpoint = run_dir / "weights" / "last.pt"
        model_kwargs = {}
        if args.resume and resume_checkpoint.is_file():
            model_kwargs["pretrain_weights"] = str(resume_checkpoint)
        model = model_class(**model_kwargs)
        train_kwargs = {
            "dataset_dir": str(training_dataset_dir),
            "epochs": int(args.epochs),
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "lr": float(args.lr0),
            "weight_decay": float(args.weight_decay),
            "resolution": int(args.imgsz),
            "device": device,
            "num_workers": max(0, int(args.workers)),
            "output_dir": str(run_dir),
            "checkpoint_interval": int(checkpoint_interval),
            "early_stopping": int(args.patience) > 0,
            "early_stopping_patience": int(args.patience),
            "lr_scheduler": "cosine" if args.cos_lr else "step",
            "warmup_epochs": float(args.warmup_epochs),
            "seed": int(args.seed),
            "progress_bar": "tqdm",
            "compute_val_loss": True,
        }
        if args.resume and (run_dir / "checkpoint.pth").is_file():
            train_kwargs["resume"] = str(run_dir / "checkpoint.pth")

        training_completed = False
        try:
            model.train(**supported_train_kwargs(model, train_kwargs))
            training_completed = True
            print(f"{WEB_PROGRESS_PREFIX} epoch={args.epochs} total={args.epochs}", flush=True)
        finally:
            finalize_rfdetr_artifacts(
                run_dir,
                model_id=args.model,
                class_names=class_names,
                epochs=args.epochs,
                log_path=run_log_path,
                dataset_audit=dataset_audit,
                training_completed=training_completed,
            )


if __name__ == "__main__":
    main()
