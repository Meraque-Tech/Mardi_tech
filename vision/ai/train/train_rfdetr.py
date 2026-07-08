#!/usr/bin/env python3
"""Train an RF-DETR object detection model for the web UI."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import shutil
from pathlib import Path

import yaml


WEB_PROGRESS_PREFIX = "WEB_TRAINING_PROGRESS"
MODEL_CLASSES = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-large": "RFDETRLarge",
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


def read_class_names(dataset_dir: Path) -> list[str]:
    yaml_path = next(
        (dataset_dir / name for name in ("data.yaml", "data.yml", "dataset.yaml") if (dataset_dir / name).is_file()),
        None,
    )
    if yaml_path is None:
        return []
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
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
                    keys = {str(key).strip() for key in row}
                    if keys & {"val/mAP_50_95", "val/mAP_50", "map_50_95", "mAP_50_95"}:
                        rows.append(row)
        except (OSError, csv.Error):
            continue
    return rows


def normalized_metric(row: dict, keys: list[str]):
    for key in keys:
        value = float_value(row, key)
        if value is not None:
            return value
    return None


def write_results_csv(run_dir: Path, epochs: int):
    rows = find_rfdetr_metric_rows(run_dir)
    results_path = run_dir / "results.csv"
    fieldnames = [
        "epoch",
        "train/loss",
        "val/loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    normalized_rows = []
    for index, row in enumerate(rows, start=1):
        epoch = normalized_metric(row, ["epoch", "step"]) or index
        normalized_rows.append({
            "epoch": int(epoch),
            "train/loss": normalized_metric(row, ["train/loss", "train_loss", "loss", "train/total_loss"]),
            "val/loss": normalized_metric(row, ["val/loss", "val_loss", "val/total_loss"]),
            "metrics/precision(B)": normalized_metric(row, ["val/precision", "precision"]),
            "metrics/recall(B)": normalized_metric(row, ["val/recall", "recall"]),
            "metrics/mAP50(B)": normalized_metric(row, ["val/mAP_50", "map_50", "mAP_50"]),
            "metrics/mAP50-95(B)": normalized_metric(row, ["val/mAP_50_95", "map_50_95", "mAP_50_95"]),
        })

    if not normalized_rows:
        normalized_rows = [{
            "epoch": max(1, int(epochs)),
            "train/loss": None,
            "val/loss": None,
            "metrics/precision(B)": None,
            "metrics/recall(B)": None,
            "metrics/mAP50(B)": None,
            "metrics/mAP50-95(B)": None,
        }]

    with results_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in normalized_rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def write_web_metrics(run_dir: Path, model_id: str, class_names: list[str]):
    payload = {
        "backend": "rfdetr",
        "model": model_id,
        "per_class": [
            {
                "class_name": class_name,
                "images": 0,
                "instances": 0,
                "precision": None,
                "recall": None,
                "f1": None,
                "map50": None,
                "map50_95": None,
            }
            for class_name in class_names
        ],
        "macro_f1": None,
        "weighted_f1": None,
        "roc_auc": {
            "mode": "not_available",
            "split": "val",
            "classes": [],
            "note": "ROC-AUC is not generated by the RF-DETR training runner.",
        },
    }
    (run_dir / "web_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    class_names = read_class_names(dataset_dir)
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
        "dataset_dir": str(dataset_dir),
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
    }
    if args.resume and (run_dir / "checkpoint.pth").is_file():
        train_kwargs["resume"] = str(run_dir / "checkpoint.pth")

    model.train(**supported_train_kwargs(model, train_kwargs))
    print(f"{WEB_PROGRESS_PREFIX} epoch={args.epochs} total={args.epochs}", flush=True)
    normalize_checkpoints(run_dir)
    write_results_csv(run_dir, args.epochs)
    write_web_metrics(run_dir, args.model, class_names)


if __name__ == "__main__":
    main()
