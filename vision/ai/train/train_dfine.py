#!/usr/bin/env python3
"""Train a D-FINE object detector for the web UI."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vision.ai.train.yolo_to_coco import convert_yolo_to_coco, normalize_names, read_dataset_yaml


WEB_PROGRESS_PREFIX = "WEB_TRAINING_PROGRESS"
DFINE_RUN_LOG = "dfine_training.log"
MODEL_CONFIGS = {
    "dfine-n": "configs/dfine/custom/dfine_hgnetv2_n_custom.yml",
}
RESULT_FIELDNAMES = [
    "epoch",
    "train/loss",
    "val/loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]
METRIC_ALIASES = {
    "epoch": ["epoch", "step", "global_step"],
    "train_loss": ["train/loss", "train_loss", "loss", "loss_epoch", "train/total_loss"],
    "val_loss": ["val/loss", "val_loss", "validation/loss", "val/total_loss"],
    "precision": ["precision", "val/precision", "metrics/precision(B)"],
    "recall": ["recall", "val/recall", "metrics/recall(B)"],
    "map50": ["map50", "mAP50", "AP50", "val/mAP50", "metrics/mAP50(B)"],
    "map50_95": ["map", "mAP", "mAP50-95", "AP", "val/mAP", "metrics/mAP50-95(B)"],
}
COCO_AP_RE = re.compile(
    r"Average Precision\s+\(AP\)\s+@\[\s*IoU=(?P<iou>0\.50:0\.95|0\.50)\s*\|[^\]]+\]\s*=\s*(?P<value>-?\d+(?:\.\d+)?)"
)
EPOCH_RE = re.compile(r"(?:epoch|Epoch)\D+(\d+)(?:\D+(\d+))?")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a D-FINE detection model.")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml or dataset directory.")
    parser.add_argument("--model", default="dfine-n", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr0", type=float, default=0.0004)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--cos-lr", action="store_true", default=False)
    parser.add_argument("--warmup-epochs", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="runs/dfine")
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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def dfine_repo_dir() -> Path:
    configured = os.getenv("DFINE_REPO_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    image_checkout = Path("/opt/D-FINE")
    if image_checkout.is_dir():
        return image_checkout
    return repo_root() / "third_party" / "D-FINE"


def require_dfine_repo(path: Path):
    train_py = path / "train.py"
    if not train_py.is_file():
        raise SystemExit(
            "D-FINE repository was not found. Set DFINE_REPO_DIR to an official D-FINE checkout "
            "or vendor it under third_party/D-FINE."
        )
    config_path = path / MODEL_CONFIGS["dfine-n"]
    if not config_path.is_file():
        raise SystemExit(f"D-FINE config is missing: {config_path}")


def class_names_from_yaml(data: str) -> list[str]:
    _yaml_path, payload = read_dataset_yaml(Path(data))
    names = normalize_names(payload.get("names"))
    return [name for _id, name in sorted(names.items())]


def normalize_device(device: str | None) -> dict[str, str]:
    requested = str(device or "").strip()
    env: dict[str, str] = {}
    if not requested:
        return env
    normalized = requested.lower()
    if normalized == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif all(part.strip().isdigit() for part in requested.split(",") if part.strip()):
        env["CUDA_VISIBLE_DEVICES"] = requested
    elif normalized.startswith("cuda:") and normalized.split(":", 1)[1].isdigit():
        env["CUDA_VISIBLE_DEVICES"] = normalized.split(":", 1)[1]
    return env


def write_dfine_config(
    run_dir: Path,
    dfine_root: Path,
    coco_dir: Path,
    class_count: int,
    args,
) -> Path:
    config_path = run_dir / "dfine_web_config.yml"
    base_config = (dfine_root / MODEL_CONFIGS[args.model]).resolve()
    train_ann = coco_dir / "annotations" / "instances_train2017.json"
    val_ann = coco_dir / "annotations" / "instances_val2017.json"
    payload: dict[str, Any] = {
        "__include__": [str(base_config)],
        "output_dir": str(run_dir),
        "num_classes": int(class_count),
        "remap_mscoco_category": False,
        "epochs": int(args.epochs),
        "train_dataloader": {
            "total_batch_size": max(1, int(args.batch)),
            "num_workers": max(0, int(args.workers)),
            "dataset": {
                "img_folder": str(coco_dir / "train2017"),
                "ann_file": str(train_ann),
                "transforms": {
                    "ops": [
                        {"type": "Resize", "size": [int(args.imgsz), int(args.imgsz)]},
                    ],
                },
            },
            "collate_fn": {"base_size": int(args.imgsz)},
        },
        "val_dataloader": {
            "total_batch_size": max(1, int(args.batch)),
            "num_workers": max(0, int(args.workers)),
            "dataset": {
                "img_folder": str(coco_dir / "val2017"),
                "ann_file": str(val_ann),
                "transforms": {
                    "ops": [
                        {"type": "Resize", "size": [int(args.imgsz), int(args.imgsz)]},
                    ],
                },
            },
            "collate_fn": {"base_size": int(args.imgsz)},
        },
        "optimizer": {
            "type": "AdamW",
            "lr": float(args.lr0),
            "betas": [0.9, 0.999],
            "weight_decay": float(args.weight_decay),
        },
        "eval_spatial_size": [int(args.imgsz), int(args.imgsz)],
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def run_dfine_training(dfine_root: Path, config_path: Path, run_dir: Path, args) -> int:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        "train.py",
        "-c",
        str(config_path),
        "--use-amp",
        "--seed",
        str(int(args.seed)),
    ]
    resume_checkpoint = run_dir / "weights" / "last.pt"
    if args.resume and resume_checkpoint.is_file():
        command.extend(["-r", str(resume_checkpoint)])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.update(normalize_device(args.device))
    print("D-FINE command: " + " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=dfine_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )
    latest_epoch = 0
    if process.stdout is not None:
        for line in process.stdout:
            clean = line.rstrip("\n")
            print(clean, flush=True)
            match = EPOCH_RE.search(clean)
            if match is not None:
                try:
                    epoch = int(match.group(1))
                except (TypeError, ValueError):
                    epoch = 0
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    print(f"{WEB_PROGRESS_PREFIX} epoch={min(epoch, args.epochs)} total={args.epochs}", flush=True)
    return process.wait()


def copy_if_exists(source: Path, destination: Path):
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def normalize_checkpoints(run_dir: Path):
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    if (weights_dir / "best.pt").is_file() and (weights_dir / "last.pt").is_file():
        return

    best_patterns = (
        "checkpoint_best*.pth",
        "best*.pth",
        "**/checkpoint_best*.pth",
        "**/best*.pth",
    )
    last_patterns = (
        "checkpoint.pth",
        "last*.pth",
        "**/checkpoint.pth",
        "**/last*.pth",
    )
    for pattern in best_patterns:
        candidate = max(run_dir.glob(pattern), key=lambda path: path.stat().st_mtime, default=None)
        if candidate is not None:
            copy_if_exists(candidate, weights_dir / "best.pt")
            break
    for pattern in last_patterns:
        candidate = max(run_dir.glob(pattern), key=lambda path: path.stat().st_mtime, default=None)
        if candidate is not None:
            copy_if_exists(candidate, weights_dir / "last.pt")
            break
    if not (weights_dir / "last.pt").is_file():
        latest = max(run_dir.glob("**/*.pth"), key=lambda path: path.stat().st_mtime, default=None)
        if latest is not None:
            copy_if_exists(latest, weights_dir / "last.pt")
    if not (weights_dir / "best.pt").is_file() and (weights_dir / "last.pt").is_file():
        copy_if_exists(weights_dir / "last.pt", weights_dir / "best.pt")


def float_value(value):
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


def normalized_metric(row: dict, keys: list[str]):
    lookup = normalized_row_lookup(row)
    for key in keys:
        value = float_value(lookup.get(key) if key in lookup else lookup.get(key.lower()))
        if value is not None:
            return value
    return None


def find_metric_rows(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in run_dir.rglob("*.csv"):
        if path.name == "results.csv" or not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if any(normalized_metric(row, aliases) is not None for aliases in METRIC_ALIASES.values()):
                        rows.append(row)
        except (OSError, csv.Error):
            continue
    return rows


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


def parse_dfine_log_metrics(log_path: Path | None) -> dict[str, Any]:
    if log_path is None or not log_path.is_file():
        return {}
    latest: dict[str, Any] = {}
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    for line in lines:
        match = COCO_AP_RE.search(line)
        if match is None:
            continue
        value = float_value(match.group("value"))
        if value is None or value < 0:
            continue
        if match.group("iou") == "0.50":
            latest["metrics/mAP50(B)"] = value
        else:
            latest["metrics/mAP50-95(B)"] = value
    return latest


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


def write_results_csv(run_dir: Path, epochs: int, log_path: Path | None = None) -> str:
    rows = normalize_csv_metric_rows(find_metric_rows(run_dir))
    source = "csv"
    log_metrics = parse_dfine_log_metrics(log_path)
    if not rows:
        rows = [placeholder_result_row(epochs)]
        source = "placeholder"
    if log_metrics:
        rows[-1].update({key: value for key, value in log_metrics.items() if value is not None})
        source = "dfine_log" if source == "placeholder" else f"{source}+dfine_log"

    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in RESULT_FIELDNAMES})
    return source


def rounded_metric(value):
    number = float_value(value)
    return round(number, 4) if number is not None else None


def latest_results_row(run_dir: Path) -> dict:
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        return {}
    try:
        with results_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    except (OSError, csv.Error):
        return {}
    return rows[-1] if rows else {}


def write_web_metrics(
    run_dir: Path,
    model_id: str,
    class_names: list[str],
    conversion_summary: dict[str, Any] | None = None,
    training_completed: bool | None = None,
):
    row = latest_results_row(run_dir)
    payload = {
        "backend": "dfine",
        "model": model_id,
        "overall": {
            "precision": rounded_metric(row.get("metrics/precision(B)")),
            "recall": rounded_metric(row.get("metrics/recall(B)")),
            "f1": None,
            "map50": rounded_metric(row.get("metrics/mAP50(B)")),
            "map50_95": rounded_metric(row.get("metrics/mAP50-95(B)")),
            "source": "results.csv",
        },
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
        "per_class_source": "not_available_from_dfine_runner",
        "per_class_note": "D-FINE web metrics are normalized from available training artifacts. Per-class metrics are not parsed by this runner yet.",
        "macro_f1": None,
        "weighted_f1": None,
        "training_completed": training_completed,
        "roc_auc": {
            "mode": "not_available",
            "split": "val",
            "classes": [],
            "note": "ROC-AUC is not generated by the D-FINE training runner.",
        },
        "conversion_summary": conversion_summary,
    }
    (run_dir / "web_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def finalize_dfine_artifacts(
    run_dir: Path,
    model_id: str = "dfine-n",
    class_names: list[str] | None = None,
    epochs: int = 1,
    log_path: Path | None = None,
    conversion_summary: dict[str, Any] | None = None,
    training_completed: bool | None = None,
    quiet: bool = False,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    normalize_checkpoints(run_dir)
    source = write_results_csv(run_dir, epochs, log_path)
    write_web_metrics(run_dir, model_id, class_names or [], conversion_summary, training_completed)
    if not quiet:
        print(f"D-FINE web results saved to {run_dir / 'results.csv'} from {source}.", flush=True)
        print(f"D-FINE web metrics saved to {run_dir / 'web_metrics.json'}.", flush=True)


def main():
    args = parse_args()
    run_dir = (Path(args.project).expanduser() / args.name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = run_dir / DFINE_RUN_LOG
    class_names: list[str] = []
    conversion_summary: dict[str, Any] | None = None
    training_completed = False

    with tee_output(run_log_path):
        print(f"Logging results to {run_dir}", flush=True)
        print(f"{WEB_PROGRESS_PREFIX} epoch=1 total={args.epochs}", flush=True)
        try:
            class_names = class_names_from_yaml(args.data)
            coco_result = convert_yolo_to_coco(
                args.data,
                output_root=run_dir / "datasets" / "coco",
                dataset_name="prepared",
                link_images=True,
            )
            conversion_summary = coco_result.summary
            dfine_root = dfine_repo_dir()
            require_dfine_repo(dfine_root)
            config_path = write_dfine_config(run_dir, dfine_root, coco_result.output_dir, len(class_names), args)
            return_code = run_dfine_training(dfine_root, config_path, run_dir, args)
            if return_code != 0:
                raise SystemExit(f"D-FINE training failed with exit code {return_code}.")
            training_completed = True
            print(f"{WEB_PROGRESS_PREFIX} epoch={args.epochs} total={args.epochs}", flush=True)
        finally:
            finalize_dfine_artifacts(
                run_dir,
                model_id=args.model,
                class_names=class_names,
                epochs=args.epochs,
                log_path=run_log_path,
                conversion_summary=conversion_summary,
                training_completed=training_completed,
            )


if __name__ == "__main__":
    main()
