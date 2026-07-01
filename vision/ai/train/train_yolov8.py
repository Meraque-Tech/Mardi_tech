#!/usr/bin/env python3
"""Train a YOLOv8 object detection model with Ultralytics."""

import argparse
import csv
from contextlib import contextmanager, nullcontext
from functools import wraps
import json
import os
from pathlib import Path


# Edit these defaults for normal training. Command-line flags can still
# override any value here when you want to run a one-off experiment.
TRAINING_CONFIG = {
    "data": "data/pineapple.yaml",
    "model": "yolov8n.pt",
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "patience": 20,
    "save_period": -1,
    "device": None,
    "workers": 2,
    "optimizer": "Adam",
    "lr0": 0.001,
    "lrf": 0.01,
    "weight_decay": 0.0005,
    "cos_lr": False,
    "warmup_epochs": 3.0,
    "freeze": None,
    "pretrained": True,
    "activation": "silu",
    "exist_ok": False,
    "seed": 42,
    "project": "runs/detect",
    "name": "train",
    "resume": False,
    "disable_ultralytics_albumentations": True,
}

# Apply the selected Ultralytics online augmentations during training. Keep
# every unlisted geometric or mixing transform explicitly disabled.
TRAINING_AUGMENTATIONS = {
    "mosaic": 1.0,
    "close_mosaic": 10,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "auto_augment": None,
    "erasing": 0.0,
}

WEB_PROGRESS_PREFIX = "WEB_TRAINING_PROGRESS"
IMAGE_EXTENSIONS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def confusion_matrix_axis_label(label):
    """Use clearer terminology for the ground-truth confusion-matrix axis."""
    return "Actual" if label == "True" else label


@contextmanager
def use_actual_confusion_matrix_axis_label():
    """Temporarily customize Ultralytics' matplotlib confusion-matrix label."""
    try:
        from matplotlib.axes import Axes
    except ImportError:
        yield
        return

    original_set_xlabel = Axes.set_xlabel

    @wraps(original_set_xlabel)
    def set_xlabel(axes, xlabel, *args, **kwargs):
        return original_set_xlabel(
            axes,
            confusion_matrix_axis_label(xlabel),
            *args,
            **kwargs,
        )

    Axes.set_xlabel = set_xlabel
    try:
        yield
    finally:
        if Axes.set_xlabel is set_xlabel:
            Axes.set_xlabel = original_set_xlabel


@contextmanager
def use_disabled_ultralytics_albumentations():
    """Prevent Ultralytics from adding its optional Albumentations defaults."""
    try:
        from ultralytics.data import augment as ultralytics_augment
    except ImportError:
        yield
        return

    original_albumentations = ultralytics_augment.Albumentations

    def empty_albumentations(*args, **kwargs):
        return ultralytics_augment.Compose([])

    ultralytics_augment.Albumentations = empty_albumentations
    try:
        yield
    finally:
        if ultralytics_augment.Albumentations is empty_albumentations:
            ultralytics_augment.Albumentations = original_albumentations


def report_epoch_start(trainer):
    """Emit a stable progress marker for the training web UI."""
    current_epoch = int(getattr(trainer, "epoch", 0)) + 1
    total_epochs = int(getattr(trainer, "epochs", current_epoch))
    print(
        f"{WEB_PROGRESS_PREFIX} epoch={current_epoch} total={total_epochs}",
        flush=True,
    )


def str_to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def none_or_text(value: str):
    normalized = value.strip()
    if normalized.lower() in {"", "none", "null"}:
        return None
    return normalized


def set_activation(name: str):
    normalized = (name or "silu").strip().lower().replace("-", "_")
    try:
        import torch.nn as nn
        from ultralytics.nn.modules import Conv
    except ImportError as exc:
        raise SystemExit(f"Could not configure activation function: {exc}") from exc

    activations = {
        "silu": nn.SiLU,
        "relu": nn.ReLU,
        "leaky_relu": lambda: nn.LeakyReLU(0.1, inplace=True),
        "mish": nn.Mish,
        "gelu": nn.GELU,
        "hardswish": nn.Hardswish,
    }
    if normalized not in activations:
        choices = ", ".join(sorted(activations))
        raise SystemExit(f"Unknown activation '{name}'. Choose one of: {choices}")

    Conv.default_act = activations[normalized]()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 image detection model."
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to the dataset YAML file. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Base YOLOv8 model or checkpoint. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Training image size. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size. Use -1 for Ultralytics auto-batch. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience in epochs. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=None,
        help="Save an extra checkpoint every N epochs. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Training device, for example 0, 0,1, cpu, or mps. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of dataloader workers. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument(
        "--optimizer",
        default=None,
        help="Optimizer to use, for example auto, SGD, Adam, or AdamW. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=None,
        help="Initial learning rate. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=None,
        help="Final learning-rate factor. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay regularization. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument(
        "--cos-lr",
        action="store_true",
        default=None,
        help="Use cosine learning-rate scheduling. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=None,
        help="Number of learning-rate warmup epochs. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Freeze the first N model layers. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument(
        "--pretrained",
        type=str_to_bool,
        default=None,
        help="Whether to use pretrained weights. Use true or false. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument(
        "--activation",
        default=None,
        help="Activation function for YOLO Conv layers: silu, relu, leaky_relu, mish, gelu, or hardswish. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        default=None,
        help="Allow reusing the same project/name output folder. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible training. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Directory where training runs are saved. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Name for this training run. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="Resume training from the configured or given checkpoint.",
    )
    parser.add_argument(
        "--disable-ultralytics-albumentations",
        type=str_to_bool,
        default=None,
        help="Disable Ultralytics' optional Albumentations defaults. Overrides TRAINING_CONFIG.",
    )
    parser.add_argument("--mosaic", type=float, default=None, help="Mosaic augmentation probability.")
    parser.add_argument("--close-mosaic", type=int, default=None, help="Disable mosaic for the final N epochs.")
    parser.add_argument("--hsv-h", type=float, default=None, help="HSV hue augmentation gain.")
    parser.add_argument("--hsv-s", type=float, default=None, help="HSV saturation augmentation gain.")
    parser.add_argument("--hsv-v", type=float, default=None, help="HSV value augmentation gain.")
    parser.add_argument("--degrees", type=float, default=None, help="Image rotation degrees.")
    parser.add_argument("--translate", type=float, default=None, help="Image translation fraction.")
    parser.add_argument("--scale", type=float, default=None, help="Image scale gain.")
    parser.add_argument("--shear", type=float, default=None, help="Image shear degrees.")
    parser.add_argument("--perspective", type=float, default=None, help="Image perspective gain.")
    parser.add_argument("--flipud", type=float, default=None, help="Vertical flip probability.")
    parser.add_argument("--fliplr", type=float, default=None, help="Horizontal flip probability.")
    parser.add_argument("--bgr", type=float, default=None, help="BGR channel swap probability.")
    parser.add_argument("--mixup", type=float, default=None, help="MixUp augmentation probability.")
    parser.add_argument("--cutmix", type=float, default=None, help="CutMix augmentation probability.")
    parser.add_argument("--copy-paste", type=float, default=None, help="Copy-paste augmentation probability.")
    parser.add_argument("--auto-augment", type=none_or_text, default=None, help="AutoAugment policy, or blank/none/null to disable.")
    parser.add_argument("--erasing", type=float, default=None, help="Random erasing probability.")
    return parser.parse_args()


def get_training_config(args):
    config = TRAINING_CONFIG.copy()

    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    return config


def get_training_augmentations(config):
    augmentations = TRAINING_AUGMENTATIONS.copy()
    for key in augmentations:
        if key in config and config[key] is not None:
            augmentations[key] = config[key]
    return augmentations


def row_float(row: dict, key: str):
    value = row.get(key)
    if value is None:
        value = row.get(f" {key}")
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def rounded_metric(value, digits: int = 4):
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def loss_components(row: dict, prefix: str):
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
        value = row_float(row, normalized)
        if value is not None:
            components[component] = value
    return components


def infer_task_from_results(rows: list[dict]) -> str:
    keys = {str(key).strip() for row in rows for key in row}
    if "metrics/mAP50(M)" in keys or "metrics/mAP50-95(M)" in keys:
        return "segment"
    if "metrics/accuracy_top1" in keys or "metrics/accuracy_top5" in keys:
        return "classify"
    return "detect"


def comparable_loss_component_names(task: str, row: dict):
    train = loss_components(row, "train")
    val = loss_components(row, "val")
    if task == "classify":
        candidates = ["loss"] if "loss" in train and "loss" in val else ["cls"]
    elif task == "segment":
        candidates = ["box", "seg", "cls", "dfl"]
    else:
        candidates = ["box", "cls", "dfl"]
    return [component for component in candidates if component in train and component in val]


def comparable_loss(row: dict, prefix: str, task: str):
    components = loss_components(row, prefix)
    names = comparable_loss_component_names(task, row)
    if not names:
        return None
    return sum(components[name] for name in names)


def auxiliary_loss(row: dict, prefix: str, task: str):
    components = loss_components(row, prefix)
    comparable_names = set(comparable_loss_component_names(task, row))
    values = [value for name, value in components.items() if name not in comparable_names]
    return sum(values) if values else None


def metric_suffix(rows: list[dict]) -> str:
    keys = {str(key).strip() for row in rows for key in row}
    if "metrics/mAP50(M)" in keys or "metrics/mAP50-95(M)" in keys:
        return "M"
    return "B"


def accuracy_metric_series(rows: list[dict], epochs: list[int]):
    keys = {str(key).strip() for row in rows for key in row}
    if "metrics/accuracy_top1" in keys or "metrics/accuracy_top5" in keys:
        return "Classification Metrics by Epoch", [
            ("Top-1 Accuracy", epochs, [row_float(row, "metrics/accuracy_top1") for row in rows]),
            ("Top-5 Accuracy", epochs, [row_float(row, "metrics/accuracy_top5") for row in rows]),
        ]

    suffix = metric_suffix(rows)
    metric_prefix = "Mask " if suffix == "M" else ""
    title = "Segmentation Mask Metrics by Epoch" if suffix == "M" else "Accuracy Metrics by Epoch"
    return title, [
        (f"{metric_prefix}mAP50", epochs, [row_float(row, f"metrics/mAP50({suffix})") for row in rows]),
        (f"{metric_prefix}mAP50-95", epochs, [row_float(row, f"metrics/mAP50-95({suffix})") for row in rows]),
        (f"{metric_prefix}F1", epochs, [f1_score(row, suffix) for row in rows]),
    ]


def f1_score(row: dict, suffix: str = "B"):
    precision = row_float(row, f"metrics/precision({suffix})")
    recall = row_float(row, f"metrics/recall({suffix})")
    if precision is None or recall is None or precision + recall <= 0:
        return None
    return 2 * precision * recall / (precision + recall)


def plot_metric_series(output_path: Path, title: str, xlabel: str, ylabel: str, series: list[tuple[str, list[int], list[float]]]):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    for label, epochs, values in series:
        if epochs and values:
            plt.plot(epochs, values, marker="o", linewidth=2, markersize=3, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_roc_auc_curves(output_path: Path, curves: list[dict]):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1.5, label="Chance")
    for curve in curves:
        plt.plot(
            curve["fpr"],
            curve["tpr"],
            linewidth=2,
            label=f'{curve["class_name"]} (AUC {curve["auc"]:.3f})',
        )

    plt.title("Validation ROC-AUC by Class")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def normalize_class_names(names) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def build_per_class_metrics(metrics) -> dict:
    if metrics is None or not hasattr(metrics, "summary"):
        return {"macro_f1": None, "weighted_f1": None, "per_class": []}

    classes = []
    for row in metrics.summary():
        use_mask = "Mask-P" in row or "Mask-R" in row or "Mask-F1" in row
        metric_prefix = "Mask" if use_mask else "Box"
        precision = rounded_metric(row.get(f"{metric_prefix}-P"))
        recall = rounded_metric(row.get(f"{metric_prefix}-R"))
        f1 = rounded_metric(row.get(f"{metric_prefix}-F1"))
        images = int(row.get("Images") or 0)
        instances = int(row.get("Instances") or 0)
        classes.append(
            {
                "class_name": str(row.get("Class", "")),
                "images": images,
                "instances": instances,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "map50": rounded_metric(row.get(f"{metric_prefix}-mAP50", row.get("mAP50"))),
                "map50_95": rounded_metric(row.get(f"{metric_prefix}-mAP50-95", row.get("mAP50-95"))),
            }
        )

    macro_f1 = None
    if classes:
        macro_f1 = sum(row["f1"] or 0 for row in classes) / len(classes)

    total_instances = sum(row["instances"] for row in classes)
    weighted_f1 = None
    if total_instances:
        weighted_f1 = sum((row["f1"] or 0) * row["instances"] for row in classes) / total_instances

    return {
        "macro_f1": rounded_metric(macro_f1),
        "weighted_f1": rounded_metric(weighted_f1),
        "per_class": classes,
    }


def resolve_dataset_entries(data_config: dict, split_name: str) -> list[Path]:
    base_path = Path(str(data_config.get("path") or ".")).expanduser()
    raw_value = data_config.get(split_name)
    if raw_value is None:
        return []
    values = raw_value if isinstance(raw_value, list) else [raw_value]
    resolved = []
    for value in values:
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = base_path / candidate
        resolved.append(candidate)
    return resolved


def collect_split_images(entries: list[Path]) -> list[Path]:
    images = []
    for entry in entries:
        if entry.is_dir():
            images.extend(
                path
                for path in sorted(entry.rglob("*"))
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            continue
        if entry.is_file() and entry.suffix.lower() == ".txt":
            for line in entry.read_text(encoding="utf-8", errors="replace").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                path = Path(stripped).expanduser()
                if not path.is_absolute():
                    path = entry.parent / path
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(path)
            continue
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(entry)
    unique_images = []
    seen = set()
    for path in images:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_images.append(resolved)
    return unique_images


def image_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        index = parts.index("images")
        return Path(*parts[:index], "labels", *parts[index + 1 :]).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def read_image_classes(label_path: Path) -> set[int]:
    if not label_path.is_file():
        return set()
    classes = set()
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        class_id = stripped.split(maxsplit=1)[0]
        try:
            classes.add(int(float(class_id)))
        except ValueError:
            continue
    return classes


def build_image_level_roc_auc(run_dir: Path, weights_path: Path, data_config: dict, config: dict) -> dict:
    names = normalize_class_names(data_config.get("names"))
    if not names:
        return {
            "mode": "image_presence",
            "split": "val",
            "classes": [],
            "note": "ROC-AUC is unavailable because class names were not found in the dataset config.",
        }

    val_entries = resolve_dataset_entries(data_config, "val")
    image_paths = collect_split_images(val_entries)
    if not image_paths:
        return {
            "mode": "image_presence",
            "split": "val",
            "classes": [],
            "note": "ROC-AUC is unavailable because no validation images were found.",
        }

    from sklearn.metrics import auc, roc_curve
    from ultralytics import YOLO

    device = config.get("device")
    predictor = YOLO(str(weights_path))
    batch_size = config.get("batch")
    if not isinstance(batch_size, int) or batch_size <= 0:
        batch_size = 16

    y_true_by_class = {class_id: [] for class_id in names}
    y_score_by_class = {class_id: [] for class_id in names}
    prediction_stream = predictor.predict(
        source=[str(path) for path in image_paths],
        stream=True,
        imgsz=config["imgsz"],
        conf=0.001,
        iou=0.7,
        batch=batch_size,
        device=device,
        verbose=False,
    )

    for image_path, result in zip(image_paths, prediction_stream):
        gt_classes = read_image_classes(image_label_path(image_path))
        scores = {class_id: 0.0 for class_id in names}
        boxes = getattr(result, "boxes", None)
        if boxes is not None and boxes.cls is not None and boxes.conf is not None:
            predicted_classes = boxes.cls.tolist()
            confidences = boxes.conf.tolist()
            for raw_class, raw_confidence in zip(predicted_classes, confidences):
                class_id = int(raw_class)
                if class_id not in scores:
                    continue
                confidence = float(raw_confidence)
                if confidence > scores[class_id]:
                    scores[class_id] = confidence
        for class_id in names:
            y_true_by_class[class_id].append(1 if class_id in gt_classes else 0)
            y_score_by_class[class_id].append(scores[class_id])

    curves = []
    summary = []
    for class_id, class_name in names.items():
        y_true = y_true_by_class[class_id]
        y_score = y_score_by_class[class_id]
        positives = sum(y_true)
        negatives = len(y_true) - positives
        entry = {
            "class_name": class_name,
            "positive_images": positives,
            "negative_images": negatives,
            "auc": None,
        }
        if positives == 0 or negatives == 0:
            summary.append(entry)
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_value = float(auc(fpr, tpr))
        entry["auc"] = rounded_metric(auc_value)
        summary.append(entry)
        curves.append(
            {
                "class_name": class_name,
                "auc": auc_value,
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
            }
        )

    if curves:
        plot_roc_auc_curves(run_dir / "roc_auc_curve.png", curves)

    return {
        "mode": "image_presence",
        "split": "val",
        "classes": summary,
        "note": "ROC-AUC is calculated per class from validation image-level class presence using the best checkpoint.",
    }


def save_web_metrics(run_dir: Path, metrics, data_config: dict, config: dict):
    payload = build_per_class_metrics(metrics)
    weights_path = run_dir / "weights" / "best.pt"
    if not weights_path.is_file():
        weights_path = run_dir / "weights" / "last.pt"

    if weights_path.is_file():
        try:
            payload["roc_auc"] = build_image_level_roc_auc(run_dir, weights_path, data_config, config)
        except Exception as exc:
            print(f"Could not generate ROC-AUC artifacts: {exc}")
            payload["roc_auc"] = {
                "mode": "image_presence",
                "split": "val",
                "classes": [],
                "note": f"ROC-AUC is unavailable: {exc}",
            }
    else:
        payload["roc_auc"] = {
            "mode": "image_presence",
            "split": "val",
            "classes": [],
            "note": "ROC-AUC is unavailable because no trained weights were found.",
        }

    output_path = run_dir / "web_metrics.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved web metrics to {output_path}")


def save_training_graphs(run_dir: Path):
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        print(f"No results.csv found for graph generation: {results_path}")
        return

    with results_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        print(f"No rows found in results.csv: {results_path}")
        return

    epochs = [int(row_float(row, "epoch") or index + 1) for index, row in enumerate(rows)]
    task = infer_task_from_results(rows)
    accuracy_title, accuracy_series = accuracy_metric_series(rows, epochs)
    loss_series = [
        ("Training loss", epochs, [comparable_loss(row, "train", task) for row in rows]),
        ("Validation loss", epochs, [comparable_loss(row, "val", task) for row in rows]),
    ]

    accuracy_series = [
        (label, [epoch for epoch, value in zip(epochs, values) if value is not None], [value for value in values if value is not None])
        for label, epochs, values in accuracy_series
    ]
    loss_series = [
        (label, [epoch for epoch, value in zip(epochs, values) if value is not None], [value for value in values if value is not None])
        for label, epochs, values in loss_series
    ]

    plot_metric_series(
        run_dir / "accuracy_by_epoch.png",
        accuracy_title,
        "Epoch",
        "Score",
        accuracy_series,
    )
    plot_metric_series(
        run_dir / "loss_by_epoch.png",
        "Loss by Epoch",
        "Epoch",
        "Loss",
        loss_series,
    )
    print(f"Saved accuracy graph to {run_dir / 'accuracy_by_epoch.png'}")
    print(f"Saved loss graph to {run_dir / 'loss_by_epoch.png'}")


def main():
    args = parse_args()
    config = get_training_config(args)
    data_path = Path(config["data"]).expanduser()

    if not config["resume"] and not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Install it with: pip install ultralytics"
        ) from exc

    set_activation(config["activation"])
    model = YOLO(config["model"])
    model.add_callback("on_train_epoch_start", report_epoch_start)
    train_kwargs = {
        "epochs": config["epochs"],
        "imgsz": config["imgsz"],
        "batch": config["batch"],
        "patience": config["patience"],
        "save_period": config["save_period"],
        "workers": config["workers"],
        "optimizer": config["optimizer"],
        "lr0": config["lr0"],
        "lrf": config["lrf"],
        "weight_decay": config["weight_decay"],
        "cos_lr": config["cos_lr"],
        "warmup_epochs": config["warmup_epochs"],
        "pretrained": config["pretrained"],
        "plots": True,
        "exist_ok": config["exist_ok"],
        "seed": config["seed"],
        "project": config["project"],
        "name": config["name"],
        "resume": config["resume"],
        **get_training_augmentations(config),
    }

    if not config["resume"]:
        train_kwargs["data"] = str(data_path)

    if config["freeze"] is not None:
        train_kwargs["freeze"] = config["freeze"]

    if config["device"] is not None:
        train_kwargs["device"] = config["device"]

    albumentations_context = (
        use_disabled_ultralytics_albumentations()
        if config["disable_ultralytics_albumentations"]
        else nullcontext()
    )
    with albumentations_context, use_actual_confusion_matrix_axis_label():
        metrics = model.train(**train_kwargs)

    run_dir = Path(getattr(model.trainer, "save_dir", Path(config["project"]) / config["name"]))
    try:
        save_training_graphs(run_dir)
    except Exception as exc:
        print(f"Could not save training graphs: {exc}")
    try:
        save_web_metrics(run_dir, metrics, getattr(model.trainer, "data", {}), config)
    except Exception as exc:
        print(f"Could not save web metrics: {exc}")


if __name__ == "__main__":
    main()
