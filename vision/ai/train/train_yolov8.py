#!/usr/bin/env python3
"""Train a YOLOv8 object detection model with Ultralytics."""

import argparse
import csv
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
    "patience": 50,
    "save_period": -1,
    "device": None,
    "workers": 8,
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "weight_decay": 0.0005,
    "cos_lr": False,
    "warmup_epochs": 3.0,
    "freeze": None,
    "pretrained": True,
    "activation": "silu",
    "exist_ok": False,
    "seed": 0,
    "project": "runs/detect",
    "name": "train",
    "resume": False,
}


def str_to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


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
    return parser.parse_args()


def get_training_config(args):
    config = TRAINING_CONFIG.copy()

    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    return config


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


def total_loss(row: dict, prefix: str):
    values = [
        row_float(row, f"{prefix}/box_loss"),
        row_float(row, f"{prefix}/cls_loss"),
        row_float(row, f"{prefix}/dfl_loss"),
    ]
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def f1_score(row: dict):
    precision = row_float(row, "metrics/precision(B)")
    recall = row_float(row, "metrics/recall(B)")
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

    accuracy_series = [
        ("mAP50", epochs, [row_float(row, "metrics/mAP50(B)") for row in rows]),
        ("mAP50-95", epochs, [row_float(row, "metrics/mAP50-95(B)") for row in rows]),
        ("F1", epochs, [f1_score(row) for row in rows]),
    ]
    loss_series = [
        ("Training loss", epochs, [total_loss(row, "train") for row in rows]),
        ("Validation loss", epochs, [total_loss(row, "val") for row in rows]),
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
        "Accuracy Metrics by Epoch",
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

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Install it with: pip install ultralytics"
        ) from exc

    set_activation(config["activation"])
    model = YOLO(config["model"])
    train_kwargs = {
        "data": str(data_path),
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
        "exist_ok": config["exist_ok"],
        "seed": config["seed"],
        "project": config["project"],
        "name": config["name"],
        "resume": config["resume"],
    }

    if config["freeze"] is not None:
        train_kwargs["freeze"] = config["freeze"]

    if config["device"] is not None:
        train_kwargs["device"] = config["device"]

    model.train(**train_kwargs)

    run_dir = Path(getattr(model.trainer, "save_dir", Path(config["project"]) / config["name"]))
    try:
        save_training_graphs(run_dir)
    except Exception as exc:
        print(f"Could not save training graphs: {exc}")


if __name__ == "__main__":
    main()
