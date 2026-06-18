#!/usr/bin/env python3
"""Train a YOLOv8 object detection model with Ultralytics."""

import argparse
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
    "project": "runs/detect",
    "name": "train",
    "resume": False,
}


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

    model = YOLO(config["model"])
    train_kwargs = {
        "data": str(data_path),
        "epochs": config["epochs"],
        "imgsz": config["imgsz"],
        "batch": config["batch"],
        "patience": config["patience"],
        "save_period": config["save_period"],
        "workers": config["workers"],
        "project": config["project"],
        "name": config["name"],
        "resume": config["resume"],
    }

    if config["device"] is not None:
        train_kwargs["device"] = config["device"]

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
