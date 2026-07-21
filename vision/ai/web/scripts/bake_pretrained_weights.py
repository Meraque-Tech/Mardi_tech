#!/usr/bin/env python3
"""Download pretrained model weights into image-local cache directories."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


YOLO_WEIGHTS = (
    "yolo26n.pt",
    "yolo26s.pt",
    "yolov8n.pt",
    "yolov8s.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    "yolov8n-seg.pt",
    "yolo26n-seg.pt",
)
SAM_WEIGHTS = (
    "sam2.1_s.pt",
    "sam2.1_t.pt",
)
RFDETR_WEIGHTS = (
    "rf-detr-nano.pth",
)


def log(message: str):
    print(f"[bake-weights] {message}", flush=True)


def configured_ultralytics_weights_dir() -> Path:
    weights_dir = Path(
        os.getenv("ULTRALYTICS_WEIGHTS_DIR")
        or Path.home() / ".cache" / "ultralytics" / "weights"
    ).expanduser()
    weights_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics.utils import SETTINGS
    except Exception as exc:
        raise RuntimeError("Ultralytics is not available; cannot bake YOLO/SAM weights.") from exc

    SETTINGS.update({"weights_dir": str(weights_dir)})
    log(f"Ultralytics weights_dir={weights_dir}")
    return weights_dir


def download_ultralytics_asset(name: str, weights_dir: Path) -> Path:
    from ultralytics.utils.downloads import attempt_download_asset

    target = weights_dir / name
    if target.is_file():
        log(f"{name}: already present at {target}")
        return target

    previous_cwd = Path.cwd()
    try:
        os.chdir(weights_dir)
        resolved = Path(attempt_download_asset(name))
    finally:
        os.chdir(previous_cwd)

    if not resolved.is_absolute():
        resolved = (weights_dir / resolved).resolve()
    if resolved.is_file() and resolved != target:
        shutil.copy2(resolved, target)
    if not target.is_file():
        raise FileNotFoundError(f"{name} was not downloaded to {target}")
    log(f"{name}: baked at {target}")
    return target


def bake_ultralytics_weights():
    weights_dir = configured_ultralytics_weights_dir()
    for name in (*YOLO_WEIGHTS, *SAM_WEIGHTS):
        download_ultralytics_asset(name, weights_dir)


def bake_rfdetr_weights():
    cache_dir = Path(
        os.getenv("RFDETR_CACHE_DIR")
        or Path.home() / ".roboflow" / "models"
    ).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / RFDETR_WEIGHTS[0]
    if target.is_file():
        log(f"rf-detr-nano.pth: already present at {target}")
        return

    try:
        import rfdetr
    except Exception as exc:
        raise RuntimeError("RF-DETR is not available; cannot bake RF-DETR weights.") from exc

    model_class = getattr(rfdetr, "RFDETRNano", None)
    if model_class is None:
        raise RuntimeError("The installed rfdetr package does not provide RFDETRNano.")

    log("rf-detr-nano.pth: downloading through RFDETRNano()")
    model_class()
    if not target.is_file():
        raise FileNotFoundError(f"RF-DETR did not create expected checkpoint at {target}")
    log(f"rf-detr-nano.pth: baked at {target}")


def main() -> int:
    bake_ultralytics_weights()
    bake_rfdetr_weights()
    log("Pretrained weight bake complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"failed: {exc}")
        raise
