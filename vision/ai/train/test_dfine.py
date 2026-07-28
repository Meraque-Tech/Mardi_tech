#!/usr/bin/env python3
"""Evaluate a D-FINE model on a labeled YOLO split for the web UI."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

try:
    from test_rfdetr import (
        class_metrics,
        collect_split_images,
        confusion_matrix_counts,
        image_label_path,
        read_yolo_labels,
        resolve_split_image_entries,
        rounded_metric,
        save_confusion_matrix,
        save_qualitative_artifacts,
        seconds_since,
        summarize_metrics,
    )
    from train_dfine import dfine_repo_dir
except ModuleNotFoundError:
    from vision.ai.train.test_rfdetr import (
        class_metrics,
        collect_split_images,
        confusion_matrix_counts,
        image_label_path,
        read_yolo_labels,
        resolve_split_image_entries,
        rounded_metric,
        save_confusion_matrix,
        save_qualitative_artifacts,
        seconds_since,
        summarize_metrics,
    )
    from vision.ai.train.train_dfine import dfine_repo_dir


WEB_TEST_PROGRESS_PREFIX = "WEB_TEST_PROGRESS"
WEB_TEST_RUN_DIR_PREFIX = "WEB_TEST_RUN_DIR"


def report_progress(percent: int, stage: str, detail: str):
    print(
        f"{WEB_TEST_PROGRESS_PREFIX} percent={int(percent)} stage={stage} detail={detail}",
        flush=True,
    )


def report_run_dir(run_dir: Path):
    print(f"{WEB_TEST_RUN_DIR_PREFIX} path={run_dir}", flush=True)


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_data_config(data_path: Path) -> dict:
    try:
        payload = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Could not read dataset YAML at {data_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Dataset YAML must contain a mapping: {data_path}")
    return payload


def normalize_class_names(names) -> list[str]:
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


def synchronize_accelerator():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def checkpoint_path_from_run(run_dir: Path) -> Path | None:
    for candidate in (run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"):
        if candidate.is_file():
            return candidate
    return None


def config_path_from_run(run_dir: Path) -> Path | None:
    candidate = run_dir / "dfine_web_config.yml"
    return candidate if candidate.is_file() else None


def model_id_from_run(run_dir: Path) -> str:
    context = read_json(run_dir / "training_report_context.json")
    hyperparameters = context.get("hyperparameters") if isinstance(context.get("hyperparameters"), dict) else {}
    return str(context.get("model") or hyperparameters.get("model") or hyperparameters.get("model_size") or "dfine-n")


def dfine_device(device: str | None) -> str:
    requested = str(device or "").strip()
    if not requested or requested == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if requested.lower().startswith("cuda"):
        return requested
    if requested.isdigit():
        return f"cuda:{requested}"
    return requested


def load_dfine_model(config_path: Path, checkpoint_path: Path, dfine_root: Path, device: str):
    import torch
    import torch.nn as nn

    dfine_root = Path(dfine_root).expanduser().resolve()
    if not (dfine_root / "train.py").is_file():
        raise RuntimeError(
            "D-FINE repository was not found. Set DFINE_REPO_DIR to an official D-FINE checkout."
        )
    if str(dfine_root) not in sys.path:
        sys.path.insert(0, str(dfine_root))

    try:
        from src.core import YAMLConfig
    except Exception as exc:
        raise RuntimeError(f"Could not import D-FINE YAMLConfig from {dfine_root}: {exc}") from exc

    cfg = YAMLConfig(str(config_path), resume=str(checkpoint_path))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = (checkpoint.get("ema") or {}).get("module") or checkpoint.get("model")
    if state is None:
        raise RuntimeError(f"D-FINE checkpoint does not contain ema.module or model state: {checkpoint_path}")
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model().to(device)
    model.eval()
    return model


def collect_dfine_predictions(model, image_bgr, device: str, imgsz: int, threshold: float = 0.001) -> list[dict]:
    import cv2
    import torch
    import torchvision.transforms as transforms
    from PIL import Image

    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    transform = transforms.Compose([
        transforms.Resize((int(imgsz), int(imgsz))),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    orig_size = torch.tensor([[width, height]], dtype=torch.float32, device=device)
    with torch.no_grad():
        labels, boxes, scores = model(image_tensor, orig_size)

    labels = labels[0].detach().cpu().tolist() if len(labels) else []
    boxes = boxes[0].detach().cpu().tolist() if len(boxes) else []
    scores = scores[0].detach().cpu().tolist() if len(scores) else []
    rows = []
    for class_id, box, confidence in zip(labels, boxes, scores):
        confidence = float(confidence)
        if confidence < threshold:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        rows.append({
            "class_id": int(class_id),
            "confidence": confidence,
            "box": [
                max(0.0, min(float(width), x1)),
                max(0.0, min(float(height), y1)),
                max(0.0, min(float(width), x2)),
                max(0.0, min(float(height), y2)),
            ],
        })
    return rows


def build_speed_payload(image_count: int, evaluation_seconds, total_seconds) -> dict:
    return {
        "image_count": image_count,
        "preprocess_ms_per_image": None,
        "inference_ms_per_image": None,
        "postprocess_ms_per_image": None,
        "model_pipeline_ms_per_image": None,
        "evaluation_seconds": rounded_metric(evaluation_seconds, 3),
        "roc_auc_seconds": None,
        "total_seconds": rounded_metric(total_seconds, 3),
        "evaluation_ms_per_image": rounded_metric((evaluation_seconds * 1000) / image_count, 3) if image_count else None,
        "total_ms_per_image": rounded_metric((total_seconds * 1000) / image_count, 3) if image_count else None,
        "note": "D-FINE timing is measured as wall-clock evaluation time for the web evaluation workflow.",
    }


def evaluate_dfine_split(
    run_dir: Path,
    data_path: Path,
    output_dir: Path,
    *,
    split: str = "val",
    conf: float = 0.25,
    imgsz: int = 640,
    device: str | None = None,
    metrics_filename: str = "validation_metrics.json",
    emit_progress: bool = True,
) -> dict:
    total_start = time.perf_counter()
    run_dir = Path(run_dir).expanduser().resolve()
    data_path = Path(data_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_path_from_run(run_dir)
    config_path = config_path_from_run(run_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(f"D-FINE checkpoint was not found in {run_dir / 'weights'}")
    if config_path is None:
        raise FileNotFoundError(f"D-FINE web config was not found in {run_dir}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    def progress(percent: int, stage: str, detail: str):
        if emit_progress:
            report_progress(percent, stage, detail)

    data_config = load_data_config(data_path)
    class_names = normalize_class_names(data_config.get("names"))
    if not class_names:
        raise RuntimeError("Class names were not found in the dataset YAML.")
    image_entries = resolve_split_image_entries(data_path, data_config, split)
    image_paths = collect_split_images(image_entries)
    if not image_paths:
        raise RuntimeError(f"No images were found for the {split} split.")

    progress(5, "initializing", "Loading D-FINE checkpoint and dataset config.")
    selected_device = dfine_device(device)
    model = load_dfine_model(config_path, checkpoint_path, dfine_repo_dir(), selected_device)

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is required for D-FINE evaluation.") from exc

    predictions_by_class: dict[int, list[dict]] = defaultdict(list)
    gts_by_class: dict[int, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    predictions_by_image: dict[int, list[dict]] = defaultdict(list)
    gts_by_image: dict[int, list[dict]] = defaultdict(list)
    images_by_class: defaultdict[int, set[int]] = defaultdict(set)
    qualitative_examples = []

    progress(15, "evaluating", f"Running D-FINE evaluation on the {split} split.")
    synchronize_accelerator()
    evaluation_start = time.perf_counter()
    total_images = len(image_paths)
    for image_index, image_path in enumerate(image_paths):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue
        height, width = image_bgr.shape[:2]
        image_targets = read_yolo_labels(image_label_path(image_path), width, height)
        for target in image_targets:
            class_id = int(target["class_id"])
            gts_by_class[class_id][image_index].append(target)
            gts_by_image[image_index].append(target)
            images_by_class[class_id].add(image_index)
        image_predictions = collect_dfine_predictions(model, image_bgr, selected_device, int(imgsz), threshold=0.001)
        for prediction in image_predictions:
            prediction["image_index"] = image_index
            predictions_by_class[int(prediction["class_id"])].append(prediction)
            predictions_by_image[image_index].append(prediction)
        if len(qualitative_examples) < 4 and (image_targets or image_predictions):
            qualitative_examples.append({
                "image_path": image_path,
                "ground_truths": image_targets,
                "predictions": image_predictions,
            })
        if image_index == 0 or image_index + 1 == total_images or (image_index + 1) % 10 == 0:
            percent = 15 + int(((image_index + 1) / total_images) * 70)
            progress(percent, "evaluating", f"Evaluated {image_index + 1} of {total_images} images.")

    synchronize_accelerator()
    evaluation_seconds = seconds_since(evaluation_start)
    progress(90, "saving_metrics", "Building D-FINE metrics and report artifacts.")

    observed_class_ids = set(range(len(class_names))) | set(gts_by_class) | set(predictions_by_class)
    per_class = [
        class_metrics(
            class_id=class_id,
            class_name=class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id),
            predictions=predictions_by_class.get(class_id, []),
            ground_truths=gts_by_class.get(class_id, {}),
            image_count_with_class=len(images_by_class.get(class_id, set())),
            confidence_threshold=float(conf),
        )
        for class_id in sorted(observed_class_ids)
    ]
    summary = summarize_metrics(per_class)
    matrix = confusion_matrix_counts(
        predictions_by_image,
        gts_by_image,
        class_count=len(class_names),
        confidence_threshold=float(conf),
    )
    raw_matrix_path = output_dir / "confusion_matrix.png"
    normalized_matrix_path = output_dir / "confusion_matrix_normalized.png"
    save_confusion_matrix(matrix, class_names, raw_matrix_path, normalize=False)
    save_confusion_matrix(matrix, class_names, normalized_matrix_path, normalize=True)
    qualitative_paths = save_qualitative_artifacts(output_dir, split, qualitative_examples, class_names)

    total_seconds = seconds_since(total_start)
    payload = {
        "backend": "dfine",
        "model": model_id_from_run(run_dir),
        "precision": summary["precision"],
        "recall": summary["recall"],
        "map50": summary["map50"],
        "map50_95": summary["map50_95"],
        "macro_f1": summary["macro_f1"],
        "weighted_f1": summary["weighted_f1"],
        "per_class": per_class,
        "split": split,
        "weights": str(checkpoint_path),
        "dataset_yaml": str(data_path),
        "run_dir": str(output_dir),
        "artifacts": {
            "confusion_matrix": str(raw_matrix_path) if raw_matrix_path.is_file() else "",
            "confusion_matrix_normalized": str(normalized_matrix_path) if normalized_matrix_path.is_file() else "",
            "qualitative_images": qualitative_paths,
        },
        "roc_auc": {
            "mode": "not_available",
            "split": split,
            "classes": [],
            "note": "ROC-AUC is not generated by the D-FINE evaluation runner.",
        },
        "timing": build_speed_payload(
            image_count=total_images,
            evaluation_seconds=evaluation_seconds,
            total_seconds=total_seconds,
        ),
        "note": "D-FINE evaluation metrics are computed from D-FINE predictions matched to YOLO labels.",
    }

    output_path = output_dir / metrics_filename
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved D-FINE evaluation metrics to {output_path}", flush=True)
    progress(100, "complete", "D-FINE evaluation artifacts complete.")
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a D-FINE model on a labeled split.")
    parser.add_argument("--run-dir", required=True, help="D-FINE training run directory.")
    parser.add_argument("--data", required=True, help="Path to the dataset YAML file.")
    parser.add_argument("--output-dir", default="", help="Directory where evaluation artifacts are saved.")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for precision/recall/F1.")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size.")
    parser.add_argument("--device", default=None, help="Device for D-FINE inference.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir
    report_run_dir(output_dir)
    evaluate_dfine_split(
        run_dir,
        Path(args.data),
        output_dir,
        split=args.split,
        conf=float(args.conf),
        imgsz=int(args.imgsz),
        device=args.device,
        metrics_filename="validation_metrics.json",
        emit_progress=True,
    )


if __name__ == "__main__":
    main()
