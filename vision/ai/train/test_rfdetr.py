#!/usr/bin/env python3
"""Evaluate an RF-DETR model on a labeled YOLO test split for the web UI."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import yaml

try:
    from train_rfdetr import (
        MODEL_CLASSES,
        class_name_for_id,
        infer_label_dir,
        read_class_names,
        resolve_yaml_dataset_root,
    )
except ModuleNotFoundError:
    from vision.ai.train.train_rfdetr import (
        MODEL_CLASSES,
        class_name_for_id,
        infer_label_dir,
        read_class_names,
        resolve_yaml_dataset_root,
    )


WEB_TEST_PROGRESS_PREFIX = "WEB_TEST_PROGRESS"
WEB_TEST_RUN_DIR_PREFIX = "WEB_TEST_RUN_DIR"
IMAGE_EXTENSIONS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
IOU_THRESHOLDS = [round(0.5 + index * 0.05, 2) for index in range(10)]
BACKGROUND_LABEL = "background"


def report_progress(percent: int, stage: str, detail: str):
    print(
        f"{WEB_TEST_PROGRESS_PREFIX} percent={int(percent)} stage={stage} detail={detail}",
        flush=True,
    )


def report_run_dir(run_dir: Path):
    print(f"{WEB_TEST_RUN_DIR_PREFIX} path={run_dir}", flush=True)


def rounded_metric(value, digits: int = 4):
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return round(parsed, digits) if math.isfinite(parsed) else None


def synchronize_accelerator():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def seconds_since(start: float | None, digits: int = 3):
    if start is None:
        return None
    return rounded_metric(time.perf_counter() - start, digits)


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def model_id_from_weights(weights_path: Path) -> str:
    context = read_json(weights_path.parent.parent / "training_report_context.json")
    hyperparameters = context.get("hyperparameters") if isinstance(context.get("hyperparameters"), dict) else {}
    model_id = context.get("model") or hyperparameters.get("model") or hyperparameters.get("model_size")
    model_id = str(model_id or "rfdetr-nano").strip().lower()
    return model_id if model_id in MODEL_CLASSES else "rfdetr-nano"


def import_model_class(model_id: str):
    try:
        import rfdetr
    except ImportError as exc:
        raise SystemExit(
            "RF-DETR is not installed. Rebuild the Docker image after updating requirements.txt."
        ) from exc
    try:
        return getattr(rfdetr, MODEL_CLASSES[model_id])
    except AttributeError as exc:
        raise SystemExit(f"The installed rfdetr package does not provide {MODEL_CLASSES[model_id]}.") from exc


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


def resolve_split_image_entries(data_path: Path, data_config: dict, split_name: str) -> list[Path]:
    dataset_root = resolve_yaml_dataset_root(data_path, data_config)
    value = data_config.get(split_name)
    if value is None and split_name == "val":
        value = data_config.get("valid")
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    entries = []
    for item in values:
        candidate = Path(str(item)).expanduser()
        if not candidate.is_absolute():
            candidate = dataset_root / candidate
        entries.append(candidate.resolve())
    return entries


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
        index = len(parts) - 1 - parts[::-1].index("images")
        return Path(*parts[:index], "labels", *parts[index + 1 :]).with_suffix(".txt")
    try:
        return (infer_label_dir(image_path.parent) / image_path.name).with_suffix(".txt")
    except FileNotFoundError:
        return image_path.with_suffix(".txt")


def clip_box(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]


def read_yolo_labels(label_path: Path, width: int, height: int) -> list[dict]:
    if not label_path.is_file():
        return []
    rows = []
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = [float(value) for value in parts[1:5]]
        except ValueError:
            continue
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        rows.append({"class_id": class_id, "box": clip_box([x1, y1, x2, y2], width, height)})
    return rows


def detection_arrays(detections):
    xyxy = getattr(detections, "xyxy", None)
    class_ids = getattr(detections, "class_id", None)
    confidences = getattr(detections, "confidence", None)
    return xyxy if xyxy is not None else [], class_ids if class_ids is not None else [], confidences if confidences is not None else []


def collect_predictions(detections, width: int, height: int) -> list[dict]:
    xyxy, class_ids, confidences = detection_arrays(detections)
    rows = []
    for index, coords in enumerate(xyxy):
        class_id = int(class_ids[index]) if index < len(class_ids) else -1
        confidence = float(confidences[index]) if index < len(confidences) else 0.0
        rows.append({
            "class_id": class_id,
            "confidence": confidence,
            "box": clip_box([float(value) for value in coords], width, height),
        })
    return rows


def color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = (
        (20, 145, 120),
        (220, 90, 70),
        (85, 120, 230),
        (230, 170, 45),
        (165, 95, 210),
        (70, 170, 210),
    )
    return palette[class_id % len(palette)]


def draw_boxes(image_bgr, rows: list[dict], class_names: list[str], include_confidence: bool):
    import cv2

    canvas = image_bgr.copy()
    height, width = canvas.shape[:2]
    line_width = max(2, round(min(width, height) / 220))
    font_scale = max(0.45, min(width, height) / 900)
    for row in rows:
        class_id = int(row.get("class_id", -1))
        x1, y1, x2, y2 = [int(round(value)) for value in row.get("box", [0, 0, 0, 0])]
        color = color_for_class(max(0, class_id))
        label = class_name_for_id(class_id, class_names)
        if include_confidence and row.get("confidence") is not None:
            label = f"{label} {float(row.get('confidence') or 0):.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_width)
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        text_w, text_h = text_size
        top = max(0, y1 - text_h - baseline - 4)
        cv2.rectangle(canvas, (x1, top), (min(width, x1 + text_w + 6), top + text_h + baseline + 4), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 3, top + text_h + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def make_contact_sheet(images, cell_size: int = 512):
    import cv2
    import numpy as np

    if not images:
        return None
    resized = []
    for image in images:
        height, width = image.shape[:2]
        scale = min(cell_size / max(1, width), cell_size / max(1, height))
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        canvas = np.full((cell_size, cell_size, 3), 245, dtype=np.uint8)
        thumb = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        y = (cell_size - new_height) // 2
        x = (cell_size - new_width) // 2
        canvas[y:y + new_height, x:x + new_width] = thumb
        resized.append(canvas)
    rows = []
    for offset in range(0, len(resized), 2):
        row = resized[offset:offset + 2]
        if len(row) == 1:
            row.append(np.full_like(row[0], 245))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def save_qualitative_artifacts(
    output_dir: Path,
    split_name: str,
    examples: list[dict],
    class_names: list[str],
):
    import cv2

    if not examples:
        return []
    pred_images = []
    label_images = []
    for example in examples[:4]:
        image = cv2.imread(str(example["image_path"]))
        if image is None:
            continue
        pred_images.append(draw_boxes(image, example.get("predictions") or [], class_names, True))
        label_images.append(draw_boxes(image, example.get("ground_truths") or [], class_names, False))
    written = []
    for suffix, images in (("pred", pred_images), ("labels", label_images)):
        sheet = make_contact_sheet(images)
        if sheet is None:
            continue
        path = output_dir / f"{split_name}_batch0_{suffix}.jpg"
        cv2.imwrite(str(path), sheet)
        written.append(str(path))
    return written


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions(predictions: list[dict], ground_truths: dict[int, list[dict]], iou_threshold: float) -> tuple[list[int], list[int]]:
    matched = {image_index: set() for image_index in ground_truths}
    tp = []
    fp = []
    for prediction in sorted(predictions, key=lambda row: row["confidence"], reverse=True):
        image_index = int(prediction["image_index"])
        candidates = ground_truths.get(image_index, [])
        best_index = None
        best_iou = 0.0
        for gt_index, target in enumerate(candidates):
            if gt_index in matched.setdefault(image_index, set()):
                continue
            iou = box_iou(prediction["box"], target["box"])
            if iou > best_iou:
                best_iou = iou
                best_index = gt_index
        if best_index is not None and best_iou >= iou_threshold:
            matched[image_index].add(best_index)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    return tp, fp


def confusion_matrix_counts(
    predictions_by_image: dict[int, list[dict]],
    ground_truths_by_image: dict[int, list[dict]],
    class_count: int,
    confidence_threshold: float,
    iou_threshold: float = 0.5,
) -> list[list[int]]:
    background_index = int(class_count)
    matrix = [[0 for _ in range(class_count + 1)] for _ in range(class_count + 1)]
    image_indexes = set(predictions_by_image) | set(ground_truths_by_image)
    for image_index in image_indexes:
        predictions = [
            row
            for row in predictions_by_image.get(image_index, [])
            if float(row.get("confidence") or 0.0) >= confidence_threshold
        ]
        predictions.sort(key=lambda row: float(row.get("confidence") or 0.0), reverse=True)
        targets = ground_truths_by_image.get(image_index, [])
        matched_targets = set()
        for prediction in predictions:
            best_index = None
            best_iou = 0.0
            for target_index, target in enumerate(targets):
                if target_index in matched_targets:
                    continue
                iou = box_iou(prediction["box"], target["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_index = target_index
            predicted_class = int(prediction.get("class_id", background_index))
            if predicted_class < 0 or predicted_class >= class_count:
                predicted_class = background_index
            if best_index is not None and best_iou >= iou_threshold:
                true_class = int(targets[best_index].get("class_id", background_index))
                if true_class < 0 or true_class >= class_count:
                    true_class = background_index
                matrix[true_class][predicted_class] += 1
                matched_targets.add(best_index)
            else:
                matrix[background_index][predicted_class] += 1
        for target_index, target in enumerate(targets):
            if target_index in matched_targets:
                continue
            true_class = int(target.get("class_id", background_index))
            if true_class < 0 or true_class >= class_count:
                true_class = background_index
            matrix[true_class][background_index] += 1
    return matrix


def save_confusion_matrix(matrix: list[list[int]], class_names: list[str], output_path: Path, normalize: bool = False) -> bool:
    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    labels = list(class_names) + [BACKGROUND_LABEL]
    values = np.array(matrix, dtype=float)
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        display_values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)
    else:
        display_values = values

    figure_size = max(6.0, min(12.0, 1.0 + len(labels) * 0.7))
    plt.figure(figsize=(figure_size, figure_size))
    plt.imshow(display_values, interpolation="nearest", cmap="Blues")
    plt.title("Normalized confusion matrix" if normalize else "Confusion matrix")
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    threshold = display_values.max() / 2.0 if display_values.size and display_values.max() > 0 else 0.0
    for row_index in range(display_values.shape[0]):
        for col_index in range(display_values.shape[1]):
            value = display_values[row_index, col_index]
            text = f"{value:.2f}" if normalize else str(int(value))
            plt.text(
                col_index,
                row_index,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )
    plt.ylabel("Actual class")
    plt.xlabel("Predicted class")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def average_precision(tp: list[int], fp: list[int], total_gt: int):
    if total_gt <= 0:
        return None
    if not tp:
        return 0.0
    cumulative_tp = []
    cumulative_fp = []
    tp_sum = 0
    fp_sum = 0
    for tp_value, fp_value in zip(tp, fp):
        tp_sum += tp_value
        fp_sum += fp_value
        cumulative_tp.append(tp_sum)
        cumulative_fp.append(fp_sum)
    recalls = [value / total_gt for value in cumulative_tp]
    precisions = [
        cumulative_tp[index] / max(1, cumulative_tp[index] + cumulative_fp[index])
        for index in range(len(cumulative_tp))
    ]
    ap = 0.0
    for step in range(101):
        recall_threshold = step / 100.0
        precision_at_recall = [
            precision for recall, precision in zip(recalls, precisions) if recall >= recall_threshold
        ]
        ap += max(precision_at_recall) if precision_at_recall else 0.0
    return ap / 101.0


def class_metrics(
    class_id: int,
    class_name: str,
    predictions: list[dict],
    ground_truths: dict[int, list[dict]],
    image_count_with_class: int,
    confidence_threshold: float,
) -> dict:
    class_predictions = [row for row in predictions if row["class_id"] == class_id]
    total_gt = sum(len(rows) for rows in ground_truths.values())
    ap_by_threshold = []
    for threshold in IOU_THRESHOLDS:
        tp, fp = match_predictions(class_predictions, ground_truths, threshold)
        ap_by_threshold.append(average_precision(tp, fp, total_gt))

    scoring_predictions = [row for row in class_predictions if row["confidence"] >= confidence_threshold]
    tp_50, fp_50 = match_predictions(scoring_predictions, ground_truths, 0.5)
    true_positives = sum(tp_50)
    false_positives = sum(fp_50)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else None
    recall = true_positives / total_gt if total_gt > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    valid_aps = [value for value in ap_by_threshold if value is not None]
    return {
        "class_id": class_id,
        "class_name": class_name,
        "images": image_count_with_class,
        "instances": total_gt,
        "precision": rounded_metric(precision),
        "recall": rounded_metric(recall),
        "f1": rounded_metric(f1),
        "map50": rounded_metric(ap_by_threshold[0]),
        "map50_95": rounded_metric(sum(valid_aps) / len(valid_aps)) if valid_aps else None,
    }


def summarize_metrics(per_class: list[dict]) -> dict:
    rows = [row for row in per_class if int(row.get("instances") or 0) > 0]
    if not rows:
        return {
            "precision": None,
            "recall": None,
            "map50": None,
            "map50_95": None,
            "macro_f1": None,
            "weighted_f1": None,
        }
    total_instances = sum(int(row.get("instances") or 0) for row in rows)

    def average(key: str):
        values = [row.get(key) for row in rows if row.get(key) is not None]
        return rounded_metric(sum(values) / len(values)) if values else None

    weighted_f1 = None
    weighted_values = [row for row in rows if row.get("f1") is not None]
    if weighted_values and total_instances > 0:
        weighted_f1 = sum((row["f1"] or 0) * int(row.get("instances") or 0) for row in weighted_values) / total_instances

    return {
        "precision": average("precision"),
        "recall": average("recall"),
        "map50": average("map50"),
        "map50_95": average("map50_95"),
        "macro_f1": average("f1"),
        "weighted_f1": rounded_metric(weighted_f1),
    }


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
        "note": "RF-DETR test timing is measured as wall-clock evaluation time for the web test workflow.",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an RF-DETR model on a test split.")
    parser.add_argument("--weights", required=True, help="Path to the RF-DETR weights (.pt/.pth).")
    parser.add_argument("--data", required=True, help="Path to the dataset YAML file.")
    parser.add_argument("--imgsz", type=int, default=512, help="Evaluation image size.")
    parser.add_argument("--batch", type=int, default=4, help="Accepted for UI compatibility; RF-DETR evaluates one image at a time.")
    parser.add_argument("--workers", type=int, default=2, help="Accepted for UI compatibility.")
    parser.add_argument("--device", default=None, help="Accepted for UI compatibility.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    parser.add_argument("--project", default="runs/test", help="Directory where test runs are saved.")
    parser.add_argument("--name", default="test", help="Name for this test run.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for precision/recall/F1.")
    return parser.parse_args()


def evaluate_rfdetr_split(
    weights_path: Path,
    data_path: Path,
    output_dir: Path,
    *,
    split: str = "test",
    conf: float = 0.25,
    metrics_filename: str = "test_metrics.json",
    emit_progress: bool = True,
) -> dict:
    total_start = time.perf_counter()
    weights_path = Path(weights_path).expanduser().resolve()
    data_path = Path(data_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    def progress(percent: int, stage: str, detail: str):
        if emit_progress:
            report_progress(percent, stage, detail)

    data_config = load_data_config(data_path)
    class_names = normalize_class_names(data_config.get("names")) or read_class_names(data_path.parent)
    if not class_names:
        raise RuntimeError("Class names were not found in the dataset YAML.")
    image_entries = resolve_split_image_entries(data_path, data_config, split)
    image_paths = collect_split_images(image_entries)
    if not image_paths:
        raise RuntimeError(f"No images were found for the {split} split.")

    progress(5, "initializing", "Loading RF-DETR weights and dataset config.")
    model_id = model_id_from_weights(weights_path)
    model = import_model_class(model_id)(pretrain_weights=str(weights_path))

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is required for RF-DETR test evaluation.") from exc

    predictions_by_class: dict[int, list[dict]] = defaultdict(list)
    gts_by_class: dict[int, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    predictions_by_image: dict[int, list[dict]] = defaultdict(list)
    gts_by_image: dict[int, list[dict]] = defaultdict(list)
    images_by_class: defaultdict[int, set[int]] = defaultdict(set)
    qualitative_examples = []

    progress(15, "evaluating", f"Running RF-DETR evaluation on the {split} split.")
    synchronize_accelerator()
    evaluation_start = time.perf_counter()
    prediction_threshold = 0.001
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
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(image_rgb, threshold=prediction_threshold)
        image_predictions = []
        for prediction in collect_predictions(detections, width, height):
            prediction["image_index"] = image_index
            predictions_by_class[int(prediction["class_id"])].append(prediction)
            predictions_by_image[image_index].append(prediction)
            image_predictions.append(prediction)
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
    progress(90, "saving_metrics", "Building RF-DETR metrics and report artifacts.")

    observed_class_ids = set(range(len(class_names))) | set(gts_by_class) | set(predictions_by_class)
    per_class = [
        class_metrics(
            class_id=class_id,
            class_name=class_name_for_id(class_id, class_names),
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
        "backend": "rfdetr",
        "model": model_id,
        "precision": summary["precision"],
        "recall": summary["recall"],
        "map50": summary["map50"],
        "map50_95": summary["map50_95"],
        "macro_f1": summary["macro_f1"],
        "weighted_f1": summary["weighted_f1"],
        "per_class": per_class,
        "split": split,
        "weights": str(weights_path),
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
            "note": "ROC-AUC is not generated by the RF-DETR evaluation runner.",
        },
        "timing": build_speed_payload(
            image_count=total_images,
            evaluation_seconds=evaluation_seconds,
            total_seconds=total_seconds,
        ),
        "note": "RF-DETR evaluation metrics are computed from RF-DETR predictions matched to YOLO labels.",
    }

    output_path = output_dir / metrics_filename
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved RF-DETR evaluation metrics to {output_path}", flush=True)
    progress(100, "complete", "RF-DETR evaluation artifacts complete.")
    return payload


def main():
    args = parse_args()
    weights_path = Path(args.weights).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    run_dir = (Path(args.project).expanduser() / args.name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    report_run_dir(run_dir)
    evaluate_rfdetr_split(
        weights_path,
        data_path,
        run_dir,
        split=args.split,
        conf=float(args.conf),
        metrics_filename="test_metrics.json",
        emit_progress=True,
    )


if __name__ == "__main__":
    main()
