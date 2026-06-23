#!/usr/bin/env python3
"""Evaluate a YOLOv8 detection model on a labeled test split."""

import argparse
import json
import os
from pathlib import Path

from train_yolov8 import IMAGE_EXTENSIONS, use_actual_confusion_matrix_axis_label


WEB_TEST_PROGRESS_PREFIX = "WEB_TEST_PROGRESS"
WEB_TEST_RUN_DIR_PREFIX = "WEB_TEST_RUN_DIR"


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
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def build_per_class_metrics(metrics) -> dict:
    if metrics is None or not hasattr(metrics, "summary"):
        return {"macro_f1": None, "weighted_f1": None, "per_class": []}

    classes = []
    for row in metrics.summary():
        precision = rounded_metric(row.get("Box-P"))
        recall = rounded_metric(row.get("Box-R"))
        f1 = rounded_metric(row.get("Box-F1"))
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
                "map50": rounded_metric(row.get("mAP50")),
                "map50_95": rounded_metric(row.get("mAP50-95")),
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


def normalize_class_names(names) -> dict[int, str]:
    if isinstance(names, dict):
        normalized = {}
        for key, value in names.items():
            try:
                normalized[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return normalized
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


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


def plot_roc_auc_curves(output_path: Path, curves: list[dict], split_name: str):
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

    plt.title(f"{split_name.capitalize()} ROC-AUC by Class")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def build_image_level_roc_auc(
    run_dir: Path,
    weights_path: Path,
    data_config: dict,
    split_name: str,
    imgsz: int,
    batch: int,
    device: str | None,
) -> dict:
    names = normalize_class_names(data_config.get("names"))
    if not names:
        return {
            "mode": "image_presence",
            "split": split_name,
            "classes": [],
            "note": "ROC-AUC is unavailable because class names were not found in the dataset config.",
        }

    split_entries = resolve_dataset_entries(data_config, split_name)
    image_paths = collect_split_images(split_entries)
    if not image_paths:
        return {
            "mode": "image_presence",
            "split": split_name,
            "classes": [],
            "note": f"ROC-AUC is unavailable because no {split_name} images were found.",
        }

    from sklearn.metrics import auc, roc_curve
    from ultralytics import YOLO

    predictor = YOLO(str(weights_path))
    batch_size = batch if isinstance(batch, int) and batch > 0 else 16

    y_true_by_class = {class_id: [] for class_id in names}
    y_score_by_class = {class_id: [] for class_id in names}
    prediction_stream = predictor.predict(
        source=[str(path) for path in image_paths],
        stream=True,
        imgsz=imgsz,
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
        plot_roc_auc_curves(run_dir / "roc_auc_curve.png", curves, split_name)

    return {
        "mode": "image_presence",
        "split": split_name,
        "classes": summary,
        "note": f"ROC-AUC is calculated per class from {split_name} image-level class presence using the evaluated checkpoint.",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv8 model on a test split.")
    parser.add_argument("--weights", required=True, help="Path to the model weights (.pt).")
    parser.add_argument("--data", required=True, help="Path to the dataset YAML file.")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader workers.")
    parser.add_argument("--device", default=None, help="Evaluation device, for example 0 or cpu.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    parser.add_argument("--project", default="runs/test", help="Directory where test runs are saved.")
    parser.add_argument("--name", default="test", help="Name for this test run.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing the output folder.")
    return parser.parse_args()


def main():
    args = parse_args()
    weights_path = Path(args.weights).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Install it with: pip install ultralytics"
        ) from exc

    report_progress(5, "initializing", "Loading weights and dataset config.")
    model = YOLO(str(weights_path))
    report_progress(15, "evaluating", f"Running evaluation on the {args.split} split.")
    with use_actual_confusion_matrix_axis_label():
        metrics = model.val(
            data=str(data_path),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            plots=True,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            verbose=True,
        )

    run_dir = Path(getattr(metrics, "save_dir", Path(args.project) / args.name)).expanduser().resolve()
    report_run_dir(run_dir)
    report_progress(85, "saving_metrics", "Building per-class metrics and ROC-AUC artifacts.")

    data_config = getattr(metrics, "data", None)
    if not isinstance(data_config, dict):
        try:
            import yaml

            data_config = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
        except Exception:
            data_config = {}

    payload = build_per_class_metrics(metrics)
    payload.update(
        {
            "precision": rounded_metric(getattr(metrics.box, "mp", None)),
            "recall": rounded_metric(getattr(metrics.box, "mr", None)),
            "map50": rounded_metric(getattr(metrics.box, "map50", None)),
            "map50_95": rounded_metric(getattr(metrics.box, "map", None)),
            "split": args.split,
            "weights": str(weights_path),
            "dataset_yaml": str(data_path),
            "run_dir": str(run_dir),
        }
    )

    try:
        payload["roc_auc"] = build_image_level_roc_auc(
            run_dir=run_dir,
            weights_path=weights_path,
            data_config=data_config,
            split_name=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
    except Exception as exc:
        print(f"Could not generate ROC-AUC artifacts: {exc}", flush=True)
        payload["roc_auc"] = {
            "mode": "image_presence",
            "split": args.split,
            "classes": [],
            "note": f"ROC-AUC is unavailable: {exc}",
        }

    output_path = run_dir / "test_metrics.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved test metrics to {output_path}", flush=True)
    report_progress(100, "complete", "Model testing complete.")


if __name__ == "__main__":
    main()
