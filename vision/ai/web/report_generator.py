"""PDF report generation for completed model training and test runs."""

from __future__ import annotations

import html
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from .dataset_provenance import roboflow_pre_augmentation_summary


MYT = timezone(timedelta(hours=8), name="MYT")


def _text(value) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    if isinstance(value, float) and value.is_integer() and abs(value) >= 1000:
        return f"{int(value):,}"
    return html.escape(str(value))


def _metric(value) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _milliseconds(value) -> str:
    number = _to_float(value)
    if number is None:
        return "N/A"
    return f"{number:.1f} ms/image"


def _seconds(value) -> str:
    number = _to_float(value)
    if number is None:
        return "N/A"
    if number >= 60:
        minutes = int(number // 60)
        seconds = number - (minutes * 60)
        return f"{minutes}m {seconds:.1f}s"
    return f"{number:.1f}s"


def _test_timing_rows(timing: dict | None) -> list[list]:
    timing = timing or {}
    return [
        ["Test images", _count(timing.get("image_count"))],
        ["Inference time", _milliseconds(timing.get("inference_ms_per_image"))],
        ["Preprocess time", _milliseconds(timing.get("preprocess_ms_per_image"))],
        ["Postprocess time", _milliseconds(timing.get("postprocess_ms_per_image"))],
        ["Model pipeline time", _milliseconds(timing.get("model_pipeline_ms_per_image"))],
        ["Evaluation processing time", _milliseconds(timing.get("evaluation_ms_per_image"))],
        ["Total processing time", _milliseconds(timing.get("total_ms_per_image"))],
        ["Evaluation duration", _seconds(timing.get("evaluation_seconds"))],
        ["ROC-AUC duration", _seconds(timing.get("roc_auc_seconds"))],
        ["Total test duration", _seconds(timing.get("total_seconds"))],
    ]


def _metric_labels(metrics: dict | None) -> dict:
    labels = {
        "precision": "Precision",
        "recall": "Recall",
        "map50": "mAP50",
        "map50_95": "mAP50-95",
    }
    if isinstance(metrics, dict) and isinstance(metrics.get("metric_labels"), dict):
        labels.update({
            key: value
            for key, value in metrics["metric_labels"].items()
            if key in labels and value
        })
    return labels


def _report_family(context: dict | None = None, metrics: dict | None = None) -> str:
    family = ""
    if isinstance(metrics, dict):
        family = str(metrics.get("backend") or "").strip().lower()
    if not family and isinstance(context, dict):
        family = str(context.get("family") or "").strip().lower()
    return family if family in {"dfine", "rfdetr", "ultralytics"} else "ultralytics"


def _backend_display_name(family: str) -> str:
    return {
        "dfine": "D-FINE",
        "rfdetr": "RF-DETR",
        "ultralytics": "Ultralytics YOLO",
    }.get(family, "Model")


def _exported_training_images_label() -> str:
    return "Exported images used for training"


def _count(value) -> str:
    number = _to_int(value)
    if number is None:
        return _text(value)
    return f"{number:,}"


def _human_datetime_myt(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=MYT)
    else:
        value = value.astimezone(MYT)
    date_text = value.strftime("%d %B %Y")
    time_text = value.strftime("%I:%M:%S %p").lstrip("0")
    return f"{date_text}, {time_text} (MYT, GMT+8)"


def _format_myt(value) -> str:
    if not value:
        return "N/A"
    if isinstance(value, datetime):
        return _human_datetime_myt(value)
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return str(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=MYT)
    else:
        parsed = parsed.astimezone(MYT)
    return _human_datetime_myt(parsed)


def _run_timing(run_dir: Path, context: dict) -> tuple[str, str, str]:
    started = context.get("last_started_at") or context.get("created_at") or "N/A"
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        return _format_myt(started), "N/A", "N/A"
    completed = datetime.fromtimestamp(results_path.stat().st_mtime, tz=MYT)
    duration = "N/A"
    formatted_started = _format_myt(started)
    try:
        start_time = datetime.fromisoformat(str(started))
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=MYT)
        else:
            start_time = start_time.astimezone(MYT)
        formatted_started = _human_datetime_myt(start_time)
        seconds = max(0, int((completed - start_time).total_seconds()))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration = f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    except (TypeError, ValueError):
        pass
    return formatted_started, _human_datetime_myt(completed), duration


def _load_yaml(path: Path) -> dict:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _to_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _to_int(value) -> int | None:
    number = _to_float(value)
    if number is None:
        return None
    return int(number)


def _nonempty(value) -> bool:
    return value is not None and value != "" and value != "N/A"


def _join_limited(values: list[str], limit: int = 4) -> str:
    values = [str(value) for value in values if value]
    if not values:
        return "None identified"
    if len(values) <= limit:
        return ", ".join(values)
    return f"{', '.join(values[:limit])}, and {len(values) - limit} more"


def _join_all(values: list[str]) -> str:
    values = [str(value) for value in values if value]
    return ", ".join(values) if values else "None identified"


def _first_present(*values):
    for value in values:
        if _nonempty(value):
            return value
    return None


def _metric_value(metrics: dict, *keys: str) -> float | None:
    for key in keys:
        value = _to_float(metrics.get(key))
        if value is not None:
            return value
    return None


def _f1_label(metrics: dict) -> str:
    if _nonempty(metrics.get("macro_f1")):
        return "macro F1"
    if _nonempty(metrics.get("weighted_f1")):
        return "weighted F1"
    return "F1"


def _f1_value(metrics: dict) -> float | None:
    return _metric_value(metrics, "macro_f1", "weighted_f1")


def _best_epoch(metrics: dict) -> dict:
    best = metrics.get("best") or {}
    return (
        best.get("best_map50_95")
        or best.get("best_map50")
        or best.get("lowest_validation_loss")
        or {}
    )


def _checkpoint_label(run_dir: Path) -> str:
    if (run_dir / "weights" / "best.pt").is_file():
        return "best.pt"
    if (run_dir / "weights" / "last.pt").is_file():
        return "last.pt"
    return "Unavailable"


def _primary_evidence(validation_metrics: dict, test_metrics: dict | None = None) -> tuple[str, dict]:
    if test_metrics and test_metrics.get("available", True):
        return "independent test set", test_metrics
    return "validation set", validation_metrics


def _metric_band(metrics: dict) -> str:
    map50 = _metric_value(metrics, "map50")
    map95 = _metric_value(metrics, "map50_95")
    recall = _metric_value(metrics, "recall")
    f1 = _f1_value(metrics)
    decisive = [value for value in (map95, f1, recall) if value is not None]
    if map50 is None and not decisive:
        return "incomplete"
    if (map95 is not None and map95 >= 0.70) and (recall is None or recall >= 0.80) and (f1 is None or f1 >= 0.80):
        return "strong"
    if (map50 is not None and map50 >= 0.75) and (map95 is None or map95 >= 0.45):
        return "usable"
    return "needs_review"


def _weak_classes(metrics: dict, limit: int | None = None) -> list[dict]:
    weak = []
    for row in metrics.get("per_class") or []:
        reasons = []
        recall = _to_float(row.get("recall"))
        f1 = _to_float(row.get("f1"))
        map95 = _to_float(row.get("map50_95"))
        instances = _to_int(row.get("instances")) or 0
        if instances and instances < 10:
            reasons.append("few evaluation instances")
        if recall is not None and recall < 0.60:
            reasons.append("low recall")
        if f1 is not None and f1 < 0.60:
            reasons.append("low F1")
        if map95 is not None and map95 < 0.40:
            reasons.append("low AP50-95")
        if reasons:
            weak.append({
                "class_name": row.get("class_name", "Unknown"),
                "instances": instances,
                "recall": recall,
                "f1": f1,
                "map50_95": map95,
                "reasons": ", ".join(reasons),
            })
    weak.sort(key=lambda item: (
        item["f1"] if item["f1"] is not None else 1.0,
        item["recall"] if item["recall"] is not None else 1.0,
        item["instances"],
    ))
    return weak[:limit] if limit else weak


def _recommendation(validation_metrics: dict, test_metrics: dict | None = None) -> str:
    evidence_label, evidence = _primary_evidence(validation_metrics, test_metrics)
    band = _metric_band(evidence)
    weak = _weak_classes(evidence)
    if band == "strong" and not weak:
        return f"Use the model for a controlled deployment pilot, with continued monitoring against the {evidence_label} baseline."
    if band in {"strong", "usable"}:
        if weak:
            weak_text = _join_all([item["class_name"] for item in weak])
            return f"Proceed to real-world testing while collecting more examples for weaker classes before broader deployment: {weak_text}."
        return f"Proceed to real-world testing and monitor performance against the {evidence_label} baseline before broader deployment."
    if band == "incomplete":
        return "Do not make a deployment decision yet because the report does not include enough completed evaluation metrics."
    return "Keep the model in development and improve data coverage, labels, or training configuration before deployment."


def _result_sentence(validation_metrics: dict, test_metrics: dict | None = None) -> str:
    evidence_label, evidence = _primary_evidence(validation_metrics, test_metrics)
    labels = _metric_labels(evidence)
    map50 = _metric(evidence.get("map50"))
    map95 = _metric(evidence.get("map50_95"))
    precision = _metric(evidence.get("precision"))
    recall = _metric(evidence.get("recall"))
    f1_label = _f1_label(evidence)
    f1 = _metric(_f1_value(evidence))
    recommendation = _recommendation(validation_metrics, test_metrics)
    return (
        f"On the {evidence_label}, the model achieved {labels['map50']} {map50}, "
        f"{labels['map50_95']} {map95}, {labels['precision'].lower()} {precision}, "
        f"{labels['recall'].lower()} {recall}, and "
        f"{f1_label} {f1}; {recommendation[0].lower() + recommendation[1:]}"
    )


def _dataset_summary_text(context: dict) -> str:
    summary = context.get("dataset_summary") or {}
    pre_augmentation = summary.get("pre_augmentation") or {}
    total_images = summary.get("total_images", "N/A")
    class_count = summary.get("class_count", len(summary.get("classes") or []))
    split_ratios = summary.get("split_ratios") or {}
    split_text = ", ".join(
        f"{split} {split_ratios.get(split)}%"
        for split in ("train", "val", "test")
        if split in split_ratios
    )
    original_text = ""
    if pre_augmentation.get("available") and _nonempty(pre_augmentation.get("total_images")):
        original_text = f"The original source dataset contains {_count(pre_augmentation.get('total_images'))} images before augmentation. "
    if split_text:
        return f"{original_text}The exported dataset contains {_count(total_images)} images across {_count(class_count)} classes with split ratios of {split_text}."
    return f"{original_text}The exported dataset contains {_count(total_images)} images across {_count(class_count)} classes."


def _imbalance_summary(summary: dict) -> tuple[str, list[dict]]:
    distribution = summary.get("class_distribution") or []
    populated = [row for row in distribution if _to_int(row.get("images")) or _to_int(row.get("instances"))]
    if not populated:
        return "Class distribution could not be assessed from the available dataset summary.", []
    max_images = max((_to_int(row.get("images")) or 0) for row in populated)
    min_images = min((_to_int(row.get("images")) or 0) for row in populated)
    underrepresented = [
        row for row in populated
        if (_to_int(row.get("images")) or 0) < max(10, max_images * 0.25)
    ]
    if min_images == 0:
        text = "At least one class has no labeled images, so class-level evaluation is high risk."
    elif max_images >= min_images * 3:
        text = f"The largest class has {_count(max_images)} images versus {_count(min_images)} in the smallest class, indicating material class imbalance."
    else:
        text = f"Class image counts range from {_count(min_images)} to {_count(max_images)}, with no severe image-count imbalance detected."
    return text, underrepresented


def _label_quality_rows(summary: dict) -> list[list]:
    rows = [["Check", "Finding"]]
    rows.append(["Missing label files", summary.get("missing_labels", 0)])
    warnings = summary.get("warnings") or []
    issue_warnings = [
        warning for warning in warnings
        if any(token in warning.lower() for token in ("missing", "malformed", "unknown class", "no labeled"))
    ]
    rows.append(["Label/data warnings", _join_limited(issue_warnings, 3)])
    return rows


def _short_config_rows(context: dict, run_dir: Path) -> list[list]:
    hyperparameters = context.get("hyperparameters") or _load_yaml(run_dir / "args.yaml")
    rows = [["Setting", "Value"]]
    family = _report_family(context, {"backend": hyperparameters.get("family")})
    if family == "rfdetr":
        settings = (
            ("Epochs", "epochs"),
            ("Image size", "imgsz"),
            ("Batch size", "batch"),
            ("Workers", "workers"),
            ("Initial LR", "lr0"),
            ("Weight decay", "weight_decay"),
            ("Warmup epochs", "warmup_epochs"),
            ("Cosine LR", "cos_lr"),
            ("Early-stopping patience", "patience"),
            ("Transfer-learning checkpoint", "model"),
            ("Random seed", "seed"),
        )
    elif family == "dfine":
        settings = (
            ("Epochs", "epochs"),
            ("Image size", "imgsz"),
            ("Batch size", "batch"),
            ("Workers", "workers"),
            ("Initial LR", "lr0"),
            ("Weight decay", "weight_decay"),
            ("Transfer-learning checkpoint", "model"),
            ("Random seed", "seed"),
        )
    else:
        settings = (
            ("Epochs", "epochs"),
            ("Image size", "imgsz"),
            ("Batch size", "batch"),
            ("Optimizer", "optimizer"),
            ("Initial LR", "lr0"),
            ("Final LR factor", "lrf"),
            ("Cosine LR", "cos_lr"),
            ("Early-stopping patience", "patience"),
            ("Transfer-learning checkpoint", "model"),
            ("Random seed", "seed"),
            ("Deterministic mode", "deterministic"),
        )
    for label, key in settings:
        value = _first_present(hyperparameters.get(key), context.get(key))
        if _nonempty(value):
            rows.append([label, value])
    if family in {"dfine", "rfdetr"}:
        return rows
    augmentation_keys = ("mosaic", "mixup", "copy_paste", "degrees", "translate", "scale", "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v")
    augmentation = [
        f"{key}={hyperparameters[key]}"
        for key in augmentation_keys
        if key in hyperparameters and _nonempty(hyperparameters.get(key))
    ]
    if augmentation:
        rows.append(["Runtime augmentation", ", ".join(augmentation)])
    return rows


def _filtered_hyperparameters(context: dict, run_dir: Path) -> dict:
    hyperparameters = context.get("hyperparameters") or _load_yaml(run_dir / "args.yaml")
    family = _report_family(context, {"backend": hyperparameters.get("family")})
    ignored = {"dataset_yaml"}
    if family == "rfdetr":
        allowed = {
            "family",
            "task",
            "model",
            "model_size",
            "epochs",
            "imgsz",
            "batch",
            "workers",
            "lr0",
            "weight_decay",
            "warmup_epochs",
            "cos_lr",
            "optimizer",
            "patience",
            "seed",
            "device",
            "project",
            "name",
            "requested_name",
            "resume",
            "save_period",
        }
        return {key: value for key, value in hyperparameters.items() if key in allowed and key not in ignored}
    if family == "dfine":
        allowed = {
            "family",
            "task",
            "model",
            "model_size",
            "epochs",
            "imgsz",
            "batch",
            "workers",
            "lr0",
            "weight_decay",
            "optimizer",
            "seed",
            "device",
            "project",
            "name",
            "requested_name",
            "resume",
            "save_period",
        }
        return {key: value for key, value in hyperparameters.items() if key in allowed and key not in ignored}
    return {key: value for key, value in hyperparameters.items() if key not in ignored}


def _training_behaviour_text(metrics: dict) -> str:
    labels = _metric_labels(metrics)
    history = metrics.get("history") or []
    if len(history) < 2:
        return "Training behaviour could not be interpreted because epoch history is unavailable or incomplete."
    first = history[0]
    last = history[-1]
    best = _best_epoch(metrics)
    best_epoch = best.get("epoch", "N/A")
    first_map = _to_float(first.get("map50_95"))
    last_map = _to_float(last.get("map50_95"))
    train_loss = _to_float(last.get("training_loss"))
    val_loss = _to_float(last.get("testing_loss"))
    trend = "improved" if first_map is not None and last_map is not None and last_map >= first_map else "did not clearly improve"
    loss_note = str(metrics.get("note") or "").split("Macro and weighted F1 are calculated from final per-class validation rows when available.")[-1].strip()
    if train_loss is not None and val_loss is not None and val_loss > train_loss * 1.5:
        fit = "validation loss is materially higher than training loss, so possible overfitting should be reviewed"
    elif first_map is not None and last_map is not None and last_map < 0.30:
        fit = f"{labels['map50_95']} remains low, so possible underfitting or dataset issues should be reviewed"
    else:
        fit = "no obvious overfitting or underfitting signal is visible from the final loss relationship"
    suffix = f" {loss_note}" if loss_note else ""
    return f"Across {len(history)} completed epochs, {labels['map50_95']} {trend}; the best tracked epoch is {best_epoch}, and {fit}.{suffix}"


def _validation_test_text(validation: dict, test: dict) -> str:
    deltas = []
    for key, label in (("map50", "mAP50"), ("map50_95", "mAP50-95"), ("recall", "recall")):
        val_value = _to_float(validation.get(key))
        test_value = _to_float(test.get(key))
        if val_value is not None and test_value is not None:
            deltas.append((label, test_value - val_value))
    if not deltas:
        return "Validation and test results could not be compared because one or more metrics are unavailable."
    largest = max(deltas, key=lambda item: abs(item[1]))
    direction = "higher" if largest[1] >= 0 else "lower"
    return f"The largest validation-to-test shift is {largest[0]}, which is {abs(largest[1]):.4f} {direction} on the test set."


def _has_metric_value(rows: list[dict], key: str) -> bool:
    return any(_to_float(row.get(key)) is not None for row in rows)


def _per_class_table_rows(metrics: dict) -> tuple[list[list], list[float]]:
    classes = metrics.get("per_class") or []
    columns = [
        ("class_name", "Class", 42 * 1.0, "text"),
        ("instances", "Instances", 24 * 1.0, "count"),
    ]
    optional = (
        ("precision", "Precision", 26 * 1.0),
        ("recall", "Recall", 24 * 1.0),
        ("f1", "F1", 22 * 1.0),
        ("map50", "AP50", 24 * 1.0),
        ("map50_95", "AP50-95", 28 * 1.0),
    )
    for key, label, width in optional:
        if _has_metric_value(classes, key):
            columns.append((key, label, width, "metric"))
    if len(columns) <= 2:
        return [], []
    rows = [[label for _key, label, _width, _kind in columns]]
    for row in classes:
        formatted = []
        for key, _label, _width, kind in columns:
            if kind == "metric":
                formatted.append(_metric(row.get(key)))
            elif kind == "count":
                formatted.append(row.get(key, 0))
            else:
                formatted.append(row.get(key, "N/A"))
        rows.append(formatted)
    widths = [width for _key, _label, width, _kind in columns]
    total_width = sum(widths)
    if total_width > 175:
        scale = 175 / total_width
        widths = [width * scale for width in widths]
    return rows, widths


def _runtime_environment_rows(context: dict) -> list[list]:
    environment = context.get("environment") or {}
    if not environment:
        return []
    family = _report_family(context)
    rows = [["Python", environment.get("python")]]
    if family == "rfdetr":
        rows.append(["RF-DETR", environment.get("rfdetr")])
    elif family == "dfine":
        rows.append(["D-FINE repo", environment.get("dfine_repo")])
    else:
        rows.append(["Ultralytics", environment.get("ultralytics")])
    rows.append(["PyTorch", environment.get("torch")])
    gpu_names = ", ".join(str(item.get("name")) for item in environment.get("gpus", []) if item.get("name")) or "Unavailable"
    rows.append(["GPU", gpu_names])
    rows.append(["Platform", environment.get("platform")])
    return rows


def _metric_source_rows(metrics: dict) -> list[list]:
    family = _report_family(metrics=metrics)
    if family == "ultralytics":
        return []
    rows = [["Metric group", "Source"]]
    sources = metrics.get("metric_sources") or {}
    labels = {
        "train_loss": "Training loss",
        "val_loss": "Validation loss",
        "precision": "Precision",
        "recall": "Recall",
        "map50": "mAP50",
        "map50_95": "mAP50-95",
        "per_class": "Per-class metrics",
    }
    for key, label in labels.items():
        value = sources.get(key)
        if _nonempty(value):
            rows.append([label, value])
    if len(rows) == 1 and _nonempty(metrics.get("overall_metric_source")):
        rows.append(["Overall metrics", metrics.get("overall_metric_source")])
    if _nonempty(metrics.get("per_class_source")) and not any(row[0] == "Per-class metrics" for row in rows[1:]):
        rows.append(["Per-class metrics", metrics.get("per_class_source")])
    roc_note = (metrics.get("roc_auc") or {}).get("note")
    if _nonempty(roc_note):
        rows.append(["ROC-AUC", roc_note])
    return rows if len(rows) > 1 else []


def _safe_plot_metric_series(output_path: Path, title: str, ylabel: str, series: list[tuple[str, list[int], list[float]]]) -> bool:
    series = [(label, epochs, values) for label, epochs, values in series if epochs and values]
    if not series:
        return False
    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4.8))
        for label, epochs, values in series:
            plt.plot(epochs, values, marker="o", linewidth=2, markersize=3, label=label)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160)
        plt.close()
        return True
    except Exception:
        return False


def _ensure_report_metric_plots(run_dir: Path, metrics: dict) -> None:
    history = metrics.get("history") or []
    if not history:
        return
    metric_path = run_dir / "accuracy_by_epoch.png"
    if not metric_path.is_file():
        map50_epochs, map50_values = [], []
        map95_epochs, map95_values = [], []
        for row in history:
            epoch = _to_int(row.get("epoch"))
            if epoch is None:
                continue
            map50 = _to_float(row.get("map50"))
            map95 = _to_float(row.get("map50_95"))
            if map50 is not None:
                map50_epochs.append(epoch)
                map50_values.append(map50)
            if map95 is not None:
                map95_epochs.append(epoch)
                map95_values.append(map95)
        _safe_plot_metric_series(
            metric_path,
            f"{metrics.get('metric_label') or 'Detection performance'} by epoch",
            metrics.get("metric_label") or "Metric",
            [("mAP50", map50_epochs, map50_values), ("mAP50-95", map95_epochs, map95_values)],
        )

    loss_path = run_dir / "loss_by_epoch.png"
    if not loss_path.is_file():
        train_epochs, train_values = [], []
        val_epochs, val_values = [], []
        for row in history:
            epoch = _to_int(row.get("epoch"))
            if epoch is None:
                continue
            train_loss = _to_float(row.get("training_loss"))
            val_loss = _to_float(row.get("testing_loss"))
            if train_loss is not None:
                train_epochs.append(epoch)
                train_values.append(train_loss)
            if val_loss is not None:
                val_epochs.append(epoch)
                val_values.append(val_loss)
        _safe_plot_metric_series(
            loss_path,
            "Training and validation loss by epoch",
            "Loss",
            [("Training loss", train_epochs, train_values), ("Validation loss", val_epochs, val_values)],
        )


def _dataset_root(yaml_path: Path, config: dict) -> Path:
    value = Path(str(config.get("path") or yaml_path.parent)).expanduser()
    if not value.is_absolute():
        value = yaml_path.parent / value
    return value.resolve()


def _class_names(config: dict, summary: dict) -> list[str]:
    names = config.get("names")
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    if isinstance(names, list):
        return [str(item) for item in names]
    return [str(item) for item in summary.get("classes") or []]


def _training_images(yaml_path: Path, config: dict, summary: dict) -> tuple[list[Path], Path | None]:
    split = (summary.get("splits") or {}).get("train") or {}
    image_path = Path(split.get("image_path")) if split.get("image_path") else None
    label_path = Path(split.get("label_path")) if split.get("label_path") else None
    root = _dataset_root(yaml_path, config)
    if image_path is None:
        raw = config.get("train")
        if isinstance(raw, str):
            image_path = Path(raw).expanduser()
            if not image_path.is_absolute():
                image_path = root / image_path
    if image_path is None or not image_path.is_dir():
        return [], label_path
    if label_path is None:
        parts = list(image_path.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            label_path = Path(*parts)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in sorted(image_path.rglob("*")) if p.is_file() and p.suffix.lower() in extensions], label_path


def _sample_images_by_class(
    yaml_path: Path,
    summary: dict,
    limit: int = 3,
) -> tuple[list[str], dict[int, list[tuple[Path, Path]]]]:
    config = _load_yaml(yaml_path)
    names = _class_names(config, summary)
    images, labels_root = _training_images(yaml_path, config, summary)
    selected = {class_id: [] for class_id in range(len(names))}
    if labels_root is None:
        return names, selected
    for image_path in images:
        if all(len(items) >= limit for items in selected.values()):
            break
        try:
            relative = image_path.relative_to(Path((summary.get("splits") or {}).get("train", {}).get("image_path", image_path.parent)))
            label_path = (labels_root / relative).with_suffix(".txt")
        except ValueError:
            label_path = labels_root / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        class_ids = set()
        for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if not fields:
                continue
            try:
                class_ids.add(int(float(fields[0])))
            except ValueError:
                continue
        for class_id in class_ids:
            if class_id in selected and len(selected[class_id]) < limit:
                selected[class_id].append((image_path, label_path))
    return names, selected


def _annotated_thumbnail(image_path: Path, label_path: Path, names: list[str], output_path: Path) -> Path | None:
    try:
        from PIL import Image as PILImage, ImageDraw, ImageFont, ImageOps

        with PILImage.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        line_width = max(2, round(min(width, height) / 180))
        font = ImageFont.load_default()
        for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                class_id = int(float(fields[0]))
                center_x, center_y, box_width, box_height = map(float, fields[1:5])
            except ValueError:
                continue
            x1 = max(0, (center_x - box_width / 2) * width)
            y1 = max(0, (center_y - box_height / 2) * height)
            x2 = min(width, (center_x + box_width / 2) * width)
            y2 = min(height, (center_y + box_height / 2) * height)
            color = (20, 145, 120)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
            label = names[class_id] if 0 <= class_id < len(names) else str(class_id)
            text_box = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle(text_box, fill=color)
            draw.text((x1, y1), label, fill="white", font=font)
        image.thumbnail((800, 520))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, "JPEG", quality=82, optimize=True)
        return output_path
    except (OSError, ValueError):
        return None


class _ReportBuilder:
    def __init__(self, output_path: Path, title: str):
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate

        self.colors = colors
        self.mm = mm
        self.output_path = output_path
        self.story = []
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#173b36"), spaceAfter=14))
        styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], textColor=colors.HexColor("#176b5b"), spaceBefore=12, spaceAfter=7, keepWithNext=True))
        styles.add(ParagraphStyle(name="Subsection", parent=styles["Heading3"], textColor=colors.HexColor("#24333f"), spaceBefore=8, spaceAfter=5, keepWithNext=True))
        styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
        self.styles = styles
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.document = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=15 * mm, leftMargin=15 * mm, topMargin=16 * mm, bottomMargin=16 * mm, title=title)
        self.title = title

    def paragraph(self, value, style="BodyText"):
        from reportlab.platypus import Paragraph

        self.story.append(Paragraph(str(value), self.styles[style]))

    def heading(self, value, level=2):
        self.paragraph(_text(value), "Section" if level == 2 else "Subsection")

    def table(self, rows, widths=None, header=True):
        from reportlab.platypus import Paragraph, Table, TableStyle

        if not rows:
            return
        formatted = []
        for row_index, row in enumerate(rows):
            formatted.append([
                Paragraph(
                    f"<b>{_text(cell)}</b>" if header and row_index == 0 else _text(cell),
                    self.styles["Small"],
                )
                for cell in row
            ])
        table = Table(formatted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
        commands = [
            ("GRID", (0, 0), (-1, -1), 0.35, self.colors.HexColor("#cbd5df")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("LEADING", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [self.colors.white, self.colors.HexColor("#f7f9fa")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        if header:
            commands.extend([
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), self.colors.HexColor("#e7f3ef")),
            ])
        table.setStyle(TableStyle(commands))
        self.story.append(table)

    def image(self, path: Path, max_width_mm=175, max_height_mm=105):
        from reportlab.lib.utils import ImageReader
        from reportlab.platypus import Image as ReportImage

        if not path.is_file():
            return False
        reader = ImageReader(str(path))
        width, height = reader.getSize()
        scale = min(max_width_mm * self.mm / width, max_height_mm * self.mm / height)
        self.story.append(ReportImage(str(path), width=width * scale, height=height * scale))
        return True

    def page_break(self):
        from reportlab.platypus import PageBreak

        self.story.append(PageBreak())

    def build(self):
        from reportlab.lib.units import mm

        def footer(canvas, document):
            canvas.saveState()
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(self.colors.HexColor("#687987"))
            canvas.drawString(15 * mm, 9 * mm, self.title)
            canvas.drawRightString(195 * mm, 9 * mm, f"Page {document.page}")
            canvas.restoreState()

        self.document.build(self.story, onFirstPage=footer, onLaterPages=footer)


def _add_dataset(builder: _ReportBuilder, run_dir: Path, context: dict):
    summary = context.get("dataset_summary") or {}
    pre_augmentation = summary.get("pre_augmentation") or {}
    if not pre_augmentation and summary.get("dataset_root"):
        pre_augmentation = roboflow_pre_augmentation_summary(
            Path(summary["dataset_root"]),
            summary.get("splits") or {},
        )
    yaml_value = context.get("dataset_yaml") or ""
    builder.heading("Dataset Information")
    builder.table([
        ["Dataset YAML", yaml_value or "Unavailable"],
        ["Dataset root", summary.get("dataset_root", "Unavailable")],
        [_exported_training_images_label(), summary.get("total_images", "N/A")],
        ["Classes", summary.get("class_count", len(summary.get("classes") or []))],
        ["Split strategy", summary.get("split_strategy", "N/A")],
        ["Split seed", summary.get("split_seed", "N/A")],
    ], widths=[42 * builder.mm, 133 * builder.mm])

    if pre_augmentation.get("available"):
        builder.heading("Original Dataset Before Augmentation", 3)
        count_basis = "Estimated" if pre_augmentation.get("counts_are_estimated") else "Reconstructed by exact division"
        builder.table([
            ["Original source images", pre_augmentation.get("total_images", "N/A")],
            ["Training outputs per source image", pre_augmentation.get("augmentation_multiplier", "N/A")],
            ["Count basis", count_basis],
            ["Metadata source", pre_augmentation.get("metadata_source", "N/A")],
        ], widths=[70 * builder.mm, 105 * builder.mm], header=False)
        original_rows = [["Original split", "Source images", "Original ratio"]]
        original_ratios = pre_augmentation.get("split_ratios") or {}
        for split in ("train", "val", "test"):
            entry = (pre_augmentation.get("splits") or {}).get(split) or {}
            original_rows.append([split.title(), entry.get("images", 0), f"{original_ratios.get(split, 0)}%"])
        builder.table(original_rows, widths=[58 * builder.mm, 58 * builder.mm, 59 * builder.mm])
        if pre_augmentation.get("note"):
            builder.paragraph(pre_augmentation["note"], "Small")

    builder.heading("Exported Dataset After Augmentation", 3)
    split_rows = [["Exported split", "Images", "Missing labels", "Exported ratio"]]
    ratios = summary.get("split_ratios") or {}
    for split in ("train", "val", "test"):
        entry = (summary.get("splits") or {}).get(split) or {}
        split_rows.append([split.title(), entry.get("images", 0), entry.get("missing_labels", 0), f"{ratios.get(split, 0)}%"])
    builder.table(split_rows, widths=[42 * builder.mm] * 4)

    distribution = summary.get("class_distribution") or []
    if distribution:
        builder.heading("Class Distribution", 3)
        builder.table(
            [["Class", "Images", "Instances"]]
            + [[row.get("class_name"), row.get("images", 0), row.get("instances", 0)] for row in distribution],
            widths=[85 * builder.mm, 45 * builder.mm, 45 * builder.mm],
        )

    yaml_path = Path(yaml_value).expanduser() if yaml_value else None
    if yaml_path is None or not yaml_path.is_file():
        builder.paragraph("Dataset samples are unavailable because the original dataset path cannot be accessed.")
        return
    builder.heading("Training Dataset Samples by Class")
    names, selections = _sample_images_by_class(yaml_path, summary)
    assets_dir = run_dir / "report_assets" / "samples"
    from reportlab.platypus import Image as ReportImage, Table, TableStyle
    from PIL import Image as PILImage

    for class_id, class_name in enumerate(names):
        builder.heading(class_name, 3)
        cells = []
        for index, (image_path, label_path) in enumerate(selections.get(class_id, []), start=1):
            target = assets_dir / f"class-{class_id}-{index}.jpg"
            if not target.is_file():
                _annotated_thumbnail(image_path, label_path, names, target)
            if target.is_file():
                with PILImage.open(target) as sample_image:
                    width, height = sample_image.size
                scale = min(54 * builder.mm / width, 36 * builder.mm / height)
                cells.append(ReportImage(str(target), width=width * scale, height=height * scale))
        if not cells:
            builder.paragraph("No labeled training sample was available for this class.", "Small")
            continue
        sample_table = Table([cells], colWidths=[58 * builder.mm] * len(cells), hAlign="LEFT")
        sample_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        builder.story.append(sample_table)


def _add_executive_summary(
    builder: _ReportBuilder,
    run_dir: Path,
    context: dict,
    metrics: dict,
    test_metrics: dict | None = None,
):
    summary = context.get("dataset_summary") or {}
    evidence_label, evidence = _primary_evidence(metrics, test_metrics)
    labels = _metric_labels(evidence)
    best = _best_epoch(metrics)
    builder.heading("Executive Summary")
    builder.table([
        ["Item", "Summary"],
        ["Dataset", f"{_count(summary.get('total_images', 'N/A'))} images, {_count(summary.get('class_count', len(summary.get('classes') or [])))} classes"],
        ["Best checkpoint", _checkpoint_label(run_dir)],
        ["Best tracked epoch", best.get("epoch", "N/A")],
        ["Primary evidence", evidence_label.title()],
        [labels["precision"], _metric(evidence.get("precision"))],
        [labels["recall"], _metric(evidence.get("recall"))],
        [labels["map50"], _metric(evidence.get("map50"))],
        [labels["map50_95"], _metric(evidence.get("map50_95"))],
        [_f1_label(evidence), _metric(_f1_value(evidence))],
    ], widths=[55 * builder.mm, 120 * builder.mm])
    builder.paragraph(_text(_result_sentence(metrics, test_metrics)))


def _add_model_dataset_overview(builder: _ReportBuilder, context: dict):
    summary = context.get("dataset_summary") or {}
    family = _report_family(context)
    pre_augmentation = summary.get("pre_augmentation") or {}
    ratios = summary.get("split_ratios") or {}
    splits = summary.get("splits") or {}
    original_ratios = pre_augmentation.get("split_ratios") or {}
    original_splits = pre_augmentation.get("splits") or {}
    builder.heading("Model and Dataset Overview")
    builder.paragraph(_text(_dataset_summary_text(context)))
    rows = [
        ["Item", "Value"],
        ["Model family", _backend_display_name(family)],
        ["Model variant", context.get("model", "N/A")],
        ["Original images", pre_augmentation.get("total_images", "Unavailable")],
        [_exported_training_images_label(), summary.get("total_images", "N/A")],
        ["Augmentation multiplier", pre_augmentation.get("augmentation_multiplier", "N/A")],
        ["Dataset source", context.get("dataset_source", "Prepared dataset")],
        ["Dataset version", context.get("dataset_version", "N/A")],
        ["Preparation date", _format_myt(context.get("created_at"))],
    ]
    builder.table(rows, widths=[64 * builder.mm, 111 * builder.mm])

    if pre_augmentation.get("available"):
        split_rows = [["Split", "Original images", "Original percent", "Exported images", "Exported percent"]]
        for split in ("train", "val", "test"):
            original_entry = original_splits.get(split) or {}
            exported_entry = splits.get(split) or {}
            split_rows.append([
                split.title(),
                original_entry.get("images", "N/A"),
                f"{original_ratios.get(split, 0)}%",
                exported_entry.get("images", 0),
                f"{ratios.get(split, 0)}%",
            ])
        builder.table(split_rows, widths=[35 * builder.mm, 36 * builder.mm, 36 * builder.mm, 34 * builder.mm, 34 * builder.mm])
    else:
        split_rows = [["Split", "Exported images", "Exported percent"]]
        for split in ("train", "val", "test"):
            entry = splits.get(split) or {}
            split_rows.append([split.title(), entry.get("images", 0), f"{ratios.get(split, 0)}%"])
        builder.table(split_rows, widths=[58 * builder.mm, 58 * builder.mm, 59 * builder.mm])

    classes = summary.get("classes") or []
    if classes:
        builder.paragraph(_text(f"Classes: {', '.join(str(item) for item in classes)}."), "Small")


def _add_dataset_quality(builder: _ReportBuilder, context: dict):
    summary = context.get("dataset_summary") or {}
    if not summary:
        builder.heading("Dataset Quality and Risks")
        builder.paragraph("Dataset quality could not be assessed because no dataset summary was available.")
        return

    builder.heading("Dataset Quality and Risks")
    imbalance_text, underrepresented = _imbalance_summary(summary)
    builder.paragraph(_text(imbalance_text))
    builder.table(_label_quality_rows(summary), widths=[62 * builder.mm, 113 * builder.mm])

    distribution = summary.get("class_distribution") or []
    if distribution:
        builder.table(
            [["Class", "Images", "Instances"]]
            + [[row.get("class_name"), row.get("images", 0), row.get("instances", 0)] for row in distribution],
            widths=[85 * builder.mm, 45 * builder.mm, 45 * builder.mm],
        )
    if underrepresented:
        names = _join_all([row.get("class_name") for row in underrepresented])
        builder.paragraph(
            _text(
                "Underrepresented classes may have unstable per-class metrics and higher "
                f"field risk: {names}."
            ),
            "Small",
        )


def _add_training_configuration(builder: _ReportBuilder, run_dir: Path, context: dict):
    builder.heading("Training Configuration")
    rows = _short_config_rows(context, run_dir)
    if len(rows) == 1:
        builder.paragraph("Training configuration was unavailable.")
        return
    builder.table(rows, widths=[70 * builder.mm, 105 * builder.mm])


def _add_training_behaviour(builder: _ReportBuilder, run_dir: Path, metrics: dict):
    labels = _metric_labels(metrics)
    _ensure_report_metric_plots(run_dir, metrics)
    builder.heading("Training Behaviour")
    builder.paragraph(_text(_training_behaviour_text(metrics)))
    best = metrics.get("best") or {}
    best_rows = [["Criterion", "Epoch", "Value"]]
    for key, label, value_key in (
        ("best_map50", f"Best {labels['map50']}", "map50"),
        ("best_map50_95", f"Best {labels['map50_95']}", "map50_95"),
        ("lowest_training_loss", "Lowest train loss", "training_loss"),
        ("lowest_validation_loss", "Lowest val loss", "testing_loss"),
    ):
        row = best.get(key)
        if row:
            best_rows.append([label, row.get("epoch"), _metric(row.get(value_key))])
    if len(best_rows) > 1:
        builder.table(best_rows, widths=[85 * builder.mm, 35 * builder.mm, 55 * builder.mm])

    for filename, caption in (
        ("loss_by_epoch.png", "Training and validation loss by epoch"),
        ("accuracy_by_epoch.png", f"{metrics.get('metric_label') or 'mAP'} progression by epoch"),
    ):
        path = run_dir / filename
        if path.is_file():
            builder.heading(caption, 3)
            builder.image(path)


def _add_validation_performance(builder: _ReportBuilder, run_dir: Path, metrics: dict):
    labels = _metric_labels(metrics)
    family = _report_family(metrics=metrics)
    builder.heading("Validation Performance")
    if metrics.get("magic_adjusted"):
        builder.paragraph(
            "Adjusted report-preview metrics are active. These values are generated from the original run metrics and are not raw validation results.",
            "Small",
        )
        adjustments = metrics.get("magic_adjustments") or []
        if adjustments:
            adjustment_rows = [["Adjusted score", "Target"]]
            for adjustment in adjustments:
                scope = str(adjustment.get("scope") or "overall")
                metric_key = str(adjustment.get("metric_key") or "")
                class_name = str(adjustment.get("class_name") or "")
                metric_label = {
                    "map50_95": "AP50-95" if scope == "per_class" else labels["map50_95"],
                    "map50": "AP50" if scope == "per_class" else labels["map50"],
                    "precision": labels["precision"],
                    "recall": labels["recall"],
                    "macro_f1": "Macro F1",
                    "weighted_f1": "Weighted F1",
                    "f1": "F1",
                }.get(metric_key, metric_key)
                adjustment_label = f"{class_name} {metric_label}".strip() if scope == "per_class" else metric_label
                adjustment_rows.append([adjustment_label, _metric(adjustment.get("target"))])
            builder.table(adjustment_rows, widths=[120 * builder.mm, 55 * builder.mm])
    builder.table([
        ["Metric", "Final validation value"],
        [labels["precision"], _metric(metrics.get("precision"))],
        [labels["recall"], _metric(metrics.get("recall"))],
        [labels["map50"], _metric(metrics.get("map50"))],
        [labels["map50_95"], _metric(metrics.get("map50_95"))],
        ["Macro F1", _metric(metrics.get("macro_f1"))],
        ["Weighted F1", _metric(metrics.get("weighted_f1"))],
    ], widths=[90 * builder.mm, 85 * builder.mm])
    if family in {"dfine", "rfdetr"} and metrics.get("per_class_note"):
        builder.paragraph(
            _text(metrics.get("per_class_note")),
            "Small",
        )

    classes = metrics.get("per_class") or []
    weak = _weak_classes(metrics)
    if weak:
        builder.paragraph(
            _text(
                "Weak classes requiring review: "
                + _join_all([f"{row['class_name']} ({row['reasons']})" for row in weak])
                + "."
            ),
            "Small",
        )
    if classes:
        rows, widths = _per_class_table_rows(metrics)
        if rows:
            builder.table(rows, widths=[width * builder.mm for width in widths])
        elif metrics.get("per_class_note"):
            builder.paragraph(_text(metrics.get("per_class_note")), "Small")

    for filename, caption in (
        ("confusion_matrix_normalized.png", "Normalized validation confusion matrix"),
        ("confusion_matrix.png", "Validation confusion matrix (raw counts)"),
    ):
        path = run_dir / filename
        if path.is_file():
            builder.heading(caption, 3)
            builder.image(path)

    auc_classes = (metrics.get("roc_auc") or {}).get("classes") or []
    if auc_classes:
        builder.heading("Per-Class Validation ROC-AUC", 3)
        builder.table(
            [["Class", "Positive images", "Negative images", "AUC"]]
            + [[row.get("class_name"), row.get("positive_images", 0), row.get("negative_images", 0), _metric(row.get("auc"))] for row in auc_classes],
            widths=[70 * builder.mm, 38 * builder.mm, 38 * builder.mm, 29 * builder.mm],
        )
        roc_path = run_dir / "roc_auc_curve.png"
        if roc_path.is_file():
            builder.image(roc_path)


def _add_qualitative_results(
    builder: _ReportBuilder,
    run_dir: Path,
    test_dir: Path | None = None,
    metrics: dict | None = None,
):
    builder.heading("Qualitative Results")
    candidates = []
    for directory, label in ((run_dir, "Validation"), (test_dir, "Test")):
        if directory is None:
            continue
        for pattern in ("val_batch*_pred.jpg", "val_batch*_labels.jpg", "test_batch*_pred.jpg", "test_batch*_labels.jpg"):
            for path in sorted(directory.glob(pattern))[:2]:
                candidates.append((path, f"{label} example: {path.name}"))
    if not candidates:
        family = _report_family(metrics=metrics)
        if family in {"dfine", "rfdetr"}:
            builder.paragraph(
                f"No representative prediction images were available for this {_backend_display_name(family)} training run. "
                "Training-only reports include prediction examples only when validation visualization artifacts exist; "
                "use Model Testing for test-set prediction artifacts."
            )
        else:
            builder.paragraph(
                "No representative prediction images were available. Review confusion matrices and per-class metrics for error analysis."
            )
        return
    for path, caption in candidates[:6]:
        builder.heading(caption, 3)
        builder.image(path)


def _add_conclusion_and_recommendation(
    builder: _ReportBuilder,
    context: dict,
    metrics: dict,
    test_metrics: dict | None = None,
):
    builder.heading("Conclusion and Recommendation")
    evidence_label, evidence = _primary_evidence(metrics, test_metrics)
    weak = _weak_classes(evidence)
    strengths = []
    if _metric_value(evidence, "map50") is not None:
        strengths.append(f"mAP50 {_metric(evidence.get('map50'))}")
    if _metric_value(evidence, "recall") is not None:
        strengths.append(f"recall {_metric(evidence.get('recall'))}")
    if _f1_value(evidence) is not None:
        strengths.append(f"{_f1_label(evidence)} {_metric(_f1_value(evidence))}")
    builder.paragraph(_text(f"Overall performance is based primarily on the {evidence_label}."))
    builder.paragraph(_text(f"Key strengths: {_join_limited(strengths, 4)}."))
    if weak:
        builder.paragraph(
            _text(
                "Classes or scenarios requiring additional attention: "
                + _join_all([item["class_name"] for item in weak])
                + "."
            )
        )
    else:
        builder.paragraph("No weak classes were automatically flagged by the report thresholds.")
    builder.paragraph(_text(_training_behaviour_text(metrics)))
    builder.paragraph(_text(_recommendation(metrics, test_metrics)))


def _add_technical_appendix(
    builder: _ReportBuilder,
    run_dir: Path,
    context: dict,
    metrics: dict,
    test_dir: Path | None = None,
    test_context: dict | None = None,
    test_metrics: dict | None = None,
):
    builder.page_break()
    builder.paragraph("Technical Appendix", "ReportTitle")
    builder.heading("Internal Run Metadata")
    started_at, completed_at, duration = _run_timing(run_dir, context)
    builder.table([
        ["Run directory", run_dir],
        ["Started", started_at],
        ["Completed", completed_at],
        ["Duration", duration],
        ["Dataset YAML", context.get("dataset_yaml")],
        ["Device", context.get("device")],
        ["Completed epoch", metrics.get("epoch")],
    ], widths=[45 * builder.mm, 130 * builder.mm])

    environment_rows = _runtime_environment_rows(context)
    if environment_rows:
        builder.heading("Runtime Environment", 3)
        builder.table(environment_rows, widths=[45 * builder.mm, 130 * builder.mm], header=False)

    metric_source_rows = _metric_source_rows(metrics)
    if metric_source_rows:
        builder.heading("Metric Sources", 3)
        builder.table(metric_source_rows, widths=[55 * builder.mm, 120 * builder.mm])

    builder.heading("Complete Hyperparameters")
    hyperparameters = _filtered_hyperparameters(context, run_dir)
    rows = [["Parameter", "Value"]] + [[key, value] for key, value in sorted(hyperparameters.items())]
    builder.table(rows, widths=[70 * builder.mm, 105 * builder.mm])

    _add_dataset(builder, run_dir, context)

    _ensure_report_metric_plots(run_dir, metrics)
    builder.heading("Additional Training Plots")
    found_plot = False
    for filename, caption in (
        ("accuracy_by_epoch.png", "Detection performance by epoch"),
        ("loss_by_epoch.png", "Training and validation loss by epoch"),
        ("confusion_matrix_normalized.png", "Normalized validation confusion matrix"),
        ("confusion_matrix.png", "Validation confusion matrix (raw counts)"),
        ("roc_auc_curve.png", "Validation ROC-AUC by class"),
    ):
        path = run_dir / filename
        if path.is_file():
            found_plot = True
            builder.heading(caption, 3)
            builder.image(path)
    if not found_plot:
        builder.paragraph(
            "No additional training plots were available for this run. Some backends only expose scalar metrics unless plot artifacts are generated during training or testing."
        )

    if test_dir and test_metrics:
        builder.heading("Test Run Metadata")
        parameters = (test_context or {}).get("parameters") or {}
        builder.table([
            ["Test run directory", test_dir],
            ["Test started", _format_myt((test_context or {}).get("created_at"))],
            ["Dataset YAML", test_metrics.get("dataset_yaml", (test_context or {}).get("dataset_yaml"))],
            ["Weights", test_metrics.get("weights", (test_context or {}).get("weights_label"))],
            ["Image size", parameters.get("imgsz")],
            ["Batch size", parameters.get("batch")],
            ["Workers", parameters.get("workers")],
            ["Device", parameters.get("device")],
        ], widths=[45 * builder.mm, 130 * builder.mm])


def _add_training(
    builder: _ReportBuilder,
    run_dir: Path,
    context: dict,
    metrics: dict,
    test_metrics: dict | None = None,
    include_conclusion: bool = True,
    include_appendix: bool = True,
):
    builder.paragraph("Model Training Report", "ReportTitle")
    builder.paragraph(
        f"Generated {_human_datetime_myt(datetime.now(MYT))}",
        "Small",
    )
    _add_executive_summary(builder, run_dir, context, metrics, test_metrics)
    _add_model_dataset_overview(builder, context)
    _add_dataset_quality(builder, context)
    _add_training_configuration(builder, run_dir, context)
    _add_training_behaviour(builder, run_dir, metrics)
    _add_validation_performance(builder, run_dir, metrics)
    _add_qualitative_results(builder, run_dir, metrics=metrics)
    if include_conclusion:
        _add_conclusion_and_recommendation(builder, context, metrics, test_metrics)
    if include_appendix:
        _add_technical_appendix(builder, run_dir, context, metrics)


def _add_test(builder: _ReportBuilder, test_dir: Path, context: dict, metrics: dict, validation: dict):
    labels = _metric_labels(metrics)
    validation_labels = _metric_labels(validation)
    builder.page_break()
    builder.paragraph("Independent Test Evaluation", "ReportTitle")
    builder.heading("Final Test-Set Performance")
    builder.paragraph(_text(_result_sentence(validation, metrics)))
    builder.table([
        ["Metric", "Final test value"],
        [labels["precision"], _metric(metrics.get("precision"))],
        [labels["recall"], _metric(metrics.get("recall"))],
        [labels["map50"], _metric(metrics.get("map50"))],
        [labels["map50_95"], _metric(metrics.get("map50_95"))],
        ["Macro F1", _metric(metrics.get("macro_f1"))],
        ["Weighted F1", _metric(metrics.get("weighted_f1"))],
    ], widths=[90 * builder.mm, 85 * builder.mm])

    builder.heading("Test Run Overview")
    parameters = context.get("parameters") or {}
    builder.table([
        ["Test run directory", test_dir],
        ["Test started", _format_myt(context.get("created_at"))],
        ["Evaluated split", metrics.get("split", context.get("dataset_split", "test"))],
        ["Dataset source", context.get("dataset_source")],
        ["Dataset YAML", metrics.get("dataset_yaml", context.get("dataset_yaml"))],
        ["Weights", metrics.get("weights", context.get("weights_label"))],
        ["Image size", parameters.get("imgsz")],
        ["Batch size", parameters.get("batch")],
        ["Workers", parameters.get("workers")],
        ["Device", parameters.get("device")],
    ], widths=[45 * builder.mm, 130 * builder.mm])

    builder.heading("Test Timing")
    builder.table(
        [["Metric", "Value"]] + _test_timing_rows(metrics.get("timing")),
        widths=[75 * builder.mm, 100 * builder.mm],
    )

    test_summary = context.get("dataset_summary") or {}
    test_split = (test_summary.get("splits") or {}).get("test") or {}
    if test_summary:
        builder.heading("Test Dataset Summary")
        builder.table([
            ["Test images", test_split.get("images", test_summary.get("total_images", "N/A"))],
            ["Classes", test_summary.get("class_count", len(test_summary.get("classes") or []))],
            ["Missing labels", test_split.get("missing_labels", test_summary.get("missing_labels", 0))],
        ], widths=[75 * builder.mm, 100 * builder.mm], header=False)
        distribution = test_split.get("class_distribution") or test_summary.get("class_distribution") or []
        if distribution:
            builder.table(
                [["Class", "Images", "Instances"]]
                + [[row.get("class_name"), row.get("images", 0), row.get("instances", 0)] for row in distribution],
                widths=[85 * builder.mm, 45 * builder.mm, 45 * builder.mm],
            )
    builder.heading("Validation vs Test Comparison")
    builder.paragraph(_text(_validation_test_text(validation, metrics)))
    builder.table([
        ["Metric", "Validation", "Test"],
        [validation_labels["precision"], _metric(validation.get("precision")), _metric(metrics.get("precision"))],
        [validation_labels["recall"], _metric(validation.get("recall")), _metric(metrics.get("recall"))],
        [validation_labels["map50"], _metric(validation.get("map50")), _metric(metrics.get("map50"))],
        [validation_labels["map50_95"], _metric(validation.get("map50_95")), _metric(metrics.get("map50_95"))],
        ["Macro F1", _metric(validation.get("macro_f1")), _metric(metrics.get("macro_f1"))],
        ["Weighted F1", _metric(validation.get("weighted_f1")), _metric(metrics.get("weighted_f1"))],
    ], widths=[70 * builder.mm, 52 * builder.mm, 53 * builder.mm])
    classes = metrics.get("per_class") or []
    if classes:
        builder.heading("Per-Class Test Metrics")
        weak = _weak_classes(metrics)
        if weak:
            builder.paragraph(
                _text(
                    "Weak test classes requiring review: "
                    + _join_all([f"{row['class_name']} ({row['reasons']})" for row in weak])
                    + "."
                ),
                "Small",
            )
        rows, widths = _per_class_table_rows(metrics)
        if rows:
            builder.table(rows, widths=[width * builder.mm for width in widths])
        elif metrics.get("per_class_note"):
            builder.paragraph(_text(metrics.get("per_class_note")), "Small")
    auc_classes = (metrics.get("roc_auc") or {}).get("classes") or []
    if auc_classes:
        builder.heading("Per-Class Test ROC-AUC")
        builder.table(
            [["Class", "Positive images", "Negative images", "AUC"]]
            + [[row.get("class_name"), row.get("positive_images", 0), row.get("negative_images", 0), _metric(row.get("auc"))] for row in auc_classes],
            widths=[70 * builder.mm, 38 * builder.mm, 38 * builder.mm, 29 * builder.mm],
        )
    builder.heading("Test Evaluation Plots")
    found = False
    for filename, caption in (
        ("confusion_matrix_normalized.png", "Normalized test confusion matrix"),
        ("confusion_matrix.png", "Test confusion matrix (raw counts)"),
        ("roc_auc_curve.png", "Test ROC-AUC by class"),
    ):
        path = test_dir / filename
        if path.is_file():
            found = True
            builder.heading(caption, 3)
            builder.image(path)
    if not found:
        builder.paragraph("No test plots were available for this run.")


def generate_training_report(run_dir: Path, context: dict, metrics: dict) -> Path:
    output_path = run_dir / "training_report.pdf"
    builder = _ReportBuilder(output_path, "Training Report")
    _add_training(builder, run_dir, context, metrics)
    builder.build()
    return output_path


def generate_combined_report(
    training_dir: Path,
    training_context: dict,
    training_metrics: dict,
    test_dir: Path,
    test_context: dict,
    test_metrics: dict,
) -> Path:
    output_path = test_dir / "training_and_test_report.pdf"
    builder = _ReportBuilder(output_path, "Training and Test Report")
    _add_training(
        builder,
        training_dir,
        training_context,
        training_metrics,
        test_metrics,
        include_conclusion=False,
        include_appendix=False,
    )
    _add_test(builder, test_dir, test_context, test_metrics, training_metrics)
    _add_conclusion_and_recommendation(builder, training_context, training_metrics, test_metrics)
    _add_technical_appendix(
        builder,
        training_dir,
        training_context,
        training_metrics,
        test_dir,
        test_context,
        test_metrics,
    )
    builder.build()
    return output_path
