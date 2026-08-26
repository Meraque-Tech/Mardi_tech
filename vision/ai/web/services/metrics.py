"""Training metric parsing and non-Magic report-artifact services."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..common.files import file_signature
from vision.ai.train.train_dfine import generate_dfine_report_artifacts
from vision.ai.train.train_rfdetr import RFDETR_RUN_LOG, finalize_rfdetr_artifacts, generate_rfdetr_report_artifacts


@dataclass(frozen=True)
class MetricsDependencies:
    repo_root: Path
    log_dir: Path
    log_file: Path
    magic_metrics_filename: str
    training_context_filename: str
    rfdetr_defaults: dict
    run_metrics_cache: object
    clean_log_line: Callable[[str], str]
    current_status: Callable[[], dict]
    family_for_runs_root: Callable[[Path], str]
    load_training_report_context: Callable[[Path], dict]
    magic_overlay: Callable[[Path, dict], dict]
    read_json_object: Callable[[Path], dict]
    read_log_file: Callable[[Path], str]
    training_log_file: Callable[[], Optional[Path]]


def configure_metrics(deps: MetricsDependencies) -> None:
    globals().update({
        "REPO_ROOT": deps.repo_root,
        "LOG_DIR": deps.log_dir,
        "LOG_FILE": deps.log_file,
        "MAGIC_METRICS_FILE": deps.magic_metrics_filename,
        "TRAINING_REPORT_CONTEXT_FILE": deps.training_context_filename,
        "RFDETR_DEFAULTS": deps.rfdetr_defaults,
        "run_metrics_cache": deps.run_metrics_cache,
        "clean_log_line": deps.clean_log_line,
        "current_status": deps.current_status,
        "family_for_runs_root": deps.family_for_runs_root,
        "load_training_report_context": deps.load_training_report_context,
        "apply_magic_metrics_overlay": deps.magic_overlay,
        "read_json_object": deps.read_json_object,
        "read_log_file": deps.read_log_file,
        "training_log_file": deps.training_log_file,
    })


def float_value(row: dict, key: str) -> Optional[float]:
    value = row.get(key)
    if value is None:
        value = row.get(f" {key}")
    if value is None:
        return None
    try:
        parsed = float(str(value).strip())
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def sum_values(row: dict, keys: list[str]) -> Optional[float]:
    values = [float_value(row, key) for key in keys]
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def loss_components(row: dict, prefix: str) -> dict[str, float]:
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
        parsed = float_value(row, normalized)
        if parsed is not None:
            components[component] = parsed
    return components


def raw_loss_sum(row: dict, prefix: str) -> Optional[float]:
    components = loss_components(row, prefix)
    if components:
        return sum(components.values())
    return sum_values(row, [f"{prefix}/box_loss", f"{prefix}/cls_loss", f"{prefix}/dfl_loss"])


def comparable_loss_component_names(task: str, row: dict) -> list[str]:
    train = loss_components(row, "train")
    val = loss_components(row, "val")
    if task == "classify":
        candidates = ["loss"] if "loss" in train and "loss" in val else ["cls"]
    elif task == "segment":
        candidates = ["box", "seg", "cls", "dfl"]
    else:
        candidates = ["box", "cls", "dfl"]
    return [component for component in candidates if component in train and component in val]


def loss_summary(row: dict, task: str) -> dict:
    train = loss_components(row, "train")
    val = loss_components(row, "val")
    comparable_components = comparable_loss_component_names(task, row)
    train_display_components = list(comparable_components)
    val_display_components = list(comparable_components)
    if not comparable_components and "loss" in train:
        train_display_components = ["loss"]
        val_display_components = ["loss"] if "loss" in val else []

    def component_sum(components: dict[str, float], names: list[str]) -> Optional[float]:
        if not names:
            return None
        return sum(components[name] for name in names)

    auxiliary_train = {
        key: value for key, value in train.items() if key not in train_display_components
    }
    auxiliary_val = {
        key: value for key, value in val.items() if key not in val_display_components
    }
    return {
        "training_loss": component_sum(train, train_display_components),
        "testing_loss": component_sum(val, val_display_components),
        "raw_training_loss": sum(train.values()) if train else None,
        "raw_testing_loss": sum(val.values()) if val else None,
        "auxiliary_training_loss": sum(auxiliary_train.values()) if auxiliary_train else None,
        "auxiliary_testing_loss": sum(auxiliary_val.values()) if auxiliary_val else None,
        "loss_components": {
            "train": train,
            "val": val,
            "comparable": comparable_components,
            "auxiliary": {
                "train": auxiliary_train,
                "val": auxiliary_val,
            },
        },
    }


def loss_note(summary: dict) -> str:
    components = summary.get("loss_components") or {}
    auxiliary = components.get("auxiliary") or {}
    auxiliary_names = sorted({
        f"{prefix}/{name}_loss" if name != "loss" else f"{prefix}/loss"
        for prefix in ("train", "val")
        for name in (auxiliary.get(prefix) or {})
    })
    if auxiliary_names:
        return (
            "Train and val loss use matching loss parts only; extra model-specific losses are excluded: "
            f"{', '.join(auxiliary_names)}."
        )
    return "Train and val loss use matching loss parts."


def format_metric(value: Optional[float], digits: int = 4):
    return round(value, digits) if value is not None and math.isfinite(value) else None


def f1_from_precision_recall(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None or precision + recall <= 0:
        return None
    return 2 * precision * recall / (precision + recall)


def parse_metric_row(line: str) -> Optional[dict]:
    parts = clean_log_line(line).split()
    if len(parts) < 7:
        return None
    if parts[0].lower() == "all":
        return None

    numeric_tail = []
    for part in reversed(parts):
        value = float_value({"value": part}, "value")
        if value is None:
            break
        numeric_tail.append(value)
    numeric_tail.reverse()

    if len(numeric_tail) >= 10:
        metric_values = numeric_tail[-10:]
        class_parts = parts[: -10]
        images, instances = metric_values[0], metric_values[1]
        precision, recall, map50, map50_95 = metric_values[-4:]
    elif len(numeric_tail) >= 6:
        metric_values = numeric_tail[-6:]
        class_parts = parts[: -6]
        images, instances, precision, recall, map50, map50_95 = metric_values
    else:
        return None

    class_name = " ".join(class_parts).strip()
    if not class_name or class_name.lower() in {"all", "class", "epoch"}:
        return None

    return {
        "class_name": class_name,
        "images": int(images),
        "instances": int(instances),
        "precision": format_metric(precision),
        "recall": format_metric(recall),
        "f1": format_metric(f1_from_precision_recall(precision, recall)),
        "map50": format_metric(map50),
        "map50_95": format_metric(map50_95),
    }


def parse_class_metrics_from_log(log_path: Path) -> dict:
    if not log_path.is_file():
        return {"classes": [], "macro_f1": None, "weighted_f1": None}

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    validating_indexes = [index for index, line in enumerate(lines) if "Validating " in clean_log_line(line)]
    search_lines = lines[validating_indexes[-1] + 1:] if validating_indexes else lines

    classes_by_name = {}
    for line in search_lines:
        row = parse_metric_row(line)
        if not row:
            continue
        if row["class_name"] == "all":
            continue
        classes_by_name[row["class_name"]] = row

    classes = list(classes_by_name.values())
    macro_f1 = None
    if classes:
        macro_f1 = sum(row["f1"] or 0 for row in classes) / len(classes)

    total_instances = sum(row["instances"] for row in classes)
    weighted_f1 = None
    if total_instances:
        weighted_f1 = sum((row["f1"] or 0) * row["instances"] for row in classes) / total_instances

    return {
        "classes": classes,
        "macro_f1": format_metric(macro_f1),
        "weighted_f1": format_metric(weighted_f1),
    }


def find_training_log_for_run(run_dir: Path) -> Optional[Path]:
    """Find a log that belongs to run_dir without falling back across runs."""
    candidates = []
    active_training_log = training_log_file()
    if active_training_log is not None:
        candidates.append(active_training_log)
    candidates.extend(
        sorted(
            LOG_DIR.glob("train-*.log"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )

    run_path = str(run_dir)
    project_path = str(run_dir.parent)
    run_name_marker = f"--name {run_dir.name}"
    project_marker = f"--project {project_path}"
    seen = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        content = read_log_file(candidate)
        if run_path in content or (run_name_marker in content and project_marker in content):
            return candidate
    return None

def infer_task_from_results_columns(row: dict) -> str:
    keys = {str(key).strip() for key in row}
    if "metrics/mAP50(M)" in keys or "metrics/mAP50-95(M)" in keys:
        return "segment"
    if "metrics/accuracy_top1" in keys or "metrics/accuracy_top5" in keys:
        return "classify"
    return "detect"


def infer_run_task(run_dir: Path, row: Optional[dict] = None) -> str:
    context = load_training_report_context(run_dir)
    task = str(context.get("task") or context.get("hyperparameters", {}).get("task") or "").lower()
    if task in {"detect", "segment", "semantic", "classify"}:
        return task
    parts = {part.lower() for part in run_dir.parts}
    if "segment" in parts:
        return "segment"
    if "semantic" in parts:
        return "semantic"
    if "classify" in parts:
        return "classify"
    if row:
        return infer_task_from_results_columns(row)
    return "detect"


def metric_profile(task: str, row: dict) -> dict:
    if task == "segment":
        use_mask = float_value(row, "metrics/mAP50(M)") is not None or float_value(row, "metrics/mAP50-95(M)") is not None
        suffix = "M" if use_mask else "B"
        metric_type = "mask" if use_mask else "box"
        prefix = "Mask " if use_mask else "Box "
        return {
            "task": "segment",
            "metric_type": metric_type,
            "metric_label": "Segmentation Mask" if use_mask else "Segmentation Box",
            "chart_title": "Segmentation Mask Performance by Epoch" if use_mask else "Segmentation Box Performance by Epoch",
            "precision_key": f"metrics/precision({suffix})",
            "recall_key": f"metrics/recall({suffix})",
            "map50_key": f"metrics/mAP50({suffix})",
            "map50_95_key": f"metrics/mAP50-95({suffix})",
            "labels": {
                "precision": f"{prefix}Precision",
                "recall": f"{prefix}Recall",
                "map50": f"{prefix}mAP50",
                "map50_95": f"{prefix}mAP50-95",
            },
        }
    if task == "classify":
        return {
            "task": "classify",
            "metric_type": "classification",
            "metric_label": "Classification",
            "chart_title": "Classification Performance by Epoch",
            "precision_key": None,
            "recall_key": None,
            "map50_key": "metrics/accuracy_top1",
            "map50_95_key": "metrics/accuracy_top5",
            "labels": {
                "precision": "Precision",
                "recall": "Recall",
                "map50": "Top-1 Accuracy",
                "map50_95": "Top-5 Accuracy",
            },
        }
    return {
        "task": "detect",
        "metric_type": "box",
        "metric_label": "Detection",
        "chart_title": "Detection Performance by Epoch",
        "precision_key": "metrics/precision(B)",
        "recall_key": "metrics/recall(B)",
        "map50_key": "metrics/mAP50(B)",
        "map50_95_key": "metrics/mAP50-95(B)",
        "labels": {
            "precision": "Precision",
            "recall": "Recall",
            "map50": "mAP50",
            "map50_95": "mAP50-95",
        },
    }


def build_metric_history(rows: list[dict], profile: dict) -> list[dict]:
    history = []
    for row in rows:
        losses = loss_summary(row, profile["task"])
        history.append({
            "epoch": int(float_value(row, "epoch") or 0),
            "map50": format_metric(float_value(row, profile["map50_key"])) if profile.get("map50_key") else None,
            "map50_95": format_metric(float_value(row, profile["map50_95_key"])) if profile.get("map50_95_key") else None,
            "training_loss": format_metric(losses["training_loss"]),
            "testing_loss": format_metric(losses["testing_loss"]),
            "raw_training_loss": format_metric(losses["raw_training_loss"]),
            "raw_testing_loss": format_metric(losses["raw_testing_loss"]),
            "auxiliary_training_loss": format_metric(losses["auxiliary_training_loss"]),
            "auxiliary_testing_loss": format_metric(losses["auxiliary_testing_loss"]),
            "loss_components": losses["loss_components"],
        })
    return history


def best_metric_summary(history: list[dict]) -> dict:
    def best_by(key: str, higher_is_better: bool = True):
        candidates = [row for row in history if row.get(key) is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda row: row[key]) if higher_is_better else min(candidates, key=lambda row: row[key])

    best_map95 = best_by("map50_95")
    best_map50 = best_by("map50")
    best_train_loss = best_by("training_loss", higher_is_better=False)
    best_val_loss = best_by("testing_loss", higher_is_better=False)
    return {
        "best_map50_95": best_map95,
        "best_map50": best_map50,
        "lowest_training_loss": best_train_loss,
        "lowest_validation_loss": best_val_loss,
    }


def artifact_status(path: Path) -> dict:
    available = path.is_file()
    stat = path.stat() if available else None
    return {
        "available": available,
        "path": str(path),
        "size": stat.st_size if stat else 0,
        "modified_at": str(stat.st_mtime_ns) if stat else "0",
    }


def run_artifact_statuses(run_dir: Path) -> dict:
    return {
        "results_csv": artifact_status(run_dir / "results.csv"),
        "accuracy_graph": artifact_status(run_dir / "accuracy_by_epoch.png"),
        "loss_graph": artifact_status(run_dir / "loss_by_epoch.png"),
        "confusion_matrix": artifact_status(run_dir / "confusion_matrix.png"),
        "confusion_matrix_normalized": artifact_status(
            run_dir / "confusion_matrix_normalized.png"
        ),
        "roc_auc_curve": artifact_status(run_dir / "roc_auc_curve.png"),
    }


def read_web_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "web_metrics.json"
    if not metrics_path.is_file():
        return {}
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def is_rfdetr_run(run_dir: Path) -> bool:
    parts = {part.lower() for part in run_dir.parts}
    if "rfdetr" in parts:
        return True
    context = read_json_object(run_dir / TRAINING_REPORT_CONTEXT_FILE)
    family = str(context.get("family") or context.get("hyperparameters", {}).get("family") or "").lower()
    return family == "rfdetr"


def is_dfine_run(run_dir: Path) -> bool:
    parts = {part.lower() for part in run_dir.parts}
    if "dfine" in parts:
        return True
    context = read_json_object(run_dir / TRAINING_REPORT_CONTEXT_FILE)
    family = str(context.get("family") or context.get("hyperparameters", {}).get("family") or "").lower()
    return family == "dfine"


def rfdetr_class_names_from_context(context: dict) -> list[str]:
    summary = context.get("dataset_summary") if isinstance(context.get("dataset_summary"), dict) else {}
    classes = summary.get("classes")
    if isinstance(classes, list):
        return [str(name) for name in classes]
    distribution = summary.get("class_distribution")
    if isinstance(distribution, list):
        return [str(row.get("class_name")) for row in distribution if row.get("class_name")]
    return []


def rfdetr_dataset_audit_from_context(context: dict) -> dict:
    class_names = rfdetr_class_names_from_context(context)
    summary = context.get("dataset_summary") if isinstance(context.get("dataset_summary"), dict) else {}
    summary_splits = summary.get("splits") if isinstance(summary.get("splits"), dict) else {}
    splits = {}
    for source_name, target_name in (("train", "train"), ("val", "valid"), ("valid", "valid"), ("test", "test")):
        split = summary_splits.get(source_name)
        if not isinstance(split, dict) or target_name in splits:
            continue
        rows = []
        for row in split.get("class_distribution") or []:
            rows.append({
                "class_id": row.get("class_id"),
                "class_name": row.get("class_name"),
                "images": int(row.get("images") or 0),
                "instances": int(row.get("instances") or 0),
            })
        missing_ids = [
            index
            for index, row in enumerate(rows)
            if int(row.get("instances") or 0) == 0
        ]
        splits[target_name] = {
            "split": target_name,
            "label_path": split.get("label_path", ""),
            "classes": rows,
            "present_class_ids": [row.get("class_id") for row in rows if int(row.get("instances") or 0) > 0],
            "missing_class_ids": missing_ids,
            "malformed_labels": 0,
            "warnings": split.get("warnings") or [],
        }
    return {
        "dataset_dir": summary.get("dataset_root", ""),
        "classes": class_names,
        "splits": splits,
    }


def rfdetr_model_id_from_context(context: dict) -> str:
    hyperparameters = context.get("hyperparameters") if isinstance(context.get("hyperparameters"), dict) else {}
    value = context.get("model") or hyperparameters.get("model") or hyperparameters.get("model_size")
    return str(value or RFDETR_DEFAULTS["model"])


def rfdetr_epochs_from_context(context: dict) -> int:
    hyperparameters = context.get("hyperparameters") if isinstance(context.get("hyperparameters"), dict) else {}
    try:
        return max(1, int(hyperparameters.get("epochs") or 1))
    except (TypeError, ValueError):
        return 1


def find_rfdetr_log_for_run(run_dir: Path) -> Optional[Path]:
    candidates = [run_dir / RFDETR_RUN_LOG]
    active_training_log = training_log_file()
    if active_training_log is not None:
        candidates.append(active_training_log)
    candidates.append(LOG_FILE)
    candidates.extend(sorted(LOG_DIR.glob("train-*.log"), key=lambda path: path.stat().st_mtime, reverse=True))

    seen = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        data = read_log_file(path)
        if run_dir.name in data or str(run_dir) in data:
            return path
    return None


def ensure_rfdetr_web_artifacts(run_dir: Path, force: bool = False):
    if not is_rfdetr_run(run_dir):
        return
    if not force and (run_dir / "results.csv").is_file() and (run_dir / "web_metrics.json").is_file():
        return
    log_path = find_rfdetr_log_for_run(run_dir)
    if log_path is None:
        return
    context = read_json_object(run_dir / TRAINING_REPORT_CONTEXT_FILE)
    existing_metrics = read_web_metrics(run_dir)
    try:
        finalize_rfdetr_artifacts(
            run_dir,
            model_id=rfdetr_model_id_from_context(context),
            class_names=rfdetr_class_names_from_context(context),
            epochs=rfdetr_epochs_from_context(context),
            log_path=log_path,
            dataset_audit=rfdetr_dataset_audit_from_context(context),
            training_completed=existing_metrics.get("training_completed"),
            quiet=True,
        )
    except Exception:
        return


def rfdetr_report_artifacts_ready(run_dir: Path) -> bool:
    return (
        (run_dir / "validation_metrics.json").is_file()
        and (run_dir / "confusion_matrix.png").is_file()
        and (run_dir / "confusion_matrix_normalized.png").is_file()
        and any(run_dir.glob("val_batch*_pred.jpg"))
        and any(run_dir.glob("val_batch*_labels.jpg"))
    )


def rfdetr_report_dataset_yaml(run_dir: Path, context: dict | None = None) -> Path | None:
    context = context if isinstance(context, dict) else load_training_report_context(run_dir)
    dataset_value = context.get("dataset_yaml")
    if not dataset_value:
        return None
    dataset_path = Path(str(dataset_value)).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = (REPO_ROOT / dataset_path).resolve()
    return dataset_path if dataset_path.is_file() else None


def rfdetr_report_weights_available(run_dir: Path) -> bool:
    return (run_dir / "weights" / "best.pt").is_file() or (run_dir / "weights" / "last.pt").is_file()


def dfine_report_artifacts_ready(run_dir: Path) -> bool:
    return (
        (run_dir / "validation_metrics.json").is_file()
        and (run_dir / "confusion_matrix.png").is_file()
        and (run_dir / "confusion_matrix_normalized.png").is_file()
        and any(run_dir.glob("val_batch*_pred.jpg"))
        and any(run_dir.glob("val_batch*_labels.jpg"))
    )


def dfine_report_dataset_yaml(run_dir: Path, context: dict | None = None) -> Path | None:
    context = context if isinstance(context, dict) else load_training_report_context(run_dir)
    dataset_value = context.get("dataset_yaml")
    if not dataset_value:
        return None
    dataset_path = Path(str(dataset_value)).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = (REPO_ROOT / dataset_path).resolve()
    return dataset_path if dataset_path.is_file() else None


def dfine_report_weights_available(run_dir: Path) -> bool:
    return (
        ((run_dir / "weights" / "best.pt").is_file() or (run_dir / "weights" / "last.pt").is_file())
        and (run_dir / "dfine_web_config.yml").is_file()
    )


def ensure_rfdetr_report_artifacts_for_report(run_dir: Path) -> bool:
    if not is_rfdetr_run(run_dir):
        return False
    if rfdetr_report_artifacts_ready(run_dir):
        return True
    if current_status()["running"]:
        return False
    ensure_rfdetr_web_artifacts(run_dir)
    if rfdetr_report_artifacts_ready(run_dir):
        return True

    existing_metrics = read_web_metrics(run_dir)
    if existing_metrics.get("training_completed") is False:
        return False
    if not rfdetr_report_weights_available(run_dir):
        return False

    context = load_training_report_context(run_dir)
    dataset_yaml = rfdetr_report_dataset_yaml(run_dir, context)
    if dataset_yaml is None:
        return False

    try:
        generate_rfdetr_report_artifacts(run_dir, dataset_yaml)
    except Exception:
        return False
    return rfdetr_report_artifacts_ready(run_dir)


def ensure_dfine_report_artifacts_for_report(run_dir: Path) -> bool:
    if not is_dfine_run(run_dir):
        return False
    if dfine_report_artifacts_ready(run_dir):
        return True
    if current_status()["running"]:
        return False

    existing_metrics = read_web_metrics(run_dir)
    if existing_metrics.get("training_completed") is False:
        return False
    if not dfine_report_weights_available(run_dir):
        return False

    context = load_training_report_context(run_dir)
    dataset_yaml = dfine_report_dataset_yaml(run_dir, context)
    if dataset_yaml is None:
        return False

    try:
        generate_dfine_report_artifacts(run_dir, dataset_yaml)
    except Exception:
        return False
    return dfine_report_artifacts_ready(run_dir)


def ensure_model_report_artifacts_for_report(run_dir: Path) -> bool:
    return (
        ensure_rfdetr_report_artifacts_for_report(run_dir)
        or ensure_dfine_report_artifacts_for_report(run_dir)
    )


def read_run_metrics(run_dir: Path, include_magic: bool = True) -> dict:
    ensure_rfdetr_web_artifacts(run_dir, force=is_rfdetr_run(run_dir))
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        return {
            "available": False,
            "run_dir": str(run_dir),
            "results_csv": "",
            "artifacts": run_artifact_statuses(run_dir),
        }

    with results_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {
            "available": False,
            "run_dir": str(run_dir),
            "results_csv": str(results_path),
            "artifacts": run_artifact_statuses(run_dir),
        }

    row = rows[-1]
    task = infer_run_task(run_dir, row)
    profile = metric_profile(task, row)
    precision = float_value(row, profile["precision_key"]) if profile.get("precision_key") else None
    recall = float_value(row, profile["recall_key"]) if profile.get("recall_key") else None
    losses = loss_summary(row, profile["task"])
    map50 = float_value(row, profile["map50_key"]) if profile.get("map50_key") else None
    map50_95 = float_value(row, profile["map50_95_key"]) if profile.get("map50_95_key") else None
    web_metrics = read_web_metrics(run_dir)
    backend = web_metrics.get("backend") or family_for_runs_root(run_dir.parent)
    web_overall = web_metrics.get("overall") if isinstance(web_metrics.get("overall"), dict) else {}
    web_precision = float_value(web_overall, "precision")
    web_recall = float_value(web_overall, "recall")
    web_map50 = float_value(web_overall, "map50")
    web_map50_95 = float_value(web_overall, "map50_95")
    precision = web_precision if web_precision is not None else precision
    recall = web_recall if web_recall is not None else recall
    map50 = web_map50 if web_map50 is not None else map50
    map50_95 = web_map50_95 if web_map50_95 is not None else map50_95
    class_metrics = {
        "macro_f1": web_metrics.get("macro_f1"),
        "weighted_f1": web_metrics.get("weighted_f1"),
        "classes": web_metrics.get("per_class"),
    }
    explicit_metric_schema = int(web_metrics.get("metric_schema_version") or 0) >= 2
    if task == "segment" and not explicit_metric_schema:
        class_metrics = {"macro_f1": None, "weighted_f1": None, "classes": []}
    if (
        task != "segment"
        and (not isinstance(class_metrics["classes"], list) or not class_metrics["classes"])
    ):
        run_log = find_training_log_for_run(run_dir)
        if run_log is not None:
            class_metrics = parse_class_metrics_from_log(run_log)
    roc_auc = web_metrics.get("roc_auc")
    if not isinstance(roc_auc, dict):
        roc_auc = {
            "mode": "image_presence",
            "split": "val",
            "classes": [],
            "note": "ROC-AUC will appear after web metrics are generated for this run.",
        }
    history = build_metric_history(rows, profile)
    metrics_note = (
        web_metrics.get("per_class_note")
        if web_metrics.get("per_class_note")
        else "Macro and weighted F1 are calculated from per-class validation rows when available. "
        + loss_note(losses)
    )
    if task == "segment" and not explicit_metric_schema:
        metrics_note = (
            "Legacy per-class segmentation metrics are hidden because their box/mask family is unverified. "
            "Revalidate best.pt to generate explicit mask and box metrics. "
            + metrics_note
        )
    training_completed = web_metrics.get("training_completed")
    if training_completed is False:
        metrics_note = "Training did not complete successfully; checkpoints and metrics may be partial. " + metrics_note

    result = {
        "available": True,
        "run_dir": str(run_dir),
        "results_csv": str(results_path),
        "backend": backend,
        "training_completed": training_completed,
        "overall_metric_source": (web_metrics.get("overall") or {}).get("source"),
        "metric_sources": web_metrics.get("metric_sources"),
        "per_class_source": web_metrics.get("per_class_source"),
        "per_class_note": web_metrics.get("per_class_note"),
        "metric_schema_version": web_metrics.get("metric_schema_version"),
        "primary_metric_type": web_metrics.get("primary_metric_type") or profile["metric_type"],
        "metric_families": web_metrics.get("metric_families"),
        "overall_by_type": web_metrics.get("overall_by_type"),
        "per_class_box": web_metrics.get("per_class_box"),
        "per_class_mask": web_metrics.get("per_class_mask"),
        "metric_warnings": (
            web_metrics.get("metric_warnings") or []
        ) + (["Legacy segmentation metrics require best.pt revalidation."] if task == "segment" and not explicit_metric_schema else []),
        "epoch": int(float_value(row, "epoch") or 0),
        "task": profile["task"],
        "metric_type": profile["metric_type"],
        "metric_label": profile["metric_label"],
        "metric_labels": profile["labels"],
        "chart_title": profile["chart_title"],
        "macro_f1": class_metrics["macro_f1"],
        "weighted_f1": class_metrics["weighted_f1"],
        "per_class": class_metrics["classes"],
        "roc_auc": roc_auc,
        "training_loss": format_metric(losses["training_loss"]),
        "testing_loss": format_metric(losses["testing_loss"]),
        "raw_training_loss": format_metric(losses["raw_training_loss"]),
        "raw_testing_loss": format_metric(losses["raw_testing_loss"]),
        "auxiliary_training_loss": format_metric(losses["auxiliary_training_loss"]),
        "auxiliary_testing_loss": format_metric(losses["auxiliary_testing_loss"]),
        "loss_components": losses["loss_components"],
        "precision": format_metric(precision),
        "recall": format_metric(recall),
        "map50": format_metric(map50),
        "map50_95": format_metric(map50_95),
        "history": history,
        "best": best_metric_summary(history),
        "artifacts": run_artifact_statuses(run_dir),
        "note": metrics_note,
    }
    return apply_magic_metrics_overlay(run_dir, result) if include_magic else result


def run_metrics_signature(run_dir: Path, include_magic: bool = True) -> tuple:
    """Track only files that can alter the normalized metrics response."""
    paths = [
        run_dir / "results.csv",
        run_dir / "web_metrics.json",
        run_dir / "validation_metrics.json",
        run_dir / "args.json",
        run_dir / "args.yaml",
    ]
    if include_magic:
        paths.append(run_dir / MAGIC_METRICS_FILE)
    return tuple(file_signature(path) for path in paths)


def cached_run_metrics(run_dir: Path, include_magic: bool = True) -> dict:
    key = (str(run_dir.resolve()), bool(include_magic))
    signature = run_metrics_signature(run_dir, include_magic)
    cached = run_metrics_cache.get(key, signature)
    if cached is not None:
        return cached
    result = read_run_metrics(run_dir, include_magic=include_magic)
    return run_metrics_cache.put(key, signature, result)

__all__ = [
    "MetricsDependencies",
    "configure_metrics",
    "float_value",
    "sum_values",
    "loss_components",
    "raw_loss_sum",
    "comparable_loss_component_names",
    "loss_summary",
    "loss_note",
    "format_metric",
    "f1_from_precision_recall",
    "parse_metric_row",
    "parse_class_metrics_from_log",
    "find_training_log_for_run",
    "infer_task_from_results_columns",
    "infer_run_task",
    "metric_profile",
    "build_metric_history",
    "best_metric_summary",
    "artifact_status",
    "run_artifact_statuses",
    "read_web_metrics",
    "is_rfdetr_run",
    "is_dfine_run",
    "rfdetr_class_names_from_context",
    "rfdetr_dataset_audit_from_context",
    "rfdetr_model_id_from_context",
    "rfdetr_epochs_from_context",
    "find_rfdetr_log_for_run",
    "ensure_rfdetr_web_artifacts",
    "rfdetr_report_artifacts_ready",
    "rfdetr_report_dataset_yaml",
    "rfdetr_report_weights_available",
    "dfine_report_artifacts_ready",
    "dfine_report_dataset_yaml",
    "dfine_report_weights_available",
    "ensure_rfdetr_report_artifacts_for_report",
    "ensure_dfine_report_artifacts_for_report",
    "ensure_model_report_artifacts_for_report",
    "read_run_metrics",
    "run_metrics_signature",
    "cached_run_metrics",
]
