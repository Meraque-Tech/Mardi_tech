"""Task-aware extraction of Ultralytics box and mask validation metrics."""

from __future__ import annotations

import math
from typing import Optional


METRIC_SCHEMA_VERSION = 2
CONSISTENCY_TOLERANCE = 0.001


def rounded_metric(value, digits: int = 4):
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return round(parsed, digits) if math.isfinite(parsed) else None


def _values(value) -> list:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        value = [value]
    return value


def _summary_rows(metrics) -> list[dict]:
    if metrics is None or not hasattr(metrics, "summary"):
        return []
    try:
        rows = metrics.summary()
    except Exception:
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _metric_component(metrics, family: str):
    return getattr(metrics, "seg" if family == "mask" else "box", None)


def _component_series(component, name: str) -> list:
    return _values(getattr(component, name, None)) if component is not None else []


def _summary_value(row: dict, family: str, field: str):
    prefixes = {"box": "Box", "mask": "Mask"}
    suffixes = {
        "precision": "P",
        "recall": "R",
        "f1": "F1",
        "map50": "mAP50",
        "map50_95": "mAP50-95",
    }
    value = row.get(f"{prefixes[family]}-{suffixes[field]}")
    # Older Ultralytics summaries expose generic AP keys for boxes only.
    if value is None and family == "box" and field in {"map50", "map50_95"}:
        value = row.get("mAP50" if field == "map50" else "mAP50-95")
    return rounded_metric(value)


def _class_indexes(metrics, component, count: int) -> list[int]:
    indexes = _values(getattr(component, "ap_class_index", None))
    if not indexes:
        indexes = _values(getattr(metrics, "ap_class_index", None))
    if len(indexes) != count:
        return list(range(count))
    normalized = []
    for index in indexes:
        try:
            normalized.append(int(index))
        except (TypeError, ValueError):
            return list(range(count))
    return normalized


def _class_name(metrics, class_id: int, row: Optional[dict]) -> str:
    names = getattr(metrics, "names", {})
    if isinstance(names, dict):
        value = names.get(class_id, names.get(str(class_id)))
        if value is not None:
            return str(value)
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    if row and row.get("Class") is not None:
        return str(row["Class"])
    return str(class_id)


def _count_for_class(metrics, attribute: str, class_id: int, row: Optional[dict], row_key: str) -> int:
    if row:
        try:
            return int(row.get(row_key) or 0)
        except (TypeError, ValueError):
            pass
    counts = _values(getattr(metrics, attribute, None))
    if 0 <= class_id < len(counts):
        try:
            return int(counts[class_id])
        except (TypeError, ValueError):
            pass
    return 0


def _overall_metrics(component) -> dict:
    if component is None:
        return {}
    return {
        "precision": rounded_metric(getattr(component, "mp", None)),
        "recall": rounded_metric(getattr(component, "mr", None)),
        "map50": rounded_metric(getattr(component, "map50", None)),
        "map50_95": rounded_metric(getattr(component, "map", None)),
    }


def _average(rows: list[dict], key: str):
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _family_summary(metrics, family: str, summary_rows: list[dict]) -> dict:
    component = _metric_component(metrics, family)
    series = {
        "precision": _component_series(component, "p"),
        "recall": _component_series(component, "r"),
        "f1": _component_series(component, "f1"),
        "map50": _component_series(component, "ap50"),
        "map50_95": _component_series(component, "ap"),
    }
    count = max([len(summary_rows), *(len(values) for values in series.values())], default=0)
    indexes = _class_indexes(metrics, component, count)
    rows = []
    for position in range(count):
        summary_row = summary_rows[position] if position < len(summary_rows) else None
        class_id = indexes[position] if position < len(indexes) else position
        values = {}
        for field, items in series.items():
            value = rounded_metric(items[position]) if position < len(items) else None
            if value is None and summary_row is not None:
                value = _summary_value(summary_row, family, field)
            values[field] = value
        if values["f1"] is None:
            precision, recall = values["precision"], values["recall"]
            if precision is not None and recall is not None and precision + recall > 0:
                values["f1"] = rounded_metric(2 * precision * recall / (precision + recall))
        if not any(value is not None for value in values.values()):
            continue
        rows.append({
            "class_id": class_id,
            "class_name": _class_name(metrics, class_id, summary_row),
            "images": _count_for_class(metrics, "nt_per_image", class_id, summary_row, "Images"),
            "instances": _count_for_class(metrics, "nt_per_class", class_id, summary_row, "Instances"),
            **values,
        })

    overall = _overall_metrics(component)
    warnings = []
    for key in ("map50", "map50_95"):
        observed = _average(rows, key)
        expected = overall.get(key)
        if observed is None or expected is None:
            continue
        if abs(observed - expected) > CONSISTENCY_TOLERANCE:
            warnings.append(
                f"{family} per-class {key} mean {observed:.4f} does not match overall {expected:.4f}."
            )

    available = bool(rows) and not warnings
    if family == "mask" and component is None:
        warnings.append("Mask metrics are unavailable from this validation result.")
        available = False
    elif family == "mask" and not rows:
        warnings.append("Mask per-class metrics are unavailable; box metrics were not substituted.")
        available = False

    macro_f1 = rounded_metric(_average(rows, "f1")) if available else None
    total_instances = sum(row["instances"] for row in rows)
    weighted_f1 = None
    if available and total_instances:
        weighted_f1 = rounded_metric(
            sum((row.get("f1") or 0) * row["instances"] for row in rows) / total_instances
        )
    return {
        "available": available,
        "metric_type": family,
        "per_class": rows if available else [],
        "overall": overall if available else {},
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "warnings": warnings,
    }


def build_yolo_metric_families(metrics, task: Optional[str] = None) -> dict:
    """Return explicit box/mask families and backward-compatible primary fields."""
    summary_rows = _summary_rows(metrics)
    normalized_task = str(task or "").strip().lower()
    if not normalized_task:
        normalized_task = "segment" if getattr(metrics, "seg", None) is not None else "detect"
    primary_type = "mask" if normalized_task == "segment" else "box"
    box = _family_summary(metrics, "box", summary_rows)
    mask = _family_summary(metrics, "mask", summary_rows) if normalized_task == "segment" else {
        "available": False,
        "metric_type": "mask",
        "per_class": [],
        "overall": {},
        "macro_f1": None,
        "weighted_f1": None,
        "warnings": [],
    }
    primary = mask if primary_type == "mask" else box
    warnings = [*box["warnings"], *mask["warnings"]]
    summary_keys = sorted({str(key) for row in summary_rows for key in row})
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "primary_metric_type": primary_type,
        "metric_families": {"box": box, "mask": mask},
        "per_class_box": box["per_class"],
        "per_class_mask": mask["per_class"],
        "overall_by_type": {"box": box["overall"], "mask": mask["overall"]},
        "per_class": primary["per_class"],
        "overall": primary["overall"],
        "macro_f1": primary["macro_f1"],
        "weighted_f1": primary["weighted_f1"],
        "metric_warnings": warnings,
        "metric_summary_keys": summary_keys,
    }
