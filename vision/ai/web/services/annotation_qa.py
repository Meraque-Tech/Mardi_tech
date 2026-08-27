"""Annotation QA domain service.

The computational and persistence-heavy QA workflow lives here so the FastAPI
composition module only owns request validation and route wiring.  Integrations
with dataset preparation are explicit and configured once during app startup.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Callable, Optional

import yaml
from fastapi import HTTPException

from ..infer_yolo import InferenceStopped
from ..roboflow_sync import (
    ROBOFLOW_SYNC_LOG_FILE,
    ROBOFLOW_SYNC_PREVIEW_FILE,
    annotation_digest,
    canonical_local_annotation,
    canonical_remote_annotation,
    load_roboflow_provenance,
    redact_secret,
    roboflow_image_details,
    roboflow_upload_annotation,
    sync_preview_digest,
    write_roboflow_provenance,
)
from ..sam_qa_runtime import SamQaRuntime
from ..stratified_split import SPLIT_NAMES


@dataclass(frozen=True)
class AnnotationQaDependencies:
    annotation_qa_root: Path
    data_root: Path
    dataset_prepared_root: Path
    timezone: object
    model_default: str
    report_version: int
    safe_mapping_version: int
    sam3_model_path: Path
    sam3_min_ultralytics_version: tuple[int, int, int]
    model_registry: dict
    summary_filename: str
    report_json_filename: str
    report_csv_filename: str
    review_filename: str
    fix_summary_filename: str
    jobs: dict[str, dict]
    jobs_lock: threading.Lock
    clean_name: Callable
    dataset_response: Callable
    image_files: Callable
    label_folder_for_images: Callable
    normalize_yaml_names: Callable
    persist_job: Callable
    prepared_dataset_yaml: Callable
    read_json_object: Callable
    read_yaml_class_names: Callable
    split_image_folder: Callable
    write_json_object: Callable


def configure_annotation_qa(deps: AnnotationQaDependencies) -> None:
    """Bind the QA service to application-owned paths and dataset helpers."""
    bindings = {
        "ANNOTATION_QA_ROOT": deps.annotation_qa_root,
        "DATA_ROOT": deps.data_root,
        "DATASET_PREPARED_ROOT": deps.dataset_prepared_root,
        "MYT": deps.timezone,
        "ANNOTATION_QA_MODEL_DEFAULT": deps.model_default,
        "ANNOTATION_QA_REPORT_VERSION": deps.report_version,
        "ANNOTATION_QA_SAFE_MAPPING_VERSION": deps.safe_mapping_version,
        "SAM3_QA_MODEL_PATH": deps.sam3_model_path,
        "SAM3_MIN_ULTRALYTICS_VERSION": deps.sam3_min_ultralytics_version,
        "SAM_QA_MODEL_REGISTRY": deps.model_registry,
        "ANNOTATION_QA_SUMMARY_FILE": deps.summary_filename,
        "ANNOTATION_QA_REPORT_JSON_FILE": deps.report_json_filename,
        "ANNOTATION_QA_REPORT_CSV_FILE": deps.report_csv_filename,
        "ANNOTATION_QA_REVIEW_FILE": deps.review_filename,
        "ANNOTATION_QA_FIX_SUMMARY_FILE": deps.fix_summary_filename,
        "annotation_qa_jobs": deps.jobs,
        "annotation_qa_jobs_lock": deps.jobs_lock,
        "clean_name": deps.clean_name,
        "dataset_response": deps.dataset_response,
        "image_files": deps.image_files,
        "label_folder_for_images": deps.label_folder_for_images,
        "normalize_yaml_names": deps.normalize_yaml_names,
        "persist_job": deps.persist_job,
        "prepared_dataset_yaml": deps.prepared_dataset_yaml,
        "read_json_object": deps.read_json_object,
        "read_yaml_class_names": deps.read_yaml_class_names,
        "split_image_folder": deps.split_image_folder,
        "write_json_object": deps.write_json_object,
    }
    globals().update(bindings)


def annotation_qa_job_id() -> str:
    return f"{datetime.now(MYT).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def normalize_annotation_qa_model_name(model: str) -> str:
    value = str(model or "").strip() or ANNOTATION_QA_MODEL_DEFAULT
    aliases = {
        "sam2.1_hiera_tiny": "sam2.1_t.pt",
        "sam2.1_hiera_tiny.pt": "sam2.1_t.pt",
        "sam2.1_hiera_small": "sam2.1_s.pt",
        "sam2.1_hiera_small.pt": "sam2.1_s.pt",
        "sam2.1_hiera_base_plus": "sam2.1_b.pt",
        "sam2.1_hiera_base_plus.pt": "sam2.1_b.pt",
        "sam2.1_hiera_large": "sam2.1_l.pt",
        "sam2.1_hiera_large.pt": "sam2.1_l.pt",
        "sam3.pt": "sam3",
        str(SAM3_QA_MODEL_PATH): "sam3",
    }
    normalized = aliases.get(value, value)
    if normalized not in SAM_QA_MODEL_REGISTRY:
        raise ValueError(f"Unsupported Annotation QA model: {value}")
    return normalized


def annotation_qa_version_tuple(value: str) -> tuple[int, int, int]:
    numbers = [int(part) for part in re.findall(r"\d+", str(value))[:3]]
    return tuple((numbers + [0, 0, 0])[:3])


def annotation_qa_model_status(model_name: str) -> dict:
    normalized = normalize_annotation_qa_model_name(model_name)
    config = dict(SAM_QA_MODEL_REGISTRY[normalized])
    available = True
    reason = ""
    installed_version = ""
    if config["backend"] == "sam3":
        path = Path(str(config["path"]))
        if not path.is_file():
            available = False
            reason = f"Checkpoint not found at {path}."
        elif path.stat().st_size < 1_000_000_000:
            available = False
            reason = f"Checkpoint at {path} is unexpectedly small and may not be an Ultralytics SAM 3 checkpoint."
        try:
            installed_version = package_version("ultralytics")
        except PackageNotFoundError:
            available = False
            reason = "Ultralytics is not installed."
        else:
            if annotation_qa_version_tuple(installed_version) < SAM3_MIN_ULTRALYTICS_VERSION:
                available = False
                reason = "SAM 3 requires Ultralytics 8.3.237 or newer."
    return {
        "id": normalized,
        "label": config["label"],
        "backend": config["backend"],
        "available": available,
        "reason": reason,
        "automatic_allowed": bool(config.get("automatic_allowed", True)),
        "max_side": int(config["max_side"]),
        "prompt_chunk": int(config["prompt_chunk"]),
        "recommended_vram_gb": config.get("recommended_vram_gb"),
        "checkpoint": str(config["path"]) if config["backend"] == "sam3" else str(config["path"]),
        "ultralytics_version": installed_version,
    }


def annotation_qa_model_config(model_name: str) -> dict:
    status = annotation_qa_model_status(model_name)
    if not status["available"]:
        raise RuntimeError(status["reason"])
    return {**SAM_QA_MODEL_REGISTRY[status["id"]], **status}


def ensure_annotation_qa_path(path: Path):
    qa_root = ANNOTATION_QA_ROOT.resolve()
    try:
        path.resolve().relative_to(qa_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Annotation QA files can only be read from runs/annotation_qa.",
        ) from exc


def annotation_qa_payload(job: dict) -> dict:
    hidden = {"stop_event", "thread"}
    return {key: value for key, value in job.items() if key not in hidden}


def update_annotation_qa_job(job_id: str, **updates):
    persisted = None
    with annotation_qa_jobs_lock:
        job = annotation_qa_jobs.get(job_id)
        if job is None:
            return
        job.update(updates)
        job["updated_at"] = time.time()
        persisted = annotation_qa_payload(job)
    persist_job(job_id, "annotation_qa", str(persisted.get("status") or "running"), persisted)


def annotation_qa_thresholds(
    preset: str,
    box_tolerance_percent: Optional[float] = None,
    sam_max_difference_percent: Optional[float] = None,
) -> dict:
    presets = {
        "lenient": {
            "bbox_iou": 0.4,
            "center_shift": 0.3,
            "mask_area_ratio_min": 0.08,
            "loose_area_ratio": 0.35,
            "tight_edge_count": 3,
            "duplicate_iou": 0.9,
            "sam_confidence_min": 0.25,
            "box_tolerance_percent": 8.0,
            "sam_max_difference_percent": 35.0,
            "segment_keep_iou": 0.82,
            "segment_review_iou": 0.45,
            "segment_boundary_f1_min": 0.35,
            "segment_area_ratio_min": 0.35,
            "segment_area_ratio_max": 2.5,
        },
        "strict": {
            "bbox_iou": 0.65,
            "center_shift": 0.15,
            "mask_area_ratio_min": 0.2,
            "loose_area_ratio": 0.6,
            "tight_edge_count": 2,
            "duplicate_iou": 0.8,
            "sam_confidence_min": 0.25,
            "box_tolerance_percent": 3.0,
            "sam_max_difference_percent": 15.0,
            "segment_keep_iou": 0.92,
            "segment_review_iou": 0.65,
            "segment_boundary_f1_min": 0.60,
            "segment_area_ratio_min": 0.60,
            "segment_area_ratio_max": 1.6,
        },
    }
    thresholds = dict(presets.get(str(preset or "").lower(), {
        "bbox_iou": 0.55,
        "center_shift": 0.2,
        "mask_area_ratio_min": 0.15,
        "loose_area_ratio": 0.5,
        "tight_edge_count": 2,
        "duplicate_iou": 0.85,
        "sam_confidence_min": 0.25,
        "box_tolerance_percent": 5.0,
        "sam_max_difference_percent": 25.0,
        "segment_keep_iou": 0.88,
        "segment_review_iou": 0.55,
        "segment_boundary_f1_min": 0.45,
        "segment_area_ratio_min": 0.50,
        "segment_area_ratio_max": 2.0,
    }))
    if box_tolerance_percent is not None:
        thresholds["box_tolerance_percent"] = max(0.0, min(50.0, float(box_tolerance_percent)))
    if sam_max_difference_percent is not None:
        thresholds["sam_max_difference_percent"] = max(
            0.0,
            min(100.0, float(sam_max_difference_percent)),
        )
    if thresholds["sam_max_difference_percent"] <= thresholds["box_tolerance_percent"]:
        raise ValueError("Maximum SAM difference must be greater than the YOLO box tolerance.")
    return thresholds


def annotation_qa_split_names(scope: str, payload: dict) -> list[str]:
    normalized = str(scope or "val").lower()
    available = [split for split in SPLIT_NAMES if payload.get(split)]
    if normalized == "all":
        return available
    if normalized in SPLIT_NAMES and payload.get(normalized):
        return [normalized]
    if payload.get("val"):
        return ["val"]
    return available[:1]


def yolo_label_path(image_path: Path, labels_path: Optional[Path]) -> Optional[Path]:
    return labels_path / f"{image_path.stem}.txt" if labels_path else None


def pixel_bbox_from_yolo(fields: list[str], width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    if len(fields) != 5:
        return None
    try:
        _class_id = int(float(fields[0]))
        x_center, y_center, box_width, box_height = [float(value) for value in fields[1:5]]
    except ValueError:
        return None
    if box_width <= 0 or box_height <= 0:
        return None
    x1 = int(round((x_center - box_width / 2) * width))
    y1 = int(round((y_center - box_height / 2) * height))
    x2 = int(round((x_center + box_width / 2) * width))
    y2 = int(round((y_center + box_height / 2) * height))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def yolo_bbox_from_pixels(class_id, box: list[int] | tuple[int, int, int, int], width: int, height: int) -> str:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Corrected bounding box is empty.")
    x_center = ((x1 + x2) / 2) / width
    y_center = ((y1 + y2) / 2) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return " ".join([
        str(int(float(class_id))),
        f"{x_center:.6f}",
        f"{y_center:.6f}",
        f"{box_width:.6f}",
        f"{box_height:.6f}",
    ])


def normalize_annotation_qa_task(value: str) -> str:
    task = str(value or "auto").strip().lower()
    aliases = {"detect": "detect", "detection": "detect", "segment": "segment", "segmentation": "segment", "auto": "auto"}
    if task not in aliases:
        raise ValueError("Annotation task must be auto, detection, or segmentation.")
    return aliases[task]


def annotation_task_for_fields(fields: list[str]) -> Optional[str]:
    if len(fields) == 5:
        return "detect"
    # YOLO segmentation is a class id followed by at least three x/y pairs.
    if len(fields) >= 7 and (len(fields) - 1) % 2 == 0:
        try:
            [float(value) for value in fields]
        except ValueError:
            return None
        return "segment"
    return None


def detect_annotation_qa_task(split_contexts: list[tuple], requested_task: str = "auto") -> str:
    requested = normalize_annotation_qa_task(requested_task)
    observed: set[str] = set()
    files_checked = 0
    for _split, _images_path, labels_path, images in split_contexts:
        if labels_path is None:
            continue
        for image_path in images:
            label_path = yolo_label_path(image_path, labels_path)
            if label_path is None or not label_path.is_file():
                continue
            files_checked += 1
            for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
                fields = line.strip().split()
                if not fields:
                    continue
                task = annotation_task_for_fields(fields)
                if task:
                    observed.add(task)
            if files_checked >= 200 or len(observed) > 1:
                break
        if files_checked >= 200 or len(observed) > 1:
            break
    if len(observed) > 1:
        raise RuntimeError("Annotation QA does not accept mixed detection and segmentation rows in one dataset.")
    detected = next(iter(observed), None)
    if requested != "auto":
        if detected and detected != requested:
            raise RuntimeError(f"The selected {requested} task does not match the detected {detected} annotations.")
        return requested
    if detected is None:
        raise RuntimeError("Could not detect YOLO detection boxes or segmentation polygons in the selected splits.")
    return detected


def pixel_polygon_from_yolo(fields: list[str], width: int, height: int) -> Optional[list[list[int]]]:
    if annotation_task_for_fields(fields) != "segment":
        return None
    try:
        coordinates = [float(value) for value in fields[1:]]
    except ValueError:
        return None
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in coordinates):
        return None
    points = [
        [
            max(0, min(width - 1, int(round(coordinates[index] * width)))),
            max(0, min(height - 1, int(round(coordinates[index + 1] * height)))),
        ]
        for index in range(0, len(coordinates), 2)
    ]
    if len({tuple(point) for point in points}) < 3:
        return None
    area = abs(sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )) / 2
    return points if area >= 1.0 else None


def polygon_bbox(points: list[list[int]]) -> tuple[int, int, int, int]:
    xs = [int(point[0]) for point in points]
    ys = [int(point[1]) for point in points]
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def polygon_mask(points: list[list[int]], width: int, height: int):
    import cv2
    import numpy as np

    mask = np.zeros((height, width), dtype="uint8")
    cv2.fillPoly(mask, [np.asarray(points, dtype="int32")], 1)
    return mask


def mask_to_polygon(mask, width: int, height: int, max_points: int = 256) -> Optional[list[list[int]]]:
    import cv2
    import numpy as np

    array = mask_to_uint8(mask, width, height)
    contours, _hierarchy = cv2.findContours(array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= 1.0]
    if not contours:
        return None
    contours.sort(key=cv2.contourArea, reverse=True)
    largest_area = float(cv2.contourArea(contours[0]))
    if len(contours) > 1 and float(cv2.contourArea(contours[1])) > largest_area * 0.05:
        return None
    perimeter = cv2.arcLength(contours[0], True)
    epsilon = max(0.5, perimeter * 0.0025)
    simplified = cv2.approxPolyDP(contours[0], epsilon, True).reshape(-1, 2)
    while len(simplified) > max_points and epsilon < perimeter * 0.05:
        epsilon *= 1.5
        simplified = cv2.approxPolyDP(contours[0], epsilon, True).reshape(-1, 2)
    points = [[int(point[0]), int(point[1])] for point in simplified]
    return points if 3 <= len(points) <= max_points else None


def yolo_polygon_from_pixels(class_id, points: list[list[int]], width: int, height: int) -> str:
    if width <= 0 or height <= 0 or len(points) < 3:
        raise ValueError("Corrected segmentation polygon is invalid.")
    values = [str(int(float(class_id)))]
    for x, y in points:
        values.extend((
            f"{max(0.0, min(1.0, float(x) / width)):.6f}",
            f"{max(0.0, min(1.0, float(y) / height)):.6f}",
        ))
    return " ".join(values)


def annotation_qa_segmentation_metrics(original_mask, sam_mask) -> dict:
    import cv2
    import numpy as np

    original = np.asarray(original_mask) > 0
    candidate = np.asarray(sam_mask) > 0
    intersection = int(np.logical_and(original, candidate).sum())
    original_area = int(original.sum())
    sam_area = int(candidate.sum())
    union = original_area + sam_area - intersection
    mask_iou = intersection / union if union else 0.0
    coverage = intersection / original_area if original_area else 0.0
    precision = intersection / sam_area if sam_area else 0.0
    area_ratio = sam_area / original_area if original_area else 0.0

    def centroid(binary):
        moments = cv2.moments(binary.astype("uint8"))
        if moments["m00"] <= 0:
            return 0.0, 0.0
        return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]

    ox, oy = centroid(original)
    sx, sy = centroid(candidate)
    height, width = original.shape[:2]
    centroid_shift = math.hypot(ox - sx, oy - sy) / max(1.0, width, height)
    kernel = np.ones((3, 3), dtype="uint8")
    original_edge = cv2.morphologyEx(original.astype("uint8"), cv2.MORPH_GRADIENT, kernel) > 0
    sam_edge = cv2.morphologyEx(candidate.astype("uint8"), cv2.MORPH_GRADIENT, kernel) > 0
    expanded_original = cv2.dilate(original_edge.astype("uint8"), kernel) > 0
    expanded_sam = cv2.dilate(sam_edge.astype("uint8"), kernel) > 0
    original_edge_count = int(original_edge.sum())
    sam_edge_count = int(sam_edge.sum())
    boundary_recall = int(np.logical_and(original_edge, expanded_sam).sum()) / original_edge_count if original_edge_count else 0.0
    boundary_precision = int(np.logical_and(sam_edge, expanded_original).sum()) / sam_edge_count if sam_edge_count else 0.0
    boundary_f1 = (
        2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
        if boundary_precision + boundary_recall > 0 else 0.0
    )
    return {
        "mask_iou": round(mask_iou, 4),
        "original_coverage": round(coverage, 4),
        "sam_precision": round(precision, 4),
        "mask_area_ratio": round(area_ratio, 4),
        "centroid_shift": round(centroid_shift, 4),
        "boundary_f1": round(boundary_f1, 4),
        "original_mask_area": original_area,
        "sam_mask_area": sam_area,
    }


def bbox_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = bbox_area(box_a) + bbox_area(box_b) - intersection
    return intersection / union if union > 0 else 0.0


def bbox_center_shift(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax = (box_a[0] + box_a[2]) / 2
    ay = (box_a[1] + box_a[3]) / 2
    bx = (box_b[0] + box_b[2]) / 2
    by = (box_b[1] + box_b[3]) / 2
    scale = max(1.0, box_a[2] - box_a[0], box_a[3] - box_a[1])
    return math.hypot(ax - bx, ay - by) / scale


def bbox_edge_differences(
    yolo_box: tuple[int, int, int, int],
    sam_box: tuple[int, int, int, int],
    tolerance_percent: float,
    max_difference_percent: Optional[float] = None,
    pixel_floor: int = 2,
) -> dict:
    yolo_width = max(1, yolo_box[2] - yolo_box[0])
    yolo_height = max(1, yolo_box[3] - yolo_box[1])
    differences = {
        "left": abs(sam_box[0] - yolo_box[0]),
        "right": abs(sam_box[2] - yolo_box[2]),
        "top": abs(sam_box[1] - yolo_box[1]),
        "bottom": abs(sam_box[3] - yolo_box[3]),
    }
    percentages = {
        "left": differences["left"] / yolo_width * 100,
        "right": differences["right"] / yolo_width * 100,
        "top": differences["top"] / yolo_height * 100,
        "bottom": differences["bottom"] / yolo_height * 100,
    }
    tolerance = float(tolerance_percent)
    floor = max(0, int(pixel_floor))
    tolerance_pixels = {
        "left": max(floor, yolo_width * tolerance / 100),
        "right": max(floor, yolo_width * tolerance / 100),
        "top": max(floor, yolo_height * tolerance / 100),
        "bottom": max(floor, yolo_height * tolerance / 100),
    }
    maximum = float(max_difference_percent) if max_difference_percent is not None else None
    maximum_pixels = None
    if maximum is not None:
        maximum_pixels = {
            "left": max(floor, yolo_width * maximum / 100),
            "right": max(floor, yolo_width * maximum / 100),
            "top": max(floor, yolo_height * maximum / 100),
            "bottom": max(floor, yolo_height * maximum / 100),
        }
    within_tolerance = all(
        differences[key] <= tolerance_pixels[key]
        for key in differences
    )
    within_maximum = (
        all(differences[key] <= maximum_pixels[key] for key in differences)
        if maximum_pixels is not None
        else None
    )
    return {
        "pixels": differences,
        "percent": {key: round(value, 4) for key, value in percentages.items()},
        "max_percent": round(max(percentages.values()), 4),
        "tolerance_percent": round(tolerance, 4),
        "max_difference_percent": round(maximum, 4) if maximum is not None else None,
        "pixel_floor": floor,
        "tolerance_pixels": {key: round(value, 4) for key, value in tolerance_pixels.items()},
        "max_difference_pixels": (
            {key: round(value, 4) for key, value in maximum_pixels.items()}
            if maximum_pixels is not None
            else None
        ),
        "within_tolerance": within_tolerance,
        "within_max_difference": within_maximum,
    }


def annotation_qa_difference_band(edge_differences: dict) -> str:
    if edge_differences.get("within_tolerance"):
        return "within_tolerance"
    if edge_differences.get("within_max_difference"):
        return "reviewable"
    return "large_disagreement"


def annotation_qa_prompt_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    expansion_percent: float = 0.0,
    shift_x_percent: float = 0.0,
    shift_y_percent: float = 0.0,
) -> tuple[int, int, int, int]:
    box_width = max(1, box[2] - box[0])
    box_height = max(1, box[3] - box[1])
    expand_x = box_width * max(0.0, float(expansion_percent)) / 100
    expand_y = box_height * max(0.0, float(expansion_percent)) / 100
    shift_x = box_width * float(shift_x_percent) / 100
    shift_y = box_height * float(shift_y_percent) / 100
    x1 = max(0, min(width - 1, int(round(box[0] - expand_x + shift_x))))
    y1 = max(0, min(height - 1, int(round(box[1] - expand_y + shift_y))))
    x2 = max(x1 + 1, min(width, int(round(box[2] + expand_x + shift_x))))
    y2 = max(y1 + 1, min(height, int(round(box[3] + expand_y + shift_y))))
    return x1, y1, x2, y2


def annotation_qa_prompt_plan(
    labels: list[dict],
    width: int,
    height: int,
    expansion_percent: float,
    jitter_percent: float,
) -> tuple[list[tuple[int, int, int, int]], list[dict]]:
    prompts: list[tuple[int, int, int, int]] = []
    references: list[dict] = []
    for label_index, label in enumerate(labels):
        original = tuple(label["bbox"])
        direction = -1.0 if label_index % 2 else 1.0
        variants = (
            ("original", annotation_qa_prompt_box(original, width, height)),
            ("expanded", annotation_qa_prompt_box(
                original, width, height, expansion_percent,
            )),
            ("jittered", annotation_qa_prompt_box(
                original, width, height, expansion_percent,
                direction * jitter_percent, -direction * jitter_percent,
            )),
        )
        for variant, prompt_box in variants:
            references.append({
                "label_index": label_index,
                "label_row": label.get("row_index"),
                "variant": variant,
                "prompt_bbox": prompt_box,
            })
            prompts.append(prompt_box)
    return prompts, references


def annotation_qa_mask_iou(mask_a, mask_b) -> float:
    import numpy as np

    first = np.asarray(mask_a) > 0
    second = np.asarray(mask_b) > 0
    intersection = int(np.logical_and(first, second).sum())
    union = int(np.logical_or(first, second).sum())
    return intersection / union if union > 0 else 0.0


def annotation_qa_prompt_stability(
    candidates: list[dict],
    yolo_box: tuple[int, int, int, int],
    bbox_iou_min: float,
    edge_percent_max: float,
) -> dict:
    expected_variants = {"original", "expanded", "jittered"}
    valid = [candidate for candidate in candidates if candidate.get("bbox") is not None]
    variants = {str(candidate.get("variant")) for candidate in valid}
    complete = expected_variants.issubset(variants)
    pairwise_bbox_ious = []
    pairwise_mask_ious = []
    for left in range(len(valid)):
        for right in range(left + 1, len(valid)):
            pairwise_bbox_ious.append(bbox_iou(valid[left]["bbox"], valid[right]["bbox"]))
            if valid[left].get("mask") is not None and valid[right].get("mask") is not None:
                pairwise_mask_ious.append(annotation_qa_mask_iou(valid[left]["mask"], valid[right]["mask"]))
    yolo_width = max(1, yolo_box[2] - yolo_box[0])
    yolo_height = max(1, yolo_box[3] - yolo_box[1])
    edge_spreads = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0}
    if valid:
        boxes = [candidate["bbox"] for candidate in valid]
        edge_spreads = {
            "left": (max(box[0] for box in boxes) - min(box[0] for box in boxes)) / yolo_width * 100,
            "right": (max(box[2] for box in boxes) - min(box[2] for box in boxes)) / yolo_width * 100,
            "top": (max(box[1] for box in boxes) - min(box[1] for box in boxes)) / yolo_height * 100,
            "bottom": (max(box[3] for box in boxes) - min(box[3] for box in boxes)) / yolo_height * 100,
        }
    max_edge_spread = max(edge_spreads.values())
    minimum_bbox_iou = min(pairwise_bbox_ious) if pairwise_bbox_ious else 0.0
    minimum_mask_iou = min(pairwise_mask_ious) if pairwise_mask_ious else 0.0
    confidences = [
        float(candidate["confidence"])
        for candidate in valid
        if candidate.get("confidence") is not None
    ]
    confidence_complete = len(confidences) == len(valid) and complete
    expanded_edge_clipped = False
    for candidate in valid:
        if candidate.get("variant") not in {"expanded", "jittered"}:
            continue
        prompt = candidate.get("prompt_bbox")
        candidate_box = candidate.get("bbox")
        if not prompt or not candidate_box:
            continue
        margin_x = max(2, int((prompt[2] - prompt[0]) * 0.02))
        margin_y = max(2, int((prompt[3] - prompt[1]) * 0.02))
        if any((
            abs(candidate_box[0] - prompt[0]) <= margin_x,
            abs(candidate_box[2] - prompt[2]) <= margin_x,
            abs(candidate_box[1] - prompt[1]) <= margin_y,
            abs(candidate_box[3] - prompt[3]) <= margin_y,
        )):
            expanded_edge_clipped = True
            break
    passed = (
        complete
        and minimum_bbox_iou >= float(bbox_iou_min)
        and max_edge_spread <= float(edge_percent_max)
    )
    return {
        "complete": complete,
        "variants_expected": sorted(expected_variants),
        "variants_available": sorted(variants),
        "minimum_bbox_iou": round(minimum_bbox_iou, 4),
        "minimum_mask_iou": round(minimum_mask_iou, 4),
        "edge_spread_percent": {key: round(value, 4) for key, value in edge_spreads.items()},
        "max_edge_spread_percent": round(max_edge_spread, 4),
        "bbox_iou_minimum_required": round(float(bbox_iou_min), 4),
        "edge_percent_maximum_allowed": round(float(edge_percent_max), 4),
        "confidence_min": round(min(confidences), 4) if confidences else None,
        "confidence_mean": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "confidence_complete": confidence_complete,
        "expanded_edge_clipped": expanded_edge_clipped,
        "passed": passed,
    }


def annotation_qa_select_candidate(candidates: list[dict]) -> Optional[dict]:
    valid = [candidate for candidate in candidates if candidate.get("bbox") is not None]
    if not valid:
        return None
    variant_priority = {"expanded": 2, "original": 1, "jittered": 0}
    return max(
        valid,
        key=lambda candidate: (
            float(candidate.get("confidence")) if candidate.get("confidence") is not None else -1.0,
            variant_priority.get(str(candidate.get("variant")), -1),
        ),
    )


def annotation_qa_max_neighbor_iou(labels: list[dict], label_index: int) -> float:
    if label_index < 0 or label_index >= len(labels):
        return 0.0
    box = labels[label_index]["bbox"]
    overlaps = [
        bbox_iou(box, other["bbox"])
        for index, other in enumerate(labels)
        if index != label_index
    ]
    return max(overlaps) if overlaps else 0.0


def annotation_qa_auto_gate(
    *,
    stability: dict,
    sam_confidence: Optional[float],
    bbox_overlap: float,
    center_shift: float,
    neighbor_iou: float,
    quality_gate_passed: bool,
    thresholds: dict,
) -> tuple[bool, list[str]]:
    reasons = []
    if not quality_gate_passed:
        reasons.append("review_quality_gate_failed")
    if not stability.get("passed"):
        reasons.append("prompt_stability_failed")
    if stability.get("expanded_edge_clipped"):
        reasons.append("expanded_prompt_edge_clipped")
    if sam_confidence is None or sam_confidence < thresholds["sam_auto_quality_min"]:
        reasons.append("sam_quality_below_auto_threshold")
    if not stability.get("confidence_complete", True):
        reasons.append("sam_quality_missing_for_prompt_variant")
    if bbox_overlap < thresholds["sam_auto_yolo_iou_min"]:
        reasons.append("yolo_sam_iou_below_auto_threshold")
    if center_shift > thresholds["sam_auto_center_shift_max"]:
        reasons.append("center_shift_above_auto_threshold")
    if neighbor_iou > thresholds["sam_auto_neighbor_iou_max"]:
        reasons.append("neighbor_overlap_ambiguous")
    return not reasons, reasons


def annotation_qa_audit_required(
    job_id: str,
    image_name: str,
    class_id: Optional[int],
    label_row: Optional[int],
    decision: str,
    audit_percent: float,
) -> bool:
    percentage = max(0.0, min(100.0, float(audit_percent)))
    if percentage <= 0:
        return False
    key = f"{job_id}:{image_name}:{class_id}:{label_row}:{decision}".encode("utf-8")
    bucket = int(hashlib.sha256(key).hexdigest()[:8], 16) / 0xFFFFFFFF * 100
    return bucket < percentage


def issue_is_safe_sam_replacement(issue: dict) -> bool:
    edge_differences = (issue.get("metrics") or {}).get("edge_differences") or {}
    prompt_stability = (issue.get("metrics") or {}).get("prompt_stability") or {}
    recommended_bbox = issue.get("recommended_bbox")
    return (
        issue.get("auto_fix_eligible") is True
        and issue.get("quality_gate_passed") is True
        and issue.get("difference_band") == "reviewable"
        and issue.get("fix_type") == "replace_box"
        and prompt_stability.get("passed", True) is True
        and isinstance(recommended_bbox, list)
        and len(recommended_bbox) == 4
        and edge_differences.get("within_tolerance") is False
        and edge_differences.get("within_max_difference") is True
        and issue.get("sam_max_difference_percent") is not None
    )


def issue_is_reviewable_sam_polygon(issue: dict) -> bool:
    prompt_stability = (issue.get("metrics") or {}).get("prompt_stability") or {}
    polygon = issue.get("recommended_polygon")
    return (
        issue.get("annotation_task") == "segment"
        and issue.get("auto_fix_eligible") is True
        and issue.get("quality_gate_passed") is True
        and issue.get("difference_band") == "reviewable"
        and issue.get("fix_type") == "replace_polygon"
        and prompt_stability.get("passed", True) is True
        and isinstance(polygon, list)
        and len(polygon) >= 3
    )


def mask_bbox(mask) -> Optional[tuple[int, int, int, int]]:
    import numpy as np

    rows, cols = np.where(mask > 0)
    if rows.size == 0 or cols.size == 0:
        return None
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def mask_to_uint8(mask, width: int, height: int):
    import cv2
    import numpy as np

    array = np.asarray(mask)
    if array.shape[:2] != (height, width):
        array = cv2.resize(array.astype("float32"), (width, height), interpolation=cv2.INTER_NEAREST)
    return (array > 0.5).astype("uint8")


def annotation_issue(
    job_id: str,
    index: int,
    image_path: Path,
    split: str,
    class_id: Optional[int],
    class_name: str,
    issue_type: str,
    severity: str,
    score: float,
    message: str,
    original_bbox: Optional[tuple[int, int, int, int]] = None,
    sam_bbox: Optional[tuple[int, int, int, int]] = None,
    metrics: Optional[dict] = None,
    preview: str = "",
    label_row: Optional[int] = None,
    recommended_bbox: Optional[tuple[int, int, int, int]] = None,
    annotation_task: str = "detect",
    original_polygon: Optional[list[list[int]]] = None,
    recommended_polygon: Optional[list[list[int]]] = None,
) -> dict:
    return {
        "issue_id": f"{job_id}-{index:06d}",
        "image": str(image_path),
        "image_name": image_path.name,
        "split": split,
        "class_id": class_id,
        "class_name": class_name,
        "issue_type": issue_type,
        "severity": severity,
        "score": round(float(score), 4),
        "message": message,
        "original_bbox": list(original_bbox) if original_bbox else None,
        "sam_bbox": list(sam_bbox) if sam_bbox else None,
        "recommended_bbox": list(recommended_bbox) if recommended_bbox else None,
        "original_polygon": original_polygon or None,
        "recommended_polygon": recommended_polygon or None,
        "annotation_task": annotation_task,
        "fix_type": "replace_polygon" if recommended_polygon else ("replace_box" if recommended_bbox else ""),
        "accepted_fix": "",
        "accepted_class_id": None,
        "accepted_class_name": "",
        "label_row": label_row,
        "metrics": metrics or {},
        "preview": preview,
        "review_status": "unreviewed",
    }


def draw_annotation_qa_preview(
    image_path: Path,
    output_path: Path,
    issue: dict,
    mask=None,
    original_mask=None,
):
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        return
    height, width = image.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_name(f"{output_path.stem}.raw.jpg")
    cv2.imwrite(str(raw_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    issue["raw_preview"] = f"previews/{raw_path.name}"
    if original_mask is not None:
        original_mask_array = mask_to_uint8(original_mask, width, height)
        original_mask_path = output_path.with_name(f"{output_path.stem}.original-mask.jpg")
        cv2.imwrite(str(original_mask_path), original_mask_array * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        issue["original_mask_preview"] = f"previews/{original_mask_path.name}"
        original_overlay = image.copy()
        original_overlay[original_mask_array > 0] = (40, 210, 255)
        image = cv2.addWeighted(original_overlay, 0.22, image, 0.78, 0)
    if mask is not None:
        mask_array = mask_to_uint8(mask, width, height)
        mask_path = output_path.with_name(f"{output_path.stem}.mask.jpg")
        cv2.imwrite(str(mask_path), mask_array * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        issue["mask_preview"] = f"previews/{mask_path.name}"
        overlay = image.copy()
        overlay[mask_array > 0] = (255, 220, 70)
        image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)
    original = issue.get("original_bbox")
    original_polygon = issue.get("original_polygon")
    if original_polygon:
        cv2.polylines(image, [np.asarray(original_polygon, dtype="int32")], True, (0, 210, 255), 2)
    elif original:
        x1, y1, x2, y2 = [int(value) for value in original]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 210, 255), 2)
    sam_box = issue.get("sam_bbox")
    recommended_polygon = issue.get("recommended_polygon") or issue.get("sam_polygon")
    if recommended_polygon:
        cv2.polylines(image, [np.asarray(recommended_polygon, dtype="int32")], True, (255, 150, 0), 2)
    elif sam_box:
        x1, y1, x2, y2 = [int(value) for value in sam_box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 150, 0), 2)
    caption = f"{issue['severity'].upper()} {issue['issue_type']} {issue['score']:.2f}"
    cv2.rectangle(image, (8, 8), (min(width - 8, 16 + len(caption) * 9), 38), (20, 32, 40), -1)
    cv2.putText(image, caption, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 84])


def annotation_qa_summary(
    issues: list[dict],
    images_scanned: int,
    labels_checked: int,
    model: str,
    scope: str,
    preset: str,
    box_tolerance_percent: float = 5.0,
    sam_max_difference_percent: float = 25.0,
    yolo_boxes_accepted: int = 0,
    policy: Optional[dict] = None,
    class_label_counts: Optional[dict] = None,
    annotation_task: str = "detect",
) -> dict:
    severity_counts = {"high": 0, "medium": 0, "low": 0}
    type_counts: dict[str, int] = {}
    difference_band_counts: dict[str, int] = {}
    replacements_available = 0
    replacements_blocked = 0
    decision_counts: dict[str, int] = {"auto_keep_yolo": yolo_boxes_accepted}
    automatic_fixes_queued = 0
    audits_pending = 0
    calibration: dict[str, dict] = {
        str(class_id): {
            "labels": int(count),
            "issues": 0,
            "decisions": {},
            "audits": {"pending": 0, "passed": 0, "failed": 0},
        }
        for class_id, count in (class_label_counts or {}).items()
    }
    for issue in issues:
        severity = issue.get("severity", "low")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        issue_type = issue.get("issue_type", "unknown")
        type_counts[issue_type] = type_counts.get(issue_type, 0) + 1
        difference_band = issue.get("difference_band")
        if difference_band:
            difference_band_counts[difference_band] = difference_band_counts.get(difference_band, 0) + 1
        decision = str(issue.get("qa_decision") or "human_review")
        if decision != "auto_keep_yolo":
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        if issue.get("accepted_fix") in {"sam_box", "sam_polygon"} and issue.get("accepted_fix_source") == "automatic":
            automatic_fixes_queued += 1
        if issue.get("audit_required") and issue.get("audit_status") == "pending":
            audits_pending += 1
        if issue.get("class_id") is not None:
            class_key = str(issue.get("class_id"))
            class_row = calibration.setdefault(class_key, {
                "labels": 0,
                "issues": 0,
                "decisions": {},
                "audits": {"pending": 0, "passed": 0, "failed": 0},
            })
            class_row["issues"] += 1
            class_row["decisions"][decision] = class_row["decisions"].get(decision, 0) + 1
            if issue.get("audit_required"):
                audit_status = str(issue.get("audit_status") or "pending")
                class_row["audits"][audit_status] = class_row["audits"].get(audit_status, 0) + 1
        if issue_is_safe_sam_replacement(issue) or issue_is_reviewable_sam_polygon(issue):
            replacements_available += 1
        elif (issue.get("sam_bbox") or issue.get("recommended_polygon")) and difference_band in {"reviewable", "large_disagreement"}:
            replacements_blocked += 1
    return {
        "report_version": ANNOTATION_QA_REPORT_VERSION,
        "model": model,
        "scope": scope,
        "preset": preset,
        "annotation_task": annotation_task,
        "box_tolerance_percent": round(float(box_tolerance_percent), 4),
        "sam_max_difference_percent": round(float(sam_max_difference_percent), 4),
        "images_scanned": images_scanned,
        "labels_checked": labels_checked,
        "boxes_checked": labels_checked if annotation_task == "detect" else 0,
        "masks_checked": labels_checked if annotation_task == "segment" else 0,
        "yolo_boxes_accepted": yolo_boxes_accepted,
        "original_annotations_accepted": yolo_boxes_accepted,
        "segmentation_report_version": 1 if annotation_task == "segment" else None,
        "moderate_disagreements": difference_band_counts.get("reviewable", 0),
        "large_disagreements": difference_band_counts.get("large_disagreement", 0),
        "sam_replacements_available": replacements_available,
        "sam_replacements_blocked": replacements_blocked,
        "qa_decisions": decision_counts,
        "automatic_fixes_queued": automatic_fixes_queued,
        "audits_pending": audits_pending,
        "auto_correction_policy": dict(policy or {}),
        "calibration_by_class": calibration,
        "issues": len(issues),
        "high": severity_counts.get("high", 0),
        "medium": severity_counts.get("medium", 0),
        "low": severity_counts.get("low", 0),
        "issue_types": type_counts,
        "difference_bands": difference_band_counts,
        "completed_at": datetime.now(MYT).isoformat(),
    }


def write_annotation_qa_report(run_dir: Path, issues: list[dict], summary: dict):
    write_json_object(run_dir / ANNOTATION_QA_SUMMARY_FILE, summary)
    write_json_object(run_dir / ANNOTATION_QA_REPORT_JSON_FILE, {"summary": summary, "issues": issues})
    write_json_object(run_dir / ANNOTATION_QA_REVIEW_FILE, {
        issue["issue_id"]: issue.get("review_status", "unreviewed")
        for issue in issues
    })
    csv_path = run_dir / ANNOTATION_QA_REPORT_CSV_FILE
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "issue_id", "image_name", "split", "class_id", "class_name", "issue_type",
        "severity", "score", "message", "preview", "review_status", "accepted_fix",
        "accepted_class_id", "accepted_class_name", "sam_prompt_index", "sam_confidence",
        "box_tolerance_percent", "sam_max_difference_percent", "difference_band",
        "quality_gate_passed", "auto_fix_eligible", "applied", "corrected_label_path",
        "qa_decision", "decision_reasons", "automatic_fix_eligible", "accepted_fix_source",
        "audit_required", "audit_status", "sam_selected_variant",
        "annotation_task", "original_polygon", "recommended_polygon",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for issue in issues:
            writer.writerow({field: issue.get(field, "") for field in fields})


def annotation_qa_run_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
        raise HTTPException(status_code=404, detail="Annotation QA job not found.")
    run_dir = (ANNOTATION_QA_ROOT / job_id).resolve()
    ensure_annotation_qa_path(run_dir)
    return run_dir


def load_annotation_qa_report(run_dir: Path) -> dict:
    report = read_json_object(run_dir / ANNOTATION_QA_REPORT_JSON_FILE)
    if not isinstance(report.get("issues"), list):
        raise HTTPException(status_code=404, detail="Annotation QA results are not available.")
    if not isinstance(report.get("summary"), dict):
        report["summary"] = read_json_object(run_dir / ANNOTATION_QA_SUMMARY_FILE)
    return report


def set_annotation_qa_issue_fix(run_dir: Path, issue_id: str, fix: str, class_id: Optional[int] = None) -> dict:
    report = load_annotation_qa_report(run_dir)
    issue = next(
        (item for item in report["issues"] if isinstance(item, dict) and item.get("issue_id") == issue_id),
        None,
    )
    if issue is None:
        raise HTTPException(status_code=404, detail="Annotation QA issue not found.")

    fix_value = str(fix or "").strip()
    if fix_value in {"", "none", "clear"}:
        issue["accepted_fix"] = ""
        issue["accepted_class_id"] = None
        issue["accepted_class_name"] = ""
    elif fix_value == "sam_box":
        report_version = int((report.get("summary") or {}).get("report_version", 1))
        if report_version < ANNOTATION_QA_SAFE_MAPPING_VERSION:
            raise HTTPException(
                status_code=409,
                detail="This QA report was created before safe SAM prompt mapping was available. Run QA again before accepting SAM box fixes.",
            )
        if report_version < ANNOTATION_QA_REPORT_VERSION:
            raise HTTPException(
                status_code=409,
                detail="This QA report predates the maximum SAM-difference safety gate. Run QA again before accepting SAM box fixes.",
            )
        if not issue_is_safe_sam_replacement(issue):
            raise HTTPException(
                status_code=400,
                detail="This SAM result is outside the safe correction band or is not reliable enough for replacement.",
            )
        issue["accepted_fix"] = "sam_box"
        issue["review_status"] = "fix_accepted"
        if issue.get("audit_required"):
            issue["audit_status"] = "passed"
            issue["accepted_fix_source"] = "human_audited"
    elif fix_value == "sam_polygon":
        if not issue_is_reviewable_sam_polygon(issue):
            raise HTTPException(
                status_code=400,
                detail="This SAM polygon is not a stable, representable segmentation suggestion.",
            )
        issue["accepted_fix"] = "sam_polygon"
        issue["accepted_fix_source"] = "human"
        issue["review_status"] = "fix_accepted"
    elif fix_value == "class":
        if class_id is None:
            raise HTTPException(status_code=400, detail="Class fix requires a class id.")
        summary = report.get("summary") or {}
        dataset_yaml = summary.get("dataset_yaml")
        if not dataset_yaml:
            raise HTTPException(status_code=400, detail="Annotation QA report is missing dataset class metadata.")
        class_names = read_yaml_class_names(Path(dataset_yaml))
        if class_id < 0 or class_id >= len(class_names):
            raise HTTPException(status_code=400, detail="Selected class is not in the dataset class list.")
        issue["accepted_class_id"] = class_id
        issue["accepted_class_name"] = class_names[class_id]
        issue["review_status"] = "fix_accepted"
    else:
        raise HTTPException(status_code=400, detail="Unknown annotation QA fix.")

    write_annotation_qa_report(run_dir, report["issues"], report.get("summary") or {})
    return issue


def issue_label_path_in_copy(issue: dict, dataset_root: Path, corrected_root: Path) -> Path:
    image_path = Path(str(issue.get("image", ""))).expanduser().resolve()
    try:
        relative_image = image_path.relative_to(dataset_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="QA issue image is outside the source dataset.") from exc

    parts = list(relative_image.parts)
    try:
        images_index = parts.index("images")
    except ValueError:
        split = str(issue.get("split") or "").strip() or "train"
        label_relative = Path("labels") / split / f"{image_path.stem}.txt"
    else:
        parts[images_index] = "labels"
        label_relative = Path(*parts).with_suffix(".txt")
    return corrected_root / label_relative


def apply_annotation_qa_fix(issue: dict, dataset_root: Path, corrected_root: Path) -> dict:
    has_box_fix = issue.get("accepted_fix") == "sam_box"
    has_polygon_fix = issue.get("accepted_fix") == "sam_polygon"
    has_class_fix = issue.get("accepted_class_id") is not None
    if not has_box_fix and not has_polygon_fix and not has_class_fix:
        return {"applied": False, "reason": "No accepted fix."}
    if has_box_fix and not issue_is_safe_sam_replacement(issue):
        return {"applied": False, "reason": "Issue is outside the safe SAM correction band."}
    if has_polygon_fix and not issue_is_reviewable_sam_polygon(issue):
        return {"applied": False, "reason": "SAM polygon is not a safe manual suggestion."}
    label_row = issue.get("label_row")
    if not isinstance(label_row, int) or label_row < 1:
        return {"applied": False, "reason": "Issue has no label row reference."}

    label_path = issue_label_path_in_copy(issue, dataset_root, corrected_root)
    if not label_path.is_file():
        return {"applied": False, "reason": f"Copied label file not found: {label_path}"}
    try:
        relative_image = Path(str(issue.get("image"))).expanduser().resolve().relative_to(dataset_root)
    except ValueError:
        return {"applied": False, "reason": "Issue image is outside the source dataset."}
    image_path = corrected_root / relative_image
    try:
        import cv2
        image = cv2.imread(str(image_path))
    except Exception:
        image = None
    if image is None:
        return {"applied": False, "reason": f"Copied image could not be read: {image_path}"}
    height, width = image.shape[:2]

    lines = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
    line_index = label_row - 1
    if line_index >= len(lines):
        return {"applied": False, "reason": "Referenced label row no longer exists."}

    original_row = lines[line_index]
    original_fields = original_row.strip().split()
    original_task = annotation_task_for_fields(original_fields)
    expected_task = "segment" if has_polygon_fix or issue.get("annotation_task") == "segment" else "detect"
    if original_task != expected_task:
        return {"applied": False, "reason": f"Referenced label row is not a YOLO {expected_task} annotation."}
    corrected_class_id = issue.get("accepted_class_id") if has_class_fix else issue.get("class_id")
    try:
        if has_box_fix:
            corrected_row = yolo_bbox_from_pixels(corrected_class_id, issue["recommended_bbox"], width, height)
        elif has_polygon_fix:
            corrected_row = yolo_polygon_from_pixels(corrected_class_id, issue["recommended_polygon"], width, height)
        else:
            corrected_row = " ".join([str(int(float(corrected_class_id))), *original_fields[1:]])
    except (TypeError, ValueError) as exc:
        return {"applied": False, "reason": str(exc)}
    lines[line_index] = corrected_row
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    issue["applied"] = True
    issue["corrected_label_path"] = str(label_path)
    issue["original_yolo_row"] = original_row
    issue["corrected_yolo_row"] = corrected_row
    if has_class_fix:
        issue["class_id"] = issue.get("accepted_class_id")
        issue["class_name"] = issue.get("accepted_class_name") or issue.get("class_name", "")
    return {"applied": True, "label_path": str(label_path)}


def apply_annotation_qa_fixes(job_id: str) -> dict:
    run_dir = annotation_qa_run_dir(job_id)
    report = load_annotation_qa_report(run_dir)
    summary = report.get("summary") or {}
    has_legacy_sam_fix = any(
        isinstance(item, dict) and item.get("accepted_fix") == "sam_box"
        for item in report["issues"]
    )
    report_version = int(summary.get("report_version", 1))
    pending_audits = [
        item for item in report["issues"]
        if isinstance(item, dict)
        and item.get("audit_required")
        and item.get("audit_status") == "pending"
    ]
    if pending_audits:
        raise HTTPException(
            status_code=409,
            detail=f"Review the {len(pending_audits)} sampled automatic SAM correction(s) before creating the corrected dataset.",
        )
    if has_legacy_sam_fix and report_version < ANNOTATION_QA_SAFE_MAPPING_VERSION:
        raise HTTPException(
            status_code=409,
            detail="This QA report predates safe SAM prompt mapping. Run annotation QA again before creating a corrected dataset.",
        )
    if has_legacy_sam_fix and report_version < ANNOTATION_QA_REPORT_VERSION:
        raise HTTPException(
            status_code=409,
            detail="This QA report predates the maximum SAM-difference safety gate. Run annotation QA again before creating a corrected dataset.",
        )
    dataset_yaml = summary.get("dataset_yaml")
    if not dataset_yaml:
        raise HTTPException(status_code=400, detail="Annotation QA report is missing the source dataset YAML.")
    yaml_path, dataset_root, portable_payload = prepared_dataset_yaml(dataset_yaml)

    corrected_name = clean_name(f"{dataset_root.name}_qa_corrected_{job_id}", "qa_corrected_dataset")
    corrected_root = (DATASET_PREPARED_ROOT / corrected_name).resolve()
    try:
        corrected_root.relative_to(DATA_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="Corrected dataset path escaped the data workspace.") from exc
    if corrected_root.exists():
        shutil.rmtree(corrected_root)
    shutil.copytree(dataset_root, corrected_root, ignore=shutil.ignore_patterns(ANNOTATION_QA_FIX_SUMMARY_FILE))

    corrected_yaml = corrected_root / "data.yaml"
    corrected_payload = dict(portable_payload)
    corrected_payload["path"] = "."
    corrected_yaml.write_text(yaml.safe_dump(corrected_payload, sort_keys=False), encoding="utf-8")

    def issue_has_accepted_fix(item: dict) -> bool:
        return item.get("accepted_fix") in {"sam_box", "sam_polygon"} or item.get("accepted_class_id") is not None

    applied = 0
    skipped = []
    for issue in report["issues"]:
        if not isinstance(issue, dict) or not issue_has_accepted_fix(issue):
            continue
        result = apply_annotation_qa_fix(issue, dataset_root, corrected_root)
        if result.get("applied"):
            applied += 1
        else:
            issue["applied"] = False
            issue["apply_error"] = result.get("reason", "Fix was not applied.")
            skipped.append({"issue_id": issue.get("issue_id"), "reason": issue["apply_error"]})

    fix_summary = {
        "job_id": job_id,
        "source_dataset_yaml": str(yaml_path),
        "source_dataset_root": str(dataset_root),
        "corrected_dataset_yaml": str(corrected_yaml),
        "corrected_dataset_root": str(corrected_root),
        "accepted_fixes": sum(1 for item in report["issues"] if isinstance(item, dict) and issue_has_accepted_fix(item)),
        "applied_fixes": applied,
        "skipped_fixes": skipped,
        "applied_at": datetime.now(MYT).isoformat(),
    }
    write_json_object(corrected_root / ANNOTATION_QA_FIX_SUMMARY_FILE, fix_summary)
    write_json_object(run_dir / ANNOTATION_QA_FIX_SUMMARY_FILE, fix_summary)
    report["summary"]["corrected_dataset_yaml"] = str(corrected_yaml)
    report["summary"]["corrected_dataset_root"] = str(corrected_root)
    report["summary"]["applied_fixes"] = applied
    corrected_provenance = load_roboflow_provenance(corrected_root)
    if corrected_provenance.get("provider") == "roboflow":
        report["summary"]["roboflow_sync"] = {
            "workspace": corrected_provenance.get("workspace", ""),
            "project": corrected_provenance.get("project", ""),
            "source_version": corrected_provenance.get("source_version", ""),
            **(corrected_provenance.get("mapping") or {}),
        }
    write_annotation_qa_report(run_dir, report["issues"], report["summary"])

    response = dataset_response(
        corrected_yaml,
        f"Applied {applied} accepted SAM fix{'es' if applied != 1 else ''}. Training will use the corrected dataset copy.",
    )
    response.update(fix_summary)
    if report["summary"].get("roboflow_sync"):
        response["roboflow_sync"] = report["summary"]["roboflow_sync"]
    response["download_available"] = True
    return response


def annotation_qa_roboflow_context(job_id: str) -> tuple[Path, dict, Path, dict, list[str]]:
    run_dir = annotation_qa_run_dir(job_id)
    report = load_annotation_qa_report(run_dir)
    if (report.get("summary") or {}).get("annotation_task") == "segment":
        raise HTTPException(
            status_code=409,
            detail="Roboflow publishing for segmentation corrections is disabled until polygon conflict synchronization is validated.",
        )
    fix_summary = read_json_object(run_dir / ANNOTATION_QA_FIX_SUMMARY_FILE)
    corrected_root_value = fix_summary.get("corrected_dataset_root")
    if not corrected_root_value:
        raise HTTPException(
            status_code=409,
            detail="Create the corrected dataset before publishing annotations to Roboflow.",
        )
    corrected_root = Path(str(corrected_root_value)).expanduser().resolve()
    try:
        corrected_root.relative_to(DATA_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Corrected dataset is outside the web dataset workspace.") from exc
    if not corrected_root.is_dir():
        raise HTTPException(status_code=404, detail="The corrected dataset is no longer available.")

    provenance = load_roboflow_provenance(corrected_root)
    if not provenance:
        source_root_value = fix_summary.get("source_dataset_root")
        source_root = Path(str(source_root_value)).expanduser().resolve() if source_root_value else Path()
        provenance = load_roboflow_provenance(source_root) if source_root_value else {}
    if provenance.get("provider") != "roboflow":
        raise HTTPException(
            status_code=409,
            detail="This QA dataset has no Roboflow source binding. Fetch it from Roboflow again before publishing.",
        )
    if not provenance.get("workspace") or not provenance.get("project"):
        raise HTTPException(status_code=409, detail="Roboflow source workspace or project metadata is missing.")
    class_names = [str(name) for name in provenance.get("class_names") or []]
    if not class_names:
        corrected_yaml = Path(str(fix_summary.get("corrected_dataset_yaml") or corrected_root / "data.yaml"))
        class_names = read_yaml_class_names(corrected_yaml)
    return run_dir, report, corrected_root, provenance, class_names


def build_annotation_qa_roboflow_preview(job_id: str, api_key: str) -> dict:
    run_dir, report, corrected_root, provenance, class_names = annotation_qa_roboflow_context(job_id)
    workspace = str(provenance["workspace"])
    project = str(provenance["project"])
    target = {
        "workspace": workspace,
        "project": project,
        "source_version": str(provenance.get("source_version") or ""),
    }
    changed_images: dict[str, list[str]] = {}
    source_root_value = (report.get("summary") or {}).get("dataset_root")
    source_root = Path(str(source_root_value)).expanduser().resolve() if source_root_value else None
    for issue in report.get("issues") or []:
        if not isinstance(issue, dict) or not issue.get("applied"):
            continue
        image_value = issue.get("image")
        if not image_value or source_root is None:
            continue
        try:
            relative = Path(str(image_value)).expanduser().resolve().relative_to(source_root).as_posix()
        except ValueError:
            continue
        changed_images.setdefault(relative, []).append(str(issue.get("issue_id") or ""))

    manifest_images = provenance.get("images") if isinstance(provenance.get("images"), dict) else {}
    items: list[dict] = []
    for relative, issue_ids in sorted(changed_images.items()):
        entry = manifest_images.get(relative) if isinstance(manifest_images.get(relative), dict) else {}
        item = {
            "local_image": relative,
            "issue_ids": issue_ids,
            "roboflow_image_id": str(entry.get("roboflow_image_id") or ""),
            "roboflow_image_name": str(entry.get("roboflow_image_name") or ""),
            "baseline_digest": str(entry.get("baseline_digest") or ""),
            "status": "ineligible",
            "reason": str(entry.get("reason") or "Image is not bound to a unique Roboflow image ID."),
        }
        if not entry.get("eligible") or not item["roboflow_image_id"]:
            items.append(item)
            continue
        label_relative = Path(str(entry.get("local_label") or ""))
        label_path = (corrected_root / label_relative).resolve()
        try:
            label_path.relative_to(corrected_root)
        except ValueError:
            item["reason"] = "Corrected label path escaped the dataset root."
            items.append(item)
            continue
        corrected = canonical_local_annotation(label_path, class_names)
        item["corrected_digest"] = annotation_digest(corrected)
        item["annotation_name"] = f"{Path(relative).stem}.txt"
        if item["corrected_digest"] == item["baseline_digest"]:
            item.update(status="unchanged", reason="The corrected annotation is identical to the imported annotation.")
            items.append(item)
            continue
        try:
            remote_payload = roboflow_image_details(
                api_key, workspace, project, item["roboflow_image_id"]
            )
            remote = canonical_remote_annotation(remote_payload, class_names)
            item["remote_digest"] = annotation_digest(remote)
        except Exception as exc:
            item.update(status="error", reason=redact_secret(exc, api_key))
            items.append(item)
            continue
        if item["remote_digest"] == item["corrected_digest"]:
            item.update(status="already_synced", reason="Roboflow already contains this corrected annotation.")
        elif item["remote_digest"] != item["baseline_digest"]:
            item.update(status="conflict", reason="The Roboflow annotation changed after this dataset was imported.")
        else:
            item.update(status="ready", reason="Ready to replace the source annotation in the bound Roboflow project.")
        items.append(item)

    counts = {
        status: sum(1 for item in items if item.get("status") == status)
        for status in ("ready", "already_synced", "conflict", "ineligible", "unchanged", "error")
    }
    preview = {
        "job_id": job_id,
        "target": target,
        "items": items,
        "counts": counts,
        "created_at": datetime.now(MYT).isoformat(),
    }
    preview["preview_id"] = sync_preview_digest(target, items)
    write_json_object(run_dir / ROBOFLOW_SYNC_PREVIEW_FILE, preview)
    return preview


def publish_annotation_qa_to_roboflow(job_id: str, api_key: str, preview_id: str) -> dict:
    run_dir, _report, corrected_root, provenance, class_names = annotation_qa_roboflow_context(job_id)
    preview = build_annotation_qa_roboflow_preview(job_id, api_key)
    if not preview_id or preview_id != preview.get("preview_id"):
        raise HTTPException(
            status_code=409,
            detail="The Roboflow preview changed. Review the latest conflicts before publishing.",
        )
    workspace = str(provenance["workspace"])
    project = str(provenance["project"])
    results = []
    manifest_images = provenance.get("images") if isinstance(provenance.get("images"), dict) else {}
    for item in preview["items"]:
        status = item.get("status")
        result = {
            "local_image": item.get("local_image"),
            "roboflow_image_id": item.get("roboflow_image_id"),
            "status": status,
            "reason": item.get("reason", ""),
        }
        if status != "ready":
            results.append(result)
            continue
        entry = manifest_images.get(str(item["local_image"])) or {}
        label_path = (corrected_root / str(entry.get("local_label") or "")).resolve()
        try:
            label_path.relative_to(corrected_root)
            annotation_text = label_path.read_text(encoding="utf-8") if label_path.is_file() else ""
            roboflow_upload_annotation(
                api_key=api_key,
                project=project,
                image_id=str(item["roboflow_image_id"]),
                annotation_text=annotation_text,
                class_names=class_names,
                annotation_name=str(item.get("annotation_name") or f"{label_path.stem}.txt"),
            )
            result.update(status="published", reason="Roboflow accepted the corrected annotation.")
            entry["baseline_annotation"] = canonical_local_annotation(label_path, class_names)
            entry["baseline_digest"] = item.get("corrected_digest")
            entry["last_published_at"] = datetime.now(MYT).isoformat()
        except Exception as exc:
            result.update(status="failed", reason=redact_secret(exc, api_key))
        results.append(result)

    provenance["images"] = manifest_images
    provenance["last_published_at"] = datetime.now(MYT).isoformat()
    write_roboflow_provenance(corrected_root, provenance)
    counts = {
        status: sum(1 for result in results if result.get("status") == status)
        for status in ("published", "already_synced", "conflict", "ineligible", "unchanged", "error", "failed")
    }
    audit = {
        "job_id": job_id,
        "target": {
            "workspace": workspace,
            "project": project,
            "source_version": str(provenance.get("source_version") or ""),
        },
        "preview_id": preview_id,
        "published_at": datetime.now(MYT).isoformat(),
        "counts": counts,
        "results": results,
    }
    write_json_object(run_dir / ROBOFLOW_SYNC_LOG_FILE, audit)
    return audit


def load_sam_model(model_name: str):
    try:
        from ultralytics import SAM
    except Exception as exc:
        raise RuntimeError("Ultralytics SAM is not available in this environment.") from exc
    return SAM(model_name)


def sam_masks_for_image(
    model,
    image_path: Path,
    bboxes: list[tuple[int, int, int, int]],
    device: str,
) -> Optional[list[Optional[dict]]]:
    if not bboxes:
        return []
    # Keep every prompt in the result so that Ultralytics cannot compact away a
    # low-confidence middle prompt and shift later masks onto the wrong label.
    kwargs = {
        "source": str(image_path),
        "bboxes": [list(box) for box in bboxes],
        "conf": 0.0,
        "verbose": False,
    }
    if device:
        kwargs["device"] = device
    results = model.predict(**kwargs)
    if not results:
        return [None] * len(bboxes)
    result = results[0]
    masks = getattr(result, "masks", None)
    data = getattr(masks, "data", None)
    if data is None:
        return [None] * len(bboxes)
    try:
        mask_data = [item.detach().cpu().numpy() for item in data]
    except AttributeError:
        mask_data = list(data)

    boxes = getattr(result, "boxes", None)
    prompt_indices = getattr(boxes, "cls", None)
    if prompt_indices is None or len(prompt_indices) < len(mask_data):
        # A positional fallback would recreate the original corruption.
        return None
    confidences = getattr(boxes, "conf", None)
    mapped: list[Optional[dict]] = [None] * len(bboxes)
    for result_index, mask in enumerate(mask_data):
        try:
            prompt_index = int(prompt_indices[result_index].item())
        except (AttributeError, TypeError, ValueError):
            try:
                prompt_index = int(prompt_indices[result_index])
            except (TypeError, ValueError):
                return None
        if prompt_index < 0 or prompt_index >= len(bboxes) or mapped[prompt_index] is not None:
            return None
        confidence = None
        if confidences is not None and result_index < len(confidences):
            try:
                confidence = float(confidences[result_index].item())
            except (AttributeError, TypeError, ValueError):
                try:
                    confidence = float(confidences[result_index])
                except (TypeError, ValueError):
                    confidence = None
        mapped[prompt_index] = {
            "mask": mask,
            "confidence": confidence,
            "prompt_index": prompt_index,
        }
    return mapped


def annotation_qa_candidates_for_image(
    runtime: SamQaRuntime,
    image,
    labels: list[dict],
    width: int,
    height: int,
    thresholds: dict,
    stop_event: threading.Event,
    force_stability: bool = False,
) -> Optional[list[list[dict]]]:
    """Run original prompts first and stability prompts for reviewable boxes or masks."""
    candidates_by_label: list[list[dict]] = [[] for _ in labels]
    if not labels:
        return candidates_by_label

    def add_candidates(results, references, prompt_offset=0):
        if results is None:
            return False
        for index, reference in enumerate(references):
            result = results[index] if index < len(results) else None
            if result is None or result.get("mask") is None:
                continue
            candidate_mask = mask_to_uint8(result["mask"], width, height)
            candidate_box = mask_bbox(candidate_mask)
            if candidate_box is None:
                continue
            candidates_by_label[reference["label_index"]].append({
                "variant": reference["variant"],
                "prompt_bbox": reference["prompt_bbox"],
                "bbox": candidate_box,
                "mask": candidate_mask,
                "confidence": result.get("confidence"),
                "prompt_index": prompt_offset + index,
            })
        return True

    try:
        runtime.set_image(image)
        original_boxes = [tuple(label["bbox"]) for label in labels]
        original_refs = [
            {
                "label_index": index,
                "label_row": label.get("row_index"),
                "variant": "original",
                "prompt_bbox": tuple(label["bbox"]),
            }
            for index, label in enumerate(labels)
        ]
        original_results = runtime.predict_prompts(original_boxes, stop_event)
        if not add_candidates(original_results, original_refs):
            return None

        stability_boxes = []
        stability_refs = []
        for label_index, label in enumerate(labels):
            original_candidate = annotation_qa_select_candidate(candidates_by_label[label_index])
            if original_candidate is None:
                continue
            differences = bbox_edge_differences(
                tuple(label["bbox"]),
                tuple(original_candidate["bbox"]),
                thresholds["box_tolerance_percent"],
                thresholds["sam_max_difference_percent"],
            )
            if not force_stability and annotation_qa_difference_band(differences) != "reviewable":
                continue
            direction = -1.0 if label_index % 2 else 1.0
            variants = (
                ("expanded", annotation_qa_prompt_box(
                    tuple(label["bbox"]), width, height,
                    thresholds["sam_prompt_expansion_percent"],
                )),
                ("jittered", annotation_qa_prompt_box(
                    tuple(label["bbox"]), width, height,
                    thresholds["sam_prompt_expansion_percent"],
                    direction * thresholds["sam_prompt_jitter_percent"],
                    -direction * thresholds["sam_prompt_jitter_percent"],
                )),
            )
            for variant, prompt_box in variants:
                stability_boxes.append(prompt_box)
                stability_refs.append({
                    "label_index": label_index,
                    "label_row": label.get("row_index"),
                    "variant": variant,
                    "prompt_bbox": prompt_box,
                })
        stability_results = runtime.predict_prompts(stability_boxes, stop_event)
        if not add_candidates(stability_results, stability_refs, len(original_boxes)):
            return None
        return candidates_by_label
    except InterruptedError as exc:
        raise InferenceStopped("Annotation QA was stopped.") from exc
    finally:
        runtime.reset_image()


def run_annotation_qa_job(job_id: str, request_payload: dict, stop_event: threading.Event):
    run_dir = (ANNOTATION_QA_ROOT / job_id).resolve()
    ensure_annotation_qa_path(run_dir)
    preview_dir = run_dir / "previews"
    issues: list[dict] = []
    issue_index = 0
    labels_checked = 0
    images_scanned = 0
    yolo_boxes_accepted = 0
    class_label_counts: dict[str, int] = {}
    sam_runtime: Optional[SamQaRuntime] = None
    policy: dict = {}
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        yaml_path, dataset_root, payload = prepared_dataset_yaml(request_payload["dataset_yaml"])
        class_names = normalize_yaml_names(payload.get("names"))
        split_names = annotation_qa_split_names(request_payload.get("scope", "val"), payload)
        if not split_names:
            raise RuntimeError("No dataset split was available for annotation QA.")
        split_contexts = []
        for split in split_names:
            images_path = split_image_folder(dataset_root, payload.get(split))
            labels_path = label_folder_for_images(dataset_root, images_path)
            images = image_files(images_path) if images_path and images_path.is_dir() else []
            split_contexts.append((split, images_path, labels_path, images))
        annotation_task = detect_annotation_qa_task(
            split_contexts,
            request_payload.get("task", "auto"),
        )
        max_images = request_payload.get("max_images")
        total_images = sum(len(images) for _split, _images_path, _labels_path, images in split_contexts)
        if max_images:
            total_images = min(total_images, int(max_images))
        if total_images <= 0:
            raise RuntimeError("No images were found for the selected QA scope.")

        update_annotation_qa_job(
            job_id,
            status="running",
            stage="loading_model",
            percent=2,
            detail=f"Loading {request_payload['model']}.",
            annotation_task=annotation_task,
            total_images=total_images,
            run_dir=str(run_dir),
            report_available=False,
        )
        device = os.getenv("SAM_QA_DEVICE") or os.getenv("TRAINING_DEVICE") or ""
        model_config = annotation_qa_model_config(request_payload["model"])
        sam_runtime = SamQaRuntime(
            model_config,
            device=device,
            requested_max_side=int(request_payload.get("max_side") or 1280),
        ).load()
        thresholds = annotation_qa_thresholds(
            request_payload.get("preset", "balanced"),
            request_payload.get("box_tolerance_percent"),
            request_payload.get("sam_max_difference_percent"),
        )
        thresholds.update({
            "sam_prompt_expansion_percent": float(request_payload.get("sam_prompt_expansion_percent", 8.0)),
            "sam_prompt_jitter_percent": float(request_payload.get("sam_prompt_jitter_percent", 2.0)),
            "sam_stability_bbox_iou_min": float(request_payload.get("sam_stability_bbox_iou_min", 0.90)),
            "sam_stability_edge_percent_max": float(request_payload.get("sam_stability_edge_percent_max", 3.0)),
            "sam_auto_quality_min": float(request_payload.get("sam_auto_quality_min", 0.85)),
            "sam_auto_yolo_iou_min": float(request_payload.get("sam_auto_yolo_iou_min", 0.70)),
            "sam_auto_center_shift_max": float(request_payload.get("sam_auto_center_shift_max", 0.10)),
            "sam_auto_neighbor_iou_max": float(request_payload.get("sam_auto_neighbor_iou_max", 0.15)),
            "auto_audit_percent": float(request_payload.get("auto_audit_percent", 5.0)),
        })
        auto_correction_mode = str(request_payload.get("auto_correction_mode", "shadow")).lower()
        policy = {
            "mode": auto_correction_mode,
            "annotation_task": annotation_task,
            "automatic_polygon_replacement": False,
            "model": model_config["label"],
            "model_id": model_config["id"],
            "automatic_allowed": model_config["automatic_allowed"],
            **{
                key: thresholds[key]
                for key in (
                    "sam_prompt_expansion_percent",
                    "sam_prompt_jitter_percent",
                    "sam_stability_bbox_iou_min",
                    "sam_stability_edge_percent_max",
                    "sam_auto_quality_min",
                    "sam_auto_yolo_iou_min",
                    "sam_auto_center_shift_max",
                    "sam_auto_neighbor_iou_max",
                    "auto_audit_percent",
                )
            },
        }
        processed_limit = int(max_images) if max_images else None
        audited_classes: set[str] = set()

        def should_audit_decision(
            image_name: str,
            class_id: Optional[int],
            label_row: Optional[int],
            decision: str,
        ) -> bool:
            class_key = str(class_id)
            required = annotation_qa_audit_required(
                job_id,
                image_name,
                class_id,
                label_row,
                decision,
                thresholds["auto_audit_percent"],
            )
            if thresholds["auto_audit_percent"] > 0 and class_key not in audited_classes:
                required = True
            if required:
                audited_classes.add(class_key)
            return required

        def append_qa_issue(
            issue: dict,
            *,
            metrics: Optional[dict] = None,
            sam_prompt_index: Optional[int] = None,
            sam_prompt_variant: str = "",
            sam_confidence: Optional[float] = None,
            difference_band: str = "",
            quality_gate_passed: bool = False,
            auto_fix_eligible: bool = False,
            automatic_fix_eligible: bool = False,
            qa_decision: str = "human_review",
            decision_reasons: Optional[list[str]] = None,
            accepted_fix_source: str = "",
            audit_required: bool = False,
            audit_status: str = "not_required",
            mask=None,
            original_mask=None,
        ):
            issue.update({
                "sam_prompt_index": sam_prompt_index,
                "sam_selected_variant": sam_prompt_variant,
                "sam_confidence": round(sam_confidence, 4) if sam_confidence is not None else None,
                "box_tolerance_percent": thresholds["box_tolerance_percent"],
                "sam_max_difference_percent": thresholds["sam_max_difference_percent"],
                "difference_band": difference_band,
                "quality_gate_passed": quality_gate_passed,
                "auto_fix_eligible": auto_fix_eligible,
                "automatic_fix_eligible": automatic_fix_eligible,
                "qa_decision": qa_decision,
                "decision_reasons": list(decision_reasons or []),
                "accepted_fix_source": accepted_fix_source,
                "audit_required": audit_required,
                "audit_status": audit_status,
            })
            issue["preview"] = f"previews/{issue['issue_id']}.jpg"
            draw_annotation_qa_preview(
                image_path,
                run_dir / issue["preview"],
                issue,
                mask,
                original_mask,
            )
            issues.append(issue)

        for split, _images_path, labels_path, images in split_contexts:
            for image_path in images:
                if stop_event.is_set():
                    raise InferenceStopped("Annotation QA was stopped.")
                if processed_limit is not None and images_scanned >= processed_limit:
                    break
                images_scanned += 1
                percent = 5 + int((images_scanned / max(1, total_images)) * 90)
                update_annotation_qa_job(
                    job_id,
                    stage="scanning",
                    percent=min(95, percent),
                    detail=f"Scanning {images_scanned} of {total_images}: {image_path.name}",
                    images_scanned=images_scanned,
                    labels_checked=labels_checked,
                    issues=len(issues),
                )
                import cv2

                image = cv2.imread(str(image_path))
                if image is None:
                    issue_index += 1
                    issues.append(annotation_issue(
                        job_id, issue_index, image_path, split, None, "",
                        "image_read_error", "high", 1.0, "Image could not be read.",
                    ))
                    continue
                height, width = image.shape[:2]
                label_path = yolo_label_path(image_path, labels_path)
                if label_path is None or not label_path.is_file():
                    continue
                labels = []
                for row_index, line in enumerate(label_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                    fields = line.strip().split()
                    if not fields:
                        continue
                    try:
                        class_id = int(float(fields[0]))
                    except ValueError:
                        issue_index += 1
                        issues.append(annotation_issue(
                            job_id, issue_index, image_path, split, None, "",
                            "invalid_label", "high", 1.0, f"Invalid class id on row {row_index}.",
                        ))
                        continue
                    class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
                    row_task = annotation_task_for_fields(fields)
                    if row_task != annotation_task:
                        issue_index += 1
                        issue = annotation_issue(
                            job_id, issue_index, image_path, split, class_id, class_name,
                            "unsupported_annotation", "low", 0.2,
                            f"This row is not a valid YOLO {annotation_task} annotation.",
                            label_row=row_index,
                            annotation_task=annotation_task,
                        )
                        append_qa_issue(issue, qa_decision="manual_only", decision_reasons=["unsupported_annotation"])
                        continue
                    polygon = None
                    original_mask = None
                    if annotation_task == "segment":
                        polygon = pixel_polygon_from_yolo(fields, width, height)
                        if polygon is not None:
                            box = polygon_bbox(polygon)
                            original_mask = polygon_mask(polygon, width, height)
                        else:
                            box = None
                    else:
                        box = pixel_bbox_from_yolo(fields, width, height)
                    if box is None:
                        issue_index += 1
                        issue = annotation_issue(
                            job_id, issue_index, image_path, split, class_id, class_name,
                            "invalid_polygon" if annotation_task == "segment" else "invalid_label",
                            "high", 1.0,
                            f"Invalid YOLO {'polygon' if annotation_task == 'segment' else 'bbox'} on row {row_index}.",
                            label_row=row_index,
                            annotation_task=annotation_task,
                        )
                        append_qa_issue(issue, qa_decision="manual_only", decision_reasons=["invalid_annotation"])
                        continue
                    labels.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "bbox": box,
                        "polygon": polygon,
                        "original_mask": original_mask,
                        "row_index": row_index,
                    })

                for left in range(len(labels)):
                    for right in range(left + 1, len(labels)):
                        if labels[left]["class_id"] != labels[right]["class_id"]:
                            continue
                        overlap = (
                            annotation_qa_mask_iou(labels[left]["original_mask"], labels[right]["original_mask"])
                            if annotation_task == "segment"
                            else bbox_iou(labels[left]["bbox"], labels[right]["bbox"])
                        )
                        if overlap >= thresholds["duplicate_iou"]:
                            issue_index += 1
                            issue = annotation_issue(
                                job_id, issue_index, image_path, split,
                                labels[left]["class_id"], labels[left]["class_name"],
                                "duplicate_segment" if annotation_task == "segment" else "duplicate_box",
                                "medium", overlap,
                                "Two same-class annotations overlap heavily.",
                                labels[left]["bbox"], labels[right]["bbox"], {"bbox_iou": overlap},
                                label_row=labels[left].get("row_index"),
                                annotation_task=annotation_task,
                                original_polygon=labels[left].get("polygon"),
                            )
                            if annotation_task == "segment":
                                issue["sam_polygon"] = labels[right].get("polygon")
                                append_qa_issue(
                                    issue,
                                    metrics={"mask_iou": round(overlap, 4)},
                                    qa_decision="human_review",
                                    decision_reasons=["possible_duplicate_segment"],
                                    mask=labels[right].get("original_mask"),
                                    original_mask=labels[left].get("original_mask"),
                                )
                            else:
                                issues.append(issue)

                candidates_by_label = annotation_qa_candidates_for_image(
                    sam_runtime,
                    image,
                    labels,
                    width,
                    height,
                    thresholds,
                    stop_event,
                    force_stability=annotation_task == "segment",
                )
                if candidates_by_label is None:
                    issue_index += 1
                    issues.append(annotation_issue(
                        job_id, issue_index, image_path, split, None, "",
                        "sam_mapping_error", "high", 1.0,
                        "SAM returned masks without reliable prompt indices; no box comparison was made.",
                    ))
                    continue
                for label_index, label in enumerate(labels):
                    labels_checked += 1
                    class_key = str(label["class_id"])
                    class_label_counts[class_key] = class_label_counts.get(class_key, 0) + 1
                    box = label["bbox"]
                    box_area = max(1, bbox_area(box))
                    candidates_for_label = candidates_by_label[label_index]
                    selected_candidate = annotation_qa_select_candidate(candidates_for_label)
                    if selected_candidate is None:
                        issue_index += 1
                        issue = annotation_issue(
                            job_id, issue_index, image_path, split,
                            label["class_id"], label["class_name"],
                            "empty_mask", "high", 1.0,
                            "SAM did not return a usable mask for one or more prompts for this box.",
                            box,
                            label_row=label.get("row_index"),
                            annotation_task=annotation_task,
                            original_polygon=label.get("polygon"),
                        )
                        if annotation_task == "segment":
                            append_qa_issue(
                                issue,
                                qa_decision="manual_only",
                                decision_reasons=["empty_sam_mask"],
                                original_mask=label.get("original_mask"),
                            )
                        else:
                            issues.append(issue)
                        continue
                    mask = selected_candidate["mask"]
                    sam_confidence = selected_candidate.get("confidence")
                    sam_prompt_index = selected_candidate.get("prompt_index")
                    sam_prompt_variant = selected_candidate.get("variant", "")
                    if mask is None:
                        issue_index += 1
                        issue = annotation_issue(
                            job_id, issue_index, image_path, split,
                            label["class_id"], label["class_name"],
                            "empty_mask", "high", 1.0,
                            "SAM returned no mask for this prompt.",
                            box,
                            label_row=label.get("row_index"),
                            annotation_task=annotation_task,
                            original_polygon=label.get("polygon"),
                        )
                        if annotation_task == "segment":
                            append_qa_issue(
                                issue,
                                qa_decision="manual_only",
                                decision_reasons=["empty_sam_mask"],
                                original_mask=label.get("original_mask"),
                            )
                        else:
                            issues.append(issue)
                        continue
                    mask_array = mask_to_uint8(mask, width, height)
                    sam_box = mask_bbox(mask_array)
                    if sam_box is None:
                        issue_index += 1
                        issue = annotation_issue(
                            job_id, issue_index, image_path, split,
                            label["class_id"], label["class_name"],
                            "empty_mask", "high", 1.0,
                            "SAM returned an empty mask for this box.",
                            box,
                            label_row=label.get("row_index"),
                            annotation_task=annotation_task,
                            original_polygon=label.get("polygon"),
                        )
                        issue["preview"] = f"previews/{issue['issue_id']}.jpg"
                        draw_annotation_qa_preview(
                            image_path,
                            run_dir / issue["preview"],
                            issue,
                            None,
                            label.get("original_mask"),
                        )
                        issues.append(issue)
                        continue
                    stability = annotation_qa_prompt_stability(
                        candidates_for_label,
                        box,
                        thresholds["sam_stability_bbox_iou_min"],
                        thresholds["sam_stability_edge_percent_max"],
                    )
                    if annotation_task == "segment":
                        original_mask = label["original_mask"]
                        metrics = annotation_qa_segmentation_metrics(original_mask, mask_array)
                        metrics["prompt_stability"] = stability
                        if sam_confidence is not None:
                            metrics["sam_confidence"] = round(sam_confidence, 4)
                        sam_polygon = mask_to_polygon(mask_array, width, height)
                        mask_iou = float(metrics["mask_iou"])
                        if mask_iou >= thresholds["segment_keep_iou"]:
                            difference_band = "within_tolerance"
                        elif mask_iou >= thresholds["segment_review_iou"]:
                            difference_band = "reviewable"
                        else:
                            difference_band = "large_disagreement"
                        metrics["difference_band"] = difference_band
                        quality_checks = {
                            "sam_confidence": sam_confidence is not None and sam_confidence >= thresholds["sam_confidence_min"],
                            "mask_iou": mask_iou >= thresholds["segment_review_iou"],
                            "boundary_f1": metrics["boundary_f1"] >= thresholds["segment_boundary_f1_min"],
                            "area_ratio": thresholds["segment_area_ratio_min"] <= metrics["mask_area_ratio"] <= thresholds["segment_area_ratio_max"],
                            "prompt_stability": bool(stability.get("passed")),
                            "representable_polygon": sam_polygon is not None,
                        }
                        quality_gate_passed = all(quality_checks.values())
                        quality_checks["passed"] = quality_gate_passed
                        metrics["sam_quality_checks"] = quality_checks

                        if difference_band == "within_tolerance":
                            yolo_boxes_accepted += 1
                            audit_required = auto_correction_mode != "manual" and should_audit_decision(
                                image_path.name, label["class_id"], label.get("row_index"), "auto_keep_yolo",
                            )
                            if audit_required:
                                issue_index += 1
                                issue = annotation_issue(
                                    job_id, issue_index, image_path, split,
                                    label["class_id"], label["class_name"],
                                    "auto_keep_audit", "low", 1.0 - mask_iou,
                                    "The YOLO polygon and SAM mask agree; this keep decision was sampled for audit.",
                                    box, sam_box, metrics,
                                    label_row=label.get("row_index"),
                                    annotation_task="segment",
                                    original_polygon=label["polygon"],
                                )
                                issue["sam_polygon"] = sam_polygon
                                append_qa_issue(
                                    issue,
                                    metrics=metrics,
                                    sam_prompt_index=sam_prompt_index,
                                    sam_prompt_variant=sam_prompt_variant,
                                    sam_confidence=sam_confidence,
                                    difference_band=difference_band,
                                    quality_gate_passed=quality_gate_passed,
                                    qa_decision="auto_keep_yolo",
                                    decision_reasons=["mask_within_tolerance"],
                                    audit_required=True,
                                    audit_status="pending",
                                    mask=mask_array,
                                    original_mask=original_mask,
                                )
                            continue

                        if difference_band == "large_disagreement":
                            issue_type = "large_mask_disagreement"
                            severity = "high"
                            message = "SAM and the YOLO polygon have low mask agreement; polygon replacement is blocked."
                        elif sam_confidence is None or sam_confidence < thresholds["sam_confidence_min"]:
                            issue_type = "low_confidence_mask"
                            severity = "medium"
                            message = "SAM mask confidence is below the review threshold."
                        elif metrics["mask_area_ratio"] < thresholds["segment_area_ratio_min"]:
                            issue_type = "missing_object_area"
                            severity = "medium"
                            message = "The SAM mask covers substantially less area than the YOLO polygon."
                        elif metrics["mask_area_ratio"] > thresholds["segment_area_ratio_max"]:
                            issue_type = "excess_mask_area"
                            severity = "medium"
                            message = "The SAM mask covers substantially more area than the YOLO polygon."
                        elif metrics["boundary_f1"] < thresholds["segment_boundary_f1_min"]:
                            issue_type = "low_boundary_agreement"
                            severity = "medium"
                            message = "The SAM boundary and YOLO polygon boundary differ materially."
                        elif not stability.get("passed"):
                            issue_type = "unstable_sam_segmentation"
                            severity = "medium"
                            message = "SAM segmentation changed materially when its box prompt was expanded or shifted."
                        else:
                            issue_type = "moderate_mask_difference"
                            severity = "low"
                            message = "The SAM mask and YOLO polygon differ enough to require human review."

                        auto_fix_eligible = difference_band == "reviewable" and quality_gate_passed
                        issue_index += 1
                        issue = annotation_issue(
                            job_id, issue_index, image_path, split,
                            label["class_id"], label["class_name"], issue_type, severity,
                            1.0 - mask_iou, message, box, sam_box, metrics,
                            label_row=label.get("row_index"),
                            annotation_task="segment",
                            original_polygon=label["polygon"],
                            recommended_polygon=sam_polygon if auto_fix_eligible else None,
                        )
                        issue["sam_polygon"] = sam_polygon
                        append_qa_issue(
                            issue,
                            metrics=metrics,
                            sam_prompt_index=sam_prompt_index,
                            sam_prompt_variant=sam_prompt_variant,
                            sam_confidence=sam_confidence,
                            difference_band=difference_band,
                            quality_gate_passed=quality_gate_passed,
                            auto_fix_eligible=auto_fix_eligible,
                            automatic_fix_eligible=False,
                            qa_decision="human_review" if auto_fix_eligible else "manual_only",
                            decision_reasons=["segmentation_manual_review_required"],
                            mask=mask_array,
                            original_mask=original_mask,
                        )
                        continue
                    neighbor_iou = annotation_qa_max_neighbor_iou(labels, label_index)
                    overlap = bbox_iou(box, sam_box)
                    center_shift = bbox_center_shift(box, sam_box)
                    mask_area = int(mask_array.sum())
                    mask_area_ratio = mask_area / box_area
                    sam_box_area_ratio = bbox_area(sam_box) / box_area
                    metrics = {
                        "bbox_iou": round(overlap, 4),
                        "center_shift": round(center_shift, 4),
                        "mask_area_ratio": round(mask_area_ratio, 4),
                        "sam_bbox_area_ratio": round(sam_box_area_ratio, 4),
                        "prompt_stability": stability,
                        "neighbor_iou": round(neighbor_iou, 4),
                    }
                    edge_differences = bbox_edge_differences(
                        box,
                        sam_box,
                        thresholds["box_tolerance_percent"],
                        thresholds["sam_max_difference_percent"],
                    )
                    difference_band = annotation_qa_difference_band(edge_differences)
                    metrics["edge_differences"] = edge_differences
                    metrics["difference_band"] = difference_band
                    if sam_confidence is not None:
                        metrics["sam_confidence"] = round(sam_confidence, 4)
                    metrics["box_agreement"] = difference_band

                    if difference_band == "within_tolerance":
                        yolo_boxes_accepted += 1
                        audit_required = auto_correction_mode != "manual" and should_audit_decision(
                            image_path.name, label["class_id"], label.get("row_index"), "auto_keep_yolo",
                        )
                        if audit_required:
                            issue_index += 1
                            issue = annotation_issue(
                                job_id, issue_index, image_path, split,
                                label["class_id"], label["class_name"],
                                "auto_keep_audit", "low", 0.0,
                                "YOLO and SAM agree within tolerance. This automatic keep was sampled for audit.",
                                box, sam_box, metrics,
                                label_row=label.get("row_index"),
                            )
                            append_qa_issue(
                                issue,
                                metrics=metrics,
                                sam_prompt_index=sam_prompt_index,
                                sam_prompt_variant=sam_prompt_variant,
                                sam_confidence=sam_confidence,
                                difference_band=difference_band,
                                quality_gate_passed=True,
                                auto_fix_eligible=False,
                                automatic_fix_eligible=False,
                                qa_decision="auto_keep_yolo",
                                decision_reasons=["within_tolerance"],
                                audit_required=True,
                                audit_status="pending",
                                mask=mask_array,
                            )
                        continue

                    if difference_band == "large_disagreement":
                        issue_index += 1
                        issue = annotation_issue(
                            job_id, issue_index, image_path, split,
                            label["class_id"], label["class_name"],
                            "large_box_disagreement", "high",
                            min(1.0, edge_differences["max_percent"] / 100),
                            "SAM and YOLO differ beyond the configured maximum; automatic replacement is blocked.",
                            box, sam_box, metrics,
                            label_row=label.get("row_index"),
                        )
                        append_qa_issue(
                            issue,
                            metrics=metrics,
                            sam_prompt_index=sam_prompt_index,
                            sam_prompt_variant=sam_prompt_variant,
                            sam_confidence=sam_confidence,
                            difference_band=difference_band,
                            qa_decision="manual_only",
                            decision_reasons=["large_disagreement"],
                            mask=mask_array,
                        )
                        continue

                    if (
                        sam_confidence is None
                        or sam_confidence < thresholds["sam_confidence_min"]
                    ):
                        issue_index += 1
                        issue = annotation_issue(
                            job_id, issue_index, image_path, split,
                            label["class_id"], label["class_name"],
                            "low_confidence_mask", "medium",
                            1 - sam_confidence if sam_confidence is not None else 1.0,
                            "SAM mask confidence is below the automatic correction threshold.",
                            box, sam_box, metrics,
                            label_row=label.get("row_index"),
                        )
                        append_qa_issue(
                            issue,
                            metrics=metrics,
                            sam_prompt_index=sam_prompt_index,
                            sam_prompt_variant=sam_prompt_variant,
                            sam_confidence=sam_confidence,
                            difference_band=difference_band,
                            qa_decision="manual_only",
                            decision_reasons=["sam_quality_below_review_threshold"],
                            mask=mask_array,
                        )
                        continue

                    quality_checks = {
                        "bbox_iou": overlap >= thresholds["bbox_iou"],
                        "center_shift": center_shift <= thresholds["center_shift"],
                        "mask_coverage": mask_area_ratio >= thresholds["mask_area_ratio_min"],
                        "sam_box_area": sam_box_area_ratio >= thresholds["loose_area_ratio"],
                    }
                    quality_gate_passed = all(quality_checks.values())
                    quality_checks["passed"] = quality_gate_passed
                    metrics["sam_quality_checks"] = quality_checks
                    automatic_gate_passed, automatic_gate_reasons = annotation_qa_auto_gate(
                        stability=stability,
                        sam_confidence=stability.get("confidence_min"),
                        bbox_overlap=overlap,
                        center_shift=center_shift,
                        neighbor_iou=neighbor_iou,
                        quality_gate_passed=quality_gate_passed,
                        thresholds=thresholds,
                    )
                    if auto_correction_mode == "manual":
                        automatic_gate_passed = False
                        automatic_gate_reasons = [*automatic_gate_reasons, "automatic_correction_disabled"]
                    metrics["automatic_quality_gate"] = {
                        "passed": automatic_gate_passed,
                        "reasons": automatic_gate_reasons,
                        "quality_min_required": thresholds["sam_auto_quality_min"],
                        "yolo_iou_min_required": thresholds["sam_auto_yolo_iou_min"],
                        "center_shift_max_allowed": thresholds["sam_auto_center_shift_max"],
                        "neighbor_iou_max_allowed": thresholds["sam_auto_neighbor_iou_max"],
                    }

                    edge_margin_x = max(2, int((box[2] - box[0]) * 0.03))
                    edge_margin_y = max(2, int((box[3] - box[1]) * 0.03))
                    edge_hits = sum([
                        abs(sam_box[0] - box[0]) <= edge_margin_x,
                        abs(sam_box[2] - box[2]) <= edge_margin_x,
                        abs(sam_box[1] - box[1]) <= edge_margin_y,
                        abs(sam_box[3] - box[3]) <= edge_margin_y,
                    ])
                    candidates = []
                    if mask_area_ratio < thresholds["mask_area_ratio_min"]:
                        candidates.append(("low_mask_coverage", "high", 1 - mask_area_ratio, "SAM mask covers very little of the labeled box."))
                    if overlap < thresholds["bbox_iou"]:
                        severity = "high" if overlap < thresholds["bbox_iou"] * 0.65 else "medium"
                        candidates.append(("low_box_agreement", severity, 1 - overlap, "SAM mask bbox disagrees with the YOLO box."))
                    if center_shift > thresholds["center_shift"]:
                        candidates.append(("shifted_box", "medium", center_shift, "SAM mask center is far from the YOLO box center."))
                    if sam_box_area_ratio < thresholds["loose_area_ratio"]:
                        candidates.append(("loose_box", "medium", 1 - sam_box_area_ratio, "YOLO box is much larger than the SAM mask bbox."))
                    if edge_hits >= thresholds["tight_edge_count"] and mask_area_ratio > 0.35:
                        candidates.append(("possibly_tight_box", "low", edge_hits / 4, "SAM mask touches multiple edges of the YOLO box."))
                    if not stability.get("passed"):
                        candidates.append((
                            "unstable_sam_prompt",
                            "medium",
                            min(1.0, max(0.0, 1.0 - stability.get("minimum_bbox_iou", 0.0))),
                            "SAM produced materially different boxes when the prompt was expanded or jittered.",
                        ))

                    if candidates:
                        issue_type, severity, score, message = sorted(
                            candidates,
                            key=lambda item: {"high": 3, "medium": 2, "low": 1}[item[1]],
                            reverse=True,
                        )[0]
                    else:
                        issue_type = "moderate_box_difference"
                        severity = "low"
                        score = min(1.0, edge_differences["max_percent"] / 100)
                        message = "SAM and YOLO differ beyond the acceptance tolerance and require review."

                    stability_passed = bool(stability.get("passed"))
                    auto_fix_eligible = quality_gate_passed and stability_passed
                    if not stability_passed:
                        qa_decision = "manual_only"
                        decision_reasons = ["prompt_stability_failed"]
                    elif automatic_gate_passed:
                        qa_decision = "auto_replace_sam"
                        decision_reasons = ["strict_automatic_gate_passed"]
                    else:
                        qa_decision = "human_review"
                        decision_reasons = automatic_gate_reasons or ["strict_automatic_gate_failed"]
                    audit_required = qa_decision == "auto_replace_sam" and should_audit_decision(
                        image_path.name, label["class_id"], label.get("row_index"), qa_decision,
                    )
                    accepted_fix_source = ""
                    accepted_fix = ""
                    review_status = "unreviewed"
                    if qa_decision == "auto_replace_sam" and auto_correction_mode == "automatic":
                        accepted_fix = "sam_box"
                        accepted_fix_source = "automatic"
                        review_status = "fix_accepted"
                    issue_index += 1
                    issue = annotation_issue(
                        job_id, issue_index, image_path, split,
                        label["class_id"], label["class_name"], issue_type, severity,
                        score, message, box, sam_box, metrics,
                        label_row=label.get("row_index"),
                        recommended_bbox=sam_box if auto_fix_eligible else None,
                    )
                    issue["accepted_fix"] = accepted_fix
                    issue["review_status"] = review_status
                    append_qa_issue(
                        issue,
                        metrics=metrics,
                        sam_prompt_index=sam_prompt_index,
                        sam_prompt_variant=sam_prompt_variant,
                        sam_confidence=sam_confidence,
                        difference_band=difference_band,
                        quality_gate_passed=quality_gate_passed,
                        auto_fix_eligible=auto_fix_eligible,
                        automatic_fix_eligible=automatic_gate_passed,
                        qa_decision=qa_decision,
                        decision_reasons=decision_reasons,
                        accepted_fix_source=accepted_fix_source,
                        audit_required=audit_required,
                        audit_status="pending" if audit_required else "not_required",
                        mask=mask_array,
                    )
            if processed_limit is not None and images_scanned >= processed_limit:
                break

        policy["runtime"] = sam_runtime.stats()
        summary = annotation_qa_summary(
            issues,
            images_scanned,
            labels_checked,
            request_payload.get("model_label", request_payload["model"]),
            request_payload.get("scope", "val"),
            request_payload.get("preset", "balanced"),
            request_payload.get("box_tolerance_percent", 5.0),
            request_payload.get("sam_max_difference_percent", 25.0),
            yolo_boxes_accepted,
            policy,
            class_label_counts,
            annotation_task,
        )
        summary.update({
            "dataset_yaml": str(yaml_path),
            "dataset_root": str(dataset_root),
            "splits": split_names,
        })
        write_annotation_qa_report(run_dir, issues, summary)
        update_annotation_qa_job(
            job_id,
            status="completed",
            stage="completed",
            percent=100,
            detail=f"QA complete: {len(issues)} issues flagged.",
            images_scanned=images_scanned,
            labels_checked=labels_checked,
            issues=len(issues),
            high=summary["high"],
            medium=summary["medium"],
            low=summary["low"],
            summary=summary,
            report_available=True,
        )
    except InferenceStopped:
        if sam_runtime is not None:
            policy["runtime"] = sam_runtime.stats()
        summary = annotation_qa_summary(
            issues,
            images_scanned,
            labels_checked,
            request_payload.get("model_label", request_payload.get("model", ANNOTATION_QA_MODEL_DEFAULT)),
            request_payload.get("scope", "val"),
            request_payload.get("preset", "balanced"),
            request_payload.get("box_tolerance_percent", 5.0),
            request_payload.get("sam_max_difference_percent", 25.0),
            yolo_boxes_accepted,
            policy,
            class_label_counts,
        )
        write_annotation_qa_report(run_dir, issues, summary)
        update_annotation_qa_job(
            job_id,
            status="stopped",
            stage="stopped",
            percent=0,
            detail="Annotation QA stopped.",
            images_scanned=images_scanned,
            labels_checked=labels_checked,
            issues=len(issues),
            summary=summary,
            report_available=True,
        )
    except Exception as exc:
        update_annotation_qa_job(
            job_id,
            status="failed",
            stage="failed",
            percent=0,
            detail=str(exc),
            error=str(exc),
            images_scanned=images_scanned,
            labels_checked=labels_checked,
            issues=len(issues),
            report_available=False,
        )
    finally:
        if sam_runtime is not None:
            sam_runtime.close()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def annotation_qa_job_from_disk(job_id: str) -> Optional[dict]:
    run_dir = (ANNOTATION_QA_ROOT / job_id).resolve()
    try:
        ensure_annotation_qa_path(run_dir)
    except HTTPException:
        return None
    report = read_json_object(run_dir / ANNOTATION_QA_REPORT_JSON_FILE)
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else read_json_object(run_dir / ANNOTATION_QA_SUMMARY_FILE)
    if not summary:
        return None
    return {
        "job_id": job_id,
        "status": "completed",
        "stage": "completed",
        "percent": 100,
        "detail": f"QA complete: {summary.get('issues', 0)} issues flagged.",
        "run_dir": str(run_dir),
        "images_scanned": summary.get("images_scanned", 0),
        "labels_checked": summary.get("labels_checked", 0),
        "issues": summary.get("issues", 0),
        "high": summary.get("high", 0),
        "medium": summary.get("medium", 0),
        "low": summary.get("low", 0),
        "summary": summary,
        "report_available": True,
        "created_at": run_dir.stat().st_mtime,
        "updated_at": run_dir.stat().st_mtime,
    }

__all__ = [
    "AnnotationQaDependencies",
    "configure_annotation_qa",
    "annotation_qa_job_id",
    "normalize_annotation_qa_model_name",
    "annotation_qa_version_tuple",
    "annotation_qa_model_status",
    "annotation_qa_model_config",
    "ensure_annotation_qa_path",
    "annotation_qa_payload",
    "update_annotation_qa_job",
    "annotation_qa_thresholds",
    "annotation_qa_split_names",
    "yolo_label_path",
    "pixel_bbox_from_yolo",
    "yolo_bbox_from_pixels",
    "normalize_annotation_qa_task",
    "annotation_task_for_fields",
    "detect_annotation_qa_task",
    "pixel_polygon_from_yolo",
    "polygon_bbox",
    "polygon_mask",
    "mask_to_polygon",
    "yolo_polygon_from_pixels",
    "annotation_qa_segmentation_metrics",
    "bbox_area",
    "bbox_iou",
    "bbox_center_shift",
    "bbox_edge_differences",
    "annotation_qa_difference_band",
    "annotation_qa_prompt_box",
    "annotation_qa_prompt_plan",
    "annotation_qa_mask_iou",
    "annotation_qa_prompt_stability",
    "annotation_qa_select_candidate",
    "annotation_qa_max_neighbor_iou",
    "annotation_qa_auto_gate",
    "annotation_qa_audit_required",
    "issue_is_safe_sam_replacement",
    "issue_is_reviewable_sam_polygon",
    "mask_bbox",
    "mask_to_uint8",
    "annotation_issue",
    "draw_annotation_qa_preview",
    "annotation_qa_summary",
    "write_annotation_qa_report",
    "annotation_qa_run_dir",
    "load_annotation_qa_report",
    "set_annotation_qa_issue_fix",
    "issue_label_path_in_copy",
    "apply_annotation_qa_fix",
    "apply_annotation_qa_fixes",
    "annotation_qa_roboflow_context",
    "build_annotation_qa_roboflow_preview",
    "publish_annotation_qa_to_roboflow",
    "load_sam_model",
    "sam_masks_for_image",
    "annotation_qa_candidates_for_image",
    "run_annotation_qa_job",
    "annotation_qa_job_from_disk",
]
