#!/usr/bin/env python3
"""Run RF-DETR inference on a single image or video."""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Callable

import cv2

from .infer_yolo import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    InferenceStopped,
    browser_transcode_video,
    encode_preview_jpeg,
    write_json_atomic,
)


MODEL_CACHE: dict[tuple[str, str], dict] = {}
MODEL_CACHE_LOCK = threading.Lock()
MODEL_CLASSES = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-large": "RFDETRLarge",
}


def media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError("Unsupported media type.")


def video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    return max(0, frames)


def device_label(_device: str = "") -> str:
    return "RF-DETR"


def progress_percent(stage: str, frames: int, total_frames: int) -> float:
    if stage == "loading_model":
        return 5.0
    if stage == "encoding":
        return 96.0
    if stage == "complete":
        return 100.0
    if total_frames > 0:
        return min(95.0, 10.0 + (frames / total_frames) * 85.0)
    return 50.0 if frames else 10.0


def write_progress(
    progress_path: Path | None,
    stage: str,
    frames: int = 0,
    total_frames: int = 0,
    detections: int = 0,
    detail: str = "",
):
    payload = {
        "stage": stage,
        "frames": frames,
        "total_frames": total_frames,
        "detections": detections,
        "percent": round(progress_percent(stage, frames, total_frames), 1),
        "detail": detail or stage.replace("_", " ").title(),
        "updated_at": time.time(),
        "task": "detect",
    }
    write_json_atomic(progress_path, payload)
    return payload


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def model_id_from_run(weights_path: Path) -> str:
    context = read_json(weights_path.parent.parent / "training_report_context.json")
    model_id = (
        context.get("model")
        or context.get("hyperparameters", {}).get("model")
        or context.get("hyperparameters", {}).get("model_size")
        or "rfdetr-nano"
    )
    model_id = str(model_id).strip().lower()
    return model_id if model_id in MODEL_CLASSES else "rfdetr-nano"


def class_names_from_run(weights_path: Path) -> list[str]:
    context = read_json(weights_path.parent.parent / "training_report_context.json")
    summary = context.get("dataset_summary") if isinstance(context.get("dataset_summary"), dict) else {}
    classes = summary.get("classes") if isinstance(summary, dict) else []
    if isinstance(classes, list):
        names = []
        for item in classes:
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("class_name") or "").strip()
            else:
                name = str(item).strip()
            if name:
                names.append(name)
        if names:
            return names
    try:
        from rfdetr.assets.coco_classes import COCO_CLASSES
    except Exception:
        return []
    return [str(name) for name in COCO_CLASSES]


def import_model_class(model_id: str):
    try:
        import rfdetr
    except ImportError as exc:
        raise RuntimeError("RF-DETR is not installed. Rebuild the Docker image after updating requirements.txt.") from exc
    try:
        return getattr(rfdetr, MODEL_CLASSES[model_id])
    except AttributeError as exc:
        raise RuntimeError(f"The installed rfdetr package does not provide {MODEL_CLASSES[model_id]}.") from exc


def get_rfdetr_model(weights_path: Path, use_cache: bool):
    model_id = model_id_from_run(weights_path)
    if not use_cache:
        return import_model_class(model_id)(pretrain_weights=str(weights_path)), threading.Lock(), False

    key = (str(weights_path.resolve()), model_id)
    with MODEL_CACHE_LOCK:
        entry = MODEL_CACHE.get(key)
        if entry is None:
            entry = {
                "model": import_model_class(model_id)(pretrain_weights=str(weights_path)),
                "lock": threading.Lock(),
            }
            MODEL_CACHE[key] = entry
            return entry["model"], entry["lock"], False
        return entry["model"], entry["lock"], True


def detection_arrays(detections):
    xyxy = getattr(detections, "xyxy", None)
    class_ids = getattr(detections, "class_id", None)
    confidences = getattr(detections, "confidence", None)
    if xyxy is None:
        xyxy = []
    if class_ids is None:
        class_ids = []
    if confidences is None:
        confidences = []
    return xyxy, class_ids, confidences


def class_name(names: list[str], class_id: int) -> str:
    if 0 <= class_id < len(names):
        return names[class_id]
    return f"class_{class_id}"


def collect_detections(detections, names: list[str]) -> list[dict]:
    xyxy, class_ids, confidences = detection_arrays(detections)
    rows = []
    for index, coords in enumerate(xyxy):
        class_id = int(class_ids[index]) if index < len(class_ids) else -1
        confidence = float(confidences[index]) if index < len(confidences) else 0.0
        x1, y1, x2, y2 = [float(value) for value in coords]
        rows.append({
            "class_id": class_id,
            "class_name": class_name(names, class_id),
            "confidence": round(confidence, 4),
            "box": [
                int(round(x1)),
                int(round(y1)),
                int(round(x2 - x1)),
                int(round(y2 - y1)),
            ],
        })
    return rows


def plot_detections(image_bgr, detections, names: list[str], show_boxes=True, show_labels=True, show_conf=True):
    image = image_bgr.copy()
    xyxy, class_ids, confidences = detection_arrays(detections)
    color = (20, 145, 120)
    text_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, coords in enumerate(xyxy):
        x1, y1, x2, y2 = [int(round(float(value))) for value in coords]
        class_id = int(class_ids[index]) if index < len(class_ids) else -1
        confidence = float(confidences[index]) if index < len(confidences) else 0.0
        label = class_name(names, class_id)
        caption = f"{label} {confidence:.2f}" if show_conf else label
        if show_boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        if not show_labels:
            continue
        text_size, baseline = cv2.getTextSize(caption, font, 0.55, 1)
        text_width, text_height = text_size
        label_top = max(0, y1 - text_height - baseline - 6)
        label_bottom = label_top + text_height + baseline + 6
        label_right = min(image.shape[1] - 1, x1 + text_width + 8)
        cv2.rectangle(image, (x1, label_top), (label_right, label_bottom), color, -1)
        cv2.putText(
            image,
            caption,
            (x1 + 4, label_bottom - baseline - 3),
            font,
            0.55,
            text_color,
            1,
            cv2.LINE_AA,
        )
    return image


def run_rfdetr_inference(
    weights_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    json_path: str | Path | None = None,
    progress_path: str | Path | None = None,
    preview_path: str | Path | None = None,
    imgsz: int = 512,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "",
    vid_stride: int = 1,
    progress_callback: Callable[[dict], None] | None = None,
    preview_callback: Callable[[bytes], None] | None = None,
    live_frame_callback: Callable[[object], None] | None = None,
    preview_fps: float = 30.0,
    preview_max_width: int = 640,
    stop_event: threading.Event | None = None,
    use_model_cache: bool = False,
    show_boxes: bool = True,
    show_labels: bool = True,
    show_conf: bool = True,
    show_masks: bool = True,
) -> dict:
    del imgsz, iou, device, show_masks
    weights_path = Path(weights_path).expanduser().resolve()
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    json_path = Path(json_path).expanduser().resolve() if json_path else None
    progress_path = Path(progress_path).expanduser().resolve() if progress_path else None
    preview_path = Path(preview_path).expanduser().resolve() if preview_path else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    media_kind = media_type(input_path)
    video_stride = max(1, int(vid_stride or 1))
    source_total_frames = video_frame_count(input_path) if media_kind == "video" else 1
    total_frames = max(1, math.ceil(source_total_frames / video_stride)) if media_kind == "video" else 1
    raw_video_output = output_path.with_name(f"{output_path.stem}.raw{output_path.suffix}")
    writer_output_path = raw_video_output if media_kind == "video" else output_path
    names = class_names_from_run(weights_path)

    def stopped():
        return stop_event is not None and stop_event.is_set()

    def emit_progress(stage: str, frames: int = 0, detections: int = 0, detail: str = "", frame_total: int = total_frames):
        payload = write_progress(
            progress_path,
            stage,
            frames=frames,
            total_frames=frame_total,
            detections=detections,
            detail=detail,
        )
        if progress_callback is not None:
            progress_callback(payload)
        return payload

    def emit_preview(image, force_disk: bool = False):
        nonlocal last_preview_disk_at
        if live_frame_callback is not None:
            live_frame_callback(image)
        jpeg = encode_preview_jpeg(image, max_width=preview_max_width)
        if jpeg is None:
            return
        if preview_callback is not None:
            preview_callback(jpeg)
        now = time.perf_counter()
        if preview_path is not None and (force_disk or now - last_preview_disk_at >= preview_disk_interval):
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            preview_path.write_bytes(jpeg)
            last_preview_disk_at = now

    started = time.perf_counter()
    emit_progress("loading_model", detail="Loading RF-DETR model.")
    if stopped():
        raise InferenceStopped()
    model, model_lock, cache_hit = get_rfdetr_model(weights_path, use_model_cache)
    frames = 0
    detections_count = 0
    image_detections = []
    video_writer = None
    transcoded = False
    last_progress_at = 0.0
    last_preview_at = 0.0
    last_preview_disk_at = 0.0
    preview_interval = 1.0 / max(1.0, float(preview_fps or 30.0))
    preview_disk_interval = 2.0

    emit_progress("processing", detail=f"Processing frames{' with cached model' if cache_hit else ''}.")
    try:
        with model_lock:
            if media_kind == "image":
                image_bgr = cv2.imread(str(input_path))
                if image_bgr is None:
                    raise RuntimeError("Could not read image input.")
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                rfdetr_detections = model.predict(image_rgb, threshold=float(conf))
                plotted = plot_detections(image_bgr, rfdetr_detections, names, show_boxes, show_labels, show_conf)
                image_detections = collect_detections(rfdetr_detections, names)
                detections_count = len(image_detections)
                frames = 1
                if not cv2.imwrite(str(output_path), plotted):
                    raise RuntimeError("Could not write annotated image output.")
                emit_preview(plotted, force_disk=True)
                emit_progress("processing", frames=frames, detections=detections_count, detail="Processed 1 of 1 frames.")
            else:
                capture = cv2.VideoCapture(str(input_path))
                if not capture.isOpened():
                    raise RuntimeError("Could not open video input.")
                fps = capture.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 25.0
                fps = max(1.0, fps / video_stride)
                source_frame_index = 0
                try:
                    while True:
                        if stopped():
                            raise InferenceStopped()
                        ok, image_bgr = capture.read()
                        if not ok:
                            break
                        source_frame_index += 1
                        if (source_frame_index - 1) % video_stride:
                            continue
                        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                        rfdetr_detections = model.predict(image_rgb, threshold=float(conf))
                        detections_count += len(collect_detections(rfdetr_detections, names))
                        plotted = plot_detections(image_bgr, rfdetr_detections, names, show_boxes, show_labels, show_conf)
                        frames += 1
                        if video_writer is None:
                            height, width = plotted.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            video_writer = cv2.VideoWriter(str(writer_output_path), fourcc, fps, (width, height))
                            if not video_writer.isOpened():
                                raise RuntimeError("Could not open annotated video output.")
                        video_writer.write(plotted)
                        now = time.perf_counter()
                        if now - last_preview_at >= preview_interval or frames == 1:
                            emit_preview(plotted, force_disk=frames == 1)
                            last_preview_at = now
                        if now - last_progress_at >= 0.35 or frames == total_frames:
                            emit_progress(
                                "processing",
                                frames=frames,
                                detections=detections_count,
                                detail=f"Processed {frames} of {total_frames} frames.",
                            )
                            last_progress_at = now
                finally:
                    capture.release()
    finally:
        if video_writer is not None:
            video_writer.release()

    if stopped():
        raise InferenceStopped()

    if media_kind == "video":
        emit_progress("encoding", frames=frames, detections=detections_count, detail="Encoding browser-compatible video.")
        transcoded = browser_transcode_video(writer_output_path, output_path)
        if transcoded:
            writer_output_path.unlink(missing_ok=True)
        elif writer_output_path != output_path:
            writer_output_path.replace(output_path)

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    payload = {
        "engine": "python-rfdetr",
        "device": device_label(),
        "model_cached": cache_hit,
        "task": "detect",
        "media_type": media_kind,
        "frames": frames,
        "detections": detections_count,
        "classifications": 0,
        "elapsed_ms": elapsed_ms,
        "output": str(output_path),
        "video_stride": video_stride,
        "preview_fps_target": preview_fps,
        "browser_video": transcoded,
        "image_detections": image_detections,
        "image_classifications": [],
        "show_boxes": show_boxes,
        "show_labels": show_labels,
        "show_conf": show_conf,
        "show_masks": False,
    }
    write_json_atomic(json_path, payload)
    emit_progress("complete", frames=frames, detections=detections_count, detail="Inference complete.", frame_total=frames or total_frames)
    return payload
