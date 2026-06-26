#!/usr/bin/env python3
"""Run YOLO inference on a single image or video and emit JSON summary output."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MODEL_CACHE: dict[tuple[str, str], dict] = {}
MODEL_CACHE_LOCK = threading.Lock()


class InferenceStopped(Exception):
    """Raised when an in-process inference job is stopped by the user."""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--progress-json", default="")
    parser.add_argument("--preview", default="")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="")
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--backend", default="pytorch", help=argparse.SUPPRESS)
    parser.add_argument("--engine-cache-dir", default="", help=argparse.SUPPRESS)
    parser.add_argument("--half", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def write_json_atomic(path: Path | None, payload: dict):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def progress_percent(stage: str, frames: int, total_frames: int) -> float:
    if stage == "loading_model":
        return 2.0
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
    }
    write_json_atomic(progress_path, payload)
    return payload


def resolve_device(device: str | None = "") -> str:
    requested = str(device or "").strip()
    if requested:
        return requested
    try:
        import torch
    except Exception:
        return ""
    return "0" if torch.cuda.is_available() else ""


def device_label(device: str) -> str:
    normalized = str(device or "").strip().lower()
    if not normalized or normalized == "cpu":
        return "CPU"
    if normalized == "mps":
        return "MPS"
    if normalized.startswith("cuda"):
        return device.upper()
    return f"GPU {device}"


def get_yolo_model(weights_path: Path, device: str, use_cache: bool):
    from ultralytics import YOLO

    if not use_cache:
        return YOLO(str(weights_path)), threading.Lock(), False

    key = (str(weights_path.resolve()), device)
    with MODEL_CACHE_LOCK:
        entry = MODEL_CACHE.get(key)
        if entry is None:
            entry = {
                "model": YOLO(str(weights_path)),
                "lock": threading.Lock(),
            }
            MODEL_CACHE[key] = entry
            return entry["model"], entry["lock"], False
        return entry["model"], entry["lock"], True


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


def encode_preview_jpeg(image, max_width: int = 960) -> bytes | None:
    height, width = image.shape[:2]
    preview = image
    if width > max_width:
        scale = max_width / width
        preview = cv2.resize(image, (max_width, max(1, int(height * scale))))
    ok, encoded = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return encoded.tobytes() if ok else None


def write_preview(preview_path: Path | None, image, max_width: int = 960) -> bytes | None:
    jpeg = encode_preview_jpeg(image, max_width=max_width)
    if preview_path is None or jpeg is None:
        return jpeg
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_bytes(jpeg)
    return jpeg


def draw_result_fast(result):
    image = result.orig_img.copy()
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return image

    xyxy = boxes.xyxy.cpu().tolist()
    cls_ids = boxes.cls.cpu().tolist() if boxes.cls is not None else []
    confs = boxes.conf.cpu().tolist() if boxes.conf is not None else []
    names = getattr(result, "names", {}) or {}
    color = (20, 145, 120)
    text_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for index, coords in enumerate(xyxy):
        x1, y1, x2, y2 = [int(round(value)) for value in coords]
        class_id = int(cls_ids[index]) if index < len(cls_ids) else -1
        confidence = float(confs[index]) if index < len(confs) else 0.0
        label = names.get(class_id, f"class_{class_id}") if isinstance(names, dict) else f"class_{class_id}"
        caption = f"{label} {confidence:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

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


def collect_image_detections(result) -> list[dict]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return []

    xyxy = boxes.xyxy.cpu().tolist()
    cls_ids = boxes.cls.cpu().tolist() if boxes.cls is not None else []
    confs = boxes.conf.cpu().tolist() if boxes.conf is not None else []
    names = getattr(result, "names", {}) or {}
    detections = []
    for index, coords in enumerate(xyxy):
        class_id = int(cls_ids[index]) if index < len(cls_ids) else -1
        confidence = float(confs[index]) if index < len(confs) else 0.0
        label = names.get(class_id, f"class_{class_id}") if isinstance(names, dict) else f"class_{class_id}"
        x1, y1, x2, y2 = coords
        detections.append({
            "class_id": class_id,
            "class_name": str(label),
            "confidence": round(confidence, 4),
            "box": [
                int(round(x1)),
                int(round(y1)),
                int(round(x2 - x1)),
                int(round(y2 - y1)),
            ],
        })
    return detections


def browser_transcode_video(input_path: Path, output_path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    return completed.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 0


def run_yolo_inference(
    weights_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    json_path: str | Path | None = None,
    progress_path: str | Path | None = None,
    preview_path: str | Path | None = None,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "",
    vid_stride: int = 1,
    progress_callback: Callable[[dict], None] | None = None,
    preview_callback: Callable[[bytes], None] | None = None,
    preview_fps: float = 5.0,
    stop_event: threading.Event | None = None,
    use_model_cache: bool = False,
) -> dict:
    weights_path = Path(weights_path).expanduser().resolve()
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    json_path = Path(json_path).expanduser().resolve() if json_path else None
    progress_path = Path(progress_path).expanduser().resolve() if progress_path else None
    preview_path = Path(preview_path).expanduser().resolve() if preview_path else None
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    media_kind = media_type(input_path)
    video_stride = max(1, int(vid_stride or 1))
    source_total_frames = video_frame_count(input_path) if media_kind == "video" else 1
    total_frames = max(1, math.ceil(source_total_frames / video_stride)) if media_kind == "video" else 1
    raw_video_output = output_path.with_name(f"{output_path.stem}.raw{output_path.suffix}")
    writer_output_path = raw_video_output if media_kind == "video" else output_path
    selected_device = resolve_device(device)

    def stopped():
        return stop_event is not None and stop_event.is_set()

    def emit_progress(
        stage: str,
        frames: int = 0,
        detections: int = 0,
        detail: str = "",
        frame_total: int = total_frames,
    ):
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

    started = time.perf_counter()
    emit_progress(
        "loading_model",
        detail=f"Loading PyTorch model on {device_label(selected_device)}.",
    )
    if stopped():
        raise InferenceStopped()

    model, model_lock, cache_hit = get_yolo_model(weights_path, selected_device, use_model_cache)
    if stopped():
        raise InferenceStopped()
    emit_progress(
        "processing",
        detail=(
            f"Processing frames on {device_label(selected_device)}"
            f"{' with cached model' if cache_hit else ''}."
        ),
    )

    frames = 0
    detections = 0
    image_detections = []
    video_writer = None
    transcoded = False
    last_progress_at = 0.0
    last_preview_at = 0.0
    preview_interval = 1.0 / max(1.0, float(preview_fps or 5.0))

    def emit_preview(image):
        jpeg = write_preview(preview_path, image)
        if jpeg is not None and preview_callback is not None:
            preview_callback(jpeg)

    try:
        with model_lock:
            results = model.predict(
                source=str(input_path),
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=selected_device or None,
                save=False,
                verbose=False,
                stream=True,
                vid_stride=video_stride,
            )
            for result in results:
                if stopped():
                    raise InferenceStopped()
                frames += 1
                boxes = getattr(result, "boxes", None)
                count = len(boxes) if boxes is not None else 0
                detections += count
                plotted = draw_result_fast(result)
                now = time.perf_counter()
                if now - last_preview_at >= preview_interval or frames == 1:
                    emit_preview(plotted)
                    last_preview_at = now
                if frames == 1 and media_kind == "image":
                    image_detections = collect_image_detections(result)
                    if not cv2.imwrite(str(output_path), plotted):
                        raise RuntimeError("Could not write annotated image output.")
                    emit_preview(plotted)
                elif media_kind == "video":
                    if video_writer is None:
                        height, width = plotted.shape[:2]
                        capture = cv2.VideoCapture(str(input_path))
                        fps = capture.get(cv2.CAP_PROP_FPS)
                        capture.release()
                        if fps <= 0:
                            fps = 25.0
                        fps = max(1.0, fps / video_stride)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(str(writer_output_path), fourcc, fps, (width, height))
                        if not video_writer.isOpened():
                            raise RuntimeError("Could not open annotated video output.")
                    video_writer.write(plotted)
                if now - last_progress_at >= 0.35 or frames == total_frames:
                    emit_progress(
                        "processing",
                        frames=frames,
                        detections=detections,
                        detail=f"Processed {frames} of {total_frames} frames.",
                    )
                    last_progress_at = now
    finally:
        if video_writer is not None:
            video_writer.release()

    if stopped():
        raise InferenceStopped()

    if video_writer is not None:
        emit_progress(
            "encoding",
            frames=frames,
            detections=detections,
            detail="Encoding browser-compatible video.",
        )
        transcoded = browser_transcode_video(writer_output_path, output_path)
        if transcoded:
            writer_output_path.unlink(missing_ok=True)
        elif writer_output_path != output_path:
            writer_output_path.replace(output_path)

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    payload = {
        "engine": "python-ultralytics",
        "device": device_label(selected_device),
        "model_cached": cache_hit,
        "media_type": media_kind,
        "frames": frames,
        "detections": detections,
        "elapsed_ms": elapsed_ms,
        "output": str(output_path),
        "video_stride": video_stride,
        "browser_video": transcoded,
        "image_detections": image_detections,
    }
    write_json_atomic(json_path, payload)
    emit_progress(
        "complete",
        frames=frames,
        detections=detections,
        detail="Inference complete.",
        frame_total=frames or total_frames,
    )
    return payload


def main():
    args = parse_args()
    run_yolo_inference(
        weights_path=args.weights,
        input_path=args.input,
        output_path=args.output,
        json_path=args.json,
        progress_path=args.progress_json or None,
        preview_path=args.preview or None,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        vid_stride=args.vid_stride,
        use_model_cache=False,
    )


if __name__ == "__main__":
    main()
