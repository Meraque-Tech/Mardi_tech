#!/usr/bin/env python3
"""Run YOLO inference on a single image or video and emit JSON summary output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


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
    parser.add_argument("--backend", choices=["pytorch", "tensorrt"], default="pytorch")
    parser.add_argument("--engine-cache-dir", default="")
    parser.add_argument("--half", action="store_true")
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
    if stage == "exporting_engine":
        return 8.0
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
    write_json_atomic(progress_path, {
        "stage": stage,
        "frames": frames,
        "total_frames": total_frames,
        "detections": detections,
        "percent": round(progress_percent(stage, frames, total_frames), 1),
        "detail": detail or stage.replace("_", " ").title(),
        "updated_at": time.time(),
    })


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


def write_preview(preview_path: Path | None, image, max_width: int = 960):
    if preview_path is None:
        return
    height, width = image.shape[:2]
    preview = image
    if width > max_width:
        scale = max_width / width
        preview = cv2.resize(image, (max_width, max(1, int(height * scale))))
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), preview)


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def engine_cache_key(weights_path: Path, imgsz: int, device: str, half: bool) -> str:
    payload = {
        "sha256": file_sha256(weights_path),
        "imgsz": imgsz,
        "device": device or "0",
        "half": bool(half),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def cached_tensorrt_engine(weights_path: Path, cache_root: Path, imgsz: int, device: str, half: bool):
    from ultralytics import YOLO

    export_device = device or "0"
    cache_key = engine_cache_key(weights_path, imgsz, export_device, half)
    export_dir = cache_root / cache_key
    engine_path = export_dir / "model.engine"
    metadata_path = export_dir / "metadata.json"
    if engine_path.is_file():
        return engine_path, True, 0.0

    export_dir.mkdir(parents=True, exist_ok=True)
    source_path = export_dir / "source.pt"
    if not source_path.is_file():
        shutil.copy2(weights_path, source_path)

    started = time.perf_counter()
    try:
        exported = YOLO(str(source_path)).export(
            format="engine",
            imgsz=imgsz,
            half=half,
            device=export_device,
            dynamic=False,
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "TensorRT export failed. Confirm the container has NVIDIA GPU access, "
            "TensorRT support, and a CUDA-compatible Ultralytics environment."
        ) from exc

    exported_path = Path(str(exported)).expanduser().resolve()
    if not exported_path.is_file():
        fallback = source_path.with_suffix(".engine")
        exported_path = fallback if fallback.is_file() else exported_path
    if not exported_path.is_file():
        raise RuntimeError("TensorRT export completed without producing a .engine file.")
    if exported_path != engine_path:
        shutil.move(str(exported_path), str(engine_path))

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    metadata_path.write_text(json.dumps({
        "source_weights": str(weights_path),
        "imgsz": imgsz,
        "device": export_device,
        "half": half,
        "export_ms": elapsed_ms,
        "created_at": time.time(),
    }, indent=2), encoding="utf-8")
    return engine_path, False, elapsed_ms


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    json_path = Path(args.json).expanduser().resolve()
    progress_path = Path(args.progress_json).expanduser().resolve() if args.progress_json else None
    preview_path = Path(args.preview).expanduser().resolve() if args.preview else None
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    media_kind = media_type(input_path)
    video_stride = max(1, int(args.vid_stride or 1))
    source_total_frames = video_frame_count(input_path) if media_kind == "video" else 1
    total_frames = max(1, math.ceil(source_total_frames / video_stride)) if media_kind == "video" else 1
    raw_video_output = output_path.with_name(f"{output_path.stem}.raw{output_path.suffix}")
    writer_output_path = raw_video_output if media_kind == "video" else output_path

    from ultralytics import YOLO

    started = time.perf_counter()
    weights_path = Path(args.weights).expanduser().resolve()
    model_path = weights_path
    engine_cached = False
    export_ms = 0.0
    write_progress(
        progress_path,
        "loading_model",
        total_frames=total_frames,
        detail=f"Loading {args.backend} model.",
    )
    if args.backend == "tensorrt":
        cache_root = Path(args.engine_cache_dir or output_dir / "engines").expanduser().resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        write_progress(
            progress_path,
            "exporting_engine",
            total_frames=total_frames,
            detail="Preparing TensorRT engine. First run can take several minutes.",
        )
        model_path, engine_cached, export_ms = cached_tensorrt_engine(
            weights_path,
            cache_root,
            args.imgsz,
            args.device or "0",
            args.half,
        )

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(input_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device or None,
        save=False,
        verbose=False,
        stream=True,
        vid_stride=video_stride,
    )

    frames = 0
    detections = 0
    image_detections = []
    video_writer = None
    transcoded = False
    last_progress_at = 0.0
    last_preview_at = 0.0
    write_progress(
        progress_path,
        "processing",
        total_frames=total_frames,
        detail="Processing frames.",
    )
    for result in results:
        frames += 1
        boxes = getattr(result, "boxes", None)
        count = len(boxes) if boxes is not None else 0
        detections += count
        plotted = result.plot()
        now = time.perf_counter()
        if now - last_preview_at >= 0.5 or frames == 1:
            write_preview(preview_path, plotted)
            last_preview_at = now
        if frames == 1 and media_kind == "image":
            image_detections = collect_image_detections(result)
            if not cv2.imwrite(str(output_path), plotted):
                raise RuntimeError("Could not write annotated image output.")
            write_preview(preview_path, plotted)
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
            write_progress(
                progress_path,
                "processing",
                frames=frames,
                total_frames=total_frames,
                detections=detections,
                detail=f"Processed {frames} of {total_frames} frames.",
            )
            last_progress_at = now

    if video_writer is not None:
        video_writer.release()
        write_progress(
            progress_path,
            "encoding",
            frames=frames,
            total_frames=total_frames,
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
        "backend": args.backend,
        "engine_cached": engine_cached,
        "engine_export_ms": export_ms,
        "media_type": media_kind,
        "frames": frames,
        "detections": detections,
        "elapsed_ms": elapsed_ms,
        "output": str(output_path),
        "video_stride": video_stride,
        "browser_video": transcoded,
        "image_detections": image_detections,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_progress(
        progress_path,
        "complete",
        frames=frames,
        total_frames=frames or total_frames,
        detections=detections,
        detail="Inference complete.",
    )


if __name__ == "__main__":
    main()
