#!/usr/bin/env python3
"""Run YOLO inference on a single image or video and emit JSON summary output."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError("Unsupported media type.")


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


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    json_path = Path(args.json).expanduser().resolve()
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    started = time.perf_counter()
    model = YOLO(str(Path(args.weights).expanduser().resolve()))
    results = model.predict(
        source=str(input_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device or None,
        save=False,
        verbose=False,
        stream=True,
    )

    media_kind = media_type(input_path)
    frames = 0
    detections = 0
    image_detections = []
    video_writer = None
    for result in results:
        frames += 1
        boxes = getattr(result, "boxes", None)
        count = len(boxes) if boxes is not None else 0
        detections += count
        plotted = result.plot()
        if frames == 1 and media_kind == "image":
            image_detections = collect_image_detections(result)
            if not cv2.imwrite(str(output_path), plotted):
                raise RuntimeError("Could not write annotated image output.")
        elif media_kind == "video":
            if video_writer is None:
                height, width = plotted.shape[:2]
                capture = cv2.VideoCapture(str(input_path))
                fps = capture.get(cv2.CAP_PROP_FPS)
                capture.release()
                if fps <= 0:
                    fps = 25.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                if not video_writer.isOpened():
                    raise RuntimeError("Could not open annotated video output.")
            video_writer.write(plotted)

    if video_writer is not None:
        video_writer.release()

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    payload = {
        "engine": "python-ultralytics",
        "media_type": media_kind,
        "frames": frames,
        "detections": detections,
        "elapsed_ms": elapsed_ms,
        "output": str(output_path),
        "image_detections": image_detections,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
