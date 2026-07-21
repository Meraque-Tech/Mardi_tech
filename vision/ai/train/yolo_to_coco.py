#!/usr/bin/env python3
"""Convert prepared YOLO detection datasets to COCO detection layout."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_OUTPUTS = {
    "train": ("train", "train2017", "instances_train2017.json"),
    "val": ("val", "val2017", "instances_val2017.json"),
    "test": ("test", "test2017", "instances_test2017.json"),
}


@dataclass
class SplitSummary:
    split: str
    image_count: int = 0
    annotation_count: int = 0
    missing_label_count: int = 0
    malformed_label_count: int = 0
    unknown_class_count: int = 0
    skipped_box_count: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class ConversionResult:
    output_dir: Path
    summary_path: Path
    annotation_paths: dict[str, Path]
    summary: dict[str, Any]


def find_dataset_yaml(dataset: Path) -> Path:
    dataset = dataset.expanduser().resolve()
    if dataset.is_file():
        return dataset
    for name in ("data.yaml", "data.yml", "dataset.yaml"):
        candidate = dataset / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find data.yaml in {dataset}.")


def read_dataset_yaml(dataset: Path) -> tuple[Path, dict[str, Any]]:
    yaml_path = find_dataset_yaml(dataset)
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Could not read dataset YAML at {yaml_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Dataset YAML must contain a mapping: {yaml_path}")
    return yaml_path, payload


def dataset_root(yaml_path: Path, payload: dict[str, Any]) -> Path:
    root = Path(str(payload.get("path") or yaml_path.parent)).expanduser()
    if not root.is_absolute():
        root = yaml_path.parent / root
    return root.resolve()


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        normalized: dict[int, str] = {}
        for key, value in names.items():
            try:
                normalized[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return normalized
    return {}


def resolve_split_images(root: Path, value: Any) -> Path | None:
    if not value:
        return None
    values = value if isinstance(value, list) else [value]
    for item in values:
        candidate = Path(str(item)).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        if candidate.is_dir():
            return candidate
    return None


def infer_label_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        candidate = Path(*parts[:index], "labels", *parts[index + 1:])
        if candidate.is_dir():
            return candidate.resolve()
    candidates = (
        image_dir.parent / "labels",
        image_dir.parent.parent / "labels" / image_dir.name,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return image_dir.parent / "labels"


def image_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    if data.startswith(b"GIF") and len(data) >= 10:
        width, height = struct.unpack("<HH", data[6:10])
        return int(width), int(height)
    if data.startswith(b"BM") and len(data) >= 26:
        width, height = struct.unpack("<II", data[18:26])
        return int(width), abs(int(height))
    if data.startswith(b"\xff\xd8"):
        index = 2
        while index + 9 < len(data):
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            index += 2
            if marker in {0xD8, 0xD9}:
                continue
            if index + 2 > len(data):
                break
            length = struct.unpack(">H", data[index:index + 2])[0]
            if marker in set(range(0xC0, 0xC4)) | set(range(0xC5, 0xC8)) | set(range(0xC9, 0xCC)) | set(range(0xCD, 0xD0)):
                if index + 7 <= len(data):
                    height, width = struct.unpack(">HH", data[index + 3:index + 7])
                    return int(width), int(height)
            index += max(length, 2)
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        pass
    try:
        import cv2

        image = cv2.imread(str(path))
        if image is not None:
            height, width = image.shape[:2]
            return int(width), int(height)
    except Exception:
        pass
    raise RuntimeError(f"Could not read image dimensions for {path}.")


def yolo_label_path(label_dir: Path, image_path: Path) -> Path:
    return label_dir / f"{image_path.stem}.txt"


def copy_or_link_image(source: Path, target: Path, link_images: bool):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if link_images:
        try:
            target.symlink_to(source.resolve())
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def convert_box(parts: list[str], width: int, height: int) -> tuple[int, list[float], float] | None:
    if len(parts) < 5:
        return None
    class_value = float(parts[0])
    class_id = int(class_value)
    if class_id != class_value:
        return None
    x_center, y_center, box_width, box_height = [float(value) for value in parts[1:5]]
    if box_width <= 0 or box_height <= 0:
        return None
    x_min = (x_center - box_width / 2) * width
    y_min = (y_center - box_height / 2) * height
    x_max = (x_center + box_width / 2) * width
    y_max = (y_center + box_height / 2) * height
    x_min = max(0.0, min(float(width), x_min))
    y_min = max(0.0, min(float(height), y_min))
    x_max = max(0.0, min(float(width), x_max))
    y_max = max(0.0, min(float(height), y_max))
    clipped_width = x_max - x_min
    clipped_height = y_max - y_min
    if clipped_width <= 0 or clipped_height <= 0:
        return None
    bbox = [round(x_min, 4), round(y_min, 4), round(clipped_width, 4), round(clipped_height, 4)]
    return class_id, bbox, round(clipped_width * clipped_height, 4)


def coco_categories(names: dict[int, str]) -> list[dict[str, Any]]:
    return [
        {"id": class_id, "name": name, "supercategory": "object"}
        for class_id, name in sorted(names.items())
    ]


def convert_split(
    split_key: str,
    image_dir: Path,
    output_dir: Path,
    names: dict[int, str],
    image_id_start: int,
    annotation_id_start: int,
    link_images: bool,
) -> tuple[dict[str, Any], SplitSummary, int, int]:
    _source_name, image_output_name, annotation_name = SPLIT_OUTPUTS[split_key]
    target_image_dir = output_dir / image_output_name
    label_dir = infer_label_dir(image_dir)
    summary = SplitSummary(split=split_key)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_id = image_id_start
    annotation_id = annotation_id_start

    for image_path in sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
        relative_name = image_path.relative_to(image_dir).as_posix()
        target_image = target_image_dir / relative_name
        try:
            width, height = image_size(image_path)
        except (OSError, RuntimeError) as exc:
            summary.warnings.append(str(exc))
            continue
        copy_or_link_image(image_path, target_image, link_images)
        images.append({
            "id": image_id,
            "file_name": relative_name,
            "width": width,
            "height": height,
        })
        summary.image_count += 1

        label_path = yolo_label_path(label_dir / image_path.relative_to(image_dir).parent, image_path)
        if not label_path.is_file():
            summary.missing_label_count += 1
            image_id += 1
            continue
        try:
            lines = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            summary.warnings.append(f"Could not read label file {label_path}: {exc}")
            image_id += 1
            continue
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                converted = convert_box(line.split(), width, height)
            except ValueError:
                converted = None
            if converted is None:
                summary.malformed_label_count += 1
                continue
            class_id, bbox, area = converted
            if class_id not in names:
                summary.unknown_class_count += 1
                summary.warnings.append(f"Unknown class id {class_id} in {label_path}:{line_number}.")
                continue
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": [],
            })
            annotation_id += 1
            summary.annotation_count += 1
        image_id += 1

    coco = {
        "info": {"description": "Converted from YOLO format"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": coco_categories(names),
    }
    annotation_path = output_dir / "annotations" / annotation_name
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    return coco, summary, image_id, annotation_id


def convert_yolo_to_coco(
    dataset: str | Path,
    output_root: str | Path | None = None,
    dataset_name: str | None = None,
    link_images: bool = False,
) -> ConversionResult:
    yaml_path, payload = read_dataset_yaml(Path(dataset))
    root = dataset_root(yaml_path, payload)
    names = normalize_names(payload.get("names"))
    if not names:
        raise RuntimeError(f"No class names found in {yaml_path}.")

    dataset_name = dataset_name or yaml_path.parent.name
    output_base = Path(output_root).expanduser() if output_root else yaml_path.parent / ".coco"
    output_dir = (output_base / dataset_name).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_values = {
        "train": payload.get("train"),
        "val": payload.get("val") or payload.get("valid"),
        "test": payload.get("test"),
    }
    image_id = 1
    annotation_id = 1
    annotation_paths: dict[str, Path] = {}
    split_summaries: dict[str, Any] = {}

    for split_key, split_value in split_values.items():
        image_dir = resolve_split_images(root, split_value)
        if image_dir is None:
            if split_key == "test":
                continue
            raise FileNotFoundError(f"Missing {split_key} image directory from {yaml_path}.")
        _coco, summary, image_id, annotation_id = convert_split(
            split_key,
            image_dir,
            output_dir,
            names,
            image_id,
            annotation_id,
            link_images,
        )
        annotation_paths[split_key] = output_dir / "annotations" / SPLIT_OUTPUTS[split_key][2]
        split_summaries[split_key] = {
            "image_count": summary.image_count,
            "annotation_count": summary.annotation_count,
            "missing_label_count": summary.missing_label_count,
            "malformed_label_count": summary.malformed_label_count,
            "unknown_class_count": summary.unknown_class_count,
            "skipped_box_count": summary.skipped_box_count,
            "warnings": summary.warnings,
        }

    summary_payload = {
        "source_yaml": str(yaml_path),
        "source_root": str(root),
        "output_dir": str(output_dir),
        "class_count": len(names),
        "classes": [{"id": key, "name": value} for key, value in sorted(names.items())],
        "splits": split_summaries,
    }
    summary_path = output_dir / "conversion_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return ConversionResult(output_dir, summary_path, annotation_paths, summary_payload)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a YOLO dataset to COCO layout.")
    parser.add_argument("--data", required=True, help="Path to data.yaml or a dataset directory.")
    parser.add_argument("--output-root", default=None, help="Root directory where the COCO dataset folder is written.")
    parser.add_argument("--name", default=None, help="Output dataset folder name.")
    parser.add_argument("--link-images", action="store_true", help="Symlink images instead of copying when possible.")
    return parser.parse_args()


def main():
    args = parse_args()
    result = convert_yolo_to_coco(args.data, args.output_root, args.name, args.link_images)
    print(json.dumps(result.summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
