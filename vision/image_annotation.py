#!/usr/bin/env python3
"""Download a Roboflow dataset and overlay YOLO annotations on its images."""

from __future__ import annotations

import argparse
import colorsys
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime for lightweight installs.
    yaml = None

try:
    from PIL import Image, ImageDraw, ImageFont, ImageOps
except ImportError as exc:  # pragma: no cover - handled at runtime.
    raise SystemExit("Pillow is required. Install it with: pip install pillow") from exc

try:
    import roboflow
except ImportError:  # pragma: no cover - handled at runtime.
    roboflow = None


ROBOFLOW_API_KEY = "BSHu0NfrIBFl58wtwWnP"
ROBOFLOW_WORKSPACE = "rnd-kyodu"
ROBOFLOW_PROJECT = "pineapple_ai_system"
ROBOFLOW_VERSION = 7
ROBOFLOW_FORMAT = "yolov8"

# Edit these paths, then run: python3 vision/image_annotation.py
INPUT_DIR = Path.home() / "roboflow_datasets" / ROBOFLOW_PROJECT
OUTPUT_DIR = Path("/home/aloy/Mardi_Annotated_Dataset")

# Set to True when INPUT_DIR already contains a downloaded YOLO dataset.
SKIP_DOWNLOAD = False
OVERWRITE_DOWNLOAD = True
MAX_IMAGES = 0
LINE_WIDTH = 0
LOG_EVERY_IMAGES = 100
LOG_FIRST_MISSING_LABELS = 10

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train", "valid", "val", "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a YOLOv8 dataset from Roboflow and write annotated preview "
            "images with bounding boxes or segmentation polygons overlaid."
        )
    )
    parser.add_argument("--api-key", default=ROBOFLOW_API_KEY, help="Roboflow API key.")
    parser.add_argument("--workspace", default=ROBOFLOW_WORKSPACE, help="Roboflow workspace slug.")
    parser.add_argument("--project", default=ROBOFLOW_PROJECT, help="Roboflow project slug.")
    parser.add_argument("--version", type=int, default=ROBOFLOW_VERSION, help="Roboflow dataset version number.")
    parser.add_argument(
        "--format",
        default=ROBOFLOW_FORMAT,
        help="Roboflow export format. Keep 'yolov8' for YOLO boxes or segmentation labels.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=INPUT_DIR,
        help=f"Folder where the dataset will be downloaded. Default: {INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Folder where annotated images will be written. Default: {OUTPUT_DIR}",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        default=SKIP_DOWNLOAD,
        help="Use an existing dataset in --download-dir instead of downloading from Roboflow.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=OVERWRITE_DOWNLOAD,
        help="Ask the Roboflow SDK to overwrite an existing download.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=MAX_IMAGES,
        help="Only annotate the first N images. Use 0 to annotate every image.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=LINE_WIDTH,
        help="Annotation stroke width in pixels. Use 0 for automatic sizing.",
    )
    return parser.parse_args()


def download_dataset(args: argparse.Namespace) -> Path:
    if args.skip_download:
        print(f"Skipping Roboflow download. Reading dataset from: {args.download_dir}")
        return find_dataset_root(args.download_dir)
    if roboflow is None:
        raise SystemExit("Roboflow is required. Install it with: pip install roboflow")

    download_dir = args.download_dir.expanduser().resolve()
    download_dir.parent.mkdir(parents=True, exist_ok=True)
    print(
        "Fetching Roboflow dataset: "
        f"workspace={args.workspace}, project={args.project}, "
        f"version={args.version}, format={args.format}"
    )
    print(f"Download folder: {download_dir}")

    rf = roboflow.Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)
    try:
        dataset = version.download(
            args.format,
            location=str(download_dir),
            overwrite=args.overwrite,
        )
    except TypeError:
        dataset = version.download(args.format, location=str(download_dir))

    dataset_location = Path(getattr(dataset, "location", download_dir))
    print(f"Roboflow returned dataset location: {dataset_location}")
    return find_dataset_root(dataset_location)


def find_dataset_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    if looks_like_dataset_root(root):
        print(f"Found YOLO dataset root: {root}")
        return root

    for candidate in sorted(root.rglob("*")):
        if candidate.is_dir() and looks_like_dataset_root(candidate):
            print(f"Found YOLO dataset root: {candidate}")
            return candidate

    raise FileNotFoundError(
        f"No YOLO dataset root found under {root}. "
        "Expected data.yaml, dataset.yaml, images/, or train/valid/test images folders. "
        f"Folder preview: {folder_preview(root)}"
    )


def folder_preview(root: Path, limit: int = 20) -> str:
    if not root.exists():
        return "folder does not exist"

    entries = []
    for index, item in enumerate(sorted(root.rglob("*"))):
        if index >= limit:
            entries.append("...")
            break
        entries.append(str(item.relative_to(root)))
    return ", ".join(entries) if entries else "folder is empty"


def looks_like_dataset_root(path: Path) -> bool:
    return (
        (path / "data.yaml").is_file()
        or (path / "dataset.yaml").is_file()
        or (path / "images").is_dir()
        or any((path / split / "images").is_dir() for split in SPLIT_ALIASES)
    )


def read_class_names(dataset_root: Path) -> list[str]:
    if yaml is None:
        return []

    yaml_path = next(
        (candidate for candidate in (dataset_root / "data.yaml", dataset_root / "dataset.yaml") if candidate.is_file()),
        None,
    )
    if yaml_path is None:
        matches = list(dataset_root.rglob("data.yaml")) + list(dataset_root.rglob("dataset.yaml"))
        yaml_path = matches[0] if matches else None
    if yaml_path is None:
        return []

    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8", errors="replace")) or {}
    names = payload.get("names", [])
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names, key=lambda value: int(value))]
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def image_files(dataset_root: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "annotated" not in {part.lower() for part in path.parts}
    )


def label_for_image(dataset_root: Path, image_path: Path) -> Path | None:
    relative = image_path.relative_to(dataset_root)
    parts = list(relative.parts)

    if "images" in parts:
        index = parts.index("images")
        label_parts = parts.copy()
        label_parts[index] = "labels"
        candidate = dataset_root.joinpath(*label_parts).with_suffix(".txt")
        if candidate.is_file():
            return candidate

    for split in SPLIT_ALIASES:
        candidate = dataset_root / split / "labels" / f"{image_path.stem}.txt"
        if candidate.is_file():
            return candidate

    candidate = dataset_root / "labels" / f"{image_path.stem}.txt"
    if candidate.is_file():
        return candidate

    matches = list(dataset_root.rglob(f"{image_path.stem}.txt"))
    return matches[0] if matches else None


def class_color(class_id: int) -> tuple[int, int, int]:
    hue = (class_id * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.74, 0.92)
    return int(red * 255), int(green * 255), int(blue * 255)


def clamp_point(x_value: float, y_value: float, width: int, height: int) -> tuple[float, float]:
    return (
        min(max(x_value * width, 0), width),
        min(max(y_value * height, 0), height),
    )


def text_size(draw: ImageDraw.ImageDraw, text: str, font: Any) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def draw_label(
    draw: ImageDraw.ImageDraw,
    label: str,
    anchor: tuple[float, float],
    color: tuple[int, int, int],
    font: Any,
) -> None:
    text_width, text_height = text_size(draw, label, font)
    x_value, y_value = anchor
    y_value = max(0, y_value - text_height - 4)
    box = (x_value, y_value, x_value + text_width + 6, y_value + text_height + 4)
    draw.rectangle(box, fill=color)
    draw.text((x_value + 3, y_value + 2), label, fill="white", font=font)


def overlay_annotations(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    class_names: list[str],
    requested_line_width: int,
) -> bool:
    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    except OSError:
        print(f"Skipping unreadable image: {image_path}", file=sys.stderr)
        return False

    width, height = image.size
    line_width = requested_line_width or max(2, round(min(width, height) / 220))
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(image, "RGBA")

    for row in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = row.split()
        if len(fields) < 5:
            continue
        try:
            class_id = int(float(fields[0]))
            values = [float(value) for value in fields[1:]]
        except ValueError:
            continue

        color = class_color(class_id)
        label = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)

        if len(values) == 4:
            center_x, center_y, box_width, box_height = values
            x1 = max(0, (center_x - box_width / 2) * width)
            y1 = max(0, (center_y - box_height / 2) * height)
            x2 = min(width, (center_x + box_width / 2) * width)
            y2 = min(height, (center_y + box_height / 2) * height)
            draw.rectangle((x1, y1, x2, y2), outline=color + (255,), width=line_width)
            draw_label(draw, label, (x1, y1), color, font)
        elif len(values) >= 6 and len(values) % 2 == 0:
            points = [
                clamp_point(values[index], values[index + 1], width, height)
                for index in range(0, len(values), 2)
            ]
            draw.polygon(points, outline=color + (255,), fill=color + (55,))
            draw.line(points + [points[0]], fill=color + (255,), width=line_width)
            draw_label(draw, label, points[0], color, font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return True


def progress_interval(total_images: int) -> int:
    if LOG_EVERY_IMAGES > 0:
        return LOG_EVERY_IMAGES
    return max(1, total_images // 20)


def annotate_dataset(dataset_root: Path, output_dir: Path, max_images: int, line_width: int) -> tuple[int, int]:
    class_names = read_class_names(dataset_root)
    images = image_files(dataset_root)
    if max_images > 0:
        images = images[:max_images]

    total_images = len(images)
    print(f"Scanning dataset images: {total_images} found")
    print(f"Class names loaded: {len(class_names)}")
    if max_images > 0:
        print(f"Max images enabled: processing first {max_images} image(s)")
    if total_images == 0:
        print("No images found to annotate.")
        return 0, 0

    annotated = 0
    missing_labels = 0
    unreadable_images = 0
    interval = progress_interval(total_images)

    print("Starting image annotation...")
    for index, image_path in enumerate(images, start=1):
        label_path = label_for_image(dataset_root, image_path)
        if label_path is None:
            missing_labels += 1
            if missing_labels <= LOG_FIRST_MISSING_LABELS:
                print(f"[{index}/{total_images}] Missing label: {image_path.relative_to(dataset_root)}")
            continue

        relative = image_path.relative_to(dataset_root)
        output_path = output_dir / relative
        if overlay_annotations(image_path, label_path, output_path, class_names, line_width):
            annotated += 1
        else:
            unreadable_images += 1

        if index == 1 or index % interval == 0 or index == total_images:
            percent = (index / total_images) * 100
            print(
                f"Progress: {index}/{total_images} ({percent:.1f}%) | "
                f"annotated={annotated}, missing_labels={missing_labels}, unreadable={unreadable_images}",
                flush=True,
            )

    print(
        "Annotation complete: "
        f"annotated={annotated}, missing_labels={missing_labels}, unreadable={unreadable_images}"
    )
    return annotated, missing_labels


def main() -> int:
    args = parse_args()
    dataset_root = download_dataset(args)
    output_dir = args.output_dir.expanduser().resolve()

    print(f"Dataset root: {dataset_root}")
    print(f"Annotated output: {output_dir}")
    annotated, missing_labels = annotate_dataset(dataset_root, output_dir, args.max_images, args.line_width)
    print(f"Annotated images: {annotated}")
    if missing_labels:
        print(f"Images without matching labels: {missing_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
