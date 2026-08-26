"""Dataset discovery, validation, splitting, and archive services."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import yaml
from fastapi import HTTPException

from ..common.uploads import replace_directory
from ..dataset_provenance import roboflow_pre_augmentation_summary
from ..stratified_split import SPLIT_NAMES, stratified_split


DatasetProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class DatasetDependencies:
    data_root: Path
    image_extensions: set[str]
    split_aliases: dict[str, str]
    split_metadata_filename: str
    summary_filename: str
    download_max_age_seconds: int
    downloads: dict[str, dict]
    downloads_lock: threading.Lock
    clean_name: Callable[[str, str], str]


def configure_datasets(deps: DatasetDependencies) -> None:
    """Bind dataset services to application paths and shared download state."""
    globals().update({
        "DATA_ROOT": deps.data_root,
        "IMAGE_EXTENSIONS": deps.image_extensions,
        "SPLIT_ALIASES": deps.split_aliases,
        "SPLIT_METADATA_FILE": deps.split_metadata_filename,
        "DATASET_SUMMARY_FILE": deps.summary_filename,
        "DATASET_DOWNLOAD_MAX_AGE_SECONDS": deps.download_max_age_seconds,
        "dataset_downloads": deps.downloads,
        "dataset_download_lock": deps.downloads_lock,
        "clean_name": deps.clean_name,
    })


def validate_classes(classes: list[str]) -> dict[int, str]:
    names = [item.strip() for item in classes if item.strip()]
    if not names:
        raise HTTPException(status_code=400, detail="At least one class name is required.")
    return {index: name for index, name in enumerate(names)}


def normalize_yaml_names(raw_names) -> list[str]:
    if isinstance(raw_names, list):
        return [str(name).strip() for name in raw_names if str(name).strip()]

    if isinstance(raw_names, dict):
        normalized = []
        def sort_key(item):
            try:
                return (0, int(item[0]))
            except (TypeError, ValueError):
                return (1, str(item[0]))

        for key, value in sorted(raw_names.items(), key=sort_key):
            name = str(value).strip()
            if name:
                normalized.append(name)
        return normalized

    return []


def read_yaml_class_names(yaml_path: Path) -> list[str]:
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not read dataset YAML: {exc}") from exc

    names = normalize_yaml_names(payload.get("names"))
    if not names:
        raise HTTPException(status_code=400, detail=f"No class names found in {yaml_path}")
    return names


def detect_dataset_classes(source: Path) -> list[str]:
    yaml_path = source if source.is_file() and source.suffix.lower() in {".yaml", ".yml"} else find_dataset_yaml(find_dataset_root(source))
    if not yaml_path:
        raise HTTPException(status_code=404, detail="No data.yaml or dataset.yaml file found.")
    return read_yaml_class_names(yaml_path)


def resolve_dataset_classes(root: Path, classes: list[str]) -> dict[int, str]:
    names = [item.strip() for item in classes if item.strip()]
    if not names:
        names = detect_dataset_classes(root)
    return validate_classes(names)


def dataset_response(
    yaml_path: Path,
    message: str,
    progress_callback: Optional[DatasetProgressCallback] = None,
    source_metadata: Optional[dict] = None,
) -> dict:
    try:
        classes = read_yaml_class_names(yaml_path)
    except HTTPException:
        classes = []
    summary = inspect_dataset_yaml(yaml_path, classes, progress_callback, source_metadata)
    try:
        (yaml_path.parent / DATASET_SUMMARY_FILE).write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    except OSError:
        pass
    return {
        "dataset_yaml": str(yaml_path),
        "classes": classes,
        "summary": summary,
        "message": message,
    }


def resolve_yaml_dataset_root(yaml_path: Path, payload: dict) -> Path:
    root = payload.get("path") or yaml_path.parent
    root_path = Path(root).expanduser()
    if not root_path.is_absolute():
        root_path = yaml_path.parent / root_path
    return root_path.resolve()


def split_image_folder(dataset_root: Path, value) -> Optional[Path]:
    if not value:
        return None
    candidates = value if isinstance(value, list) else [value]
    for item in candidates:
        path = Path(str(item)).expanduser()
        if not path.is_absolute():
            path = dataset_root / path
        if path.exists():
            return path
    return None


def prepared_dataset_yaml(dataset_yaml: str) -> tuple[Path, Path, dict]:
    yaml_path = Path(dataset_yaml).expanduser()
    if not yaml_path.is_absolute():
        yaml_path = DATA_ROOT / yaml_path
    yaml_path = yaml_path.resolve()
    data_root = DATA_ROOT.resolve()

    try:
        yaml_path.relative_to(data_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=403,
            detail="Dataset downloads are limited to the web dataset workspace.",
        ) from exc

    if not yaml_path.is_file() or yaml_path.suffix.lower() not in {".yaml", ".yml"}:
        raise HTTPException(status_code=404, detail="Prepared dataset YAML was not found.")

    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not read dataset YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Dataset YAML must contain a mapping.")

    dataset_root = resolve_yaml_dataset_root(yaml_path, payload)
    try:
        dataset_root.relative_to(data_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=403,
            detail="The prepared dataset points outside the web dataset workspace.",
        ) from exc
    if not dataset_root.is_dir():
        raise HTTPException(status_code=404, detail="Prepared dataset directory was not found.")

    portable_payload = dict(payload)
    portable_payload["path"] = "."
    for split in SPLIT_NAMES:
        split_value = payload.get(split)
        if not split_value:
            portable_payload.pop(split, None)
            continue

        split_values = split_value if isinstance(split_value, list) else [split_value]
        portable_values = []
        for value in split_values:
            split_path = Path(str(value)).expanduser()
            if not split_path.is_absolute():
                split_path = dataset_root / split_path
            split_path = split_path.resolve()
            try:
                relative_path = split_path.relative_to(dataset_root)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dataset {split} path points outside the dataset directory.",
                ) from exc
            if not split_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset {split} path was not found: {relative_path}",
                )
            portable_values.append(relative_path.as_posix())

        portable_payload[split] = portable_values if isinstance(split_value, list) else portable_values[0]

    if not portable_payload.get("train") or not portable_payload.get("val"):
        raise HTTPException(status_code=400, detail="Prepared dataset must contain train and val paths.")

    return yaml_path, dataset_root, portable_payload


def build_dataset_archive(dataset_yaml: str) -> tuple[Path, str]:
    yaml_path, dataset_root, portable_payload = prepared_dataset_yaml(dataset_yaml)
    archive_name = f"{clean_name(dataset_root.name, 'prepared_dataset')}.zip"
    source_yaml_relative = (
        yaml_path.relative_to(dataset_root)
        if yaml_path.is_relative_to(dataset_root)
        else None
    )
    descriptor, archive_value = tempfile.mkstemp(prefix="yolov8-dataset-", suffix=".zip")
    os.close(descriptor)
    archive_path = Path(archive_value)

    try:
        with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
            archive.writestr(
                "data.yaml",
                yaml.safe_dump(portable_payload, sort_keys=False),
                compress_type=zipfile.ZIP_DEFLATED,
            )
            files = sorted(
                (item for item in dataset_root.rglob("*") if item.is_file()),
                key=lambda item: item.relative_to(dataset_root).as_posix(),
            )
            for item in files:
                relative_path = item.relative_to(dataset_root)
                if (
                    relative_path.as_posix() == "data.yaml"
                    or relative_path == source_yaml_relative
                    or item.is_symlink()
                ):
                    continue
                compression = (
                    zipfile.ZIP_STORED
                    if item.suffix.lower() in IMAGE_EXTENSIONS
                    else zipfile.ZIP_DEFLATED
                )
                archive.write(item, relative_path.as_posix(), compress_type=compression)
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise

    return archive_path, archive_name


def build_annotated_dataset_archive(dataset_yaml: str) -> tuple[Path, str]:
    _yaml_path, dataset_root, portable_payload = prepared_dataset_yaml(dataset_yaml)
    try:
        from vision.image_annotation import annotate_dataset
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Annotated dataset export requires Pillow and the image_annotation module.",
        ) from exc

    output_root = Path(tempfile.mkdtemp(prefix="annotated-dataset-"))
    descriptor, archive_value = tempfile.mkstemp(prefix="annotated-yolov8-dataset-", suffix=".zip")
    os.close(descriptor)
    archive_path = Path(archive_value)
    archive_name = f"{clean_name(dataset_root.name, 'prepared_dataset')}_annotated.zip"

    try:
        annotated, missing_labels = annotate_dataset(dataset_root, output_root, max_images=0, line_width=0)
        if annotated == 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No annotated images were created. "
                    f"Images without matching labels: {missing_labels}."
                ),
            )

        with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
            archive.writestr(
                "data.yaml",
                yaml.safe_dump(portable_payload, sort_keys=False),
                compress_type=zipfile.ZIP_DEFLATED,
            )
            for item in sorted(output_root.rglob("*"), key=lambda path: path.relative_to(output_root).as_posix()):
                if not item.is_file() or item.is_symlink():
                    continue
                relative_path = item.relative_to(output_root)
                compression = (
                    zipfile.ZIP_STORED
                    if item.suffix.lower() in IMAGE_EXTENSIONS
                    else zipfile.ZIP_DEFLATED
                )
                archive.write(item, relative_path.as_posix(), compress_type=compression)
    except HTTPException:
        archive_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        archive_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Annotated dataset export failed: {exc}") from exc
    finally:
        shutil.rmtree(output_root, ignore_errors=True)

    return archive_path, archive_name


def cleanup_expired_dataset_downloads():
    cutoff = time.time() - DATASET_DOWNLOAD_MAX_AGE_SECONDS
    expired_paths = []
    with dataset_download_lock:
        for download_id, download in list(dataset_downloads.items()):
            if download["created_at"] < cutoff:
                expired_paths.append(Path(download["path"]))
                dataset_downloads.pop(download_id, None)
    for path in expired_paths:
        path.unlink(missing_ok=True)


def label_folder_for_images(dataset_root: Path, images_path: Optional[Path]) -> Optional[Path]:
    if images_path is None:
        return None
    parts = list(images_path.parts)
    if "images" in parts:
        index = parts.index("images")
        parts[index] = "labels"
        return Path(*parts)
    relative = images_path.relative_to(dataset_root) if images_path.is_relative_to(dataset_root) else images_path.name
    return dataset_root / "labels" / relative


def update_class_distribution(
    images: list[Path],
    labels_path: Optional[Path],
    distribution: dict[int, dict],
    progress_callback: Optional[Callable[[int], None]] = None,
) -> tuple[int, int, int]:
    missing_labels = 0
    malformed_rows = 0
    unknown_class_rows = 0
    for index, image in enumerate(images, start=1):
        label_path = labels_path / f"{image.stem}.txt" if labels_path else None
        if label_path is None or not label_path.is_file():
            missing_labels += 1
            if progress_callback:
                progress_callback(index)
            continue

        classes_in_image: set[int] = set()
        for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.strip().split()
            if not fields:
                continue
            try:
                class_id = int(fields[0])
            except ValueError:
                malformed_rows += 1
                continue
            if class_id not in distribution:
                unknown_class_rows += 1
                continue
            distribution[class_id]["instances"] += 1
            classes_in_image.add(class_id)

        for class_id in classes_in_image:
            distribution[class_id]["images"] += 1
        if progress_callback:
            progress_callback(index)

    return missing_labels, malformed_rows, unknown_class_rows


def read_split_metadata(dataset_root: Path, yaml_path: Path) -> dict:
    candidates = [dataset_root / SPLIT_METADATA_FILE, yaml_path.parent / SPLIT_METADATA_FILE]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def inspect_dataset_yaml(
    yaml_path: Path,
    classes: list[str],
    progress_callback: Optional[DatasetProgressCallback] = None,
    source_metadata: Optional[dict] = None,
) -> dict:
    warnings = []
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return {"warnings": [f"Could not inspect dataset YAML: {exc}"]}

    dataset_root = resolve_yaml_dataset_root(yaml_path, payload)
    splits = {}
    total_images = 0
    total_missing = 0
    malformed_rows = 0
    unknown_class_rows = 0
    class_distribution = {
        class_id: {
            "class_id": class_id,
            "class_name": class_name,
            "images": 0,
            "instances": 0,
        }
        for class_id, class_name in enumerate(classes)
    }
    split_distributions = {}
    split_contexts = []
    for split in SPLIT_NAMES:
        images_path = split_image_folder(dataset_root, payload.get(split))
        labels_path = label_folder_for_images(dataset_root, images_path)
        images = image_files(images_path) if images_path and images_path.is_dir() else []
        split_distribution = {
            class_id: {
                "class_id": class_id,
                "class_name": class_name,
                "images": 0,
                "instances": 0,
            }
            for class_id, class_name in enumerate(classes)
        }
        total_images += len(images)
        split_contexts.append((split, images_path, labels_path, images, split_distribution))

    inspected_images = 0
    if progress_callback:
        progress_callback("inspecting", 0, total_images, f"Inspecting 0 of {total_images} images")

    for split, images_path, labels_path, images, split_distribution in split_contexts:
        split_start = inspected_images
        inspection_progress = None
        if progress_callback:
            def inspection_progress(count: int, offset: int = split_start):
                current = offset + count
                progress_callback(
                    "inspecting",
                    current,
                    total_images,
                    f"Inspecting {current} of {total_images} images",
                )
        missing_labels, split_malformed, split_unknown = update_class_distribution(
            images,
            labels_path,
            split_distribution,
            inspection_progress,
        )
        inspected_images += len(images)
        total_missing += missing_labels
        malformed_rows += split_malformed
        unknown_class_rows += split_unknown
        split_distributions[split] = split_distribution
        for class_id, row in split_distribution.items():
            class_distribution[class_id]["images"] += row["images"]
            class_distribution[class_id]["instances"] += row["instances"]
        splits[split] = {
            "images": len(images),
            "missing_labels": missing_labels,
            "image_path": str(images_path) if images_path else "",
            "label_path": str(labels_path) if labels_path else "",
            "class_distribution": list(split_distribution.values()),
        }

    split_metadata = read_split_metadata(dataset_root, yaml_path)
    split_strategy = split_metadata.get("strategy") or "existing"
    configured_ratios = split_metadata.get("ratios") or {}
    ratios = {
        split: float(configured_ratios.get(split, 0))
        for split in SPLIT_NAMES
    }
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-6):
        ratios = {
            split: (splits[split]["images"] / total_images if total_images else 0.0)
            for split in SPLIT_NAMES
        }

    active_splits = [split for split in SPLIT_NAMES if ratios[split] > 0]
    class_balance = []
    for class_id, class_name in enumerate(classes):
        total_class_images = class_distribution[class_id]["images"]
        total_class_instances = class_distribution[class_id]["instances"]
        balance_splits = {}
        max_image_deviation = 0.0
        for split in SPLIT_NAMES:
            row = split_distributions[split][class_id]
            image_share = (
                row["images"] / total_class_images
                if total_class_images else 0.0
            )
            instance_share = (
                row["instances"] / total_class_instances
                if total_class_instances else 0.0
            )
            image_deviation = (image_share - ratios[split]) * 100
            max_image_deviation = max(max_image_deviation, abs(image_deviation))
            balance_splits[split] = {
                "images": row["images"],
                "instances": row["instances"],
                "image_share": round(image_share * 100, 2),
                "instance_share": round(instance_share * 100, 2),
                "target_share": round(ratios[split] * 100, 2),
                "image_deviation": round(image_deviation, 2),
            }

        class_balance.append({
            "class_id": class_id,
            "class_name": class_name,
            "images": total_class_images,
            "instances": total_class_instances,
            "splits": balance_splits,
            "max_image_deviation": round(max_image_deviation, 2),
        })

        if total_class_images == 0:
            warnings.append(f"Class '{class_name}' has no labeled images.")
        elif total_class_images < len(active_splits):
            warnings.append(
                f"Class '{class_name}' appears in only {total_class_images} "
                f"image{'s' if total_class_images != 1 else ''}; representation in all "
                f"{len(active_splits)} splits is not possible."
            )
        elif split_strategy == "multi_label_stratified":
            missing_splits = [
                split for split in active_splits
                if balance_splits[split]["images"] == 0
            ]
            if missing_splits:
                warnings.append(
                    f"Class '{class_name}' could not be represented in: "
                    f"{', '.join(missing_splits)}."
                )
            elif total_class_images >= 10 and max_image_deviation > 10:
                warnings.append(
                    f"Class '{class_name}' differs from the requested split ratio by up "
                    f"to {max_image_deviation:.1f} percentage points because of "
                    "multi-label constraints."
                )

    if not classes:
        warnings.append("No class names found.")
    if splits["train"]["images"] == 0:
        warnings.append("No training images found.")
    if splits["val"]["images"] == 0:
        warnings.append("No validation images found.")
    if total_missing:
        warnings.append(f"{total_missing} images are missing label files.")
    if malformed_rows:
        warnings.append(f"{malformed_rows} malformed annotation rows were ignored.")
    if unknown_class_rows:
        warnings.append(f"{unknown_class_rows} annotations reference unknown class IDs.")

    summary = {
        "dataset_root": str(dataset_root),
        "class_count": len(classes),
        "classes": classes,
        "class_distribution": list(class_distribution.values()),
        "class_balance": class_balance,
        "splits": splits,
        "split_strategy": split_strategy,
        "split_ratios": {split: round(ratios[split] * 100, 2) for split in SPLIT_NAMES},
        "split_seed": split_metadata.get("seed"),
        "total_images": total_images,
        "missing_labels": total_missing,
        "warnings": warnings,
    }
    pre_augmentation = roboflow_pre_augmentation_summary(dataset_root, splits, source_metadata)
    if pre_augmentation:
        summary["pre_augmentation"] = pre_augmentation
    return summary

def validate_split(split: SplitConfig) -> tuple[float, float, float]:
    total = split.train + split.val + split.test
    if total != 100:
        raise HTTPException(status_code=400, detail="Train, val, and test split values must total 100%.")
    return split.train / total, split.val / total, split.test / total


def find_dataset_yaml(root: Path) -> Optional[Path]:
    for name in ("data.yaml", "dataset.yaml"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    matches = list(root.rglob("data.yaml")) + list(root.rglob("dataset.yaml"))
    return matches[0] if matches else None


def find_dataset_root(root: Path) -> Path:
    if (root / "images").is_dir() or (root / "train").is_dir() or find_dataset_yaml(root):
        return root

    for candidate in root.rglob("*"):
        if not candidate.is_dir():
            continue
        if (candidate / "images").is_dir() or (candidate / "train").is_dir() or find_dataset_yaml(candidate):
            return candidate

    return root


def split_dirs(root: Path) -> dict[str, dict[str, Path]]:
    layouts: dict[str, dict[str, Path]] = {}

    for alias, split in SPLIT_ALIASES.items():
        images_a = root / "images" / alias
        labels_a = root / "labels" / alias
        if images_a.is_dir():
            layouts[split] = {"images": images_a, "labels": labels_a}

        images_b = root / alias / "images"
        labels_b = root / alias / "labels"
        if images_b.is_dir():
            layouts[split] = {"images": images_b, "labels": labels_b}

    return layouts


def flat_dirs(root: Path) -> Optional[dict[str, Path]]:
    images = root / "images"
    labels = root / "labels"
    if images.is_dir():
        return {"images": images, "labels": labels}
    return None


def image_files(folder: Path) -> list[Path]:
    return sorted(
        item for item in folder.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def has_image_files(folder: Optional[Path]) -> bool:
    return bool(
        folder
        and folder.is_dir()
        and any(
            item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            for item in folder.rglob("*")
        )
    )


def copy_pair(image_path: Path, label_dir: Path, output_images: Path, output_labels: Path):
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    target_image = output_images / image_path.name
    target_label = output_labels / f"{image_path.stem}.txt"
    source_label = label_dir / f"{image_path.stem}.txt"

    shutil.copy2(image_path, target_image)
    if source_label.is_file():
        shutil.copy2(source_label, target_label)
    else:
        target_label.write_text("", encoding="utf-8")


def write_dataset_yaml(output_dir: Path, dataset_root: Path, names: dict[int, str], layout: dict[str, str]):
    yaml_path = output_dir / "data.yaml"
    payload = {
        "path": str(dataset_root),
        "train": layout["train"],
        "val": layout["val"],
        "names": names,
    }
    if layout.get("test"):
        payload["test"] = layout["test"]

    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)

    return yaml_path


def prepare_existing_split(root: Path, name: str, names: dict[int, str]) -> Path:
    layouts = split_dirs(root)
    if "train" not in layouts or "val" not in layouts:
        raise HTTPException(
            status_code=400,
            detail="Dataset must contain train and val/valid image folders.",
        )

    output_dir = DATA_ROOT / "prepared" / clean_name(name, "dataset")
    staging_dir = output_dir.with_name(f".{output_dir.name}.prepare-{uuid.uuid4().hex}")
    staging_dir.mkdir(parents=True, exist_ok=True)

    train_path = layouts["train"]["images"].relative_to(root)
    val_path = layouts["val"]["images"].relative_to(root)
    test_path = layouts.get("test", {}).get("images")

    layout = {
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path.relative_to(root)) if test_path else "",
    }
    try:
        write_dataset_yaml(staging_dir, root, names, layout)
        replace_directory(staging_dir, output_dir)
        return output_dir / "data.yaml"
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def collect_source_images(root: Path) -> list[tuple[Path, Path]]:
    collected: list[tuple[Path, Path]] = []

    flat = flat_dirs(root)
    if flat:
        collected.extend((image, flat["labels"]) for image in image_files(flat["images"]))

    for layout in split_dirs(root).values():
        collected.extend((image, layout["labels"]) for image in image_files(layout["images"]))

    seen: set[Path] = set()
    unique: list[tuple[Path, Path]] = []
    for image, labels in collected:
        if image in seen:
            continue
        seen.add(image)
        unique.append((image, labels))

    return unique


def prepare_split_dataset(
    root: Path,
    name: str,
    names: dict[int, str],
    split: SplitConfig,
    progress_callback: Optional[DatasetProgressCallback] = None,
) -> Path:
    source_images = collect_source_images(root)
    if not source_images:
        raise HTTPException(status_code=400, detail="No images found in the dataset path.")

    train_ratio, val_ratio, test_ratio = validate_split(split)
    split_progress = None
    if progress_callback:
        stage_details = {
            "reading_labels": lambda current, total: f"Reading labels: {current} of {total} images",
            "calculating_targets": lambda current, total: "Calculating per-class split targets",
            "assigning": lambda current, total: f"Assigning images: {current} of {total}",
            "finalizing_split": lambda current, total: "Finalizing split assignments",
        }

        def split_progress(stage: str, current: int, total: int):
            progress_callback(
                stage,
                current,
                total,
                stage_details[stage](current, total),
            )
    groups, diagnostics = stratified_split(
        source_images,
        {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        set(names),
        seed=42,
        progress_callback=split_progress,
    )

    output_root = DATA_ROOT / "prepared" / clean_name(name, "dataset")
    staging_root = output_root.with_name(f".{output_root.name}.prepare-{uuid.uuid4().hex}")
    try:
        copied = 0
        copy_total = len(source_images)
        if progress_callback:
            progress_callback("copying", 0, copy_total, f"Copying 0 of {copy_total} images")
        for split_name, pairs in groups.items():
            for image_path, label_dir in pairs:
                copy_pair(
                    image_path,
                    label_dir,
                    staging_root / "images" / split_name,
                    staging_root / "labels" / split_name,
                )
                copied += 1
                if progress_callback:
                    progress_callback(
                        "copying",
                        copied,
                        copy_total,
                        f"Copying {copied} of {copy_total} images",
                    )

        write_dataset_yaml(
            staging_root,
            output_root,
            names,
            {"train": "images/train", "val": "images/val", "test": "images/test"},
        )
        (staging_root / SPLIT_METADATA_FILE).write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        replace_directory(staging_root, output_root)
        return output_root / "data.yaml"
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)


def prepare_dataset(
    source: Path,
    name: str,
    classes: list[str],
    split: SplitConfig,
    force_split: bool,
    progress_callback: Optional[DatasetProgressCallback] = None,
) -> Path:
    if source.is_file() and source.suffix.lower() in {".yaml", ".yml"}:
        return source

    root = find_dataset_root(source)
    names = resolve_dataset_classes(root, classes)

    if not force_split and "train" in split_dirs(root) and "val" in split_dirs(root):
        return prepare_existing_split(root, name, names)

    return prepare_split_dataset(root, name, names, split, progress_callback)


def prepare_roboflow_download(
    dataset_root: Path,
    name: str,
    classes: list[str],
    split: Optional[SplitConfig] = None,
    progress_callback: Optional[DatasetProgressCallback] = None,
) -> tuple[Path, str]:
    root = find_dataset_root(dataset_root)
    source_yaml = find_dataset_yaml(root)

    if source_yaml:
        try:
            payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Roboflow data.yaml could not be read: {exc}",
            ) from exc

        yaml_root = resolve_yaml_dataset_root(source_yaml, payload)
        train_path = split_image_folder(yaml_root, payload.get("train"))
        val_path = split_image_folder(yaml_root, payload.get("val"))
        test_value = payload.get("test")
        test_path = split_image_folder(yaml_root, test_value)
        test_is_valid = not test_value or (test_path and test_path.is_dir())
        if has_image_files(train_path) and has_image_files(val_path) and test_is_valid:
            return source_yaml, "valid"

    layouts = split_dirs(root)
    if (
        "train" in layouts
        and "val" in layouts
        and has_image_files(layouts["train"]["images"])
        and has_image_files(layouts["val"]["images"])
    ):
        names = resolve_dataset_classes(root, classes)
        return prepare_existing_split(root, name, names), "normalized"

    train_layout = layouts.get("train")
    if train_layout and has_image_files(train_layout["images"]):
        yaml_path = prepare_dataset(
            source=root,
            name=name,
            classes=classes,
            split=split or SplitConfig(),
            force_split=True,
            progress_callback=progress_callback,
        )
        return yaml_path, "rebuilt"

    if source_yaml:
        train_value = payload.get("train") or "<missing>"
        val_value = payload.get("val") or "<missing>"
        test_value = payload.get("test") or "<not configured>"
        raise HTTPException(
            status_code=400,
            detail=(
                "Roboflow dataset paths are invalid and no usable train/validation "
                f"folders were detected under {root}. data.yaml specifies "
                f"train={train_value!r}, val={val_value!r}, test={test_value!r}."
            ),
        )

    yaml_path = prepare_dataset(
        source=root,
        name=name,
        classes=classes,
        split=SplitConfig(),
        force_split=False,
    )
    return yaml_path, "valid"


def resolve_test_split_source(
    dataset_root: Path,
    payload: Optional[dict] = None,
) -> tuple[str, Optional[Path]]:
    if isinstance(payload, dict):
        for split_name in ("test", "val", "train"):
            images_path = split_image_folder(dataset_root, payload.get(split_name))
            if has_image_files(images_path):
                return split_name, images_path

    layouts = split_dirs(dataset_root)
    for split_name in ("test", "val", "train"):
        layout = layouts.get(split_name)
        if layout and has_image_files(layout["images"]):
            return split_name, layout["images"]

    flat = flat_dirs(dataset_root)
    if flat and has_image_files(flat["images"]):
        return "flat", flat["images"]

    return "", None


def build_test_dataset_yaml(
    dataset_root: Path,
    images_path: Path,
    classes: list[str],
    output_name: str,
) -> Path:
    dataset_root = dataset_root.resolve()
    images_path = images_path.resolve()
    labels_path = label_folder_for_images(dataset_root, images_path)
    if labels_path is None or not labels_path.exists():
        raise HTTPException(
            status_code=400,
            detail="The uploaded test dataset does not contain a matching labels folder.",
        )

    try:
        relative_images = images_path.relative_to(dataset_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="The detected test images are outside the dataset root.",
        ) from exc

    output_dir = DATA_ROOT / "test_prepared" / clean_name(output_name, "test_dataset")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "data.yaml"
    payload = {
        "path": str(dataset_root),
        "test": relative_images.as_posix(),
        "names": {index: name for index, name in enumerate(classes)},
    }
    yaml_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return yaml_path


def resolve_reference_classes(reference_dataset_yaml: str) -> list[str]:
    if not reference_dataset_yaml:
        return []
    reference_path, _, _ = prepared_dataset_yaml(reference_dataset_yaml)
    return read_yaml_class_names(reference_path)


def prepare_custom_test_dataset(
    source: Path,
    output_name: str,
    reference_dataset_yaml: str = "",
) -> tuple[Path, dict]:
    dataset_root = find_dataset_root(source)
    source_yaml = find_dataset_yaml(dataset_root)
    payload = None
    classes: list[str] = []
    yaml_root = dataset_root

    if source_yaml:
        try:
            payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read uploaded dataset YAML: {exc}",
            ) from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Uploaded dataset YAML must contain a mapping.")
        yaml_root = resolve_yaml_dataset_root(source_yaml, payload)
        classes = normalize_yaml_names(payload.get("names"))

    if not classes:
        classes = resolve_reference_classes(reference_dataset_yaml)
    if not classes:
        raise HTTPException(
            status_code=400,
            detail=(
                "No class names were found in the uploaded dataset. Include a data.yaml "
                "or prepare a dataset in the UI first so its class names can be reused."
            ),
        )

    selected_split, images_path = resolve_test_split_source(yaml_root, payload)
    if images_path is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "No usable labeled images were found in the uploaded test dataset. "
                "Include a YOLO images/labels layout or a data.yaml with a test split."
            ),
        )

    yaml_path = build_test_dataset_yaml(yaml_root, images_path, classes, output_name)
    return yaml_path, {
        "dataset_root": str(yaml_root),
        "source_split": selected_split,
        "classes": classes,
    }


def resolve_prepared_test_dataset_yaml(dataset_yaml: str) -> tuple[Path, dict]:
    yaml_path, dataset_root, portable_payload = prepared_dataset_yaml(dataset_yaml)
    if not portable_payload.get("test"):
        raise HTTPException(
            status_code=400,
            detail="The prepared dataset does not contain a test split to evaluate.",
        )
    return yaml_path, {
        "dataset_root": str(dataset_root),
        "source_split": "test",
        "classes": normalize_yaml_names(portable_payload.get("names")),
    }

__all__ = [
    "DatasetDependencies",
    "configure_datasets",
    "validate_classes",
    "normalize_yaml_names",
    "read_yaml_class_names",
    "detect_dataset_classes",
    "resolve_dataset_classes",
    "dataset_response",
    "resolve_yaml_dataset_root",
    "split_image_folder",
    "prepared_dataset_yaml",
    "build_dataset_archive",
    "build_annotated_dataset_archive",
    "cleanup_expired_dataset_downloads",
    "label_folder_for_images",
    "update_class_distribution",
    "read_split_metadata",
    "inspect_dataset_yaml",
    "validate_split",
    "find_dataset_yaml",
    "find_dataset_root",
    "split_dirs",
    "flat_dirs",
    "image_files",
    "has_image_files",
    "copy_pair",
    "write_dataset_yaml",
    "prepare_existing_split",
    "collect_source_images",
    "prepare_split_dataset",
    "prepare_dataset",
    "prepare_roboflow_download",
    "resolve_test_split_source",
    "build_test_dataset_yaml",
    "resolve_reference_classes",
    "prepare_custom_test_dataset",
    "resolve_prepared_test_dataset_yaml",
]
