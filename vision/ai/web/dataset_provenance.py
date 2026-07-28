"""Dataset provenance helpers shared by preparation and reporting."""

import math
import re
from pathlib import Path
from typing import Optional


SPLIT_NAMES = ("train", "val", "test")
ROBOFLOW_IMAGE_COUNT_RE = re.compile(r"The dataset includes\s+([\d,]+)\s+images", re.IGNORECASE)
ROBOFLOW_AUGMENTATION_VERSIONS_RE = re.compile(
    r"augmentation was applied to create\s+(\d+)\s+versions? of each source image",
    re.IGNORECASE,
)


def _positive_int(value) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def roboflow_augmentation_multiplier(dataset_root: Path, source_metadata: Optional[dict] = None) -> tuple[int | None, str]:
    """Read Roboflow's training-output multiplier from SDK or export metadata."""
    metadata = source_metadata or {}
    augmentation = metadata.get("augmentation") if isinstance(metadata, dict) else None
    if isinstance(augmentation, dict):
        image_settings = augmentation.get("image")
        if isinstance(image_settings, dict):
            versions = _positive_int(image_settings.get("versions"))
            if versions:
                return versions, "Roboflow API"
        versions = _positive_int(augmentation.get("versions"))
        if versions:
            return versions, "Roboflow API"

    readme_path = dataset_root / "README.roboflow.txt"
    try:
        readme = readme_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return (1, "Roboflow API") if isinstance(augmentation, dict) and not augmentation else (None, "")
    match = ROBOFLOW_AUGMENTATION_VERSIONS_RE.search(readme)
    if match:
        return int(match.group(1)), "README.roboflow.txt"
    if "No image augmentation techniques were applied" in readme:
        return 1, "README.roboflow.txt"
    if isinstance(augmentation, dict) and not augmentation:
        return 1, "Roboflow API"
    return None, ""


def roboflow_pre_augmentation_summary(
    dataset_root: Path,
    splits: dict,
    source_metadata: Optional[dict] = None,
) -> dict:
    """Reconstruct the source-image split before Roboflow expanded training."""
    readme_path = dataset_root / "README.roboflow.txt"
    if not readme_path.is_file() and not source_metadata:
        return {}

    multiplier, metadata_source = roboflow_augmentation_multiplier(dataset_root, source_metadata)
    if multiplier is None:
        return {}

    exported_counts = {
        split: int((splits.get(split) or {}).get("images") or 0)
        for split in SPLIT_NAMES
    }
    exported_train = exported_counts["train"]
    if exported_train <= 0:
        return {}

    source_train_float = exported_train / multiplier
    source_train = round(source_train_float)
    exact = math.isclose(source_train_float, source_train, abs_tol=1e-9)
    source_counts = {
        "train": source_train,
        "val": exported_counts["val"],
        "test": exported_counts["test"],
    }
    source_total = sum(source_counts.values())
    if source_total <= 0:
        return {}

    reported_exported_total = None
    try:
        readme = readme_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        readme = ""
    count_match = ROBOFLOW_IMAGE_COUNT_RE.search(readme)
    if count_match:
        reported_exported_total = int(count_match.group(1).replace(",", ""))

    note = (
        f"Roboflow generated {multiplier} total training outputs per source image; "
        "validation and test images were not multiplied."
    )
    if not exact:
        note += " The original training count is estimated because the exported count is not evenly divisible by the multiplier."

    return {
        "available": True,
        "source": "Roboflow",
        "metadata_source": metadata_source,
        "augmentation_multiplier": multiplier,
        "counts_are_estimated": not exact,
        "total_images": source_total,
        "splits": {
            split: {"images": count}
            for split, count in source_counts.items()
        },
        "split_ratios": {
            split: round(count / source_total * 100, 2)
            for split, count in source_counts.items()
        },
        "exported_total_images": sum(exported_counts.values()),
        "reported_exported_total": reported_exported_total,
        "note": note,
    }
