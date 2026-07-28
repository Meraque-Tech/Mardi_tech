"""Tests for Roboflow pre-augmentation dataset provenance."""

from pathlib import Path

import yaml
import pytest

from vision.ai.web.dataset_provenance import roboflow_pre_augmentation_summary
from vision.ai.web.report_generator import _add_dataset


def exported_splits(train: int, val: int, test: int) -> dict:
    return {
        "train": {"images": train, "missing_labels": 0},
        "val": {"images": val, "missing_labels": 0},
        "test": {"images": test, "missing_labels": 0},
    }


def test_reconstructs_original_roboflow_split(tmp_path: Path):
    (tmp_path / "README.roboflow.txt").write_text(
        """The dataset includes 52174 images.
The following augmentation was applied to create 5 versions of each source image:
* Horizontal flip
""",
        encoding="utf-8",
    )

    result = roboflow_pre_augmentation_summary(
        tmp_path,
        exported_splits(49690, 1242, 1242),
    )

    assert result["available"] is True
    assert result["counts_are_estimated"] is False
    assert result["augmentation_multiplier"] == 5
    assert result["total_images"] == 12422
    assert result["splits"] == {
        "train": {"images": 9938},
        "val": {"images": 1242},
        "test": {"images": 1242},
    }
    assert result["split_ratios"] == {"train": 80.0, "val": 10.0, "test": 10.0}
    assert result["reported_exported_total"] == 52174


def test_marks_non_divisible_training_count_as_estimated(tmp_path: Path):
    (tmp_path / "README.roboflow.txt").write_text(
        "The following augmentation was applied to create 3 versions of each source image:\n",
        encoding="utf-8",
    )

    result = roboflow_pre_augmentation_summary(
        tmp_path,
        exported_splits(100, 10, 10),
    )

    assert result["counts_are_estimated"] is True
    assert result["splits"]["train"]["images"] == 33


def test_reads_multiplier_from_roboflow_sdk_metadata(tmp_path: Path):
    result = roboflow_pre_augmentation_summary(
        tmp_path,
        exported_splits(500, 20, 20),
        {"augmentation": {"image": {"versions": 5}}},
    )

    assert result["metadata_source"] == "Roboflow API"
    assert result["augmentation_multiplier"] == 5
    assert result["splits"]["train"]["images"] == 100


def test_returns_no_provenance_without_roboflow_metadata(tmp_path: Path):
    assert roboflow_pre_augmentation_summary(
        tmp_path,
        exported_splits(80, 10, 10),
    ) == {}


def test_rebuilds_roboflow_export_when_yaml_paths_are_invalid_but_train_exists(
    tmp_path: Path,
    monkeypatch,
):
    pytest.importorskip("fastapi")
    from vision.ai.web import app as web_app

    monkeypatch.setattr(web_app, "DATA_ROOT", tmp_path / "web_datasets")
    source = tmp_path / "roboflow" / "pineapple_ai_system"
    images = source / "train" / "images"
    labels = source / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (source / "data.yaml").write_text(
        "\n".join([
            "train: ../train/images",
            "val: ../valid/images",
            "test: ../test/images",
            "names: ['flat_plant', 'no_plant', 'ok_plant']",
        ]),
        encoding="utf-8",
    )
    for index in range(10):
        stem = f"image_{index:03d}"
        (images / f"{stem}.jpg").write_bytes(b"not-a-real-jpeg")
        (labels / f"{stem}.txt").write_text(
            f"{index % 3} 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )

    yaml_path, action = web_app.prepare_roboflow_download(
        source,
        "rebuilt_pineapple",
        [],
        web_app.SplitConfig(train=60, val=20, test=20),
    )

    assert action == "rebuilt"
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    output_root = yaml_path.parent
    assert payload["train"] == "images/train"
    assert payload["val"] == "images/val"
    assert payload["test"] == "images/test"
    assert len(list((output_root / "images" / "train").glob("*.jpg"))) == 6
    assert len(list((output_root / "images" / "val").glob("*.jpg"))) == 2
    assert len(list((output_root / "images" / "test").glob("*.jpg"))) == 2


class RecordingBuilder:
    mm = 1

    def __init__(self):
        self.headings = []
        self.tables = []
        self.paragraphs = []

    def heading(self, value, level=2):
        self.headings.append((value, level))

    def table(self, rows, widths=None, header=True):
        self.tables.append(rows)

    def paragraph(self, value, style="BodyText"):
        self.paragraphs.append((value, style))


def test_report_labels_original_and_exported_counts(tmp_path: Path):
    builder = RecordingBuilder()
    summary = {
        "total_images": 52174,
        "class_count": 3,
        "split_strategy": "existing",
        "split_seed": None,
        "split_ratios": {"train": 95.24, "val": 2.38, "test": 2.38},
        "splits": exported_splits(49690, 1242, 1242),
        "pre_augmentation": {
            "available": True,
            "total_images": 12422,
            "augmentation_multiplier": 5,
            "counts_are_estimated": False,
            "metadata_source": "README.roboflow.txt",
            "splits": exported_splits(9938, 1242, 1242),
            "split_ratios": {"train": 80.0, "val": 10.0, "test": 10.0},
            "note": "Roboflow generated five training outputs.",
        },
    }

    _add_dataset(builder, tmp_path, {"dataset_summary": summary, "dataset_yaml": ""})

    assert ("Original Dataset Before Augmentation", 3) in builder.headings
    assert ("Exported Dataset After Augmentation", 3) in builder.headings
    assert any(row[0] == "Original source images" and row[1] == 12422 for table in builder.tables for row in table)
    assert any(row[0] == "Train" and row[1] == 9938 for table in builder.tables for row in table)
    assert any(row[0] == "Train" and row[1] == 49690 for table in builder.tables for row in table)
