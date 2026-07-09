"""Tests for the RF-DETR dataset layout adapter."""

from pathlib import Path

import yaml

from vision.ai.train.train_rfdetr import prepare_rfdetr_dataset


def write_label(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")


def write_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-image")


def test_prepare_rfdetr_dataset_adapts_ultralytics_yolo_layout(tmp_path):
    dataset = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        write_image(dataset / "images" / split / f"{split}.jpg")
        write_label(dataset / "labels" / split / f"{split}.txt")
    (dataset / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(dataset),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "pineapple"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    adapter = prepare_rfdetr_dataset(dataset)

    assert (adapter / "data.yaml").is_file()
    assert (adapter / "train" / "images").is_dir()
    assert (adapter / "train" / "labels").is_dir()
    assert (adapter / "valid" / "images").is_dir()
    assert (adapter / "valid" / "labels").is_dir()
    assert (adapter / "test" / "images").is_dir()
    assert (adapter / "test" / "labels").is_dir()

    payload = yaml.safe_load((adapter / "data.yaml").read_text(encoding="utf-8"))
    assert payload["path"] == "."
    assert payload["train"] == "train/images"
    assert payload["val"] == "valid/images"
    assert payload["test"] == "test/images"
    assert payload["names"] == {0: "pineapple"}


def test_prepare_rfdetr_dataset_keeps_roboflow_style_layout(tmp_path):
    dataset = tmp_path / "dataset"
    for split in ("train", "valid"):
        write_image(dataset / split / "images" / f"{split}.jpg")
        write_label(dataset / split / "labels" / f"{split}.txt")
    (dataset / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(dataset),
                "train": "train/images",
                "val": "valid/images",
                "names": ["pineapple"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    adapter = prepare_rfdetr_dataset(dataset)

    assert (adapter / "train" / "images").is_dir()
    assert (adapter / "train" / "labels").is_dir()
    assert (adapter / "valid" / "images").is_dir()
    assert (adapter / "valid" / "labels").is_dir()
