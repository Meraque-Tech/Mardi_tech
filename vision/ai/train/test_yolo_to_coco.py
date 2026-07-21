"""Tests for YOLO-to-COCO dataset conversion."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import yaml

from vision.ai.train.yolo_to_coco import convert_yolo_to_coco


def write_png(path: Path, width: int = 100, height: int = 50):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b"".join(b"\x00" + b"\xff\xff\xff" * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def write_dataset_yaml(dataset: Path, names):
    (dataset / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(dataset),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_convert_yolo_to_coco_writes_coco2017_layout(tmp_path):
    dataset = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        write_png(dataset / "images" / split / f"{split}.png")
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split / f"{split}.txt").write_text(
            "0 0.5 0.5 0.2 0.4\n",
            encoding="utf-8",
        )
    write_dataset_yaml(dataset, {0: "pineapple"})

    result = convert_yolo_to_coco(dataset / "data.yaml", tmp_path / "coco", "pineapple")

    assert (result.output_dir / "train2017" / "train.png").is_file()
    assert (result.output_dir / "val2017" / "val.png").is_file()
    assert (result.output_dir / "test2017" / "test.png").is_file()
    train = json.loads((result.output_dir / "annotations" / "instances_train2017.json").read_text(encoding="utf-8"))
    assert train["categories"] == [{"id": 0, "name": "pineapple", "supercategory": "object"}]
    assert train["images"][0]["width"] == 100
    assert train["images"][0]["height"] == 50
    assert train["annotations"][0]["category_id"] == 0
    assert train["annotations"][0]["bbox"] == [40.0, 15.0, 20.0, 20.0]
    assert result.summary["splits"]["train"]["annotation_count"] == 1


def test_convert_yolo_to_coco_reports_empty_and_malformed_labels(tmp_path):
    dataset = tmp_path / "dataset"
    write_png(dataset / "images" / "train" / "good.png", width=20, height=10)
    write_png(dataset / "images" / "train" / "empty.png", width=20, height=10)
    write_png(dataset / "images" / "val" / "val.png", width=20, height=10)
    (dataset / "labels" / "train").mkdir(parents=True)
    (dataset / "labels" / "train" / "good.txt").write_text(
        "1 0.5 0.5 0.5 0.5\n"
        "bad row\n"
        "9 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )
    (dataset / "labels" / "val").mkdir(parents=True)
    (dataset / "labels" / "val" / "val.txt").write_text("", encoding="utf-8")
    write_dataset_yaml(dataset, ["flat", "ok"])

    result = convert_yolo_to_coco(dataset, tmp_path / "coco", "audit")

    train = json.loads((result.output_dir / "annotations" / "instances_train2017.json").read_text(encoding="utf-8"))
    assert len(train["images"]) == 2
    assert len(train["annotations"]) == 1
    assert result.summary["splits"]["train"]["missing_label_count"] == 1
    assert result.summary["splits"]["train"]["malformed_label_count"] == 1
    assert result.summary["splits"]["train"]["unknown_class_count"] == 1
