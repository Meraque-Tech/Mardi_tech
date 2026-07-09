"""Tests for the RF-DETR dataset layout adapter."""

from pathlib import Path

import yaml

from vision.ai.train.train_rfdetr import (
    audit_rfdetr_dataset,
    parse_rfdetr_log_metrics,
    prepare_rfdetr_dataset,
    write_results_csv,
    write_web_metrics,
)


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


def test_rfdetr_dataset_audit_counts_each_split_class(tmp_path):
    dataset = tmp_path / "dataset"
    for split in ("train", "val"):
        write_image(dataset / "images" / split / f"{split}.jpg")
        write_label(dataset / "labels" / split / f"{split}.txt")
    (dataset / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(dataset),
                "train": "images/train",
                "val": "images/val",
                "names": ["pineapple", "missing_class"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    adapter = prepare_rfdetr_dataset(dataset)
    audit = audit_rfdetr_dataset(adapter, ["pineapple", "missing_class"])

    train_rows = audit["splits"]["train"]["classes"]
    assert train_rows[0]["instances"] == 1
    assert train_rows[0]["images"] == 1
    assert train_rows[1]["instances"] == 0
    assert audit["splits"]["train"]["missing_class_ids"] == [1]


def test_rfdetr_log_metrics_are_written_to_web_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log_path = run_dir / "rfdetr_training.log"
    log_path.write_text(
        """
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           mAP            ┃  mAR   ┃         F1 sweep         ┃
┡━━━━━━━━┯━━━━━━━━┯━━━━━━━━╇━━━━━━━━╇━━━━━━━━┯━━━━━━━━┯━━━━━━━━┩
│ 50:95  │   50   │   75   │  @500  │   F1   │  Prec  │ Recall │
├────────┼────────┼────────┼────────┼────────┼────────┼────────┤
│ 0.0283 │ 0.0821 │ 0.0041 │ 0.2421 │ 0.1566 │ 0.0905 │ 0.5789 │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┘
Val (Epoch 1/3) — Per-class Metrics
┏━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
┃ Class    ┃ AP 50:95 ┃     AR ┃     F1 ┃ Precision ┃ Recall ┃
┡━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━┩
│ ok_plant │   0.0283 │ 0.2421 │ 0.1566 │    0.0905 │ 0.5789 │
└──────────┴──────────┴────────┴────────┴───────────┴────────┘
Validation: |          | 0/? [00:00<?, ?it/s]
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           mAP            ┃  mAR   ┃         F1 sweep         ┃
┡━━━━━━━━┯━━━━━━━━┯━━━━━━━━╇━━━━━━━━╇━━━━━━━━┯━━━━━━━━┯━━━━━━━━┩
│ 50:95  │   50   │   75   │  @500  │   F1   │  Prec  │ Recall │
├────────┼────────┼────────┼────────┼────────┼────────┼────────┤
│ 0.3095 │ 0.7122 │ 0.2051 │ 0.5231 │ 0.6733 │ 0.7035 │ 0.6802 │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┘
Val (Epoch 1/3) — Per-class Metrics
┏━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
┃ Class      ┃ AP 50:95 ┃     AR ┃     F1 ┃ Precision ┃ Recall ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━┩
│ flat_plant │   0.3820 │ 0.5622 │ 0.7511 │    0.6498 │ 0.8900 │
│ ok_plant   │   0.3623 │ 0.5454 │ 0.7967 │    0.8160 │ 0.7783 │
└────────────┴──────────┴────────┴────────┴───────────┴────────┘
""",
        encoding="utf-8",
    )
    dataset_audit = {
        "splits": {
            "valid": {
                "classes": [
                    {"class_id": 0, "class_name": "flat_plant", "images": 2, "instances": 3},
                    {"class_id": 1, "class_name": "ok_plant", "images": 4, "instances": 5},
                ]
            }
        }
    }

    parsed = parse_rfdetr_log_metrics(log_path)
    assert len(parsed["overall"]) == 2
    assert parsed["overall"][0]["map50"] == 0.0821
    assert parsed["overall"][-1]["map50"] == 0.7122
    assert parsed["per_class"][0]["class_name"] == "ok_plant"

    (run_dir / "results.csv").write_text(
        "epoch,train/loss,val/loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "3,,,,,,\n",
        encoding="utf-8",
    )
    lightning_dir = run_dir / "lightning_logs" / "version_0"
    lightning_dir.mkdir(parents=True)
    (lightning_dir / "metrics.csv").write_text(
        "epoch,train/loss_epoch,val/loss_epoch\n"
        "1,2.5000,2.2500\n",
        encoding="utf-8",
    )
    source = write_results_csv(run_dir, 3, log_path)
    write_web_metrics(run_dir, "rfdetr-nano", ["flat_plant", "ok_plant"], log_path, dataset_audit, False)

    assert source == "rfdetr_log"
    rows = (run_dir / "results.csv").read_text(encoding="utf-8")
    assert "metrics/mAP50(B)" in rows
    assert "0.0821" in rows
    assert "0.7122" in rows
    assert "2.5" in rows
    assert "2.25" in rows
    payload = yaml.safe_load((run_dir / "web_metrics.json").read_text(encoding="utf-8"))
    assert payload["backend"] == "rfdetr"
    assert payload["per_class"][0]["class_name"] == "flat_plant"
    assert payload["per_class"][0]["f1"] == 0.7511
    assert payload["per_class"][1]["class_name"] == "ok_plant"
    assert payload["per_class"][1]["f1"] == 0.7967
    assert payload["macro_f1"] == 0.7739
    assert payload["roc_auc"]["mode"] == "not_available"


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
