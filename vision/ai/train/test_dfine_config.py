"""Tests for D-FINE web runner configuration."""

from argparse import Namespace
import csv

import yaml

from vision.ai.train.train_dfine import (
    dfine_progress_marker,
    parse_dfine_coco_ap_line,
    parse_dfine_progress_line,
    upsert_live_result,
    write_dfine_config,
    write_results_csv,
)


def test_dfine_web_config_converts_pil_images_to_tensors(tmp_path):
    dfine_root = tmp_path / "D-FINE"
    config_path = dfine_root / "configs" / "dfine" / "custom" / "dfine_hgnetv2_n_custom.yml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("output_dir: ./output\n", encoding="utf-8")
    coco_dir = tmp_path / "coco"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = Namespace(model="dfine-n", epochs=3, batch=4, workers=2, imgsz=640, lr0=0.0004, weight_decay=0.0001)

    generated = write_dfine_config(run_dir, dfine_root, coco_dir, 3, args)

    payload = yaml.safe_load(generated.read_text(encoding="utf-8"))
    train_ops = payload["train_dataloader"]["dataset"]["transforms"]["ops"]
    val_ops = payload["val_dataloader"]["dataset"]["transforms"]["ops"]
    assert payload["train_dataloader"]["dataset"]["transforms"]["type"] == "Compose"
    assert payload["val_dataloader"]["dataset"]["transforms"]["type"] == "Compose"
    assert {"type": "ConvertPILImage", "dtype": "float32", "scale": True} in train_ops
    assert {"type": "ConvertPILImage", "dtype": "float32", "scale": True} in val_ops
    assert {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True} in train_ops
    assert payload["train_dataloader"]["collate_fn"]["type"] == "BatchImageCollateFunction"


def test_dfine_progress_line_is_normalized_for_web_metrics():
    line = (
        "Epoch: [0/3]  [ 200/1351]  eta: 0:03:15  lr: 0.000162  "
        "loss: 26.7643 (29.0367)  loss_vfl: 0.8779 (0.6572)"
    )

    parsed = parse_dfine_progress_line(line)

    assert parsed["epoch"] == 1
    assert parsed["raw_epoch"] == 0
    assert parsed["step"] == 200
    assert parsed["total_steps"] == 1351
    assert parsed["eta"] == "0:03:15"
    assert parsed["lr"] == 0.000162
    assert parsed["train/loss"] == 29.0367
    assert parsed["train/loss_avg"] == 29.0367
    assert parsed["train/loss_step"] == 26.7643

    marker = dfine_progress_marker(parsed, 3)
    assert marker == (
        "WEB_TRAINING_PROGRESS epoch=1 total=3 step=200 steps=1351 "
        "loss=26.7643 loss_avg=29.0367 lr=0.000162 eta=0:03:15"
    )


def test_dfine_live_results_are_preserved_by_finalization(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    upsert_live_result(run_dir, 1, {"train/loss": 29.0367})
    upsert_live_result(
        run_dir,
        1,
        parse_dfine_coco_ap_line(
            "Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.790"
        ),
    )
    source = write_results_csv(run_dir, 3)

    assert source == "results_csv"
    with (run_dir / "results.csv").open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows == [
        {
            "epoch": "1",
            "train/loss": "29.0367",
            "val/loss": "",
            "metrics/precision(B)": "",
            "metrics/recall(B)": "",
            "metrics/mAP50(B)": "0.79",
            "metrics/mAP50-95(B)": "",
        }
    ]
