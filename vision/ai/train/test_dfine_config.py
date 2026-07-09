"""Tests for D-FINE web runner configuration."""

from argparse import Namespace

import yaml

from vision.ai.train.train_dfine import write_dfine_config


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
