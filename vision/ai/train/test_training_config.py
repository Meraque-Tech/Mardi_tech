"""Tests for the YOLO training augmentation configuration."""

from vision.ai.train.train_yolov8 import TRAINING_AUGMENTATIONS


def test_requested_ultralytics_augmentations_are_enabled():
    assert TRAINING_AUGMENTATIONS == {
        "mosaic": 1.0,
        "close_mosaic": 10,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "bgr": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "auto_augment": None,
        "erasing": 0.0,
    }
