"""Tests for the YOLO training augmentation configuration."""

from argparse import Namespace

from vision.ai.train.train_yolov8 import (
    DISABLED_TRAINING_AUGMENTATIONS,
    TRAINING_AUGMENTATIONS,
    TRAINING_CONFIG,
    get_optimizer_train_kwargs,
    get_training_augmentations,
    get_training_config,
    should_disable_ultralytics_albumentations,
    training_augmentation_summary,
)


def test_default_optimizer_uses_ultralytics_auto_selection():
    assert TRAINING_CONFIG["optimizer"] == "auto"


def test_augmentation_is_disabled_by_default():
    assert TRAINING_CONFIG["augmentation_enabled"] is False
    assert get_training_augmentations(TRAINING_CONFIG) == DISABLED_TRAINING_AUGMENTATIONS
    assert training_augmentation_summary(TRAINING_CONFIG).startswith("Augmentation: Off")


def test_auto_optimizer_does_not_override_ultralytics_tuning():
    config = TRAINING_CONFIG | {
        "optimizer": "auto",
        "lr0": 0.123,
        "lrf": 0.456,
        "weight_decay": 0.789,
    }

    assert get_optimizer_train_kwargs(config) == {"optimizer": "auto"}


def test_manual_optimizer_keeps_explicit_tuning():
    config = TRAINING_CONFIG | {
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0001,
    }

    assert get_optimizer_train_kwargs(config) == {
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0001,
    }


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


def test_cli_ultralytics_augmentations_override_defaults():
    args = Namespace(
        data=None,
        model=None,
        epochs=None,
        imgsz=None,
        batch=None,
        patience=None,
        save_period=None,
        device=None,
        workers=None,
        optimizer=None,
        lr0=None,
        lrf=None,
        weight_decay=None,
        cos_lr=None,
        warmup_epochs=None,
        freeze=None,
        pretrained=None,
        activation=None,
        exist_ok=None,
        seed=None,
        project=None,
        name=None,
        resume=None,
        augmentation_enabled=True,
        disable_ultralytics_albumentations=False,
        mosaic=0.25,
        close_mosaic=0,
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.2,
        scale=0.25,
        shear=1.0,
        perspective=0.001,
        flipud=0.1,
        fliplr=0.75,
        bgr=0.2,
        mixup=0.15,
        cutmix=0.05,
        copy_paste=0.1,
        auto_augment="randaugment",
        erasing=0.2,
    )

    config = get_training_config(args)
    augmentations = get_training_augmentations(config)

    assert config["disable_ultralytics_albumentations"] is False
    assert augmentations == {
        "mosaic": 0.25,
        "close_mosaic": 0,
        "hsv_h": 0.02,
        "hsv_s": 0.6,
        "hsv_v": 0.3,
        "degrees": 5.0,
        "translate": 0.2,
        "scale": 0.25,
        "shear": 1.0,
        "perspective": 0.001,
        "flipud": 0.1,
        "fliplr": 0.75,
        "bgr": 0.2,
        "mixup": 0.15,
        "cutmix": 0.05,
        "copy_paste": 0.1,
        "auto_augment": "randaugment",
        "erasing": 0.2,
    }
    assert training_augmentation_summary(config).startswith("Augmentation: On")


def test_disabled_augmentation_overrides_configured_values():
    config = TRAINING_CONFIG | {
        "augmentation_enabled": False,
        "disable_ultralytics_albumentations": False,
        "mosaic": 1.0,
        "hsv_h": 0.02,
        "translate": 0.2,
        "scale": 0.4,
        "fliplr": 0.8,
        "mixup": 0.3,
        "auto_augment": "randaugment",
        "erasing": 0.2,
    }

    assert get_training_augmentations(config) == {
        "mosaic": 0,
        "close_mosaic": 0,
        "hsv_h": 0,
        "hsv_s": 0,
        "hsv_v": 0,
        "degrees": 0,
        "translate": 0,
        "scale": 0,
        "shear": 0,
        "perspective": 0,
        "flipud": 0,
        "fliplr": 0,
        "bgr": 0,
        "mixup": 0,
        "cutmix": 0,
        "copy_paste": 0,
        "auto_augment": None,
        "erasing": 0,
    }
    assert should_disable_ultralytics_albumentations(config) is True


def test_enabled_augmentation_respects_optional_albumentations_setting():
    enabled = TRAINING_CONFIG | {
        "augmentation_enabled": True,
        "disable_ultralytics_albumentations": False,
    }

    assert should_disable_ultralytics_albumentations(enabled) is False
    enabled["disable_ultralytics_albumentations"] = True
    assert should_disable_ultralytics_albumentations(enabled) is True
