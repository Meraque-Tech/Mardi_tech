"""Tests for effective augmentation status in generated training reports."""

from vision.ai.web.report_generator import _short_config_rows


def rows_as_dict(rows):
    return {label: value for label, value in rows[1:]}


def test_report_states_when_runtime_augmentation_is_off(tmp_path):
    rows = _short_config_rows({
        "family": "ultralytics",
        "hyperparameters": {
            "family": "ultralytics",
            "augmentation_enabled": False,
            "mosaic": 1.0,
            "translate": 0.2,
            "scale": 0.4,
        },
    }, tmp_path)

    assert rows_as_dict(rows)["Runtime augmentation"] == "Off — all transforms disabled"


def test_report_lists_only_active_values_when_runtime_augmentation_is_on(tmp_path):
    rows = _short_config_rows({
        "family": "ultralytics",
        "hyperparameters": {
            "family": "ultralytics",
            "augmentation_enabled": True,
            "mosaic": 0.5,
            "translate": 0,
            "scale": 0.25,
            "auto_augment": None,
        },
    }, tmp_path)

    summary = rows_as_dict(rows)["Runtime augmentation"]
    assert summary == "On — mosaic=0.5, scale=0.25"

