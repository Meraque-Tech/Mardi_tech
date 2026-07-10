"""Behavior tests for multi-score Magic Button report overlays."""

import pytest


def load_web_app():
    pytest.importorskip("fastapi")
    from vision.ai.web import app as web_app

    return web_app


def sample_metrics():
    return {
        "available": True,
        "precision": 0.58,
        "recall": 0.52,
        "map50": 0.64,
        "map50_95": 0.42,
        "macro_f1": 0.5483,
        "weighted_f1": 0.5500,
        "per_class": [
            {"class_name": "crop", "instances": 20, "precision": 0.60, "recall": 0.50, "f1": 0.5455, "map50": 0.66, "map50_95": 0.44},
            {"class_name": "weed", "instances": 10, "precision": 0.55, "recall": 0.55, "f1": 0.55, "map50": 0.62, "map50_95": 0.40},
            {"class_name": "soil", "instances": 5, "precision": 0.59, "recall": 0.51, "f1": 0.5471, "map50": 0.64, "map50_95": 0.42},
        ],
    }


def test_parse_magic_metrics_payload_accepts_multiple_and_legacy_adjustments():
    web_app = load_web_app()

    parsed = web_app.parse_magic_metrics_payload({
        "project": "runs/rfdetr",
        "name": "train-a",
        "adjustments": [
            {"scope": "overall", "metric_key": "map50_95", "target": 0.70},
            {"scope": "per_class", "class_name": "weed", "metric_key": "recall", "target": 0.80},
        ],
    })
    assert len(parsed["adjustments"]) == 2
    assert parsed["adjustments"][1]["class_name"] == "weed"

    legacy = web_app.parse_magic_metrics_payload({"metric_key": "precision", "target": 0.75})
    assert legacy["adjustments"] == [{
        "scope": "overall",
        "metric_key": "precision",
        "target": 0.75,
        "class_name": "",
    }]


def test_multi_magic_overlay_preserves_independent_overall_targets_deterministically():
    web_app = load_web_app()
    adjustments = [
        {"scope": "overall", "metric_key": "map50_95", "target": 0.70},
        {"scope": "overall", "metric_key": "recall", "target": 0.75},
    ]

    first = web_app.build_magic_metrics_overlay(sample_metrics(), adjustments)
    second = web_app.build_magic_metrics_overlay(sample_metrics(), adjustments)

    assert first["metrics"]["map50_95"] == 0.70
    assert first["metrics"]["recall"] == 0.75
    assert first["metrics"]["per_class"] == second["metrics"]["per_class"]
    assert first["adjustments"] == adjustments
    assert len(first["metrics"]["magic_adjustments"]) == 2


def test_overall_rebalancing_keeps_explicit_per_class_target_fixed():
    web_app = load_web_app()
    overlay = web_app.build_magic_metrics_overlay(sample_metrics(), [
        {"scope": "per_class", "class_name": "weed", "metric_key": "recall", "target": 0.90},
        {"scope": "overall", "metric_key": "recall", "target": 0.70},
    ])

    rows = {row["class_name"]: row for row in overlay["metrics"]["per_class"]}
    assert rows["weed"]["recall"] == 0.90
    assert overlay["metrics"]["recall"] == 0.70
    assert sum(row["recall"] for row in rows.values()) / len(rows) == pytest.approx(0.70, abs=0.0001)


def test_multiple_per_class_precision_and_recall_targets_recalculate_f1():
    web_app = load_web_app()
    overlay = web_app.build_magic_metrics_overlay(sample_metrics(), [
        {"scope": "per_class", "class_name": "crop", "metric_key": "precision", "target": 0.80},
        {"scope": "per_class", "class_name": "crop", "metric_key": "recall", "target": 0.60},
    ])

    crop = next(row for row in overlay["metrics"]["per_class"] if row["class_name"] == "crop")
    assert crop["precision"] == 0.80
    assert crop["recall"] == 0.60
    assert crop["f1"] == pytest.approx(0.6857, abs=0.0001)


def test_magic_adjustments_reject_duplicates_and_conflicting_scores():
    web_app = load_web_app()

    with pytest.raises(web_app.HTTPException) as duplicate_error:
        web_app.parse_magic_metrics_payload({"adjustments": [
            {"scope": "overall", "metric_key": "recall", "target": 0.70},
            {"scope": "overall", "metric_key": "recall", "target": 0.80},
        ]})
    assert "Duplicate" in duplicate_error.value.detail

    with pytest.raises(web_app.HTTPException) as ap_error:
        web_app.parse_magic_metrics_payload({"adjustments": [
            {"scope": "overall", "metric_key": "map50", "target": 0.60},
            {"scope": "overall", "metric_key": "map50_95", "target": 0.70},
        ]})
    assert "cannot be lower" in ap_error.value.detail

    with pytest.raises(web_app.HTTPException) as f1_error:
        web_app.parse_magic_metrics_payload({"adjustments": [
            {"scope": "per_class", "class_name": "weed", "metric_key": "f1", "target": 0.70},
            {"scope": "per_class", "class_name": "weed", "metric_key": "precision", "target": 0.80},
        ]})
    assert "either F1 or Precision/Recall" in f1_error.value.detail
