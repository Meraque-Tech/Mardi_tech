from types import SimpleNamespace
import unittest

from vision.ai.train.yolo_metrics import build_yolo_metric_families


def metric_component(*, p, r, f1, ap50, ap, map50=None, map95=None):
    return SimpleNamespace(
        p=p,
        r=r,
        f1=f1,
        ap50=ap50,
        ap=ap,
        mp=sum(p) / len(p),
        mr=sum(r) / len(r),
        map50=sum(ap50) / len(ap50) if map50 is None else map50,
        map=sum(ap) / len(ap) if map95 is None else map95,
        ap_class_index=list(range(len(p))),
    )


def segment_metrics():
    box = metric_component(
        p=[0.8, 0.6], r=[0.7, 0.5], f1=[0.7467, 0.5455],
        ap50=[0.9, 0.7], ap=[0.6, 0.4],
    )
    mask = metric_component(
        p=[0.75, 0.55], r=[0.65, 0.45], f1=[0.6964, 0.495],
        ap50=[0.8, 0.6], ap=[0.5, 0.3],
    )
    rows = [
        {"Class": "palm-tree", "Images": 5, "Instances": 10},
        {"Class": "leaf", "Images": 3, "Instances": 4},
    ]
    return SimpleNamespace(
        box=box,
        seg=mask,
        names={0: "palm-tree", 1: "leaf"},
        nt_per_class=[10, 4],
        nt_per_image=[5, 3],
        summary=lambda: rows,
    )


def test_segmentation_returns_distinct_box_and_mask_tables_with_mask_primary():
    payload = build_yolo_metric_families(segment_metrics(), "segment")

    assert payload["primary_metric_type"] == "mask"
    assert payload["per_class"] == payload["per_class_mask"]
    assert payload["per_class_mask"][0]["map50_95"] == 0.5
    assert payload["per_class_box"][0]["map50_95"] == 0.6
    assert payload["overall"]["map50_95"] == 0.4
    assert payload["overall_by_type"]["box"]["map50_95"] == 0.5
    assert payload["metric_warnings"] == []


def test_detection_returns_box_metrics_as_primary():
    metrics = segment_metrics()
    metrics.seg = None

    payload = build_yolo_metric_families(metrics, "detect")

    assert payload["primary_metric_type"] == "box"
    assert payload["per_class"] == payload["per_class_box"]
    assert payload["per_class_mask"] == []


def test_missing_segmentation_metrics_never_substitute_box_values():
    metrics = segment_metrics()
    metrics.seg = None

    payload = build_yolo_metric_families(metrics, "segment")

    assert payload["primary_metric_type"] == "mask"
    assert payload["per_class"] == []
    assert payload["per_class_mask"] == []
    assert payload["per_class_box"]
    assert any("Mask metrics are unavailable" in warning for warning in payload["metric_warnings"])


def test_inconsistent_per_class_average_invalidates_only_that_family():
    metrics = segment_metrics()
    metrics.seg.map = 0.9

    payload = build_yolo_metric_families(metrics, "segment")

    assert payload["metric_families"]["mask"]["available"] is False
    assert payload["per_class_mask"] == []
    assert payload["metric_families"]["box"]["available"] is True
    assert any("does not match overall" in warning for warning in payload["metric_warnings"])


def test_box_values_matching_reported_regression_cannot_become_mask_values():
    metrics = segment_metrics()
    metrics.box.ap50 = [0.7145, 0.7145]
    metrics.box.ap = [0.4364, 0.4364]
    metrics.box.map50 = 0.7145
    metrics.box.map = 0.4364
    metrics.seg.ap50 = [0.7, 0.7]
    metrics.seg.ap = [0.3984, 0.3984]
    metrics.seg.map50 = 0.7
    metrics.seg.map = 0.3984

    payload = build_yolo_metric_families(metrics, "segment")

    assert payload["overall"]["map50_95"] == 0.3984
    assert payload["per_class"][0]["map50_95"] == 0.3984
    assert payload["overall_by_type"]["box"]["map50_95"] == 0.4364


def load_tests(_loader, _tests, _pattern):
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite
