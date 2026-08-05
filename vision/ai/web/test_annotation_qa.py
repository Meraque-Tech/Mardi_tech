"""Regression tests for SAM prompt mapping and box-tolerance decisions."""

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import unittest

import numpy as np



def load_qa_helpers():
    source_path = Path(__file__).with_name("app.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {
        "sam_masks_for_image",
        "bbox_edge_differences",
        "annotation_qa_difference_band",
        "issue_is_safe_sam_replacement",
        "annotation_qa_thresholds",
        "annotation_qa_summary",
    }
    module = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
        type_ignores=[],
    )
    namespace = {
        "Optional": Optional,
        "Path": Path,
        "datetime": datetime,
        "MYT": timezone(timedelta(hours=8)),
        "ANNOTATION_QA_REPORT_VERSION": 3,
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


QA_HELPERS = load_qa_helpers()


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeBoxes:
    def __init__(self, indices, confidences):
        self.cls = [Scalar(value) for value in indices]
        self.conf = [Scalar(value) for value in confidences]


class FakeMasks:
    def __init__(self, data):
        self.data = data


class FakeResult:
    def __init__(self, data, indices, confidences):
        self.masks = FakeMasks(data)
        self.boxes = FakeBoxes(indices, confidences)


class FakeModel:
    def __init__(self, result):
        self.result = result
        self.kwargs = None

    def predict(self, **kwargs):
        self.kwargs = kwargs
        return [self.result]


class AnnotationQaTests(unittest.TestCase):
    def test_masks_are_restored_to_original_prompt_indices(self):
        mask_zero = np.zeros((4, 4), dtype=np.uint8)
        mask_two = np.ones((4, 4), dtype=np.uint8)
        model = FakeModel(FakeResult([mask_zero, mask_two], [0, 2], [0.91, 0.88]))

        mapped = QA_HELPERS["sam_masks_for_image"](
            model,
            Path("image.jpg"),
            [(0, 0, 4, 4), (0, 0, 4, 4), (0, 0, 4, 4)],
            "",
        )

        self.assertEqual(model.kwargs["conf"], 0.0)
        self.assertIsNotNone(mapped)
        self.assertIs(mapped[0]["mask"], mask_zero)
        self.assertIsNone(mapped[1])
        self.assertIs(mapped[2]["mask"], mask_two)
        self.assertEqual(mapped[2]["prompt_index"], 2)
        self.assertAlmostEqual(mapped[2]["confidence"], 0.88)

    def test_missing_prompt_indices_fail_safe(self):
        result = FakeResult([np.ones((2, 2), dtype=np.uint8)], [0], [0.9])
        result.boxes.cls = None
        model = FakeModel(result)

        mapped = QA_HELPERS["sam_masks_for_image"](
            model,
            Path("image.jpg"),
            [(0, 0, 2, 2), (0, 0, 2, 2)],
            "",
        )

        self.assertIsNone(mapped)

    def test_box_tolerance_accepts_edge_differences_at_boundary(self):
        yolo = (10, 20, 110, 220)
        within = QA_HELPERS["bbox_edge_differences"](yolo, (15, 24, 106, 230), 5.0, 25.0)
        outside = QA_HELPERS["bbox_edge_differences"](yolo, (16, 24, 106, 230), 5.0, 25.0)

        self.assertTrue(within["within_tolerance"])
        self.assertEqual(within["max_percent"], 5.0)
        self.assertFalse(outside["within_tolerance"])
        self.assertTrue(outside["within_max_difference"])

    def test_difference_bands_include_maximum_boundary(self):
        yolo = (0, 0, 100, 100)
        cases = (
            ((5, 0, 100, 100), "within_tolerance"),
            ((6, 0, 100, 100), "reviewable"),
            ((25, 0, 100, 100), "reviewable"),
            ((26, 0, 100, 100), "large_disagreement"),
        )

        for sam, expected in cases:
            with self.subTest(sam=sam):
                differences = QA_HELPERS["bbox_edge_differences"](yolo, sam, 5.0, 25.0)
                self.assertEqual(QA_HELPERS["annotation_qa_difference_band"](differences), expected)

    def test_small_boxes_receive_pixel_floor(self):
        differences = QA_HELPERS["bbox_edge_differences"](
            (0, 0, 10, 10),
            (2, 0, 10, 10),
            5.0,
            25.0,
        )

        self.assertEqual(differences["max_percent"], 20.0)
        self.assertTrue(differences["within_tolerance"])
        self.assertEqual(differences["pixel_floor"], 2)

    def test_thresholds_reject_inverted_difference_band(self):
        with self.assertRaisesRegex(ValueError, "greater than"):
            QA_HELPERS["annotation_qa_thresholds"]("balanced", 5.0, 5.0)

    def test_only_reviewable_band_is_safe_for_sam_replacement(self):
        issue = {
            "auto_fix_eligible": True,
            "quality_gate_passed": True,
            "difference_band": "reviewable",
            "fix_type": "replace_box",
            "recommended_bbox": [1, 2, 3, 4],
            "sam_max_difference_percent": 25.0,
            "metrics": {
                "edge_differences": {
                    "within_tolerance": False,
                    "within_max_difference": True,
                },
            },
        }

        self.assertTrue(QA_HELPERS["issue_is_safe_sam_replacement"](issue))
        issue["quality_gate_passed"] = False
        self.assertFalse(QA_HELPERS["issue_is_safe_sam_replacement"](issue))
        issue["quality_gate_passed"] = True
        issue["difference_band"] = "large_disagreement"
        self.assertFalse(QA_HELPERS["issue_is_safe_sam_replacement"](issue))

    def test_summary_records_report_version_and_tolerance(self):
        summary = QA_HELPERS["annotation_qa_summary"](
            [], 2, 3, "sam2.1_s.pt", "val", "balanced", 7.5, 30.0, 2,
        )

        self.assertEqual(summary["report_version"], 3)
        self.assertEqual(summary["box_tolerance_percent"], 7.5)
        self.assertEqual(summary["sam_max_difference_percent"], 30.0)
        self.assertEqual(summary["yolo_boxes_accepted"], 2)

    def test_summary_counts_reviewable_and_blocked_sam_results(self):
        reviewable = {
            "issue_type": "moderate_box_difference",
            "severity": "low",
            "sam_bbox": [1, 2, 3, 4],
            "recommended_bbox": [1, 2, 3, 4],
            "fix_type": "replace_box",
            "auto_fix_eligible": True,
            "quality_gate_passed": True,
            "difference_band": "reviewable",
            "sam_max_difference_percent": 25.0,
            "metrics": {
                "edge_differences": {
                    "within_tolerance": False,
                    "within_max_difference": True,
                },
            },
        }
        blocked = {
            "issue_type": "large_box_disagreement",
            "severity": "high",
            "sam_bbox": [10, 20, 30, 40],
            "recommended_bbox": None,
            "fix_type": "",
            "auto_fix_eligible": False,
            "quality_gate_passed": False,
            "difference_band": "large_disagreement",
            "sam_max_difference_percent": 25.0,
            "metrics": {
                "edge_differences": {
                    "within_tolerance": False,
                    "within_max_difference": False,
                },
            },
        }

        summary = QA_HELPERS["annotation_qa_summary"](
            [reviewable, blocked], 1, 2, "sam2.1_s.pt", "val", "balanced",
        )

        self.assertEqual(summary["moderate_disagreements"], 1)
        self.assertEqual(summary["large_disagreements"], 1)
        self.assertEqual(summary["sam_replacements_available"], 1)
        self.assertEqual(summary["sam_replacements_blocked"], 1)


if __name__ == "__main__":
    unittest.main()
