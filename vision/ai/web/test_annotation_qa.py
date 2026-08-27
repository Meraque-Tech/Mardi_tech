"""Regression tests for SAM prompt mapping and box-tolerance decisions."""

import ast
import hashlib
import math
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
import unittest

import numpy as np



def load_qa_helpers():
    source_path = Path(__file__).with_name("services") / "annotation_qa.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {
        "sam_masks_for_image",
        "bbox_edge_differences",
        "annotation_qa_difference_band",
        "issue_is_safe_sam_replacement",
        "issue_is_reviewable_sam_polygon",
        "annotation_qa_thresholds",
        "annotation_qa_summary",
        "bbox_area",
        "bbox_iou",
        "annotation_qa_prompt_box",
        "annotation_qa_prompt_plan",
        "annotation_qa_mask_iou",
        "annotation_qa_prompt_stability",
        "annotation_qa_select_candidate",
        "annotation_qa_max_neighbor_iou",
        "annotation_qa_auto_gate",
        "annotation_qa_audit_required",
        "mask_to_uint8",
        "mask_bbox",
        "draw_annotation_qa_preview",
        "annotation_qa_candidates_for_image",
        "normalize_annotation_qa_task",
        "annotation_task_for_fields",
        "pixel_polygon_from_yolo",
        "polygon_bbox",
        "polygon_mask",
        "mask_to_polygon",
        "yolo_polygon_from_pixels",
        "annotation_qa_segmentation_metrics",
        "issue_label_path_in_copy",
        "apply_annotation_qa_fix",
    }
    module = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
        type_ignores=[],
    )
    class DummyHTTPException(Exception):
        def __init__(self, status_code=500, detail=""):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    namespace = {
        "Optional": Optional,
        "Path": Path,
        "datetime": datetime,
        "MYT": timezone(timedelta(hours=8)),
        "ANNOTATION_QA_REPORT_VERSION": 4,
        "hashlib": hashlib,
        "math": math,
        "threading": threading,
        "SamQaRuntime": object,
        "InferenceStopped": RuntimeError,
        "HTTPException": DummyHTTPException,
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
    def test_annotation_task_recognizes_detection_and_segmentation_rows(self):
        detect = "0 0.5 0.5 0.2 0.3".split()
        segment = "1 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8".split()

        self.assertEqual(QA_HELPERS["annotation_task_for_fields"](detect), "detect")
        self.assertEqual(QA_HELPERS["annotation_task_for_fields"](segment), "segment")
        self.assertIsNone(QA_HELPERS["annotation_task_for_fields"]("0 1 2 3 4 5".split()))

    def test_polygon_parse_rasterize_and_serialize_round_trip(self):
        fields = "1 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8".split()
        polygon = QA_HELPERS["pixel_polygon_from_yolo"](fields, 100, 80)

        self.assertEqual(polygon, [[10, 8], [80, 8], [80, 64], [10, 64]])
        self.assertEqual(QA_HELPERS["polygon_bbox"](polygon), (10, 8, 81, 65))
        mask = QA_HELPERS["polygon_mask"](polygon, 100, 80)
        self.assertGreater(int(mask.sum()), 3000)
        row = QA_HELPERS["yolo_polygon_from_pixels"](1, polygon, 100, 80)
        self.assertEqual(QA_HELPERS["annotation_task_for_fields"](row.split()), "segment")

    def test_segmentation_metrics_and_contour_are_mask_aware(self):
        original = np.zeros((100, 100), dtype=np.uint8)
        original[20:80, 20:80] = 1
        same = original.copy()
        shifted = np.zeros_like(original)
        shifted[35:95, 35:95] = 1

        exact = QA_HELPERS["annotation_qa_segmentation_metrics"](original, same)
        different = QA_HELPERS["annotation_qa_segmentation_metrics"](original, shifted)
        self.assertEqual(exact["mask_iou"], 1.0)
        self.assertEqual(exact["boundary_f1"], 1.0)
        self.assertLess(different["mask_iou"], exact["mask_iou"])
        contour = QA_HELPERS["mask_to_polygon"](same, 100, 100)
        self.assertIsNotNone(contour)
        self.assertGreaterEqual(len(contour), 3)

    def test_reviewable_polygon_requires_manual_quality_gates(self):
        issue = {
            "annotation_task": "segment",
            "auto_fix_eligible": True,
            "quality_gate_passed": True,
            "difference_band": "reviewable",
            "fix_type": "replace_polygon",
            "recommended_polygon": [[1, 1], [9, 1], [9, 9], [1, 9]],
            "metrics": {"prompt_stability": {"passed": True}},
        }
        self.assertTrue(QA_HELPERS["issue_is_reviewable_sam_polygon"](issue))
        issue["metrics"]["prompt_stability"]["passed"] = False
        self.assertFalse(QA_HELPERS["issue_is_reviewable_sam_polygon"](issue))

    def test_manual_polygon_fix_preserves_copy_and_unrelated_rows(self):
        import cv2

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            corrected = root / "corrected"
            source_image = source / "train" / "images" / "sample.jpg"
            corrected_image = corrected / "train" / "images" / "sample.jpg"
            source_label = source / "train" / "labels" / "sample.txt"
            corrected_label = corrected / "train" / "labels" / "sample.txt"
            for path in (source_image, corrected_image, source_label, corrected_label):
                path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(source_image), np.full((100, 100, 3), 120, dtype=np.uint8))
            cv2.imwrite(str(corrected_image), np.full((100, 100, 3), 120, dtype=np.uint8))
            original = "1 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8"
            unrelated = "0 0.2 0.2 0.3 0.2 0.3 0.3 0.2 0.3"
            source_label.write_text(f"{original}\n{unrelated}\n", encoding="utf-8")
            corrected_label.write_text(f"{original}\n{unrelated}\n", encoding="utf-8")
            issue = {
                "image": str(source_image),
                "split": "train",
                "class_id": 1,
                "label_row": 1,
                "annotation_task": "segment",
                "accepted_fix": "sam_polygon",
                "auto_fix_eligible": True,
                "quality_gate_passed": True,
                "difference_band": "reviewable",
                "fix_type": "replace_polygon",
                "recommended_polygon": [[15, 15], [75, 15], [75, 75], [15, 75]],
                "metrics": {"prompt_stability": {"passed": True}},
            }

            result = QA_HELPERS["apply_annotation_qa_fix"](issue, source, corrected)

            self.assertTrue(result["applied"])
            rows = corrected_label.read_text(encoding="utf-8").splitlines()
            self.assertEqual(rows[1], unrelated)
            self.assertEqual(rows[0], "1 0.150000 0.150000 0.750000 0.150000 0.750000 0.750000 0.150000 0.750000")
            self.assertEqual(source_label.read_text(encoding="utf-8").splitlines()[0], original)

    def test_staged_prompts_skip_stability_work_for_boxes_within_tolerance(self):
        class Runtime:
            def __init__(self):
                self.calls = []
                self.reset = False

            def set_image(self, _image):
                pass

            def predict_prompts(self, boxes, _stop_event):
                self.calls.append(list(boxes))
                masks = []
                for box in boxes:
                    mask = np.zeros((100, 100), dtype=np.uint8)
                    x1, y1, x2, y2 = box
                    if x1 == 50:
                        x1 = 53
                    mask[y1:y2, x1:x2] = 1
                    masks.append({"mask": mask, "confidence": 0.9})
                return masks

            def reset_image(self):
                self.reset = True

        runtime = Runtime()
        labels = [
            {"bbox": (10, 10, 30, 30), "row_index": 1},
            {"bbox": (50, 50, 80, 80), "row_index": 2},
        ]
        candidates = QA_HELPERS["annotation_qa_candidates_for_image"](
            runtime,
            np.zeros((100, 100, 3), dtype=np.uint8),
            labels,
            100,
            100,
            {
                "box_tolerance_percent": 5.0,
                "sam_max_difference_percent": 25.0,
                "sam_prompt_expansion_percent": 8.0,
                "sam_prompt_jitter_percent": 2.0,
            },
            threading.Event(),
        )

        self.assertEqual([len(call) for call in runtime.calls], [2, 2])
        self.assertEqual(len(candidates[0]), 1)
        self.assertEqual(len(candidates[1]), 3)
        self.assertTrue(runtime.reset)

    def test_preview_writer_creates_interactive_overlay_assets(self):
        import cv2

        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.jpg"
            preview_path = root / "previews" / "issue.jpg"
            cv2.imwrite(str(image_path), np.full((60, 80, 3), 100, dtype=np.uint8))
            mask = np.zeros((60, 80), dtype=np.uint8)
            mask[15:45, 20:60] = 1
            issue = {
                "severity": "low",
                "issue_type": "moderate_box_difference",
                "score": 0.5,
                "original_bbox": [15, 10, 65, 50],
                "sam_bbox": [20, 15, 60, 45],
            }

            QA_HELPERS["draw_annotation_qa_preview"](image_path, preview_path, issue, mask)

            self.assertTrue(preview_path.is_file())
            self.assertTrue((preview_path.parent / "issue.raw.jpg").is_file())
            self.assertTrue((preview_path.parent / "issue.mask.jpg").is_file())
            self.assertEqual(issue["raw_preview"], "previews/issue.raw.jpg")
            self.assertEqual(issue["mask_preview"], "previews/issue.mask.jpg")

    def test_segmentation_preview_writes_original_and_sam_masks(self):
        import cv2

        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.jpg"
            preview_path = root / "previews" / "segment.jpg"
            cv2.imwrite(str(image_path), np.full((60, 80, 3), 100, dtype=np.uint8))
            original = np.zeros((60, 80), dtype=np.uint8)
            original[10:50, 10:50] = 1
            sam = np.zeros((60, 80), dtype=np.uint8)
            sam[12:52, 12:52] = 1
            issue = {
                "severity": "low",
                "issue_type": "moderate_mask_difference",
                "score": 0.2,
                "original_bbox": [10, 10, 50, 50],
                "sam_bbox": [12, 12, 52, 52],
                "original_polygon": [[10, 10], [50, 10], [50, 50], [10, 50]],
                "sam_polygon": [[12, 12], [52, 12], [52, 52], [12, 52]],
            }

            QA_HELPERS["draw_annotation_qa_preview"](image_path, preview_path, issue, sam, original)

            self.assertTrue(preview_path.is_file())
            self.assertTrue((preview_path.parent / "segment.original-mask.jpg").is_file())
            self.assertEqual(issue["original_mask_preview"], "previews/segment.original-mask.jpg")

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

        self.assertEqual(summary["report_version"], 4)
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

    def test_prompt_plan_has_original_expanded_and_jittered_variants(self):
        labels = [{"bbox": (20, 20, 80, 100), "row_index": 3}]
        prompts, references = QA_HELPERS["annotation_qa_prompt_plan"](labels, 120, 140, 8.0, 2.0)

        self.assertEqual(len(prompts), 3)
        self.assertEqual({item["variant"] for item in references}, {"original", "expanded", "jittered"})
        self.assertEqual(prompts[0], (20, 20, 80, 100))
        self.assertGreater(prompts[1][2] - prompts[1][0], prompts[0][2] - prompts[0][0])

    def test_prompt_stability_rejects_changed_sam_box(self):
        mask = np.ones((20, 20), dtype=np.uint8)
        stable = [
            {"variant": "original", "prompt_bbox": (20, 20, 80, 100), "bbox": (25, 25, 75, 95), "mask": mask, "confidence": 0.9},
            {"variant": "expanded", "prompt_bbox": (15, 15, 85, 105), "bbox": (25, 25, 75, 95), "mask": mask, "confidence": 0.91},
            {"variant": "jittered", "prompt_bbox": (17, 13, 87, 103), "bbox": (25, 25, 75, 95), "mask": mask, "confidence": 0.89},
        ]
        result = QA_HELPERS["annotation_qa_prompt_stability"](stable, (20, 20, 80, 100), 0.9, 3.0)
        self.assertTrue(result["passed"])
        unstable = [*stable]
        unstable[2] = {**unstable[2], "bbox": (40, 25, 90, 95)}
        result = QA_HELPERS["annotation_qa_prompt_stability"](unstable, (20, 20, 80, 100), 0.9, 3.0)
        self.assertFalse(result["passed"])

    def test_automatic_gate_requires_stability_and_geometry(self):
        thresholds = {
            "sam_auto_quality_min": 0.85,
            "sam_auto_yolo_iou_min": 0.70,
            "sam_auto_center_shift_max": 0.10,
            "sam_auto_neighbor_iou_max": 0.15,
        }
        stable = {"passed": True, "expanded_edge_clipped": False}
        passed, reasons = QA_HELPERS["annotation_qa_auto_gate"](
            stability=stable, sam_confidence=0.9, bbox_overlap=0.8,
            center_shift=0.05, neighbor_iou=0.05,
            quality_gate_passed=True, thresholds=thresholds,
        )
        self.assertTrue(passed)
        self.assertEqual(reasons, [])
        passed, reasons = QA_HELPERS["annotation_qa_auto_gate"](
            stability={"passed": False, "expanded_edge_clipped": False}, sam_confidence=0.9,
            bbox_overlap=0.8, center_shift=0.05, neighbor_iou=0.05,
            quality_gate_passed=True, thresholds=thresholds,
        )
        self.assertFalse(passed)
        self.assertIn("prompt_stability_failed", reasons)


if __name__ == "__main__":
    unittest.main()
