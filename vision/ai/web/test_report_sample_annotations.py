import tempfile
import unittest
from pathlib import Path

import yaml

from vision.ai.web import report_generator


class ReportSampleAnnotationTests(unittest.TestCase):
    def setUp(self):
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - Pillow is part of the web image.
            self.skipTest(str(exc))
        self.Image = Image
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image_path = self.root / "sample.jpg"
        self.label_path = self.root / "sample.txt"
        Image.new("RGB", (100, 100), "white").save(self.image_path, "JPEG", quality=100)

    def tearDown(self):
        self.temporary.cleanup()

    def render(self, label_text, task, target_class_id=0):
        self.label_path.write_text(label_text, encoding="utf-8")
        output_path = self.root / f"{task}.jpg"
        warnings = []
        result = report_generator._annotated_thumbnail(
            self.image_path,
            self.label_path,
            ["target", "other"],
            output_path,
            task=task,
            target_class_id=target_class_id,
            warnings=warnings,
        )
        self.assertEqual(result, output_path)
        return self.Image.open(output_path).convert("RGB"), warnings

    def test_explicit_segmentation_task_wins_over_detection_defaults(self):
        task = report_generator._report_task(
            {"task": "segment", "model": "yolov8n-seg.pt"},
            {"task": "detect"},
            self.root / "runs" / "detect" / "legacy-name",
        )
        self.assertEqual(task, "segment")

    def test_legacy_mask_metrics_resolve_segmentation_task(self):
        self.assertEqual(
            report_generator._report_task({}, {"primary_metric_type": "mask"}),
            "segment",
        )

    def test_row_fallback_does_not_guess_ambiguous_four_point_format(self):
        self.label_path.write_text("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", encoding="utf-8")
        selections = {0: [(self.image_path, self.label_path)]}
        self.assertEqual(report_generator._infer_sample_task(selections), "")

    def test_detection_parser_keeps_bbox_geometry(self):
        parsed = report_generator._parse_yolo_sample_annotation(
            "0 0.5 0.5 0.4 0.2",
            "detect",
            100,
            80,
        )
        self.assertEqual(parsed["kind"], "box")
        self.assertEqual(parsed["box"], [30.0, 32.0, 70.0, 48.0])

    def test_segmentation_parser_uses_all_polygon_points(self):
        parsed = report_generator._parse_yolo_sample_annotation(
            "0 0.1 0.1 0.8 0.1 0.8 0.7 0.1 0.7",
            "segment",
            100,
            80,
        )
        self.assertEqual(parsed["kind"], "polygon")
        self.assertEqual(parsed["points"], [(10.0, 8.0), (80.0, 8.0), (80.0, 56.0), (10.0, 56.0)])

    def test_segmentation_thumbnail_draws_polygon_fill_and_only_target_class(self):
        image, warnings = self.render(
            "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n"
            "1 0.05 0.05 0.15 0.05 0.10 0.15\n",
            "segment",
        )
        interior = image.getpixel((50, 50))
        other_class_interior = image.getpixel((10, 10))
        self.assertGreater(interior[1], interior[0] + 5)
        self.assertGreater(min(other_class_interior), 225)
        self.assertEqual(warnings, [])

    def test_detection_thumbnail_still_draws_a_box(self):
        image, warnings = self.render("0 0.5 0.5 0.4 0.4\n", "detect")
        boundary = image.getpixel((30, 50))
        center = image.getpixel((50, 50))
        self.assertGreater(boundary[1], boundary[0] + 10)
        self.assertGreater(min(center), 225)
        self.assertEqual(warnings, [])

    def test_malformed_segmentation_row_is_skipped_with_warning(self):
        _image, warnings = self.render(
            "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n"
            "0 0.1 0.1 0.2\n",
            "segment",
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("1 malformed or unsupported segment", warnings[0])

    def test_cache_signature_changes_with_task_and_label_content(self):
        self.label_path.write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        detection = report_generator._sample_asset_signature(
            self.image_path, self.label_path, "detect", 0
        )
        segmentation_task = report_generator._sample_asset_signature(
            self.image_path, self.label_path, "segment", 0
        )
        self.label_path.write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
        changed_label = report_generator._sample_asset_signature(
            self.image_path, self.label_path, "detect", 0
        )
        self.assertNotEqual(detection, segmentation_task)
        self.assertNotEqual(detection, changed_label)

    def test_dataset_section_generates_versioned_segmentation_assets(self):
        try:
            from reportlab.platypus import SimpleDocTemplate  # noqa: F401
        except ImportError as exc:  # pragma: no cover - ReportLab is part of the web image.
            self.skipTest(str(exc))

        dataset_root = self.root / "dataset"
        images = dataset_root / "images" / "train"
        labels = dataset_root / "labels" / "train"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        image_path = images / "tree.jpg"
        label_path = labels / "tree.txt"
        self.Image.new("RGB", (120, 80), "white").save(image_path, "JPEG", quality=100)
        label_path.write_text("0 0.1 0.1 0.9 0.1 0.8 0.9 0.2 0.9\n", encoding="utf-8")
        yaml_path = dataset_root / "data.yaml"
        yaml_path.write_text(
            yaml.safe_dump({
                "path": str(dataset_root),
                "train": "images/train",
                "val": "images/train",
                "names": {0: "tree"},
            }, sort_keys=False),
            encoding="utf-8",
        )
        run_dir = self.root / "runs" / "segment" / "train"
        context = {
            "task": "segment",
            "dataset_yaml": str(yaml_path),
            "dataset_summary": {
                "dataset_root": str(dataset_root),
                "class_count": 1,
                "classes": ["tree"],
                "splits": {
                    "train": {
                        "images": 1,
                        "missing_labels": 0,
                        "image_path": str(images),
                        "label_path": str(labels),
                    }
                },
            },
        }
        output_path = self.root / "dataset-section.pdf"
        builder = report_generator._ReportBuilder(output_path, "Dataset Samples")
        report_generator._add_dataset(builder, run_dir, context, {"task": "segment"})
        builder.build()

        assets = list((run_dir / "report_assets" / "samples-v2" / "segment").glob("*.jpg"))
        self.assertTrue(output_path.is_file())
        self.assertEqual(len(assets), 1)


if __name__ == "__main__":
    unittest.main()
