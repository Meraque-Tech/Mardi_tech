"""Tests for collapsible web UI panel markup."""

import ast
from html.parser import HTMLParser
import math
from pathlib import Path
from typing import Optional
import unittest


INDEX_HTML = Path(__file__).parent / "static" / "index.html"
WEB_DIR = Path(__file__).parent
APP_PY = WEB_DIR / "app.py"
APP_JS = WEB_DIR / "static" / "app.js"
REPORT_GENERATOR = WEB_DIR / "report_generator.py"
TRAIN_YOLOV8 = WEB_DIR.parent / "train" / "train_yolov8.py"
EXPECTED_PANELS = {"dataset", "training", "gpu", "advanced", "logs", "results", "testing"}


class CollapsibleMarkupParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.panels = set()
        self.toggles = {}

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        panel_key = attributes.get("data-panel-key")
        if panel_key:
            self.panels.add(panel_key)
        toggle_key = attributes.get("data-panel-toggle")
        if toggle_key:
            self.toggles[toggle_key] = attributes


class CollapsibleSectionTests(unittest.TestCase):
    def test_each_requested_panel_has_an_accessible_toggle(self):
        parser = CollapsibleMarkupParser()
        parser.feed(INDEX_HTML.read_text(encoding="utf-8"))

        self.assertEqual(parser.panels, EXPECTED_PANELS)
        self.assertEqual(set(parser.toggles), EXPECTED_PANELS)
        for attributes in parser.toggles.values():
            self.assertIn(attributes.get("aria-expanded"), {"true", "false"})
            self.assertTrue(attributes.get("aria-label"))

    def test_activation_architecture_control_is_not_exposed(self):
        markup = INDEX_HTML.read_text(encoding="utf-8")

        self.assertNotIn('id="activation"', markup)
        self.assertNotIn("Model Architecture", markup)

    def test_class_ids_field_is_read_only(self):
        markup = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn('id="classes"', markup)
        self.assertIn("readonly", markup)

    def test_model_selector_exposes_supported_yolo_task_families(self):
        markup = INDEX_HTML.read_text(encoding="utf-8")

        expected_options = {
            "nano": "yolov8n.pt",
            "small": "yolov8s.pt",
            "medium": "yolov8m.pt",
            "large": "yolov8l.pt",
            "xlarge": "yolov8x.pt",
            "nano-seg": "yolov8n-seg.pt",
            "small-seg": "yolov8s-seg.pt",
            "medium-seg": "yolov8m-seg.pt",
            "large-seg": "yolov8l-seg.pt",
            "xlarge-seg": "yolov8x-seg.pt",
            "nano-cls": "yolov8n-cls.pt",
            "small-cls": "yolov8s-cls.pt",
            "medium-cls": "yolov8m-cls.pt",
            "large-cls": "yolov8l-cls.pt",
            "xlarge-cls": "yolov8x-cls.pt",
        }
        for family in ("yolo11", "yolo26"):
            for size_key, size_suffix in {
                "nano": "n",
                "small": "s",
                "medium": "m",
                "large": "l",
                "xlarge": "x",
            }.items():
                expected_options[f"{family}-{size_key}"] = f"{family}{size_suffix}.pt"
                expected_options[f"{family}-{size_key}-seg"] = f"{family}{size_suffix}-seg.pt"
                expected_options[f"{family}-{size_key}-cls"] = f"{family}{size_suffix}-cls.pt"

        for task in ("Detection", "Segmentation", "Classification"):
            for family in ("YOLOv8", "YOLO11", "YOLO26"):
                self.assertIn(f'optgroup label="{task} - {family}"', markup)
        for value, checkpoint in expected_options.items():
            self.assertIn(f'<option value="{value}">', markup)
            self.assertIn(checkpoint, markup)

        app_module = ast.parse(APP_PY.read_text(encoding="utf-8"))
        model_map = None
        for node in app_module.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "MODEL_MAP"
                for target in node.targets
            ):
                model_map = ast.literal_eval(node.value)
                break

        self.assertIsNotNone(model_map)
        for value, checkpoint in expected_options.items():
            self.assertEqual(model_map[value], checkpoint)

    def test_task_specific_project_defaults_are_exposed(self):
        app_module = ast.parse(APP_PY.read_text(encoding="utf-8"))
        project_defaults = None
        for node in app_module.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "TRAINING_PROJECT_DEFAULTS"
                for target in node.targets
            ):
                project_defaults = ast.literal_eval(node.value)
                break

        self.assertEqual(project_defaults, {
            "detect": "runs/detect",
            "segment": "runs/segment",
            "classify": "runs/classify",
        })

        script = APP_JS.read_text(encoding="utf-8")
        self.assertIn("function taskForModelSize", script)
        self.assertIn("function modelSizeForTask", script)
        self.assertIn("function syncProjectWithModelTask", script)
        self.assertIn('"model-size").addEventListener("change", syncProjectWithModelTask)', script)

    def test_previous_training_sessions_preserve_task_specific_project(self):
        script = APP_JS.read_text(encoding="utf-8")

        self.assertIn("function isDefaultTrainingTarget", script)
        self.assertIn("task: session.task", script)
        self.assertIn("task: latest.task", script)
        self.assertIn("modelSizeForTask(session.task", script)
        self.assertIn("defaultProjectForModelSize($(\"model-size\").value)", script)

    def test_report_headings_stay_with_following_content(self):
        source = REPORT_GENERATOR.read_text(encoding="utf-8")

        self.assertIn('name="Section"', source)
        self.assertIn('name="Subsection"', source)
        self.assertGreaterEqual(source.count("keepWithNext=True"), 2)

    def test_log_metric_parser_handles_segmentation_rows(self):
        app_module = ast.parse(APP_PY.read_text(encoding="utf-8"))
        parse_metric_row_node = next(
            node
            for node in app_module.body
            if isinstance(node, ast.FunctionDef) and node.name == "parse_metric_row"
        )
        def float_value_stub(row, key):
            try:
                value = float(str(row.get(key)).strip())
            except ValueError:
                return None
            return value if math.isfinite(value) else None

        namespace = {
            "Optional": Optional,
            "clean_log_line": lambda value: value.strip(),
            "float_value": float_value_stub,
            "format_metric": lambda value: round(value, 4) if value is not None and math.isfinite(value) else None,
            "f1_from_precision_recall": lambda precision, recall: (
                None
                if precision is None or recall is None or precision + recall <= 0
                else 2 * precision * recall / (precision + recall)
            ),
        }
        exec(compile(ast.Module([parse_metric_row_node], []), str(APP_PY), "exec"), namespace)
        parse_metric_row = namespace["parse_metric_row"]

        detection_row = parse_metric_row("ball 127 127 0.838 0.57 0.671 0.378")
        self.assertEqual(detection_row["class_name"], "ball")
        self.assertEqual(detection_row["images"], 127)
        self.assertEqual(detection_row["instances"], 127)
        self.assertEqual(detection_row["precision"], 0.838)

        segmentation_row = parse_metric_row(
            "palm-tree 2627 13174 0.699 0.645 0.701 0.671 0.706 0.669 0.697 0.434"
        )
        self.assertEqual(segmentation_row["class_name"], "palm-tree")
        self.assertEqual(segmentation_row["images"], 2627)
        self.assertEqual(segmentation_row["instances"], 13174)
        self.assertEqual(segmentation_row["precision"], 0.706)
        self.assertEqual(segmentation_row["recall"], 0.669)
        self.assertEqual(segmentation_row["map50"], 0.697)
        self.assertEqual(segmentation_row["map50_95"], 0.434)

        self.assertIsNone(parse_metric_row(
            "all 2943 19190 0.599 0.151 0 0.596 0.139 0.128 0.0559"
        ))

    def test_segmentation_metrics_use_mask_columns_and_labels(self):
        app_source = APP_PY.read_text(encoding="utf-8")
        script = APP_JS.read_text(encoding="utf-8")
        train_source = TRAIN_YOLOV8.read_text(encoding="utf-8")

        self.assertIn('"metrics/mAP50(M)"', app_source)
        self.assertIn('"metrics/mAP50-95(M)"', app_source)
        self.assertIn('prefix = "Mask "', app_source)
        self.assertIn('f"{prefix}Precision"', app_source)
        self.assertIn('"Segmentation Mask Performance by Epoch"', app_source)
        self.assertIn("applyMetricLabels(metrics.metric_labels", script)
        self.assertIn('id="performance-chart-title"', INDEX_HTML.read_text(encoding="utf-8"))
        self.assertIn('"metrics/mAP50(M)"', train_source)
        self.assertIn('"metrics/accuracy_top1"', train_source)
        self.assertIn('"Classification Metrics by Epoch"', train_source)
        self.assertIn('"Mask" if use_mask else "Box"', train_source)
        self.assertIn('"Segmentation Mask Metrics by Epoch"', train_source)

    def test_ultralytics_augmentation_controls_are_exposed(self):
        markup = INDEX_HTML.read_text(encoding="utf-8")
        script = APP_JS.read_text(encoding="utf-8")
        app_source = APP_PY.read_text(encoding="utf-8")
        train_source = TRAIN_YOLOV8.read_text(encoding="utf-8")
        expected_controls = {
            "disable-ultralytics-albumentations": "disable_ultralytics_albumentations",
            "mosaic": "mosaic",
            "close-mosaic": "close_mosaic",
            "hsv-h": "hsv_h",
            "hsv-s": "hsv_s",
            "hsv-v": "hsv_v",
            "degrees": "degrees",
            "translate": "translate",
            "scale": "scale",
            "shear": "shear",
            "perspective": "perspective",
            "flipud": "flipud",
            "fliplr": "fliplr",
            "bgr": "bgr",
            "mixup": "mixup",
            "cutmix": "cutmix",
            "copy-paste": "copy_paste",
            "auto-augment": "auto_augment",
            "erasing": "erasing",
        }

        self.assertIn("<h3>Augmentation</h3>", markup)
        for control_id, payload_key in expected_controls.items():
            self.assertIn(f'id="{control_id}"', markup)
            self.assertIn(f"{payload_key}:", script)
            self.assertIn(payload_key, app_source)

        self.assertIn("--disable-ultralytics-albumentations", app_source)
        self.assertIn("--auto-augment", app_source)
        self.assertIn("get_training_augmentations(config)", train_source)
        self.assertIn("nullcontext()", train_source)


if __name__ == "__main__":
    unittest.main()
