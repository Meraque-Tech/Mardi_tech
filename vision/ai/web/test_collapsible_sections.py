"""Tests for collapsible web UI panel markup."""

import ast
from html.parser import HTMLParser
from pathlib import Path
import unittest


INDEX_HTML = Path(__file__).parent / "static" / "index.html"
WEB_DIR = Path(__file__).parent
APP_PY = WEB_DIR / "app.py"
APP_JS = WEB_DIR / "static" / "app.js"
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

    def test_model_selector_exposes_yolov8_task_families(self):
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

        self.assertIn('optgroup label="Detection"', markup)
        self.assertIn('optgroup label="Segmentation"', markup)
        self.assertIn('optgroup label="Classification"', markup)
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
