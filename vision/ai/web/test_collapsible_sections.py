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
STYLES = WEB_DIR / "static" / "styles.css"
REPORT_GENERATOR = WEB_DIR / "report_generator.py"
DOCKERFILE_CUDA = WEB_DIR / "Dockerfile.cuda"
TRAIN_WEB_COMPOSE = WEB_DIR / "docker-compose.train_web.yml"
BAKE_WEIGHTS_SCRIPT = WEB_DIR / "scripts" / "bake_pretrained_weights.py"
TRAIN_YOLOV8 = WEB_DIR.parent / "train" / "train_yolov8.py"
EXPECTED_PANELS = {
    "dataset",
    "annotation-qa",
    "training",
    "gpu",
    "advanced",
    "logs",
    "results",
    "testing",
}


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
                if family == "yolo26":
                    expected_options[f"{family}-{size_key}-sem"] = f"{family}{size_suffix}-sem.pt"

        for task in ("Detection", "Classification"):
            for family in ("YOLOv8", "YOLO11", "YOLO26"):
                self.assertIn(f'optgroup label="{task} - {family}"', markup)
        for family in ("YOLOv8", "YOLO11", "YOLO26"):
            self.assertIn(f'optgroup label="Instance Segmentation - {family}"', markup)
        self.assertIn('optgroup label="Semantic Segmentation - YOLO26"', markup)
        self.assertIn("instance segmentation", markup)
        self.assertIn("semantic segmentation", markup)
        self.assertNotIn("YOLOv5", markup)
        self.assertNotIn("yolov5", markup)
        self.assertIn('id="model-search"', markup)
        self.assertIn('type="search"', markup)
        self.assertIn('id="model-selector-toggle"', markup)
        self.assertIn('id="model-selector-menu"', markup)
        self.assertIn('id="model-task"', markup)
        self.assertIn('id="model-family"', markup)
        self.assertNotIn('id="model-size-choice"', markup)
        self.assertIn('id="model-compatibility"', markup)
        self.assertIn('id="selected-model-summary"', markup)
        self.assertIn('id="model-options"', markup)
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

    def test_rfdetr_nano_backend_is_exposed_with_backend_specific_controls(self):
        markup = INDEX_HTML.read_text(encoding="utf-8")
        script = APP_JS.read_text(encoding="utf-8")
        app_source = APP_PY.read_text(encoding="utf-8")
        train_source = (WEB_DIR.parent / "train" / "train_rfdetr.py").read_text(encoding="utf-8")
        infer_source = (WEB_DIR / "infer_rfdetr.py").read_text(encoding="utf-8")

        self.assertIn('optgroup label="Detection - RF-DETR"', markup)
        self.assertIn('<option value="rfdetr-nano">RF-DETR Nano - rfdetr-nano</option>', markup)
        self.assertNotIn('option value="rfdetr-small"', markup)
        self.assertIn('"rfdetr-nano": {', app_source)
        self.assertIn('"model": "rfdetr-nano"', app_source)
        self.assertIn('default="rfdetr-nano"', train_source)
        self.assertIn('or "rfdetr-nano"', infer_source)

        self.assertIn('const RFDETR_DEFAULTS = {', script)
        self.assertIn('lr0: 0.0001', script)
        self.assertIn('"weight-decay": 0.0001', script)
        self.assertIn('"warmup-epochs": 0', script)
        self.assertIn('return RFDETR_MODEL_SIZE;', script)
        self.assertIn('data-ultralytics-only', markup)
        self.assertIn('function syncModelFamilyControls', script)
        self.assertIn('document.querySelectorAll("[data-ultralytics-only]")', script)

        self.assertIn('weight_family: str = Form("auto")', app_source)
        self.assertIn('id="inference-upload-family"', markup)
        self.assertIn('form.append("weight_family"', script)
        self.assertIn("MODEL_CATALOG", script)
        self.assertIn('projectTask: "rfdetr"', script)
        self.assertIn('"compute_val_loss": True', train_source)
        self.assertIn("model.train(**supported_train_kwargs(model, train_kwargs))", train_source)
        self.assertIn("RFDETR_TEST_SCRIPT", app_source)
        self.assertIn('test_backend == "rfdetr"', app_source)
        self.assertIn("test_rfdetr.py", app_source)
        self.assertIn('training_metrics.get("training_completed") is False', app_source)
        self.assertIn('training_completed=existing_metrics.get("training_completed")', app_source)
        self.assertIn("generate_rfdetr_report_artifacts", train_source)
        self.assertIn("validation_metrics.json", train_source)
        self.assertIn("generate_report_artifacts=True", train_source)
        self.assertIn("generate_rfdetr_report_artifacts", app_source)
        self.assertIn("rfdetr_report_artifacts_ready", app_source)
        self.assertIn("ensure_rfdetr_report_artifacts_for_report", app_source)

    def test_dfine_nano_backend_is_exposed_with_backend_specific_controls(self):
        markup = INDEX_HTML.read_text(encoding="utf-8")
        script = APP_JS.read_text(encoding="utf-8")
        app_source = APP_PY.read_text(encoding="utf-8")
        train_source = (WEB_DIR.parent / "train" / "train_dfine.py").read_text(encoding="utf-8")
        test_source = (WEB_DIR.parent / "train" / "test_dfine.py").read_text(encoding="utf-8")
        infer_source = (WEB_DIR / "infer_dfine.py").read_text(encoding="utf-8")

        self.assertIn('optgroup label="Detection - D-FINE"', markup)
        self.assertIn('<option value="dfine-n">D-FINE Nano - dfine-n</option>', markup)
        self.assertIn('"dfine-n": {', app_source)
        self.assertIn('"family": "dfine"', app_source)
        self.assertIn('"model": DFINE_DEFAULTS["model"]', app_source)
        self.assertIn('default="dfine-n"', train_source)
        self.assertIn('convert_yolo_to_coco', train_source)
        self.assertIn('DFINE_REPO_DIR', train_source)
        self.assertIn('run_dfine_inference', infer_source)

        self.assertIn('const DFINE_DEFAULTS = {', script)
        self.assertIn('"warmup-epochs": 500', script)
        self.assertIn('warmupLabel.textContent = isDfine ? "Warmup steps" : "Warmup epochs"', script)
        self.assertIn('warmup_duration', train_source)
        self.assertIn('CosineAnnealingLR', train_source)
        self.assertIn('projectTask: "dfine"', script)
        self.assertIn('backend: "dfine"', script)
        self.assertIn('dfine: "runs/dfine"', script)
        self.assertIn('uploadFamily === "dfine"', script)
        self.assertIn("DFINE_PROGRESS_LOG_RE", app_source)
        self.assertIn("parse_progress_fields", app_source)
        self.assertIn("current_step", app_source)
        self.assertIn("progress_percent", app_source)
        self.assertIn("dfine_progress_marker", train_source)
        self.assertIn("generate_dfine_report_artifacts", train_source)
        self.assertIn("validation_metrics.json", train_source)
        self.assertIn("generate_report_artifacts=True", train_source)
        self.assertIn("generate_dfine_report_artifacts", app_source)
        self.assertIn("dfine_report_artifacts_ready", app_source)
        self.assertIn("ensure_dfine_report_artifacts_for_report", app_source)
        self.assertIn("ensure_model_report_artifacts_for_report(run_dir)", app_source)
        self.assertIn('float_value(web_overall, "precision")', app_source)
        self.assertIn('float_value(web_overall, "map50_95")', app_source)
        self.assertIn("YAMLConfig", test_source)
        self.assertIn("confusion_matrix_counts", test_source)
        self.assertIn("save_qualitative_artifacts", test_source)
        self.assertIn("merge_post_training_validation_metrics", train_source)

    def test_rfdetr_progress_and_results_refresh_are_supported(self):
        script = APP_JS.read_text(encoding="utf-8")
        app_source = APP_PY.read_text(encoding="utf-8")
        test_source = (WEB_DIR.parent / "train" / "test_rfdetr.py").read_text(encoding="utf-8")

        self.assertIn("RFDETR_VALIDATION_PROGRESS_RE", app_source)
        self.assertIn("Val\\s+\\(Epoch\\s+(\\d+)\\s*/\\s*(\\d+)\\)", app_source)
        self.assertIn("force=is_rfdetr_run(run_dir)", app_source)
        self.assertIn("Progress updates when validation metrics are logged", app_source)
        self.assertIn("confusion_matrix_counts", test_source)
        self.assertIn("save_qualitative_artifacts", test_source)
        self.assertIn("validation_metrics.json", app_source)
        self.assertIn('run_dir.glob("val_batch*_pred.jpg")', app_source)
        self.assertIn('run_dir.glob("val_batch*_labels.jpg")', app_source)

        self.assertIn("progress.detail", script)
        self.assertIn("Running epoch", script)
        self.assertIn("finished validation", script)
        self.assertNotIn("`${completed} ${completed === 1 ? \"epoch\" : \"epochs\"} completed.`", script)

    def test_docker_image_bakes_default_pretrained_weights(self):
        dockerfile = DOCKERFILE_CUDA.read_text(encoding="utf-8")
        compose = TRAIN_WEB_COMPOSE.read_text(encoding="utf-8")
        bake_script = BAKE_WEIGHTS_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("ARG BAKE_PRETRAINED_WEIGHTS=1", dockerfile)
        self.assertIn("python3 /app/vision/ai/web/scripts/bake_pretrained_weights.py", dockerfile)
        self.assertIn("YOLO_CONFIG_DIR=/home/appuser/.config/Ultralytics", dockerfile)
        self.assertIn("ULTRALYTICS_WEIGHTS_DIR=/home/appuser/.cache/ultralytics/weights", dockerfile)
        self.assertIn("RFDETR_CACHE_DIR=/home/appuser/.roboflow/models", dockerfile)

        self.assertIn('BAKE_PRETRAINED_WEIGHTS: "${BAKE_PRETRAINED_WEIGHTS:-1}"', compose)
        self.assertIn("YOLO_CONFIG_DIR: /home/appuser/.config/Ultralytics", compose)
        self.assertIn("ULTRALYTICS_WEIGHTS_DIR: /home/appuser/.cache/ultralytics/weights", compose)
        self.assertIn("RFDETR_CACHE_DIR: /home/appuser/.roboflow/models", compose)

        for weight_name in (
            "rf-detr-nano.pth",
            "yolo26n.pt",
            "yolo26s.pt",
            "yolov8n.pt",
            "yolov8s.pt",
            "yolo11n.pt",
            "yolo11s.pt",
            "yolov8n-seg.pt",
            "yolo26n-seg.pt",
            "sam2.1_s.pt",
            "sam2.1_t.pt",
        ):
            self.assertIn(weight_name, bake_script)

    def test_optional_sam_annotation_qa_controls_are_exposed(self):
        markup = INDEX_HTML.read_text(encoding="utf-8")
        script = APP_JS.read_text(encoding="utf-8")
        app_source = APP_PY.read_text(encoding="utf-8")

        for control_id in (
            "run-annotation-qa",
            "stop-annotation-qa",
            "annotation-qa-model",
            "annotation-qa-scope",
            "annotation-qa-preset",
            "annotation-qa-tolerance",
            "annotation-qa-max-difference",
            "annotation-qa-auto-mode",
            "annotation-qa-audit",
            "annotation-qa-prompt-expansion",
            "annotation-qa-prompt-jitter",
            "annotation-qa-stability-iou",
            "annotation-qa-stability-edge",
            "annotation-qa-auto-quality",
            "annotation-qa-auto-iou",
            "annotation-qa-auto-center",
            "annotation-qa-auto-neighbor",
            "annotation-qa-status",
            "annotation-qa-issues",
        ):
            self.assertIn(f'id="{control_id}"', markup)
        self.assertIn("sam2.1_s.pt", markup)
        self.assertIn("box_tolerance_percent", script)
        self.assertIn("sam_max_difference_percent", script)
        self.assertIn("auto_correction_mode", script)
        self.assertIn("prompt_stability", script)
        self.assertIn("box_tolerance_percent", app_source)
        self.assertIn("sam_max_difference_percent", app_source)
        self.assertIn("function runAnnotationQa", script)
        self.assertIn("function syncAnnotationQaActionStates", script)
        self.assertIn("/api/annotation-qa/start", app_source)
        self.assertIn("AnnotationQaRequest", app_source)

    def test_annotation_qa_follows_top_workspace(self):
        markup = INDEX_HTML.read_text(encoding="utf-8")

        workspace_start = markup.index('<section class="workspace">')
        workspace_end = markup.index('</section>', workspace_start)
        workspace_markup = markup[workspace_start:workspace_end]
        self.assertNotIn('id="annotation-qa-panel"', workspace_markup)
        self.assertLess(markup.index('id="dataset-panel"'), markup.index('id="annotation-qa-panel"'))
        self.assertLess(markup.index('id="training-panel"'), markup.index('id="annotation-qa-panel"'))
        self.assertLess(markup.index('id="gpu-monitor-panel"'), markup.index('id="annotation-qa-panel"'))

    def test_annotation_qa_issue_column_preserves_table_layout_and_readable_labels(self):
        script = APP_JS.read_text(encoding="utf-8")
        styles = STYLES.read_text(encoding="utf-8")

        self.assertNotIn(".qa-table td:nth-child(3) {\n  display: block", styles)
        self.assertIn("function annotationQaIssueTypeLabel(issueType)", script)
        self.assertIn("annotationQaIssueTypeLabel(issue.issue_type)", script)
        self.assertIn('low_box_agreement: "Low box agreement"', script)
        self.assertIn('low_confidence_mask: "Low-confidence SAM mask"', script)

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
            "rfdetr": "runs/rfdetr",
            "dfine": "runs/dfine",
            "segment": "runs/segment",
            "semantic": "runs/semantic",
            "classify": "runs/classify",
        })

        script = APP_JS.read_text(encoding="utf-8")
        self.assertIn("function taskForModelSize", script)
        self.assertIn("function modelSizeForTask", script)
        self.assertIn("function syncProjectWithModelTask", script)
        self.assertIn("function filterModelOptions", script)
        self.assertIn("const MODEL_TASKS = [", script)
        self.assertIn("const MODEL_FAMILIES = [", script)
        self.assertIn("const MODEL_SIZES = [", script)
        self.assertIn("const MODEL_CATALOG = [", script)
        self.assertIn("const MODEL_BY_VALUE", script)
        self.assertIn("function syncGuidedControlsFromModel", script)
        self.assertIn("function chooseGuidedModel", script)
        self.assertIn("function guidedModelValue", script)
        self.assertIn("function openModelSelector", script)
        self.assertIn("function closeModelSelector", script)
        self.assertIn("function chooseModelOption", script)
        self.assertIn('taskBadge.className = "model-task-badge"', script)
        self.assertIn("taskBadge.dataset.task = task.badge", script)
        self.assertIn('familyLabel.className = "model-family-label"', script)
        self.assertIn('"model-selector-toggle").addEventListener("click"', script)
        self.assertIn('"model-search").addEventListener("input", filterModelOptions)', script)
        self.assertIn('"model-task", "model-family"', script)
        self.assertNotIn('"model-size-choice"', script)
        self.assertIn('"model-size").addEventListener("change", () =>', script)
        self.assertIn("syncProjectWithModelTask();", script)

        styles = (WEB_DIR / "static" / "styles.css").read_text(encoding="utf-8")
        self.assertIn(".model-task-badge[data-task=\"detection\"]", styles)
        self.assertIn(".model-task-badge[data-task=\"instance-segmentation\"]", styles)
        self.assertIn(".model-task-badge[data-task=\"classification\"]", styles)

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

    def test_task_aware_loss_summary_excludes_auxiliary_losses(self):
        app_module = ast.parse(APP_PY.read_text(encoding="utf-8"))
        required = {
            "float_value",
            "sum_values",
            "loss_components",
            "raw_loss_sum",
            "comparable_loss_component_names",
            "loss_summary",
            "loss_note",
        }
        nodes = [
            node for node in app_module.body
            if isinstance(node, ast.FunctionDef) and node.name in required
        ]
        namespace = {"math": math, "Optional": Optional}
        exec(compile(ast.Module(nodes, []), str(APP_PY), "exec"), namespace)

        yolo26_row = {
            "train/box_loss": "1.12065",
            "train/seg_loss": "1.68615",
            "train/cls_loss": "0.98417",
            "train/dfl_loss": "0.01301",
            "train/sem_loss": "0.51730",
            "val/box_loss": "1.13102",
            "val/seg_loss": "1.65126",
            "val/cls_loss": "0.83375",
            "val/dfl_loss": "0.01515",
            "val/sem_loss": "0",
        }
        summary = namespace["loss_summary"](yolo26_row, "segment")
        self.assertAlmostEqual(summary["training_loss"], 3.80398)
        self.assertAlmostEqual(summary["testing_loss"], 3.63118)
        self.assertAlmostEqual(summary["raw_training_loss"], 4.32128)
        self.assertEqual(summary["loss_components"]["comparable"], ["box", "seg", "cls", "dfl"])
        self.assertEqual(summary["loss_components"]["auxiliary"]["train"], {"sem": 0.5173})
        self.assertEqual(summary["loss_components"]["auxiliary"]["val"], {"sem": 0.0})
        self.assertIn("train/sem_loss", namespace["loss_note"](summary))

        detect_row = {
            "train/box_loss": "1",
            "train/cls_loss": "2",
            "train/dfl_loss": "3",
            "train/extra_loss": "99",
            "val/box_loss": "4",
            "val/cls_loss": "5",
            "val/dfl_loss": "6",
        }
        detect_summary = namespace["loss_summary"](detect_row, "detect")
        self.assertEqual(detect_summary["training_loss"], 6)
        self.assertEqual(detect_summary["testing_loss"], 15)
        self.assertEqual(detect_summary["auxiliary_training_loss"], 99)

        classify_row = {
            "train/loss": "0.8",
            "val/loss": "0.9",
            "train/cls_loss": "3",
            "val/cls_loss": "4",
        }
        classify_summary = namespace["loss_summary"](classify_row, "classify")
        self.assertEqual(classify_summary["training_loss"], 0.8)
        self.assertEqual(classify_summary["testing_loss"], 0.9)

        classify_cls_row = {
            "train/cls_loss": "0.3",
            "val/cls_loss": "0.4",
        }
        classify_cls_summary = namespace["loss_summary"](classify_cls_row, "classify")
        self.assertEqual(classify_cls_summary["training_loss"], 0.3)
        self.assertEqual(classify_cls_summary["testing_loss"], 0.4)

        rfdetr_row = {
            "train/loss": "4.485000133514404",
            "val/loss": "",
        }
        rfdetr_summary = namespace["loss_summary"](rfdetr_row, "detect")
        self.assertAlmostEqual(rfdetr_summary["training_loss"], 4.485000133514404)
        self.assertIsNone(rfdetr_summary["testing_loss"])
        self.assertIsNone(rfdetr_summary["auxiliary_training_loss"])

    def test_underrepresented_report_classes_are_not_truncated(self):
        source = REPORT_GENERATOR.read_text(encoding="utf-8")
        module = ast.parse(source)
        required = {"_to_float", "_to_int", "_count", "_nonempty", "_join_all", "_imbalance_summary"}
        nodes = [
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name in required
        ]
        namespace = {}
        exec(compile(ast.Module(nodes, []), str(REPORT_GENERATOR), "exec"), namespace)

        summary = {
            "class_distribution": [
                {"class_name": "Palm", "images": 100, "instances": 120},
                {"class_name": "Human", "images": 5, "instances": 5},
                {"class_name": "car", "images": 6, "instances": 6},
                {"class_name": "hill", "images": 7, "instances": 7},
                {"class_name": "leaf", "images": 8, "instances": 8},
                {"class_name": "pothole", "images": 9, "instances": 9},
                {"class_name": "road", "images": 4, "instances": 4},
            ]
        }
        _, underrepresented = namespace["_imbalance_summary"](summary)
        names = namespace["_join_all"]([row.get("class_name") for row in underrepresented])

        self.assertIn("Human, car, hill, leaf, pothole, road", names)
        self.assertNotIn("and 1 more", names)

    def test_weak_report_classes_are_not_truncated(self):
        source = REPORT_GENERATOR.read_text(encoding="utf-8")
        module = ast.parse(source)
        required = {
            "_to_float",
            "_to_int",
            "_nonempty",
            "_join_all",
            "_metric_value",
            "_f1_value",
            "_metric_band",
            "_primary_evidence",
            "_weak_classes",
            "_recommendation",
        }
        nodes = [
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name in required
        ]
        namespace = {}
        exec(compile(ast.Module(nodes, []), str(REPORT_GENERATOR), "exec"), namespace)

        metrics = {
            "map50": 0.8,
            "map50_95": 0.5,
            "recall": 0.7,
            "per_class": [
                {"class_name": name, "instances": 5, "recall": 0.4, "f1": 0.4, "map50_95": 0.3}
                for name in ("Human", "car", "hill", "leaf", "pothole", "road")
            ],
        }
        weak = namespace["_weak_classes"](metrics)
        names = namespace["_join_all"]([row["class_name"] for row in weak])
        recommendation = namespace["_recommendation"](metrics)

        self.assertIn("Human, car, hill, leaf, pothole, road", names)
        self.assertIn("Human, car, hill, leaf, pothole, road", recommendation)
        self.assertNotIn("and 1 more", recommendation)

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

    def test_rfdetr_report_config_uses_backend_specific_rows(self):
        source = REPORT_GENERATOR.read_text(encoding="utf-8")

        self.assertIn('family == "rfdetr"', source)
        self.assertIn('family == "dfine"', source)
        self.assertIn('("Weight decay", "weight_decay")', source)
        self.assertIn('("Warmup epochs", "warmup_epochs")', source)
        self.assertIn('if family in {"dfine", "rfdetr"}:', source)
        self.assertIn('"Exported images used for training"', source)
        self.assertNotIn('["YOLOv8 model variant"', source)
        self.assertNotIn('["Exported images used by YOLOv8"', source)
        self.assertIn("def _runtime_environment_rows", source)
        self.assertIn("def _filtered_hyperparameters", source)
        self.assertIn("def _metric_source_rows", source)
        self.assertIn("def _per_class_table_rows", source)
        self.assertIn('"D-FINE repo"', source)
        self.assertIn('"RF-DETR"', source)

    def test_report_downloads_use_response_filename(self):
        script = APP_JS.read_text(encoding="utf-8")
        app_source = APP_PY.read_text(encoding="utf-8")

        self.assertIn('responseDownloadFilename(response, "training_report.pdf")', script)
        self.assertIn('responseDownloadFilename(response, "training_and_test_report.pdf")', script)
        self.assertNotIn('saveBlobWithBrowserDownload(await response.blob(), "training_report.pdf")', script)
        self.assertNotIn('saveBlobWithBrowserDownload(await response.blob(), "training_and_test_report.pdf")', script)
        self.assertIn("ensure_model_report_artifacts_for_report(run_dir)\n    metrics = read_run_metrics(run_dir)", app_source)
        self.assertIn("ensure_model_report_artifacts_for_report(training_dir)\n    training_metrics_payload = read_run_metrics(training_dir)", app_source)

    def test_magic_button_adjusts_report_metrics_overlay(self):
        html = INDEX_HTML.read_text(encoding="utf-8")
        script = APP_JS.read_text(encoding="utf-8")
        styles = STYLES.read_text(encoding="utf-8")
        app_source = APP_PY.read_text(encoding="utf-8")

        self.assertIn('id="download-training-report"', html)
        self.assertIn('id="magic-metrics"', html)
        self.assertIn('id="magic-modal"', html)
        self.assertIn('id="magic-overall-options"', html)
        self.assertIn('id="magic-per-class-options"', html)
        self.assertIn('id="magic-class-select"', html)
        self.assertIn('id="magic-class-select" disabled', html)
        self.assertIn("Select a per-class metric first", html)
        self.assertIn('id="magic-target"', html)
        self.assertIn('id="magic-add"', html)
        self.assertIn('id="magic-clear"', html)
        self.assertIn('id="magic-adjustment-list"', html)
        self.assertIn('id="magic-reset"', html)
        self.assertLess(html.index('id="download-training-report"'), html.index('id="magic-metrics"'))
        self.assertIn("button.magic-button", styles)
        self.assertIn(".magic-dialog", styles)
        self.assertIn(".magic-option-grid", styles)
        self.assertIn(".magic-adjustment-list", styles)
        self.assertIn(".magic-adjustment-row", styles)
        self.assertIn(".magic-field-disabled", styles)
        self.assertIn("cursor: not-allowed", styles)
        self.assertIn("const MAGIC_OVERALL_OPTIONS", script)
        self.assertIn("const MAGIC_PER_CLASS_OPTIONS", script)
        self.assertIn("function openMagicMetricsModal", script)
        self.assertIn("function closeMagicMetricsModal", script)
        self.assertIn("function addMagicAdjustment", script)
        self.assertIn("function renderMagicAdjustments", script)
        self.assertIn("function resetMagicMetrics", script)
        self.assertIn("data-magic-edit", script)
        self.assertIn("function errorDetailText", script)
        self.assertIn("renderMagicOptionGroup", script)
        self.assertIn("classSelect.disabled = !isPerClass || !hasClasses", script)
        self.assertIn('classField.classList.toggle("magic-field-disabled", classSelect.disabled)', script)
        self.assertIn("const index = Number.parseInt($(\"magic-class-select\").value, 10)", script)
        self.assertIn("option.value = String(index)", script)
        self.assertIn("Select a per-class metric first", script)
        self.assertIn("setMagicChoice(button.dataset.magicScope, button.dataset.magicKey)", script)
        self.assertIn('$("magic-metrics").addEventListener("click", openMagicMetricsModal)', script)
        self.assertIn('$("magic-add").addEventListener("click", addMagicAdjustment)', script)
        self.assertIn('$("magic-apply").addEventListener("click", applyMagicMetrics)', script)
        self.assertIn("Reset service is unavailable. Restart the training web service and try again.", script)
        self.assertNotIn("window.prompt", script[script.index("function magicCurrentValue"):script.index("async function startTest")])
        self.assertNotIn("window.confirm", script[script.index("function magicCurrentValue"):script.index("async function startTest")])
        self.assertIn('"Raw logs, results.csv, and weights stay unchanged."', script)
        self.assertIn('fetch("/api/train/metrics/magic"', script)
        self.assertIn("adjustments,", script)
        self.assertIn('fetch("/api/train/metrics/magic/reset"', script)
        self.assertIn('MAGIC_METRICS_FILE = "magic_metrics.json"', app_source)
        self.assertIn("class MagicMetricAdjustment(BaseModel):", app_source)
        self.assertIn("class MagicMetricsRequest(WeightRequest):", app_source)
        self.assertIn("adjustments: list[MagicMetricAdjustment]", app_source)
        self.assertIn("target: float = Field(ge=0, le=1)", app_source)
        self.assertIn("def parse_magic_metrics_payload", app_source)
        self.assertIn("payload.get(\"target\", payload.get(\"map50_95\"))", app_source)
        self.assertIn('@app.post("/api/train/metrics/magic")', app_source)
        self.assertIn("def magic_train_metrics(payload: Optional[dict] = Body(default=None))", app_source)
        self.assertIn("MAGIC_OVERALL_METRICS", app_source)
        self.assertIn("MAGIC_PER_CLASS_METRICS", app_source)
        self.assertIn("def build_magic_metrics_overlay", app_source)
        self.assertIn("def validate_magic_adjustments", app_source)
        self.assertIn("def verify_magic_adjustment_targets", app_source)
        self.assertIn("def rebalanced_metric_values", app_source)
        self.assertIn("def rebalanced_metric_total", app_source)
        self.assertIn("def adjust_overall_magic_metric", app_source)
        self.assertIn("def adjust_per_class_magic_metric", app_source)
        self.assertIn('"magic_adjustments": adjustments', app_source)
        self.assertIn('"magic_adjusted": True', app_source)
        self.assertIn("f1_from_precision_recall(precision, recall)", app_source)
        self.assertIn("return apply_magic_metrics_overlay(run_dir, result) if include_magic else result", app_source)
        self.assertIn('@app.post("/api/train/metrics/magic/reset")', app_source)
        report_source = REPORT_GENERATOR.read_text(encoding="utf-8")
        self.assertNotIn("Adjusted report-preview metrics are active", report_source)
        self.assertNotIn('"Adjusted score", "Target"', report_source)


if __name__ == "__main__":
    unittest.main()
