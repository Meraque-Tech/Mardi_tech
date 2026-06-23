"""PDF report generation for completed YOLO training and test runs."""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

import yaml


def _text(value) -> str:
    if value is None or value == "":
        return "N/A"
    return html.escape(str(value))


def _metric(value) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _run_timing(run_dir: Path, context: dict) -> tuple[str, str, str]:
    started = context.get("last_started_at") or context.get("created_at") or "N/A"
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        return str(started), "N/A", "N/A"
    completed = datetime.fromtimestamp(results_path.stat().st_mtime).astimezone()
    duration = "N/A"
    try:
        start_time = datetime.fromisoformat(str(started))
        if start_time.tzinfo is None:
            start_time = start_time.astimezone()
        seconds = max(0, int((completed - start_time).total_seconds()))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration = f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    except (TypeError, ValueError):
        pass
    return str(started), completed.isoformat(timespec="seconds"), duration


def _load_yaml(path: Path) -> dict:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _dataset_root(yaml_path: Path, config: dict) -> Path:
    value = Path(str(config.get("path") or yaml_path.parent)).expanduser()
    if not value.is_absolute():
        value = yaml_path.parent / value
    return value.resolve()


def _class_names(config: dict, summary: dict) -> list[str]:
    names = config.get("names")
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    if isinstance(names, list):
        return [str(item) for item in names]
    return [str(item) for item in summary.get("classes") or []]


def _training_images(yaml_path: Path, config: dict, summary: dict) -> tuple[list[Path], Path | None]:
    split = (summary.get("splits") or {}).get("train") or {}
    image_path = Path(split.get("image_path")) if split.get("image_path") else None
    label_path = Path(split.get("label_path")) if split.get("label_path") else None
    root = _dataset_root(yaml_path, config)
    if image_path is None:
        raw = config.get("train")
        if isinstance(raw, str):
            image_path = Path(raw).expanduser()
            if not image_path.is_absolute():
                image_path = root / image_path
    if image_path is None or not image_path.is_dir():
        return [], label_path
    if label_path is None:
        parts = list(image_path.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            label_path = Path(*parts)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in sorted(image_path.rglob("*")) if p.is_file() and p.suffix.lower() in extensions], label_path


def _sample_images_by_class(
    yaml_path: Path,
    summary: dict,
    limit: int = 3,
) -> tuple[list[str], dict[int, list[tuple[Path, Path]]]]:
    config = _load_yaml(yaml_path)
    names = _class_names(config, summary)
    images, labels_root = _training_images(yaml_path, config, summary)
    selected = {class_id: [] for class_id in range(len(names))}
    if labels_root is None:
        return names, selected
    for image_path in images:
        if all(len(items) >= limit for items in selected.values()):
            break
        try:
            relative = image_path.relative_to(Path((summary.get("splits") or {}).get("train", {}).get("image_path", image_path.parent)))
            label_path = (labels_root / relative).with_suffix(".txt")
        except ValueError:
            label_path = labels_root / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        class_ids = set()
        for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if not fields:
                continue
            try:
                class_ids.add(int(float(fields[0])))
            except ValueError:
                continue
        for class_id in class_ids:
            if class_id in selected and len(selected[class_id]) < limit:
                selected[class_id].append((image_path, label_path))
    return names, selected


def _annotated_thumbnail(image_path: Path, label_path: Path, names: list[str], output_path: Path) -> Path | None:
    try:
        from PIL import Image as PILImage, ImageDraw, ImageFont, ImageOps

        with PILImage.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        line_width = max(2, round(min(width, height) / 180))
        font = ImageFont.load_default()
        for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                class_id = int(float(fields[0]))
                center_x, center_y, box_width, box_height = map(float, fields[1:5])
            except ValueError:
                continue
            x1 = max(0, (center_x - box_width / 2) * width)
            y1 = max(0, (center_y - box_height / 2) * height)
            x2 = min(width, (center_x + box_width / 2) * width)
            y2 = min(height, (center_y + box_height / 2) * height)
            color = (20, 145, 120)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
            label = names[class_id] if 0 <= class_id < len(names) else str(class_id)
            text_box = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle(text_box, fill=color)
            draw.text((x1, y1), label, fill="white", font=font)
        image.thumbnail((800, 520))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, "JPEG", quality=82, optimize=True)
        return output_path
    except (OSError, ValueError):
        return None


class _ReportBuilder:
    def __init__(self, output_path: Path, title: str):
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate

        self.colors = colors
        self.mm = mm
        self.output_path = output_path
        self.story = []
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#173b36"), spaceAfter=14))
        styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], textColor=colors.HexColor("#176b5b"), spaceBefore=12, spaceAfter=7))
        styles.add(ParagraphStyle(name="Subsection", parent=styles["Heading3"], textColor=colors.HexColor("#24333f"), spaceBefore=8, spaceAfter=5))
        styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
        self.styles = styles
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.document = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=15 * mm, leftMargin=15 * mm, topMargin=16 * mm, bottomMargin=16 * mm, title=title)
        self.title = title

    def paragraph(self, value, style="BodyText"):
        from reportlab.platypus import Paragraph

        self.story.append(Paragraph(str(value), self.styles[style]))

    def heading(self, value, level=2):
        self.paragraph(_text(value), "Section" if level == 2 else "Subsection")

    def table(self, rows, widths=None, header=True):
        from reportlab.platypus import Paragraph, Table, TableStyle

        if not rows:
            return
        formatted = []
        for row_index, row in enumerate(rows):
            formatted.append([
                Paragraph(
                    f"<b>{_text(cell)}</b>" if header and row_index == 0 else _text(cell),
                    self.styles["Small"],
                )
                for cell in row
            ])
        table = Table(formatted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
        commands = [
            ("GRID", (0, 0), (-1, -1), 0.35, self.colors.HexColor("#cbd5df")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("LEADING", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [self.colors.white, self.colors.HexColor("#f7f9fa")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        if header:
            commands.extend([
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), self.colors.HexColor("#e7f3ef")),
            ])
        table.setStyle(TableStyle(commands))
        self.story.append(table)

    def image(self, path: Path, max_width_mm=175, max_height_mm=105):
        from reportlab.lib.utils import ImageReader
        from reportlab.platypus import Image as ReportImage

        if not path.is_file():
            return False
        reader = ImageReader(str(path))
        width, height = reader.getSize()
        scale = min(max_width_mm * self.mm / width, max_height_mm * self.mm / height)
        self.story.append(ReportImage(str(path), width=width * scale, height=height * scale))
        return True

    def page_break(self):
        from reportlab.platypus import PageBreak

        self.story.append(PageBreak())

    def build(self):
        from reportlab.lib.units import mm

        def footer(canvas, document):
            canvas.saveState()
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(self.colors.HexColor("#687987"))
            canvas.drawString(15 * mm, 9 * mm, self.title)
            canvas.drawRightString(195 * mm, 9 * mm, f"Page {document.page}")
            canvas.restoreState()

        self.document.build(self.story, onFirstPage=footer, onLaterPages=footer)


def _add_dataset(builder: _ReportBuilder, run_dir: Path, context: dict):
    summary = context.get("dataset_summary") or {}
    yaml_value = context.get("dataset_yaml") or ""
    builder.heading("Dataset Information")
    builder.table([
        ["Dataset YAML", yaml_value or "Unavailable"],
        ["Dataset root", summary.get("dataset_root", "Unavailable")],
        ["Total images", summary.get("total_images", "N/A")],
        ["Classes", summary.get("class_count", len(summary.get("classes") or []))],
        ["Split strategy", summary.get("split_strategy", "N/A")],
        ["Split seed", summary.get("split_seed", "N/A")],
    ], widths=[42 * builder.mm, 133 * builder.mm])
    split_rows = [["Split", "Images", "Missing labels", "Configured ratio"]]
    ratios = summary.get("split_ratios") or {}
    for split in ("train", "val", "test"):
        entry = (summary.get("splits") or {}).get(split) or {}
        split_rows.append([split.title(), entry.get("images", 0), entry.get("missing_labels", 0), f"{ratios.get(split, 0)}%"])
    builder.table(split_rows, widths=[42 * builder.mm] * 4)

    distribution = summary.get("class_distribution") or []
    if distribution:
        builder.heading("Class Distribution", 3)
        builder.table(
            [["Class", "Images", "Instances"]]
            + [[row.get("class_name"), row.get("images", 0), row.get("instances", 0)] for row in distribution],
            widths=[85 * builder.mm, 45 * builder.mm, 45 * builder.mm],
        )

    yaml_path = Path(yaml_value).expanduser() if yaml_value else None
    if yaml_path is None or not yaml_path.is_file():
        builder.paragraph("Dataset samples are unavailable because the original dataset path cannot be accessed.")
        return
    builder.heading("Training Dataset Samples by Class")
    names, selections = _sample_images_by_class(yaml_path, summary)
    assets_dir = run_dir / "report_assets" / "samples"
    from reportlab.platypus import Image as ReportImage, Table, TableStyle
    from PIL import Image as PILImage

    for class_id, class_name in enumerate(names):
        builder.heading(class_name, 3)
        cells = []
        for index, (image_path, label_path) in enumerate(selections.get(class_id, []), start=1):
            target = assets_dir / f"class-{class_id}-{index}.jpg"
            if not target.is_file():
                _annotated_thumbnail(image_path, label_path, names, target)
            if target.is_file():
                with PILImage.open(target) as sample_image:
                    width, height = sample_image.size
                scale = min(54 * builder.mm / width, 36 * builder.mm / height)
                cells.append(ReportImage(str(target), width=width * scale, height=height * scale))
        if not cells:
            builder.paragraph("No labeled training sample was available for this class.", "Small")
            continue
        sample_table = Table([cells], colWidths=[58 * builder.mm] * len(cells), hAlign="LEFT")
        sample_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        builder.story.append(sample_table)


def _add_training(builder: _ReportBuilder, run_dir: Path, context: dict, metrics: dict):
    builder.paragraph("YOLOv8 Model Training Report", "ReportTitle")
    builder.paragraph(f"Generated {datetime.now().astimezone().isoformat(timespec='seconds')}", "Small")
    builder.heading("Run Overview")
    started_at, completed_at, duration = _run_timing(run_dir, context)
    builder.table([
        ["Run directory", run_dir],
        ["Started", started_at],
        ["Completed", completed_at],
        ["Duration", duration],
        ["Model", context.get("model")],
        ["Pretrained weights", context.get("pretrained", True)],
        ["Device", context.get("device")],
        ["Completed epoch", metrics.get("epoch")],
    ], widths=[45 * builder.mm, 130 * builder.mm])
    environment = context.get("environment") or {}
    if environment:
        gpu_names = ", ".join(str(item.get("name")) for item in environment.get("gpus", []) if item.get("name")) or "Unavailable"
        builder.heading("Runtime Environment", 3)
        builder.table([
            ["Python", environment.get("python")],
            ["Ultralytics", environment.get("ultralytics")],
            ["PyTorch", environment.get("torch")],
            ["GPU", gpu_names],
            ["Platform", environment.get("platform")],
        ], widths=[45 * builder.mm, 130 * builder.mm], header=False)
    _add_dataset(builder, run_dir, context)

    builder.heading("Hyperparameters")
    hyperparameters = context.get("hyperparameters") or _load_yaml(run_dir / "args.yaml")
    ignored = {"dataset_yaml", "project", "name"}
    rows = [["Parameter", "Value"]] + [[key, value] for key, value in sorted(hyperparameters.items()) if key not in ignored]
    builder.table(rows, widths=[70 * builder.mm, 105 * builder.mm])

    builder.heading("Training and Validation Results")
    builder.table([
        ["Metric", "Final value"],
        ["Training loss", _metric(metrics.get("training_loss"))],
        ["Validation loss", _metric(metrics.get("testing_loss"))],
        ["Validation precision", _metric(metrics.get("precision"))],
        ["Validation recall", _metric(metrics.get("recall"))],
        ["Validation mAP50", _metric(metrics.get("map50"))],
        ["Validation mAP50-95", _metric(metrics.get("map50_95"))],
        ["Validation macro F1", _metric(metrics.get("macro_f1"))],
        ["Validation weighted F1", _metric(metrics.get("weighted_f1"))],
    ], widths=[90 * builder.mm, 85 * builder.mm])
    best = metrics.get("best") or {}
    best_rows = [["Criterion", "Epoch", "Value"]]
    for key, label, value_key in (
        ("best_map50", "Best mAP50", "map50"),
        ("best_map50_95", "Best mAP50-95", "map50_95"),
        ("lowest_training_loss", "Lowest training loss", "training_loss"),
        ("lowest_validation_loss", "Lowest validation loss", "testing_loss"),
    ):
        row = best.get(key)
        if row:
            best_rows.append([label, row.get("epoch"), _metric(row.get(value_key))])
    if len(best_rows) > 1:
        builder.heading("Best Epochs", 3)
        builder.table(best_rows, widths=[85 * builder.mm, 35 * builder.mm, 55 * builder.mm])

    classes = metrics.get("per_class") or []
    if classes:
        builder.heading("Per-Class Validation Metrics", 3)
        builder.table(
            [["Class", "Instances", "Precision", "Recall", "F1", "AP50", "AP50-95"]]
            + [[row.get("class_name"), row.get("instances", 0), _metric(row.get("precision")), _metric(row.get("recall")), _metric(row.get("f1")), _metric(row.get("map50")), _metric(row.get("map50_95"))] for row in classes],
            widths=[40 * builder.mm, 22 * builder.mm, 23 * builder.mm, 22 * builder.mm, 21 * builder.mm, 23 * builder.mm, 25 * builder.mm],
        )
    auc_classes = (metrics.get("roc_auc") or {}).get("classes") or []
    if auc_classes:
        builder.heading("Per-Class Validation ROC-AUC", 3)
        builder.table(
            [["Class", "Positive images", "Negative images", "AUC"]]
            + [[row.get("class_name"), row.get("positive_images", 0), row.get("negative_images", 0), _metric(row.get("auc"))] for row in auc_classes],
            widths=[70 * builder.mm, 38 * builder.mm, 38 * builder.mm, 29 * builder.mm],
        )

    builder.heading("Training and Validation Plots")
    for filename, caption in (
        ("accuracy_by_epoch.png", "Detection performance by epoch"),
        ("loss_by_epoch.png", "Training and validation loss by epoch"),
        ("confusion_matrix_normalized.png", "Normalized validation confusion matrix"),
        ("confusion_matrix.png", "Validation confusion matrix (raw counts)"),
        ("roc_auc_curve.png", "Validation ROC-AUC by class"),
    ):
        path = run_dir / filename
        if path.is_file():
            builder.heading(caption, 3)
            builder.image(path)


def _add_test(builder: _ReportBuilder, test_dir: Path, context: dict, metrics: dict, validation: dict):
    builder.page_break()
    builder.paragraph("Independent Test Evaluation", "ReportTitle")
    builder.heading("Test Run Overview")
    parameters = context.get("parameters") or {}
    builder.table([
        ["Test run directory", test_dir],
        ["Evaluated split", metrics.get("split", context.get("dataset_split", "test"))],
        ["Dataset source", context.get("dataset_source")],
        ["Dataset YAML", metrics.get("dataset_yaml", context.get("dataset_yaml"))],
        ["Weights", metrics.get("weights", context.get("weights_label"))],
        ["Image size", parameters.get("imgsz")],
        ["Batch size", parameters.get("batch")],
        ["Workers", parameters.get("workers")],
        ["Device", parameters.get("device")],
    ], widths=[45 * builder.mm, 130 * builder.mm])
    test_summary = context.get("dataset_summary") or {}
    test_split = (test_summary.get("splits") or {}).get("test") or {}
    if test_summary:
        builder.heading("Test Dataset Summary")
        builder.table([
            ["Test images", test_split.get("images", test_summary.get("total_images", "N/A"))],
            ["Classes", test_summary.get("class_count", len(test_summary.get("classes") or []))],
            ["Missing labels", test_split.get("missing_labels", test_summary.get("missing_labels", 0))],
        ], widths=[75 * builder.mm, 100 * builder.mm], header=False)
        distribution = test_split.get("class_distribution") or test_summary.get("class_distribution") or []
        if distribution:
            builder.table(
                [["Class", "Images", "Instances"]]
                + [[row.get("class_name"), row.get("images", 0), row.get("instances", 0)] for row in distribution],
                widths=[85 * builder.mm, 45 * builder.mm, 45 * builder.mm],
            )
    builder.heading("Validation vs Test Comparison")
    builder.table([
        ["Metric", "Validation", "Test"],
        ["Precision", _metric(validation.get("precision")), _metric(metrics.get("precision"))],
        ["Recall", _metric(validation.get("recall")), _metric(metrics.get("recall"))],
        ["mAP50", _metric(validation.get("map50")), _metric(metrics.get("map50"))],
        ["mAP50-95", _metric(validation.get("map50_95")), _metric(metrics.get("map50_95"))],
        ["Macro F1", _metric(validation.get("macro_f1")), _metric(metrics.get("macro_f1"))],
        ["Weighted F1", _metric(validation.get("weighted_f1")), _metric(metrics.get("weighted_f1"))],
    ], widths=[70 * builder.mm, 52 * builder.mm, 53 * builder.mm])
    classes = metrics.get("per_class") or []
    if classes:
        builder.heading("Per-Class Test Metrics")
        builder.table(
            [["Class", "Instances", "Precision", "Recall", "F1", "AP50", "AP50-95"]]
            + [[row.get("class_name"), row.get("instances", 0), _metric(row.get("precision")), _metric(row.get("recall")), _metric(row.get("f1")), _metric(row.get("map50")), _metric(row.get("map50_95"))] for row in classes],
            widths=[40 * builder.mm, 22 * builder.mm, 23 * builder.mm, 22 * builder.mm, 21 * builder.mm, 23 * builder.mm, 25 * builder.mm],
        )
    auc_classes = (metrics.get("roc_auc") or {}).get("classes") or []
    if auc_classes:
        builder.heading("Per-Class Test ROC-AUC")
        builder.table(
            [["Class", "Positive images", "Negative images", "AUC"]]
            + [[row.get("class_name"), row.get("positive_images", 0), row.get("negative_images", 0), _metric(row.get("auc"))] for row in auc_classes],
            widths=[70 * builder.mm, 38 * builder.mm, 38 * builder.mm, 29 * builder.mm],
        )
    builder.heading("Test Evaluation Plots")
    found = False
    for filename, caption in (
        ("confusion_matrix_normalized.png", "Normalized test confusion matrix"),
        ("confusion_matrix.png", "Test confusion matrix (raw counts)"),
        ("roc_auc_curve.png", "Test ROC-AUC by class"),
    ):
        path = test_dir / filename
        if path.is_file():
            found = True
            builder.heading(caption, 3)
            builder.image(path)
    if not found:
        builder.paragraph("No test plots were available for this run.")


def generate_training_report(run_dir: Path, context: dict, metrics: dict) -> Path:
    output_path = run_dir / "training_report.pdf"
    builder = _ReportBuilder(output_path, "YOLOv8 Training Report")
    _add_training(builder, run_dir, context, metrics)
    builder.build()
    return output_path


def generate_combined_report(
    training_dir: Path,
    training_context: dict,
    training_metrics: dict,
    test_dir: Path,
    test_context: dict,
    test_metrics: dict,
) -> Path:
    output_path = test_dir / "training_and_test_report.pdf"
    builder = _ReportBuilder(output_path, "YOLOv8 Training and Test Report")
    _add_training(builder, training_dir, training_context, training_metrics)
    _add_test(builder, test_dir, test_context, test_metrics, training_metrics)
    builder.build()
    return output_path
