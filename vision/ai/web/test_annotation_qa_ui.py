"""Regression tests for the guided Annotation QA frontend workflow."""

from collections import Counter
from html.parser import HTMLParser
from pathlib import Path


WEB_DIR = Path(__file__).parent
INDEX_HTML = WEB_DIR / "static" / "index.html"
APP_JS = WEB_DIR / "static" / "app.js"
APP_PY = WEB_DIR / "app.py"


class IdParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []

    def handle_starttag(self, _tag, attrs):
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.append(attributes["id"])


def test_annotation_qa_has_a_dedicated_guided_workflow():
    markup = INDEX_HTML.read_text(encoding="utf-8")

    assert 'data-app-tab="annotation-qa"' in markup
    assert 'aria-controls="annotation-qa-panel"' in markup
    for stage in ("config", "run", "review", "finalize"):
        assert f'id="qa-stage-indicator-{stage}"' in markup
    assert 'id="annotation-qa-configuration-explainer"' in markup
    assert 'Advanced model and safety settings' in markup


def test_annotation_qa_exposes_actionable_queues_and_filters():
    markup = INDEX_HTML.read_text(encoding="utf-8")

    for queue in ("needs_review", "audits", "safe_suggestions", "manual", "resolved", "all"):
        assert f'data-qa-queue="{queue}"' in markup
    for control in (
        "annotation-qa-search",
        "annotation-qa-filter-severity",
        "annotation-qa-filter-split",
        "annotation-qa-filter-class",
        "annotation-qa-filter-type",
        "annotation-qa-page-prev",
        "annotation-qa-page-next",
    ):
        assert f'id="{control}"' in markup


def test_review_dialog_supports_fast_safe_decisions():
    markup = INDEX_HTML.read_text(encoding="utf-8")
    script = APP_JS.read_text(encoding="utf-8")

    assert "Keep original box" in markup
    assert "Use SAM suggestion" in markup
    assert "Needs manual correction" in markup
    assert "triage label only" in markup
    assert 'id="qa-review-auto-advance"' in markup
    assert 'id="qa-undo-button"' in markup
    assert 'id="qa-review-zoom-in"' in markup
    assert 'id="qa-overlay-yolo"' in markup
    assert 'id="qa-overlay-mask"' in markup
    assert 'id="qa-overlay-sam"' in markup
    assert "function annotationQaFilteredIssues" in script
    assert "function decideAnnotationQaStatus" in script
    assert "function undoAnnotationQaDecision" in script
    assert "function setAnnotationQaZoom" in script
    assert 'event.key === "Tab"' in script


def test_preview_assets_and_safe_bulk_review_are_supported():
    markup = INDEX_HTML.read_text(encoding="utf-8")
    script = APP_JS.read_text(encoding="utf-8")
    app_source = APP_PY.read_text(encoding="utf-8")

    assert 'id="annotation-qa-bulk-accept-sam"' in markup
    assert "function issueCanBulkAcceptSamBox" in script
    assert "function bulkAcceptAnnotationQaSamBoxes" in script
    assert 'issue.audit_status === "pending"' in script
    assert 'issue["raw_preview"]' in app_source
    assert 'issue["mask_preview"]' in app_source


def test_roboflow_publish_is_bound_previewed_and_confirmed():
    markup = INDEX_HTML.read_text(encoding="utf-8")
    script = APP_JS.read_text(encoding="utf-8")
    app_source = APP_PY.read_text(encoding="utf-8")

    for control in (
        "annotation-qa-roboflow-target",
        "preview-annotation-qa-roboflow",
        "publish-annotation-qa-roboflow",
        "annotation-qa-roboflow-results",
    ):
        assert f'id="{control}"' in markup
    assert "The destination is locked" in markup
    assert "function previewAnnotationQaRoboflow" in script
    assert "function publishAnnotationQaRoboflow" in script
    assert "window.confirm" in script
    assert 'request.preview_id' in app_source
    assert 'request.confirmed' in app_source


def test_annotation_qa_markup_has_unique_ids():
    parser = IdParser()
    parser.feed(INDEX_HTML.read_text(encoding="utf-8"))
    duplicates = [name for name, count in Counter(parser.ids).items() if count > 1]

    assert not duplicates
