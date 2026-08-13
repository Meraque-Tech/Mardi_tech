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


def test_annotation_qa_loads_pages_and_does_not_hide_json_failures():
    script = APP_JS.read_text(encoding="utf-8")
    app_source = APP_PY.read_text(encoding="utf-8")

    assert "async function loadAnnotationQaIssuePage" in script
    assert "/api/annotation-qa/issues/" in script
    assert "The server returned an unreadable JSON response" in script
    api_helper = script[script.index("async function apiJson"):script.index("function errorDetailText")]
    assert "response.json().catch(() => ({}))" not in api_helper
    assert '@app.get("/api/annotation-qa/issues/{job_id}")' in app_source
    assert "annotation_qa_query_issues" in app_source
    assert "StreamingResponse(json_rows()" in app_source


def test_review_dialog_supports_fast_safe_decisions():
    markup = INDEX_HTML.read_text(encoding="utf-8")
    script = APP_JS.read_text(encoding="utf-8")

    assert "Keep original box" in markup
    assert "Use SAM suggestion" in markup
    assert 'id="qa-review-override-sam"' in markup
    assert "Use displayed SAM box anyway" in markup
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
    assert "function decideAnnotationQaSamOverride" in script
    assert "human_override: humanOverride" in script
    assert 'accepted_fix_source === "human_override"' in script
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


def test_sam3_is_exposed_with_memory_and_safety_controls():
    markup = INDEX_HTML.read_text(encoding="utf-8")
    script = APP_JS.read_text(encoding="utf-8")
    app_source = APP_PY.read_text(encoding="utf-8")

    assert '<option value="sam3">SAM 3' in markup
    assert 'id="annotation-qa-model-status"' in markup
    assert "function updateAnnotationQaModelStatus" in script
    assert "model.automatic_allowed" in script
    assert '"prompt_chunk": 8' in app_source
    assert '"max_side": 1008' in app_source
    assert "SAM 3 automatic correction is disabled" in app_source


def test_annotation_qa_markup_has_unique_ids():
    parser = IdParser()
    parser.feed(INDEX_HTML.read_text(encoding="utf-8"))
    duplicates = [name for name, count in Counter(parser.ids).items() if count > 1]

    assert not duplicates
