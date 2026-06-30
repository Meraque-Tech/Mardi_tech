from pathlib import Path
import re
import threading

import pytest

WEB_DIR = Path(__file__).parent
APP_PY = WEB_DIR / "app.py"
INDEX_HTML = WEB_DIR / "static" / "index.html"
APP_JS = WEB_DIR / "static" / "app.js"


def load_web_app():
    pytest.importorskip("fastapi")
    from vision.ai.web import app as web_app

    return web_app


def configure_inference_workspace(monkeypatch, tmp_path: Path):
    web_app = load_web_app()
    data_root = tmp_path / "datasets"
    inference_root = data_root / "inference"
    upload_root = inference_root / "uploads"
    job_root = inference_root / "jobs"
    monkeypatch.setattr(web_app, "DATA_ROOT", data_root)
    monkeypatch.setattr(web_app, "INFERENCE_ROOT", inference_root)
    monkeypatch.setattr(web_app, "INFERENCE_UPLOAD_ROOT", upload_root)
    monkeypatch.setattr(web_app, "INFERENCE_JOB_ROOT", job_root)
    monkeypatch.setitem(web_app.STORAGE_CLEANUP_TARGETS["inference_uploads"], "path", upload_root)
    monkeypatch.setitem(web_app.STORAGE_CLEANUP_TARGETS["inference_outputs"], "path", job_root)
    with web_app.inference_jobs_lock:
        web_app.inference_jobs.clear()
    return inference_root, upload_root, job_root


def test_inference_cleanup_ui_and_routes_are_exposed():
    app_source = APP_PY.read_text(encoding="utf-8")
    markup = INDEX_HTML.read_text(encoding="utf-8")
    script = APP_JS.read_text(encoding="utf-8")

    assert '@app.get("/api/inference/storage")' in app_source
    assert '@app.delete("/api/inference/outputs")' in app_source
    assert '@app.get("/api/storage/{target_key}")' in app_source
    assert '@app.delete("/api/storage/{target_key}")' in app_source
    assert 'id="clear-inference-output"' in markup
    assert 'id="clear-inference-uploads"' in markup
    assert 'id="clear-dataset-uploads"' in markup
    assert 'id="clear-dataset-extracted"' in markup
    assert 'id="clear-dataset-prepared"' in markup
    assert "function clearInferenceOutput" in script
    assert "function clearStorageTarget" in script
    assert "/api/inference/storage" in script
    assert "/api/inference/outputs" in script
    assert "/api/storage/" in script


def test_storage_cleanup_buttons_use_backend_targets():
    web_app = load_web_app()
    script = APP_JS.read_text(encoding="utf-8")
    keys = set(re.findall(r'clearStorageTarget\("([^"]+)"', script))

    assert keys
    assert keys <= set(web_app.STORAGE_CLEANUP_TARGETS)
    for key in keys:
        assert web_app.normalize_storage_target_key(key) in web_app.STORAGE_CLEANUP_TARGETS


def test_clear_inference_outputs_deletes_only_job_outputs(monkeypatch, tmp_path: Path):
    web_app = load_web_app()
    from fastapi.testclient import TestClient

    _inference_root, upload_root, job_root = configure_inference_workspace(monkeypatch, tmp_path)
    job_a = job_root / "20260625-111727-10198537"
    job_b = job_root / "20260625-112647-cc129da5"
    job_a.mkdir(parents=True)
    job_b.mkdir(parents=True)
    upload_root.mkdir(parents=True)
    (job_a / "annotated.mp4").write_bytes(b"a" * 16)
    (job_a / "result.json").write_text("{}", encoding="utf-8")
    (job_b / "preview.jpg").write_bytes(b"b" * 8)
    (upload_root / "weights.pt").write_bytes(b"keep")

    client = TestClient(web_app.app)
    response = client.delete("/api/inference/outputs")

    assert response.status_code == 200
    payload = response.json()
    assert payload["removed_jobs"] == 2
    assert payload["freed_bytes"] >= 26
    assert job_root.is_dir()
    assert list(job_root.iterdir()) == []
    assert (upload_root / "weights.pt").read_bytes() == b"keep"


def test_clear_inference_outputs_refuses_active_job(monkeypatch, tmp_path: Path):
    web_app = load_web_app()
    from fastapi.testclient import TestClient

    _inference_root, upload_root, job_root = configure_inference_workspace(monkeypatch, tmp_path)
    job_dir = job_root / "20260625-111727-10198537"
    job_dir.mkdir(parents=True)
    (job_dir / "annotated.mp4").write_bytes(b"output")
    upload_root.mkdir(parents=True)
    (upload_root / "weights.pt").write_bytes(b"keep")
    with web_app.inference_jobs_lock:
        web_app.inference_jobs["20260625-111727-10198537"] = {
            "job_id": "20260625-111727-10198537",
            "status": "processing",
            "preview_condition": threading.Condition(),
        }

    client = TestClient(web_app.app)
    response = client.delete("/api/inference/outputs")

    assert response.status_code == 409
    assert (job_dir / "annotated.mp4").exists()
    assert (upload_root / "weights.pt").exists()
    with web_app.inference_jobs_lock:
        web_app.inference_jobs.clear()


def test_clear_storage_target_deletes_only_selected_dataset_cache(monkeypatch, tmp_path: Path):
    web_app = load_web_app()
    data_root = tmp_path / "datasets"
    uploads_root = data_root / "uploads"
    extracted_root = data_root / "extracted"
    prepared_root = data_root / "prepared"
    inference_root = data_root / "inference"
    inference_upload_root = inference_root / "uploads"
    inference_job_root = inference_root / "jobs"
    monkeypatch.setattr(web_app, "DATA_ROOT", data_root)
    monkeypatch.setattr(web_app, "DATASET_UPLOAD_ROOT", uploads_root)
    monkeypatch.setattr(web_app, "DATASET_EXTRACTED_ROOT", extracted_root)
    monkeypatch.setattr(web_app, "DATASET_PREPARED_ROOT", prepared_root)
    monkeypatch.setattr(web_app, "INFERENCE_ROOT", inference_root)
    monkeypatch.setattr(web_app, "INFERENCE_UPLOAD_ROOT", inference_upload_root)
    monkeypatch.setattr(web_app, "INFERENCE_JOB_ROOT", inference_job_root)
    monkeypatch.setitem(web_app.STORAGE_CLEANUP_TARGETS["dataset_uploads"], "path", uploads_root)
    monkeypatch.setitem(web_app.STORAGE_CLEANUP_TARGETS["dataset_extracted"], "path", extracted_root)
    monkeypatch.setitem(web_app.STORAGE_CLEANUP_TARGETS["dataset_prepared"], "path", prepared_root)
    monkeypatch.setitem(web_app.STORAGE_CLEANUP_TARGETS["inference_uploads"], "path", inference_upload_root)
    monkeypatch.setitem(web_app.STORAGE_CLEANUP_TARGETS["inference_outputs"], "path", inference_job_root)
    web_app.ensure_dirs()
    (uploads_root / "dataset_a").mkdir()
    (uploads_root / "dataset_a" / "source.zip").write_bytes(b"delete")
    (extracted_root / "dataset_a").mkdir()
    (extracted_root / "dataset_a" / "data.yaml").write_text("keep", encoding="utf-8")
    (prepared_root / "dataset_a").mkdir()
    (prepared_root / "dataset_a" / "data.yaml").write_text("keep", encoding="utf-8")
    (inference_upload_root / "weights").mkdir(parents=True)
    (inference_upload_root / "weights" / "best.pt").write_bytes(b"weights")

    result = web_app.clear_storage_target("dataset_uploads")

    assert result["removed_items"] == 1
    assert result["freed_bytes"] >= 6
    assert uploads_root.is_dir()
    assert list(uploads_root.iterdir()) == []
    assert (extracted_root / "dataset_a" / "data.yaml").exists()
    assert (prepared_root / "dataset_a" / "data.yaml").exists()

    result = web_app.clear_storage_target("dataset_extracted")

    assert result["removed_items"] == 1
    assert result["freed_bytes"] >= 4
    assert extracted_root.is_dir()
    assert list(extracted_root.iterdir()) == []
    assert (prepared_root / "dataset_a" / "data.yaml").exists()

    result = web_app.clear_storage_target("dataset_prepared")

    assert result["removed_items"] == 1
    assert result["freed_bytes"] >= 4
    assert prepared_root.is_dir()
    assert list(prepared_root.iterdir()) == []

    result = web_app.clear_storage_target("inference_uploads")

    assert result["removed_items"] == 1
    assert result["freed_bytes"] >= 7
    assert inference_upload_root.is_dir()
    assert list(inference_upload_root.iterdir()) == []
