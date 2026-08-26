from __future__ import annotations

import stat
import zipfile
from pathlib import Path

import pytest

from vision.ai.web.common.cache import SignatureCache, TimedCache
from vision.ai.web.common.files import file_signature, read_text_tail
from vision.ai.web.common.model_cache import BoundedModelCache
from vision.ai.web.common.uploads import (
    UploadLimits,
    UploadValidationError,
    extract_zip_safely,
    safe_leaf_filename,
    validate_upload_totals,
)
from vision.ai.web.runtime.job_manager import (
    PersistentJobStore,
    ResourceBusyError,
    ResourceCoordinator,
)


WEB_DIR = Path(__file__).resolve().parent


def small_limits(**overrides) -> UploadLimits:
    values = {
        "max_file_bytes": 1024,
        "max_folder_bytes": 2048,
        "max_folder_files": 4,
        "max_zip_entries": 4,
        "max_zip_uncompressed_bytes": 2048,
        "max_zip_compression_ratio": 20.0,
    }
    values.update(overrides)
    return UploadLimits(**values)


def test_text_tail_reads_trailing_unicode_without_loading_contract_changes(tmp_path: Path):
    path = tmp_path / "large.log"
    path.write_text("prefix\n" + ("a" * 50_000) + "\n শেষ", encoding="utf-8")

    result = read_text_tail(path, max_chars=12)

    assert len(result) <= 12
    assert result.endswith("শেষ")
    assert read_text_tail(tmp_path / "missing.log") == ""


def test_file_signature_changes_with_content(tmp_path: Path):
    path = tmp_path / "metrics.json"
    assert file_signature(path) == (0, 0)
    path.write_text("{}", encoding="utf-8")
    first = file_signature(path)
    path.write_text('{"value": 1}', encoding="utf-8")
    assert file_signature(path) != first


def test_signature_and_timed_caches_copy_mutable_values():
    signature_cache = SignatureCache(max_items=2)
    signature_cache.put("run", (1, 2), {"history": [1]})
    cached = signature_cache.get("run", (1, 2))
    cached["history"].append(2)
    assert signature_cache.get("run", (1, 2)) == {"history": [1]}
    assert signature_cache.get("run", (2, 2)) is None

    timed = TimedCache(ttl_seconds=60)
    timed.put({"sessions": ["one"]})
    value = timed.get()
    value["sessions"].append("two")
    assert timed.get() == {"sessions": ["one"]}


def test_upload_names_and_totals_are_bounded():
    assert safe_leaf_filename("../../weights.pt") == "weights.pt"
    assert safe_leaf_filename(r"folder\dataset.zip") == "dataset.zip"
    with pytest.raises(UploadValidationError):
        safe_leaf_filename("../")
    assert validate_upload_totals([100, 200], small_limits(), folder=True) == 300
    with pytest.raises(UploadValidationError):
        validate_upload_totals([100] * 5, small_limits(), folder=True)
    with pytest.raises(UploadValidationError):
        validate_upload_totals([1025], small_limits())


def test_safe_zip_extraction_rejects_traversal_and_symlinks(tmp_path: Path):
    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("../outside.txt", "bad")
    with pytest.raises(UploadValidationError):
        extract_zip_safely(traversal, tmp_path / "traversal-output", small_limits())
    assert not (tmp_path / "outside.txt").exists()

    symlink = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("linked.txt")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(symlink, "w") as archive:
        archive.writestr(info, "target.txt")
    with pytest.raises(UploadValidationError):
        extract_zip_safely(symlink, tmp_path / "symlink-output", small_limits())


def test_safe_zip_extraction_enforces_ratio_and_reports_progress(tmp_path: Path):
    compressed = tmp_path / "compressed.zip"
    with zipfile.ZipFile(compressed, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("zeros.txt", b"0" * 1000)
    with pytest.raises(UploadValidationError):
        extract_zip_safely(
            compressed,
            tmp_path / "compressed-output",
            small_limits(max_zip_compression_ratio=2.0),
        )

    valid = tmp_path / "valid.zip"
    with zipfile.ZipFile(valid, "w") as archive:
        archive.writestr("images/example.txt", "content")
    progress = []
    extract_zip_safely(valid, tmp_path / "valid-output", small_limits(), progress=lambda *row: progress.append(row))
    assert (tmp_path / "valid-output/images/example.txt").read_text(encoding="utf-8") == "content"
    assert progress[-1][:2] == (1, 1)


def test_resource_coordinator_is_atomic_and_owner_scoped():
    coordinator = ResourceCoordinator()
    coordinator.acquire("dataset:corn", "job-one", "first preparation")
    coordinator.acquire("dataset:corn", "job-one", "first preparation")
    with pytest.raises(ResourceBusyError):
        coordinator.acquire("dataset:corn", "job-two", "second preparation")
    assert not coordinator.release("dataset:corn", "job-two")
    assert coordinator.release("dataset:corn", "job-one")
    assert coordinator.snapshot() == []


def test_persistent_jobs_survive_restart_and_active_jobs_become_interrupted(tmp_path: Path):
    path = tmp_path / "runtime/jobs.json"
    store = PersistentJobStore(path, write_interval_seconds=0)
    store.update("training:one", "training", "running", {"pid": 123}, force=True)
    store.update("inference:two", "inference", "complete", {"frames": 1}, force=True)

    restarted = PersistentJobStore(path, write_interval_seconds=0)
    assert restarted.recover_interrupted() == 1
    records = {record["job_id"]: record for record in restarted.list()}
    assert records["training:one"]["status"] == "interrupted"
    assert records["inference:two"]["status"] == "complete"


def test_model_cache_evicts_oldest_entry_and_runs_cleanup():
    cleaned = []
    cache = BoundedModelCache(max_items=2, cleanup=lambda entry: cleaned.append(entry["model"]))
    cache.get_or_create("one", lambda: {"model": "one"})
    cache.get_or_create("two", lambda: {"model": "two"})
    cache.get_or_create("one", lambda: {"model": "replacement"})
    cache.get_or_create("three", lambda: {"model": "three"})
    assert list(cache.entries) == ["one", "three"]
    assert cleaned == ["two"]


def test_frontend_uses_module_and_adaptive_polling():
    markup = (WEB_DIR / "static/index.html").read_text(encoding="utf-8")
    script = (WEB_DIR / "static/app.js").read_text(encoding="utf-8")
    polling = (WEB_DIR / "static/js/polling.js").read_text(encoding="utf-8")

    assert 'type="module"' in markup
    assert 'from "./js/polling.js"' in script
    assert "document.hidden" in script
    assert "window.setInterval(pollStatus, 2500)" not in script
    assert "ACTIVE_POLL_MS = 2500" in polling
    assert "IDLE_POLL_MS = 10000" in polling
