from datetime import datetime, timezone

import pytest


def load_web_app():
    pytest.importorskip("fastapi")
    from vision.ai.web import app as web_app

    return web_app


def test_timestamped_training_run_name_uses_myt_time():
    web_app = load_web_app()
    utc_time = datetime(2026, 7, 7, 7, 30, tzinfo=timezone.utc)

    assert web_app.timestamped_training_run_name("train", now=utc_time) == "train-20260707-153000"
    assert web_app.timestamped_training_run_name("pineapple-v1", now=utc_time) == "pineapple-v1-20260707-153000"


def test_timestamped_training_run_name_keeps_resume_names_unchanged():
    web_app = load_web_app()

    assert web_app.timestamped_training_run_name("train-20260707-153000", resume=True) == "train-20260707-153000"


def test_timestamped_training_run_name_refreshes_existing_timestamp_suffix():
    web_app = load_web_app()
    utc_time = datetime(2026, 7, 7, 8, 0, tzinfo=timezone.utc)

    assert (
        web_app.timestamped_training_run_name("train-20260707-153000", now=utc_time)
        == "train-20260707-160000"
    )
