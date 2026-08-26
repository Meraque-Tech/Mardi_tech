"""Model test execution, status, artifacts, and report routes."""

from __future__ import annotations

import functools
import os
import signal
import subprocess
import threading
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from ..common.uploads import UploadLimits, UploadValidationError, extract_zip_safely, validate_upload_totals
from ..schemas import TestArtifactRequest


@dataclass(frozen=True)
class TestingRouterDependencies:
    data_root: Path
    log_dir: Path
    repo_root: Path
    test_script: Path
    rfdetr_test_script: Path
    test_log_file: Path
    test_context_filename: str
    training_python: str
    timezone: object
    upload_limits: UploadLimits
    compute_start_lock: object
    get_state: Callable[[], tuple]
    set_state: Callable[[object, object, object, object], None]
    callbacks: dict[str, Callable]


def create_testing_router(deps: TestingRouterDependencies) -> APIRouter:
    router = APIRouter()
    DATA_ROOT = deps.data_root
    LOG_DIR = deps.log_dir
    REPO_ROOT = deps.repo_root
    TEST_SCRIPT = deps.test_script
    RFDETR_TEST_SCRIPT = deps.rfdetr_test_script
    TEST_LOG_FILE = deps.test_log_file
    TEST_REPORT_CONTEXT_FILE = deps.test_context_filename
    TRAINING_PYTHON = deps.training_python
    MYT = deps.timezone
    for name, callback in deps.callbacks.items():
        locals()[name] = callback

    # Locals cannot be populated portably through locals().update inside a
    # function, so route helpers resolve these callbacks through module globals.
    globals().update(deps.callbacks)

    def serialized_compute_start(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with deps.compute_start_lock:
                return function(*args, **kwargs)
        return wrapper

    def upload_size(upload):
        position = upload.file.tell()
        upload.file.seek(0, os.SEEK_END)
        size = max(0, int(upload.file.tell()))
        upload.file.seek(position)
        return size

    def save_upload(upload, target, progress_callback, stage, detail_prefix):
        size = upload_size(upload)
        try:
            validate_upload_totals([size], deps.upload_limits)
        except UploadValidationError as exc:
            raise upload_validation_http_error(exc) from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        copied = 0
        upload.file.seek(0)
        with target.open("wb") as output:
            while chunk := upload.file.read(1024 * 1024):
                output.write(chunk)
                copied += len(chunk)
                progress_callback(stage, copied, size, f"{detail_prefix}: {copied} of {size} bytes")
        return copied

    @router.post("/api/test/start")
    @serialized_compute_start
    def start_test(
        weight_source: str = Form("trained"),
        project: str = Form("runs/detect"),
        name: str = Form("train"),
        prepared_dataset_yaml: str = Form(""),
        reference_dataset_yaml: str = Form(""),
        dataset_source: str = Form("prepared"),
        imgsz: int = Form(512),
        batch: int = Form(8),
        workers: int = Form(2),
        device: str = Form(""),
        weight_file: Optional[UploadFile] = File(None),
        dataset_zip: Optional[UploadFile] = File(None),
        dataset_files: Optional[list[UploadFile]] = File(None),
    ):
        test_process, test_started_at, test_log_file, test_run_info = deps.get_state()

        ensure_compute_available("model testing")

        ensure_dirs()
        test_name = f"test-{datetime.now(MYT).strftime('%Y%m%d-%H%M%S')}"
        test_project_path = normalize_training_project_path("runs/test")
        test_input_root = DATA_ROOT / "test_inputs" / test_name
        test_input_root.mkdir(parents=True, exist_ok=True)

        if weight_source == "trained":
            weights_path = resolve_weight_path(project, name, "best")
            weights_label = f"best.pt from {project}/{name}"
            source_training_run = weights_path.parent.parent.resolve()
        elif weight_source == "upload":
            if weight_file is None or not weight_file.filename:
                raise HTTPException(status_code=400, detail="Choose a .pt weights file to upload.")
            if Path(weight_file.filename).suffix.lower() != ".pt":
                raise HTTPException(status_code=400, detail="Uploaded weights must be a .pt file.")
            weights_dir = test_input_root / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            weights_path = weights_dir / Path(weight_file.filename).name
            save_upload(weight_file, weights_path, lambda *_args: None, "saving", "Saving weights")
            weights_label = f"uploaded weights {weights_path.name}"
            source_training_run = None
        else:
            raise HTTPException(status_code=400, detail="Unknown weight source.")

        test_backend = family_for_weight_path(weights_path) if source_training_run else "ultralytics"
        if test_backend == "dfine":
            raise HTTPException(
                status_code=400,
                detail="Model Testing is not available for D-FINE weights yet.",
            )
        if test_backend not in {"ultralytics", "rfdetr"}:
            test_backend = "ultralytics"
        if test_backend == "rfdetr" and source_training_run:
            training_metrics = read_web_metrics(source_training_run)
            if training_metrics.get("training_completed") is False:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "This RF-DETR run did not complete successfully. "
                        "Resume training or choose a completed RF-DETR run before testing."
                    ),
                )

        if dataset_source == "prepared":
            if not prepared_dataset_yaml:
                raise HTTPException(status_code=400, detail="Prepare a dataset before using the prepared test split.")
            dataset_yaml, dataset_info = resolve_prepared_test_dataset_yaml(prepared_dataset_yaml)
        elif dataset_source == "upload_zip":
            if dataset_zip is None or not dataset_zip.filename:
                raise HTTPException(status_code=400, detail="Choose a labeled YOLO ZIP file for testing.")
            if not dataset_zip.filename.lower().endswith(".zip"):
                raise HTTPException(status_code=400, detail="Uploaded test dataset must be a ZIP file.")
            upload_dir = test_input_root / "zip"
            extract_dir = test_input_root / "zip_extracted"
            upload_dir.mkdir(parents=True, exist_ok=True)
            extract_dir.mkdir(parents=True, exist_ok=True)
            zip_path = upload_dir / Path(dataset_zip.filename).name
            save_upload(dataset_zip, zip_path, lambda *_args: None, "saving", "Saving test ZIP")
            try:
                extract_zip_safely(zip_path, extract_dir, UPLOAD_LIMITS)
            except zipfile.BadZipFile as exc:
                raise HTTPException(status_code=400, detail="Uploaded test dataset is not a valid ZIP.") from exc
            except UploadValidationError as exc:
                raise upload_validation_http_error(exc) from exc
            dataset_yaml, dataset_info = prepare_custom_test_dataset(
                extract_dir,
                f"{test_name}-dataset",
                reference_dataset_yaml,
            )
        elif dataset_source == "upload_folder":
            files = dataset_files or []
            if not files:
                raise HTTPException(status_code=400, detail="Choose a labeled YOLO folder for testing.")
            try:
                validate_upload_totals(
                    [upload_size(upload) for upload in files],
                    UPLOAD_LIMITS,
                    folder=True,
                )
            except UploadValidationError as exc:
                raise upload_validation_http_error(exc) from exc
            upload_dir = test_input_root / "folder"
            upload_dir.mkdir(parents=True, exist_ok=True)
            upload_root = upload_dir.resolve()
            for upload in files:
                relative_path = safe_upload_path(upload.filename or "")
                target = (upload_dir / relative_path).resolve()
                try:
                    target.relative_to(upload_root)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=f"Invalid upload path: {upload.filename}") from exc
                save_upload(upload, target, lambda *_args: None, "saving", "Saving test file")
            dataset_yaml, dataset_info = prepare_custom_test_dataset(
                upload_dir,
                f"{test_name}-dataset",
                reference_dataset_yaml,
            )
        else:
            raise HTTPException(status_code=400, detail="Unknown dataset source.")

        test_run_info = {
            "project": str(test_project_path),
            "name": test_name,
            "expected_run_dir": str(test_project_path / test_name),
            "run_dir": "",
            "weights_source": weight_source,
            "weights_label": weights_label,
            "backend": test_backend,
            "dataset_source": dataset_source,
            "dataset_yaml": str(dataset_yaml),
            "dataset_root": dataset_info.get("dataset_root", ""),
            "dataset_split": dataset_info.get("source_split", ""),
            "progress_percent": 0,
            "progress_stage": "starting",
            "progress_detail": "Launching the test job.",
            "report_context": {
                "created_at": datetime.now(MYT).isoformat(),
                "weight_source": weight_source,
                "weights_label": weights_label,
                "weights_path": str(weights_path),
                "training_run_dir": str(source_training_run) if source_training_run else "",
                "backend": test_backend,
                "dataset_source": dataset_source,
                "dataset_yaml": str(dataset_yaml),
                "dataset_root": dataset_info.get("dataset_root", ""),
                "dataset_split": dataset_info.get("source_split", "test"),
                "parameters": {
                    "imgsz": imgsz,
                    "batch": batch,
                    "workers": workers,
                    "device": device or "auto",
                },
            },
        }

        TEST_LOG_FILE.write_text("", encoding="utf-8")
        timestamp = datetime.now(MYT).strftime("%Y%m%d-%H%M%S")
        test_log_file = LOG_DIR / f"test-{timestamp}.log"
        test_script = RFDETR_TEST_SCRIPT if test_backend == "rfdetr" else TEST_SCRIPT
        cmd = [
            TRAINING_PYTHON,
            str(test_script),
            "--weights",
            str(weights_path),
            "--data",
            str(dataset_yaml),
            "--imgsz",
            str(imgsz),
            "--batch",
            str(batch),
            "--workers",
            str(workers),
            "--split",
            "test",
            "--project",
            str(test_project_path),
            "--name",
            test_name,
        ]
        if device.strip():
            cmd.extend(["--device", device.strip()])

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        header = (
            "Model testing started.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Weights: {weights_label}\n"
            f"Backend: {test_backend}\n"
            f"Dataset: {dataset_info.get('source_split', 'test')} split from {dataset_info.get('dataset_root', dataset_yaml)}\n\n"
        )
        TEST_LOG_FILE.write_text(header, encoding="utf-8")
        test_log_file.write_text(header, encoding="utf-8")
        test_process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            text=True,
            bufsize=1,
        )
        persist_job(
            f"testing:{test_name}",
            "testing",
            "running",
            {"pid": test_process.pid, "run": {key: value for key, value in test_run_info.items() if key != "report_context"}},
            force=True,
        )
        threading.Thread(
            target=stream_test_logs,
            args=(test_process, [TEST_LOG_FILE, test_log_file]),
            daemon=True,
        ).start()
        test_started_at = time.time()
        deps.set_state(test_process, test_started_at, test_log_file, test_run_info)

        return {
            "message": "Model testing started.",
            "pid": test_process.pid,
            "command": cmd,
            "log_file": str(TEST_LOG_FILE),
            "history_log_file": str(test_log_file),
            "test_run": {key: value for key, value in test_run_info.items() if key != "report_context"},
        }


    @router.post("/api/test/stop")
    def stop_test():
        test_process, _started_at, _log_file, _run_info = deps.get_state()

        if test_process is None or test_process.poll() is not None:
            deps.set_state(None, _started_at, _log_file, _run_info)
            return {"message": "No model-testing process is running."}

        os.killpg(os.getpgid(test_process.pid), signal.SIGTERM)
        return {"message": "Stop signal sent to the test job."}


    @router.get("/api/test/status")
    def test_status():
        status = current_test_status()
        status["log_tail"] = read_log_tail_from(TEST_LOG_FILE, 4000)
        status["gpu"] = gpu_status()
        return status


    @router.get("/api/test/logs", response_class=PlainTextResponse)
    def test_logs():
        return read_log_tail_from(TEST_LOG_FILE)


    @router.get("/api/test/logs/download")
    def download_test_log():
        path = test_log_file if test_log_file and test_log_file.is_file() else latest_timestamped_test_log()
        if path is None and TEST_LOG_FILE.is_file():
            path = TEST_LOG_FILE
        if not path:
            raise HTTPException(status_code=404, detail="No test log found.")
        return FileResponse(path, media_type="text/plain", filename=path.name)


    @router.get("/api/test/results")
    def test_results():
        run_dir = current_test_run_dir()
        if run_dir is None:
            return {"available": False, "run_dir": "", "artifacts": {}}
        return read_test_metrics(run_dir)


    @router.post("/api/test/artifacts/download")
    def download_test_artifact(request: TestArtifactRequest):
        path = resolve_test_artifact_path(request.artifact)
        media_type = "application/json" if path.suffix == ".json" else "image/png"
        return FileResponse(path, media_type=media_type, filename=path.name)


    @router.post("/api/test/report/download")
    def download_combined_test_report():
        if current_test_status()["running"]:
            raise HTTPException(status_code=409, detail="Wait for model testing to finish before generating the report.")
        test_dir = current_test_run_dir()
        if test_dir is None:
            raise HTTPException(status_code=404, detail="No completed test run was found.")
        test_metrics_payload = read_test_metrics(test_dir)
        if not test_metrics_payload.get("available"):
            raise HTTPException(status_code=409, detail="The selected test run has no completed metrics.")
        test_context = read_json_object(test_dir / TEST_REPORT_CONTEXT_FILE)
        test_dataset_value = test_context.get("dataset_yaml") or test_metrics_payload.get("dataset_yaml")
        if test_dataset_value and not test_context.get("dataset_summary"):
            test_dataset_path = Path(str(test_dataset_value)).expanduser()
            if test_dataset_path.is_file():
                test_context["dataset_summary"] = cached_dataset_summary(test_dataset_path)
        training_dir_value = test_context.get("training_run_dir")
        if not training_dir_value:
            raise HTTPException(
                status_code=409,
                detail="A combined report is unavailable because this test used uploaded weights with no linked training run.",
            )
        training_dir = Path(training_dir_value).expanduser().resolve()
        ensure_runs_path(training_dir)
        ensure_model_report_artifacts_for_report(training_dir)
        training_metrics_payload = read_run_metrics(training_dir)
        if not training_metrics_payload.get("available"):
            raise HTTPException(status_code=409, detail="The linked training run has no completed metrics.")
        try:
            from ..report_generator import generate_combined_report

            report_path = generate_combined_report(
                training_dir,
                load_training_report_context(training_dir),
                training_metrics_payload,
                test_dir,
                test_context,
                test_metrics_payload,
            )
        except ImportError as exc:
            raise HTTPException(status_code=500, detail="PDF reporting requires the reportlab dependency. Rebuild the container image.") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not generate the combined report: {exc}") from exc
        return FileResponse(
            report_path,
            media_type="application/pdf",
            filename=combined_report_download_filename(test_dir),
        )


    @router.get("/api/test/artifacts/view/{artifact}")
    def view_test_artifact(artifact: str):
        path = resolve_test_artifact_path(artifact)
        if path.suffix.lower() != ".png":
            raise HTTPException(status_code=400, detail="Only image artifacts can be viewed.")
        return FileResponse(path, media_type="image/png")

    return router
