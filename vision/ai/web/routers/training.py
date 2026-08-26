"""Non-Magic training execution, status, artifacts, and report routes."""

from __future__ import annotations

import functools
import os
import platform
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse


@dataclass(frozen=True)
class TrainingRouterDependencies:
    repo_root: Path
    log_dir: Path
    log_file: Path
    training_python: str
    timezone: object
    model_registry: dict
    training_context_filename: str
    compute_start_lock: object
    training_sessions_cache: object
    run_metrics_cache: object
    train_request_type: type
    weight_request_type: type
    artifact_request_type: type
    get_state: Callable[[], tuple]
    set_state: Callable[[object, object, object, object], None]
    callbacks: dict[str, Callable]


def create_training_router(deps: TrainingRouterDependencies) -> APIRouter:
    router = APIRouter()
    REPO_ROOT = deps.repo_root
    LOG_DIR = deps.log_dir
    LOG_FILE = deps.log_file
    TRAINING_PYTHON = deps.training_python
    MYT = deps.timezone
    MODEL_REGISTRY = deps.model_registry
    TRAINING_REPORT_CONTEXT_FILE = deps.training_context_filename
    training_sessions_cache = deps.training_sessions_cache
    run_metrics_cache = deps.run_metrics_cache
    globals().update(deps.callbacks)
    globals().update({
        "TrainRequest": deps.train_request_type,
        "WeightRequest": deps.weight_request_type,
        "ArtifactRequest": deps.artifact_request_type,
    })

    def serialized_compute_start(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with deps.compute_start_lock:
                return function(*args, **kwargs)
        return wrapper

    @router.post("/api/train/start")
    @serialized_compute_start
    def start_training(request: TrainRequest):
        training_process, training_started_at, training_log_file, training_run_info = deps.get_state()

        ensure_compute_available("training")

        dataset_yaml = Path(request.dataset_yaml).expanduser() if request.dataset_yaml else None
        if not request.resume and (dataset_yaml is None or not dataset_yaml.is_file()):
            raise HTTPException(status_code=400, detail="Prepare a valid dataset before starting a new training run.")

        model_spec = MODEL_REGISTRY.get(request.model_size)
        if not model_spec:
            raise HTTPException(status_code=400, detail=f"Unknown model size: {request.model_size}")
        model = model_spec["model"]
        model_family = model_spec["family"]
        model_task = model_spec["task"]
        requested_project = request.project
        if is_known_training_project_default(requested_project):
            requested_project = default_project_for_model_size(request.model_size)
        training_project_path = normalize_training_project_path(requested_project)
        requested_name = (request.name or "train").strip() or "train"
        run_name = timestamped_training_run_name(requested_name, resume=request.resume)
        resume_checkpoint = None
        resume_run_dir = None
        if request.resume:
            try:
                resume_run_dir, _ = resolve_run_dir_details(requested_project, run_name)
            except HTTPException as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot resume: no existing run was found for the selected project and run name.",
                ) from exc
            resume_checkpoint = resume_run_dir / "weights" / "last.pt"
            ensure_runs_path(resume_checkpoint)
            if not resume_checkpoint.is_file():
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot resume: last.pt was not found in {resume_run_dir}.",
                )
            if model_family == "ultralytics":
                model = str(resume_checkpoint)

        training_values = {
            "imgsz": request.imgsz,
            "batch": request.batch,
            "lr0": request.lr0,
            "weight_decay": request.weight_decay,
            "warmup_epochs": request.warmup_epochs,
            "cos_lr": request.cos_lr,
        }
        if model_family in {"dfine", "rfdetr"}:
            for key in training_values:
                training_values[key] = backend_training_value(request, key, model_family)

        request_payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        request_payload.update(training_values)
        request_payload["family"] = model_family
        request_payload["task"] = model_task
        request_payload["model"] = model
        request_payload["project"] = str(training_project_path)
        request_payload["requested_name"] = requested_name
        request_payload["name"] = run_name
        report_context = {}
        if resume_run_dir:
            report_context = read_json_object(resume_run_dir / TRAINING_REPORT_CONTEXT_FILE)
            if model_family in {"dfine", "rfdetr"} and dataset_yaml is None and report_context.get("dataset_yaml"):
                dataset_yaml = Path(str(report_context["dataset_yaml"])).expanduser()
        if not report_context:
            report_dataset_yaml = dataset_yaml
            if report_dataset_yaml is not None:
                report_dataset_yaml = report_dataset_yaml.resolve()
            report_context = {
                "created_at": datetime.now(MYT).isoformat(),
                "dataset_yaml": str(report_dataset_yaml) if report_dataset_yaml else "",
                "dataset_summary": cached_dataset_summary(report_dataset_yaml) if report_dataset_yaml else {},
                "hyperparameters": request_payload,
                "task": model_task,
                "family": model_family,
                "model": model,
                "pretrained": True,
                "device": request.device or os.getenv("TRAINING_DEVICE") or "auto",
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "ultralytics": installed_version("ultralytics"),
                    "rfdetr": installed_version("rfdetr"),
                    "dfine_repo": os.getenv("DFINE_REPO_DIR") or str(REPO_ROOT / "third_party" / "D-FINE"),
                    "torch": installed_version("torch"),
                    "gpus": gpu_status().get("gpus", []),
                },
            }
        report_context["last_started_at"] = datetime.now(MYT).isoformat()
        report_context["task"] = model_task
        report_context["family"] = model_family
        report_context["model"] = model
        report_context["hyperparameters"] = request_payload
        report_context["resume"] = bool(request.resume)

        training_run_info = {
            "requested_project": requested_project,
            "requested_name": requested_name,
            "project": str(training_project_path),
            "name": run_name,
            "family": model_family,
            "task": model_task,
            "model": model,
            "expected_run_dir": str(training_project_path / run_name),
            "run_dir": str(resume_run_dir) if resume_run_dir else "",
            "resolution_type": "actual" if resume_run_dir else "pending",
            "current_epoch": 0,
            "total_epochs": request.epochs,
            "report_context": report_context,
        }

        if resume_run_dir:
            persist_training_report_context(resume_run_dir)

        ensure_dirs()
        LOG_FILE.write_text("", encoding="utf-8")
        timestamp = datetime.now(MYT).strftime("%Y%m%d-%H%M%S")
        training_log_file = LOG_DIR / f"train-{timestamp}.log"

        train_script = Path(model_spec["runner"])
        if not train_script.is_file():
            raise HTTPException(status_code=500, detail=f"Training script is missing: {train_script}")

        cmd = [
            TRAINING_PYTHON,
            str(train_script),
            "--model", model,
            "--epochs", str(request.epochs),
            "--imgsz", str(training_values["imgsz"]),
            "--batch", str(training_values["batch"]),
            "--patience", str(request.patience),
            "--save-period", str(request.save_period),
            "--workers", str(request.workers),
            "--warmup-epochs", str(training_values["warmup_epochs"]),
            "--seed", str(request.seed),
            "--project", str(training_project_path),
            "--name", run_name,
        ]
        if model_family == "ultralytics":
            cmd.extend([
                "--optimizer", request.optimizer,
                "--pretrained", "true",
                "--activation", request.activation,
                "--cls-pw", str(request.cls_pw),
                "--augmentation-enabled", str(request.augmentation_enabled).lower(),
                "--disable-ultralytics-albumentations", str(request.disable_ultralytics_albumentations).lower(),
                "--mosaic", str(request.mosaic),
                "--close-mosaic", str(request.close_mosaic),
                "--hsv-h", str(request.hsv_h),
                "--hsv-s", str(request.hsv_s),
                "--hsv-v", str(request.hsv_v),
                "--degrees", str(request.degrees),
                "--translate", str(request.translate),
                "--scale", str(request.scale),
                "--shear", str(request.shear),
                "--perspective", str(request.perspective),
                "--flipud", str(request.flipud),
                "--fliplr", str(request.fliplr),
                "--bgr", str(request.bgr),
                "--mixup", str(request.mixup),
                "--cutmix", str(request.cutmix),
                "--copy-paste", str(request.copy_paste),
                "--erasing", str(request.erasing),
            ])
            if request.optimizer.strip().lower() != "auto":
                cmd.extend([
                    "--lr0", str(training_values["lr0"]),
                    "--lrf", str(request.lrf),
                    "--weight-decay", str(training_values["weight_decay"]),
                ])
            if request.auto_augment:
                cmd.extend(["--auto-augment", request.auto_augment])
        else:
            cmd.extend([
                "--lr0", str(training_values["lr0"]),
                "--weight-decay", str(training_values["weight_decay"]),
            ])

        if dataset_yaml is not None and (not request.resume or model_family in {"dfine", "rfdetr"}):
            cmd.extend(["--data", str(dataset_yaml)])

        device = request.device or os.getenv("TRAINING_DEVICE")
        if device:
            cmd.extend(["--device", device])
        if training_values["cos_lr"]:
            cmd.append("--cos-lr")
        if model_family == "ultralytics" and request.freeze is not None:
            cmd.extend(["--freeze", str(request.freeze)])
        if model_family == "ultralytics" and request.exist_ok:
            cmd.append("--exist-ok")
        if request.resume:
            cmd.append("--resume")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(REPO_ROOT)
            if not existing_pythonpath
            else f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}"
        )

        header = (
            "Training started.\n"
            f"Command: {' '.join(cmd)}\n\n"
        )
        LOG_FILE.write_text(header, encoding="utf-8")
        training_log_file.write_text(header, encoding="utf-8")
        training_process = subprocess.Popen(
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
            f"training:{run_name}",
            "training",
            "running",
            {"pid": training_process.pid, "run": {key: value for key, value in training_run_info.items() if key != "report_context"}},
            force=True,
        )
        training_sessions_cache.clear()
        run_metrics_cache.clear()
        threading.Thread(
            target=stream_training_logs,
            args=(training_process, [LOG_FILE, training_log_file]),
            daemon=True,
        ).start()
        training_started_at = time.time()

        deps.set_state(training_process, training_started_at, training_log_file, training_run_info)

        return {
            "message": "Training started.",
            "pid": training_process.pid,
            "command": cmd,
            "log_file": str(LOG_FILE),
            "history_log_file": str(training_log_file),
            "training_run": {key: value for key, value in training_run_info.items() if key != "report_context"},
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else "",
        }


    @router.post("/api/train/stop")
    def stop_training():
        training_process, _started_at, _log_file, _run_info = deps.get_state()

        if training_process is None or training_process.poll() is not None:
            deps.set_state(None, _started_at, _log_file, _run_info)
            return {"message": "No training process is running."}

        os.killpg(os.getpgid(training_process.pid), signal.SIGTERM)
        return {"message": "Stop signal sent."}


    @router.get("/api/train/status")
    def train_status():
        status = current_status()
        status["log_tail"] = read_log_tail(4000)
        status["gpu"] = gpu_status()
        return status


    @router.get("/api/train/logs", response_class=PlainTextResponse)
    def train_logs():
        return read_log_tail()


    @router.get("/api/train/logs/full", response_class=PlainTextResponse)
    def train_logs_full():
        return read_log_file(LOG_FILE)


    @router.get("/api/train/logs/errors", response_class=PlainTextResponse)
    def train_logs_errors():
        return read_error_log(LOG_FILE)


    @router.get("/api/train/logs/download")
    def download_run_log():
        active_log_file = deps.get_state()[2]
        path = active_log_file if active_log_file and active_log_file.is_file() else latest_timestamped_log()
        if path is None and LOG_FILE.is_file():
            path = LOG_FILE
        if not path:
            raise HTTPException(status_code=404, detail="No training log found.")
        return FileResponse(path, media_type="text/plain", filename=path.name)


    @router.post("/api/train/weights/status")
    def weights_status(request: WeightRequest):
        try:
            run_dir, resolution_type = resolve_run_dir_details(request.project, request.name)
        except HTTPException:
            run_dir = None
            resolution_type = "not_found"

        result = {
            "run_dir": str(run_dir) if run_dir else "",
            "resolution_type": resolution_type,
        }
        for weight in ("best", "last"):
            path = run_dir / "weights" / f"{weight}.pt" if run_dir else None
            if path and path.is_file():
                result[weight] = {"available": True, "path": str(path), "size": path.stat().st_size}
            else:
                result[weight] = {"available": False, "path": "", "size": 0}
        return result


    @router.post("/api/train/metrics")
    def train_metrics(request: WeightRequest):
        try:
            run_dir, resolution_type = resolve_run_dir_details(request.project, request.name)
        except HTTPException:
            return {
                "available": False,
                "run_dir": "",
                "resolution_type": "not_found",
                "artifacts": {},
            }

        result = cached_run_metrics(run_dir)
        result["resolution_type"] = resolution_type
        return result

    @router.post("/api/train/artifacts/download")
    def download_artifact(request: ArtifactRequest):
        path = resolve_artifact_path(request)
        media_type = "text/csv" if path.suffix == ".csv" else "image/png" if path.suffix == ".png" else "application/octet-stream"
        return FileResponse(path, media_type=media_type, filename=path.name)


    @router.post("/api/train/report/download")
    def download_training_report(request: WeightRequest):
        if current_status()["running"]:
            raise HTTPException(status_code=409, detail="Wait for training to finish before generating the report.")
        run_dir = resolve_run_dir(request.project, request.name)
        ensure_model_report_artifacts_for_report(run_dir)
        metrics = read_run_metrics(run_dir)
        if not metrics.get("available"):
            raise HTTPException(status_code=409, detail="The selected training run has no completed metrics.")
        try:
            from ..report_generator import generate_training_report

            report_path = generate_training_report(
                run_dir,
                load_training_report_context(run_dir),
                metrics,
            )
        except ImportError as exc:
            raise HTTPException(status_code=500, detail="PDF reporting requires the reportlab dependency. Rebuild the container image.") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not generate the training report: {exc}") from exc
        return FileResponse(report_path, media_type="application/pdf", filename=training_report_download_filename(run_dir))


    @router.get("/api/train/artifacts/view/{artifact}")
    def view_artifact(artifact: str, project: str = "runs/detect", name: str = "train"):
        path = resolve_artifact_path(
            ArtifactRequest(project=project, name=name, artifact=artifact)
        )
        if path.suffix.lower() != ".png":
            raise HTTPException(status_code=400, detail="Only image artifacts can be viewed.")
        return FileResponse(path, media_type="image/png")


    @router.get("/api/train/weights/{weight}")
    def download_weight(weight: str, project: str = "runs/detect", name: str = "train"):
        path = resolve_weight_path(project, name, weight)
        timestamp = datetime.now(MYT).strftime("%Y%m%d_%H%M%S")
        return FileResponse(
            path,
            media_type="application/octet-stream",
            filename=f"{weight}_{timestamp}.pt",
        )

    return router
