"""Inference execution, artifact discovery, and storage services."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from fastapi import HTTPException

from ..infer_dfine import run_dfine_inference
from ..infer_rfdetr import run_rfdetr_inference
from ..infer_yolo import InferenceStopped, run_yolo_inference


# Temporary import-time values are replaced by configure_inference().  They
# allow the declarative storage target table below to remain close to the
# storage operations that consume it.
INFERENCE_JOB_ROOT = Path(".")
INFERENCE_UPLOAD_ROOT = Path(".")
DATASET_PREPARED_ROOT = Path(".")
DATASET_EXTRACTED_ROOT = Path(".")
DATASET_UPLOAD_ROOT = Path(".")


@dataclass(frozen=True)
class InferenceDependencies:
    repo_root: Path
    inference_script: Path
    inference_upload_root: Path
    inference_job_root: Path
    dataset_upload_root: Path
    dataset_extracted_root: Path
    dataset_prepared_root: Path
    training_runs_roots: tuple[Path, ...]
    rfdetr_runs_root: Path
    dfine_runs_root: Path
    segment_runs_root: Path
    semantic_runs_root: Path
    classify_runs_root: Path
    image_extensions: set[str]
    video_extensions: set[str]
    weight_extensions: set[str]
    training_context_filename: str
    jobs: dict[str, dict]
    jobs_lock: object
    dataset_jobs: dict[str, dict]
    dataset_jobs_lock: object
    current_status: Callable[[], dict]
    test_status: Callable[[], dict]
    ensure_dirs: Callable
    ensure_inference_path: Callable[[Path], None]
    ensure_inference_runs_path: Callable[[Path], None]
    ensure_runs_path: Callable[[Path], None]
    persist_job: Callable
    read_json_object: Callable[[Path], dict]
    read_log_file: Callable[[Path], str]
    resolve_project_path: Callable[[str], Path]
    training_run_info: Callable[[], Optional[dict]]


def configure_inference(deps: InferenceDependencies) -> None:
    globals().update({
        "REPO_ROOT": deps.repo_root,
        "INFERENCE_SCRIPT": deps.inference_script,
        "INFERENCE_UPLOAD_ROOT": deps.inference_upload_root,
        "INFERENCE_JOB_ROOT": deps.inference_job_root,
        "DATASET_UPLOAD_ROOT": deps.dataset_upload_root,
        "DATASET_EXTRACTED_ROOT": deps.dataset_extracted_root,
        "DATASET_PREPARED_ROOT": deps.dataset_prepared_root,
        "TRAINING_RUNS_ROOTS": deps.training_runs_roots,
        "RFDETR_RUNS_ROOT": deps.rfdetr_runs_root,
        "DFINE_RUNS_ROOT": deps.dfine_runs_root,
        "SEGMENT_RUNS_ROOT": deps.segment_runs_root,
        "SEMANTIC_RUNS_ROOT": deps.semantic_runs_root,
        "CLASSIFY_RUNS_ROOT": deps.classify_runs_root,
        "IMAGE_EXTENSIONS": deps.image_extensions,
        "VIDEO_EXTENSIONS": deps.video_extensions,
        "INFERENCE_WEIGHT_EXTENSIONS": deps.weight_extensions,
        "TRAINING_REPORT_CONTEXT_FILE": deps.training_context_filename,
        "inference_jobs": deps.jobs,
        "inference_jobs_lock": deps.jobs_lock,
        "dataset_preparation_jobs": deps.dataset_jobs,
        "dataset_preparation_lock": deps.dataset_jobs_lock,
        "current_status": deps.current_status,
        "test_status": deps.test_status,
        "ensure_dirs": deps.ensure_dirs,
        "ensure_inference_path": deps.ensure_inference_path,
        "ensure_inference_runs_path": deps.ensure_inference_runs_path,
        "ensure_runs_path": deps.ensure_runs_path,
        "persist_job": deps.persist_job,
        "read_json_object": deps.read_json_object,
        "read_log_file": deps.read_log_file,
        "resolve_project_path": deps.resolve_project_path,
        "training_run_info": deps.training_run_info,
    })
    for key, path in {
        "inference_outputs": deps.inference_job_root,
        "inference_uploads": deps.inference_upload_root,
        "dataset_prepared": deps.dataset_prepared_root,
        "dataset_extracted": deps.dataset_extracted_root,
        "dataset_uploads": deps.dataset_upload_root,
    }.items():
        STORAGE_CLEANUP_TARGETS[key]["path"] = path


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size

    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() or item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def inference_jobs_storage_payload() -> dict:
    ensure_dirs()
    job_dirs = [path for path in INFERENCE_JOB_ROOT.iterdir() if path.is_dir()]
    return {
        "jobs_path": str(INFERENCE_JOB_ROOT),
        "job_count": len(job_dirs),
        "jobs_size": directory_size(INFERENCE_JOB_ROOT),
    }


STORAGE_CLEANUP_TARGETS = {
    "inference_outputs": {
        "label": "Inference output jobs",
        "path": INFERENCE_JOB_ROOT,
        "requires_idle": "inference",
    },
    "inference_uploads": {
        "label": "Inference uploaded weights",
        "path": INFERENCE_UPLOAD_ROOT,
        "requires_idle": "inference",
    },
    "dataset_prepared": {
        "label": "Prepared datasets",
        "path": DATASET_PREPARED_ROOT,
        "requires_idle": "dataset",
    },
    "dataset_extracted": {
        "label": "Extracted datasets",
        "path": DATASET_EXTRACTED_ROOT,
        "requires_idle": "dataset",
    },
    "dataset_uploads": {
        "label": "Uploaded dataset ZIPs",
        "path": DATASET_UPLOAD_ROOT,
        "requires_idle": "dataset",
    },
}

STORAGE_CLEANUP_ALIASES = {
    "inference_output": "inference_outputs",
    "inference_upload": "inference_uploads",
    "uploaded_inference": "inference_uploads",
    "uploaded_inference_weight": "inference_uploads",
    "uploaded_inference_weights": "inference_uploads",
    "dataset_extract": "dataset_extracted",
    "dataset_extracts": "dataset_extracted",
    "extracted_dataset": "dataset_extracted",
    "extracted_datasets": "dataset_extracted",
    "dataset_upload": "dataset_uploads",
    "uploaded_dataset": "dataset_uploads",
    "uploaded_datasets": "dataset_uploads",
    "dataset_preparation": "dataset_prepared",
    "prepared_dataset": "dataset_prepared",
    "prepared_datasets": "dataset_prepared",
}


def normalize_storage_target_key(key: str) -> str:
    return STORAGE_CLEANUP_ALIASES.get(key, key)


def storage_target_payload(key: str) -> dict:
    ensure_dirs()
    key = normalize_storage_target_key(key)
    target = STORAGE_CLEANUP_TARGETS.get(key)
    if target is None:
        raise HTTPException(status_code=404, detail="Storage cleanup target not found.")
    root = target["path"]
    children = list(root.iterdir()) if root.is_dir() else []
    return {
        "key": key,
        "label": target["label"],
        "path": str(root),
        "item_count": len(children),
        "size": directory_size(root),
    }


def storage_payload() -> dict:
    return {
        "targets": [
            storage_target_payload(key)
            for key in STORAGE_CLEANUP_TARGETS
        ],
    }


def has_active_inference_job() -> bool:
    with inference_jobs_lock:
        return any(
            job.get("status") not in {"complete", "failed", "stopped"}
            for job in inference_jobs.values()
        )


def has_active_dataset_preparation() -> bool:
    with dataset_preparation_lock:
        return any(
            job.get("status") == "running"
            for job in dataset_preparation_jobs.values()
        )


def ensure_storage_target_idle(target: dict):
    if target.get("requires_idle") == "inference" and has_active_inference_job():
        raise HTTPException(status_code=409, detail="Stop the active inference job before clearing outputs.")
    if target.get("requires_idle") == "dataset":
        if current_status()["running"]:
            raise HTTPException(status_code=409, detail="Stop training before clearing dataset storage.")
        if test_status()["running"]:
            raise HTTPException(status_code=409, detail="Stop testing before clearing dataset storage.")
        if has_active_dataset_preparation():
            raise HTTPException(status_code=409, detail="Wait for dataset preparation to finish before clearing dataset storage.")


def clear_storage_target(key: str) -> dict:
    key = normalize_storage_target_key(key)
    target = STORAGE_CLEANUP_TARGETS.get(key)
    if target is None:
        raise HTTPException(status_code=404, detail="Storage cleanup target not found.")
    ensure_storage_target_idle(target)

    before = storage_target_payload(key)
    removed_items = 0
    root = target["path"].resolve()
    for child in list(target["path"].iterdir()):
        target = child.resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=500, detail="Refusing to delete outside storage target.") from exc
        if child.is_dir():
            shutil.rmtree(child)
            removed_items += 1
        elif child.is_file() or child.is_symlink():
            child.unlink()
            removed_items += 1

    if key == "inference_outputs":
        with inference_jobs_lock:
            inference_jobs.clear()

    after = storage_target_payload(key)
    return {
        "key": key,
        "label": before["label"],
        "path": before["path"],
        "removed_items": removed_items,
        "removed_jobs": removed_items if key == "inference_outputs" else 0,
        "freed_bytes": max(0, before["size"] - after["size"]),
        "before": before,
        "after": after,
    }


def clear_inference_outputs() -> dict:
    return clear_storage_target("inference_outputs")


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def resolve_inference_weight_path(weight_path: str) -> Path:
    value = Path(str(weight_path or "")).expanduser()
    if value.is_absolute():
        candidate = value.resolve()
    else:
        candidate = (REPO_ROOT / value).resolve()
    ensure_inference_runs_path(candidate)
    if candidate.suffix.lower() not in INFERENCE_WEIGHT_EXTENSIONS or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Selected inference weights were not found.")
    return candidate


def family_for_weight_path(weights_path: Path) -> str:
    parts = {part.lower() for part in weights_path.parts}
    if "rfdetr" in parts:
        return "rfdetr"
    if "dfine" in parts:
        return "dfine"
    context = read_json_object(weights_path.parent.parent / TRAINING_REPORT_CONTEXT_FILE)
    family = str(context.get("family") or context.get("hyperparameters", {}).get("family") or "").lower()
    return family if family in {"dfine", "rfdetr", "ultralytics"} else "ultralytics"


def task_for_runs_root(root: Path) -> str:
    if root in {RFDETR_RUNS_ROOT, DFINE_RUNS_ROOT}:
        return "detect"
    if root == SEGMENT_RUNS_ROOT:
        return "segment"
    if root == SEMANTIC_RUNS_ROOT:
        return "semantic"
    if root == CLASSIFY_RUNS_ROOT:
        return "classify"
    return "detect"


def family_for_runs_root(root: Path) -> str:
    if root == RFDETR_RUNS_ROOT:
        return "rfdetr"
    if root == DFINE_RUNS_ROOT:
        return "dfine"
    return "ultralytics"


def available_inference_weights() -> list[dict]:
    weights = []
    for root in TRAINING_RUNS_ROOTS:
        if not root.is_dir():
            continue
        task = task_for_runs_root(root)
        family = family_for_runs_root(root)
        candidates = [
            path
            for path in root.rglob("weights/*")
            if path.suffix.lower() in INFERENCE_WEIGHT_EXTENSIONS
        ]
        for path in sorted(candidates):
            try:
                ensure_inference_runs_path(path)
            except HTTPException:
                continue
            run_dir = path.parent.parent
            stat = path.stat()
            try:
                run_label = run_dir.relative_to(root).as_posix()
            except ValueError:
                run_label = run_dir.name
            weights.append({
                "label": f"{task}/{run_label}/{path.name}",
                "path": relative_to_repo(path),
                "family": family,
                "task": task,
                "run": run_label,
                "weight": path.stem,
                "format": path.suffix.lower().lstrip("."),
                "size": stat.st_size,
                "inference_supported": family != "dfine",
                "unsupported_reason": (
                    "D-FINE inference is not implemented yet."
                    if family == "dfine"
                    else ""
                ),
                "modified_at": stat.st_mtime,
            })
    weights.sort(key=lambda item: item["modified_at"], reverse=True)
    return weights


def available_detection_weights() -> list[dict]:
    return available_inference_weights()


def normalize_exported_model_path(exported, fallback: Path) -> Path:
    if isinstance(exported, (list, tuple)):
        exported = exported[0] if exported else fallback
    value = Path(str(exported or fallback)).expanduser()
    if not value.is_absolute():
        value = (fallback.parent / value).resolve()
    return value.resolve()


def export_inference_pt_to_onnx(weights_path: Path, imgsz: int) -> Path:
    if weights_path.suffix.lower() != ".pt":
        return weights_path

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("ONNX export requires ultralytics in the web container.") from exc

    fallback_path = weights_path.with_suffix(".onnx")
    try:
        exported = YOLO(str(weights_path)).export(
            format="onnx",
            imgsz=int(imgsz),
            opset=12,
            simplify=False,
            dynamic=False,
        )
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        raise RuntimeError(
            f"ONNX export requires {missing}. Rebuild the container after updating requirements.txt."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"ONNX export failed: {exc}") from exc

    exported_path = normalize_exported_model_path(exported, fallback_path)
    if not exported_path.is_file() and fallback_path.is_file():
        exported_path = fallback_path.resolve()
    if not exported_path.is_file():
        raise RuntimeError("ONNX export did not produce an .onnx file.")
    if exported_path.suffix.lower() != ".onnx":
        raise RuntimeError(f"ONNX export produced an unexpected file: {exported_path.name}")
    return exported_path


def available_training_sessions() -> list[dict]:
    sessions = []
    for root in TRAINING_RUNS_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_dir() or not is_run_dir(path):
                continue
            try:
                ensure_runs_path(path)
            except HTTPException:
                continue
            file_times = [item.stat().st_mtime for item in path.rglob("*") if item.is_file()]
            modified_at = max(file_times, default=path.stat().st_mtime)
            project_path = path.parent
            task = task_for_runs_root(root)
            family = family_for_runs_root(root)
            sessions.append({
                "label": relative_to_repo(path),
                "name": path.name,
                "project": relative_to_repo(project_path),
                "run_dir": relative_to_repo(path),
                "family": family,
                "task": task,
                "modified_at": modified_at,
                "has_results": (path / "results.csv").is_file(),
                "has_best": (path / "weights" / "best.pt").is_file(),
                "has_last": (path / "weights" / "last.pt").is_file(),
            })

    sessions.sort(key=lambda item: item["modified_at"], reverse=True)
    return sessions


def ensure_inference_script() -> Path:
    if not INFERENCE_SCRIPT.is_file():
        raise HTTPException(status_code=500, detail="Python inference script is missing.")
    return INFERENCE_SCRIPT


def inference_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    raise HTTPException(status_code=400, detail="Upload an image or video file for inference.")


def inference_job_payload(job: dict) -> dict:
    public_keys = {
        "job_id",
        "status",
        "stage",
        "percent",
        "detail",
        "frames",
        "total_frames",
        "detections",
        "classifications",
        "task",
        "elapsed_ms",
        "media_type",
        "weights",
        "preview_available",
        "preview_fps",
        "result",
        "error",
        "stoppable",
    }
    return {key: value for key, value in job.items() if key in public_keys}


def update_inference_job(job_id: str, **updates):
    condition = None
    persisted = None
    with inference_jobs_lock:
        job = inference_jobs.get(job_id)
        if job is None:
            return
        job.update(updates)
        job["updated_at"] = time.time()
        condition = job.get("preview_condition")
        persisted = inference_job_payload(job)
    persist_job(job_id, "inference", str(persisted.get("status") or "running"), persisted)
    if isinstance(condition, threading.Condition):
        with condition:
            condition.notify_all()


def inference_has_live_preview(job_id: str, preview_path: Path) -> bool:
    with inference_jobs_lock:
        job = inference_jobs.get(job_id) or {}
        if job.get("preview_frame"):
            return True
    return preview_path.is_file()


def update_inference_preview_frame(job_id: str, frame: bytes):
    condition = None
    now = time.time()
    with inference_jobs_lock:
        job = inference_jobs.get(job_id)
        if job is None:
            return
        timestamps = [
            timestamp for timestamp in job.get("preview_frame_times", [])
            if now - float(timestamp) <= 2.0
        ]
        timestamps.append(now)
        elapsed = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        job["preview_frame"] = frame
        job["preview_available"] = True
        job["preview_updated_at"] = now
        job["preview_frame_times"] = timestamps
        job["preview_fps"] = round((len(timestamps) - 1) / elapsed, 1) if elapsed > 0 else 0.0
        job["updated_at"] = now
        condition = job.get("preview_condition")
    if isinstance(condition, threading.Condition):
        with condition:
            condition.notify_all()


def update_inference_webrtc_frame(job_id: str, frame):
    with inference_jobs_lock:
        job = inference_jobs.get(job_id)
        if job is None:
            return
        job["webrtc_frame"] = frame.copy()
        job["webrtc_updated_at"] = time.time()
        job["webrtc_frame_seq"] = int(job.get("webrtc_frame_seq") or 0) + 1


def latest_inference_webrtc_frame(job_id: str):
    with inference_jobs_lock:
        job = inference_jobs.get(job_id)
        if job is None:
            return None, "missing"
        frame = job.get("webrtc_frame")
        status = job.get("status", "")
    return (frame.copy() if frame is not None else None), status


def inference_mjpeg_stream(job_id: str):
    last_updated_at = 0.0
    while True:
        with inference_jobs_lock:
            job = inference_jobs.get(job_id)
            if job is None:
                break
            frame = job.get("preview_frame")
            updated_at = float(job.get("preview_updated_at") or 0)
            status = job.get("status", "")
            condition = job.get("preview_condition")

        if frame and updated_at != last_updated_at:
            last_updated_at = updated_at
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-store\r\n"
                + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                + frame
                + b"\r\n"
            )

        if status in {"complete", "failed", "stopped"}:
            break

        if isinstance(condition, threading.Condition):
            with condition:
                condition.wait(timeout=1.0)
        else:
            time.sleep(0.2)


def read_inference_progress(progress_path: Path) -> dict:
    if not progress_path.is_file():
        return {}
    return read_json_object(progress_path)


def inference_result_payload_from_disk(job_id: str, job_dir: Path) -> dict | None:
    result_path = job_dir / "result.json"
    if not result_path.is_file():
        return None
    payload = read_json_object(result_path)
    if not payload:
        return None
    return {
        **payload,
        "job_id": job_id,
        "result_url": f"/api/inference/result/{job_id}",
        "download_url": f"/api/inference/result/{job_id}?download=1",
    }


def inference_job_from_disk(job_id: str) -> dict | None:
    job_dir = (INFERENCE_JOB_ROOT / job_id).resolve()
    try:
        ensure_inference_path(job_dir)
    except HTTPException:
        return None
    if not job_dir.is_dir():
        return None

    preview_path = job_dir / "preview.jpg"
    output_candidates = [job_dir / "annotated.jpg", job_dir / "annotated.mp4"]
    output_path = next((candidate for candidate in output_candidates if candidate.is_file()), None)
    result = inference_result_payload_from_disk(job_id, job_dir)
    if result and output_path:
        return {
            "job_id": job_id,
            "status": "complete",
            "stage": "complete",
            "percent": 100,
            "detail": "Inference complete.",
            "frames": result.get("frames", 0),
            "total_frames": result.get("frames", 0),
            "detections": result.get("detections", 0),
            "classifications": result.get("classifications", 0),
            "task": result.get("task", ""),
            "elapsed_ms": result.get("elapsed_ms", 0),
            "media_type": result.get("media_type", "video" if output_path.suffix == ".mp4" else "image"),
            "preview_available": preview_path.is_file(),
            "preview_fps": 0,
            "result": result,
        }

    progress = read_inference_progress(job_dir / "progress.json")
    if progress:
        return {
            "job_id": job_id,
            "status": "running",
            "stage": progress.get("stage", "processing"),
            "percent": progress.get("percent", 0),
            "detail": progress.get("detail", "Processing inference."),
            "frames": progress.get("frames", 0),
            "total_frames": progress.get("total_frames", 0),
            "detections": progress.get("detections", 0),
            "classifications": progress.get("classifications", 0),
            "task": progress.get("task", ""),
            "elapsed_ms": 0,
            "media_type": "video" if (job_dir / "annotated.mp4").exists() else "image",
            "preview_available": preview_path.is_file(),
            "preview_fps": 0,
        }

    log_path = job_dir / "inference.log"
    if log_path.is_file():
        return {
            "job_id": job_id,
            "status": "running",
            "stage": "starting",
            "percent": 0,
            "detail": "Inference process has started.",
            "frames": 0,
            "total_frames": 0,
            "detections": 0,
            "classifications": 0,
            "task": "",
            "elapsed_ms": 0,
            "preview_available": preview_path.is_file(),
            "preview_fps": 0,
        }
    return None


def run_inference_job(
    job_id: str,
    inference_request: dict,
    result_path: Path,
    output_path: Path,
    preview_path: Path,
    stop_event: threading.Event,
):
    log_path = result_path.with_name("inference.log")
    started = time.time()
    inference_family = str(inference_request.pop("family", "ultralytics") or "ultralytics")
    log_path.write_text(f"Starting in-process Python {inference_family} inference.\n", encoding="utf-8")

    def handle_progress(progress: dict):
        update_inference_job(
            job_id,
            status="running",
            stage=progress.get("stage", "processing"),
            percent=progress.get("percent", 0),
            detail=progress.get("detail", "Processing inference."),
            frames=progress.get("frames", 0),
            total_frames=progress.get("total_frames", 0),
            detections=progress.get("detections", 0),
            classifications=progress.get("classifications", 0),
            task=progress.get("task", ""),
            elapsed_ms=round((time.time() - started) * 1000, 2),
            preview_available=inference_has_live_preview(job_id, preview_path),
            stoppable=True,
        )

    def handle_preview(frame: bytes):
        update_inference_preview_frame(job_id, frame)

    def handle_live_frame(frame):
        update_inference_webrtc_frame(job_id, frame)

    update_inference_job(
        job_id,
        status="running",
        stage="starting",
        percent=0,
        detail="Starting inference process.",
        elapsed_ms=0,
        stoppable=True,
    )
    try:
        if inference_request.pop("convert_to_onnx", False):
            original_weights = Path(inference_request["weights_path"])
            update_inference_job(
                job_id,
                status="running",
                stage="converting",
                percent=3,
                detail="Converting uploaded .pt weights to ONNX.",
                elapsed_ms=round((time.time() - started) * 1000, 2),
                stoppable=True,
            )
            onnx_path = export_inference_pt_to_onnx(
                original_weights,
                int(inference_request.get("imgsz") or 640),
            )
            inference_request["weights_path"] = onnx_path
            if onnx_path != original_weights:
                update_inference_job(
                    job_id,
                    weights=f"{original_weights.name} -> {onnx_path.name}",
                    detail="ONNX conversion complete.",
                    elapsed_ms=round((time.time() - started) * 1000, 2),
                )

        inference_runners = {
            "dfine": run_dfine_inference,
            "rfdetr": run_rfdetr_inference,
            "ultralytics": run_yolo_inference,
        }
        inference_runner = inference_runners.get(inference_family, run_yolo_inference)
        payload = inference_runner(
            **inference_request,
            progress_callback=handle_progress,
            preview_callback=handle_preview,
            live_frame_callback=handle_live_frame,
            stop_event=stop_event,
            use_model_cache=True,
        )
    except InferenceStopped:
        update_inference_job(
            job_id,
            status="stopped",
            stage="stopped",
            percent=100,
            detail="Inference stopped by user.",
            elapsed_ms=round((time.time() - started) * 1000, 2),
            preview_available=inference_has_live_preview(job_id, preview_path),
            stoppable=False,
            stop_event=None,
        )
        return
    except Exception as exc:
        detail = f"Inference failed: {exc}"
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{detail}\n")
        update_inference_job(
            job_id,
            status="failed",
            stage="failed",
            percent=100,
            detail=detail,
            error=detail,
            elapsed_ms=round((time.time() - started) * 1000, 2),
            preview_available=inference_has_live_preview(job_id, preview_path),
            stoppable=False,
            stop_event=None,
        )
        return

    if not output_path.is_file():
        detail = "Inference completed without producing an output file."
        update_inference_job(
            job_id,
            status="failed",
            stage="failed",
            percent=100,
            detail=detail,
            error=detail,
            elapsed_ms=round((time.time() - started) * 1000, 2),
            preview_available=inference_has_live_preview(job_id, preview_path),
            stoppable=False,
            stop_event=None,
        )
        return

    with inference_jobs_lock:
        existing_job = dict(inference_jobs.get(job_id) or {})
    result_payload = {
        **payload,
        "job_id": job_id,
        "media_type": existing_job.get("media_type", payload.get("media_type")),
        "weights": existing_job.get("weights", ""),
        "stdout": read_log_file(log_path).strip(),
        "result_url": f"/api/inference/result/{job_id}",
        "download_url": f"/api/inference/result/{job_id}?download=1",
    }
    update_inference_job(
        job_id,
        status="complete",
        stage="complete",
        percent=100,
        detail="Inference complete.",
        frames=payload.get("frames", 0),
        total_frames=payload.get("frames", 0),
        detections=payload.get("detections", 0),
        classifications=payload.get("classifications", 0),
        task=payload.get("task", ""),
        elapsed_ms=payload.get("elapsed_ms", round((time.time() - started) * 1000, 2)),
        preview_available=inference_has_live_preview(job_id, preview_path),
        result=result_payload,
        stoppable=False,
        stop_event=None,
    )


def stop_inference_process(job_id: str) -> dict:
    with inference_jobs_lock:
        job = inference_jobs.get(job_id)
        if job is None:
            disk_job = inference_job_from_disk(job_id)
            if disk_job is None:
                raise HTTPException(status_code=404, detail="Inference job not found.")
            return disk_job
        process = job.get("process")
        stop_event = job.get("stop_event")
        status = job.get("status")
        if status in {"complete", "failed", "stopped"}:
            return inference_job_payload(dict(job))
        job["stop_requested"] = True
        job["status"] = "stopping"
        job["stage"] = "stopping"
        job["detail"] = "Stopping inference process."
        job["stoppable"] = False
        job["updated_at"] = time.time()

    if isinstance(stop_event, threading.Event):
        stop_event.set()

    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    with inference_jobs_lock:
        return inference_job_payload(dict(inference_jobs[job_id]))


def is_run_dir(path: Path) -> bool:
    return (
        (path / "weights" / "best.pt").is_file()
        or (path / "weights" / "last.pt").is_file()
        or (path / "results.csv").is_file()
        or (path / TRAINING_REPORT_CONTEXT_FILE).is_file()
        or any(path.glob("checkpoint*.pth"))
    )


def find_named_run_dir(search_root: Path, name: str) -> Optional[Path]:
    if not search_root.exists() or not name:
        return None

    candidates = [
        path
        for path in search_root.rglob("*")
        if path.is_dir()
        and is_run_dir(path)
        and (path.name == name or path.parent.name == name)
    ]
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: max((item.stat().st_mtime for item in path.rglob("*") if item.is_file()), default=0),
    )


def resolve_run_dir_details(project: str, name: str) -> tuple[Path, str]:
    project_path = resolve_project_path(project)
    ensure_runs_path(project_path)

    current_info = training_run_info() or {}
    current_run_dir = current_info.get("run_dir")
    current_project = current_info.get("requested_project") or current_info.get("project")
    if current_run_dir and current_info.get("name") == name and current_project:
        try:
            same_project = resolve_project_path(current_project) == project_path
        except (OSError, RuntimeError):
            same_project = False
        actual = Path(current_run_dir).expanduser().resolve()
        if same_project and actual.is_dir():
            ensure_runs_path(actual)
            return actual, "actual"

    exact = (project_path / name).resolve()
    ensure_runs_path(exact)
    if is_run_dir(exact):
        return exact, "exact"

    named = find_named_run_dir(project_path, name)
    if named:
        return named, "legacy"

    raise HTTPException(
        status_code=404,
        detail=f"No completed run matching project '{project}' and name '{name}' was found.",
    )


def resolve_run_dir(project: str, name: str) -> Path:
    run_dir, _ = resolve_run_dir_details(project, name)
    return run_dir

__all__ = [
    "InferenceDependencies",
    "configure_inference",
    "STORAGE_CLEANUP_TARGETS",
    "directory_size",
    "inference_jobs_storage_payload",
    "normalize_storage_target_key",
    "storage_target_payload",
    "storage_payload",
    "has_active_inference_job",
    "has_active_dataset_preparation",
    "ensure_storage_target_idle",
    "clear_storage_target",
    "clear_inference_outputs",
    "relative_to_repo",
    "resolve_inference_weight_path",
    "family_for_weight_path",
    "task_for_runs_root",
    "family_for_runs_root",
    "available_inference_weights",
    "available_detection_weights",
    "normalize_exported_model_path",
    "export_inference_pt_to_onnx",
    "available_training_sessions",
    "ensure_inference_script",
    "inference_media_type",
    "inference_job_payload",
    "update_inference_job",
    "inference_has_live_preview",
    "update_inference_preview_frame",
    "update_inference_webrtc_frame",
    "latest_inference_webrtc_frame",
    "inference_mjpeg_stream",
    "read_inference_progress",
    "inference_result_payload_from_disk",
    "inference_job_from_disk",
    "run_inference_job",
    "stop_inference_process",
    "is_run_dir",
    "find_named_run_dir",
    "resolve_run_dir_details",
    "resolve_run_dir",
]
