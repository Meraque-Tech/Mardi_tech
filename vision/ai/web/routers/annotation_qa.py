"""HTTP routes for the Annotation QA workflow."""

from __future__ import annotations

import functools
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..roboflow_sync import ROBOFLOW_SYNC_LOG_FILE, ROBOFLOW_SYNC_PREVIEW_FILE, redact_secret
from ..schemas import (
    AnnotationQaFixRequest,
    AnnotationQaMarkRequest,
    AnnotationQaRequest,
    AnnotationQaRoboflowRequest,
)
from ..services.annotation_qa import (
    annotation_qa_job_from_disk,
    annotation_qa_job_id,
    annotation_qa_model_status,
    annotation_qa_payload,
    annotation_qa_run_dir,
    apply_annotation_qa_fixes,
    build_annotation_qa_roboflow_preview,
    ensure_annotation_qa_path,
    normalize_annotation_qa_model_name,
    normalize_annotation_qa_task,
    publish_annotation_qa_to_roboflow,
    run_annotation_qa_job,
    set_annotation_qa_issue_fix,
    write_annotation_qa_report,
)


@dataclass(frozen=True)
class AnnotationQaRouterDependencies:
    annotation_qa_root: Path
    report_json_filename: str
    report_csv_filename: str
    summary_filename: str
    jobs: dict[str, dict]
    jobs_lock: threading.Lock
    compute_start_lock: threading.RLock
    ensure_compute_available: Callable[[str], None]
    persist_job: Callable
    prepared_dataset_yaml: Callable
    read_json_object: Callable[[Path], dict]
    training_active: Callable[[], bool]
    test_active: Callable[[], bool]


def create_annotation_qa_router(deps: AnnotationQaRouterDependencies) -> APIRouter:
    router = APIRouter()
    ANNOTATION_QA_ROOT = deps.annotation_qa_root
    ANNOTATION_QA_REPORT_JSON_FILE = deps.report_json_filename
    ANNOTATION_QA_REPORT_CSV_FILE = deps.report_csv_filename
    ANNOTATION_QA_SUMMARY_FILE = deps.summary_filename
    annotation_qa_jobs = deps.jobs
    annotation_qa_jobs_lock = deps.jobs_lock
    ensure_compute_available = deps.ensure_compute_available
    persist_job = deps.persist_job
    prepared_dataset_yaml = deps.prepared_dataset_yaml
    read_json_object = deps.read_json_object

    def serialized_compute_start(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with deps.compute_start_lock:
                return function(*args, **kwargs)
        return wrapper

    @router.post("/api/annotation-qa/start")
    @serialized_compute_start
    def start_annotation_qa(request: AnnotationQaRequest):
        ensure_compute_available("annotation QA")
        try:
            annotation_task = normalize_annotation_qa_task(request.task)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if request.sam_max_difference_percent <= request.box_tolerance_percent:
            raise HTTPException(
                status_code=422,
                detail="Maximum SAM difference must be greater than the YOLO box tolerance.",
            )
        auto_correction_mode = str(request.auto_correction_mode or "shadow").strip().lower()
        if auto_correction_mode not in {"manual", "shadow", "automatic"}:
            raise HTTPException(
                status_code=422,
                detail="Automatic correction mode must be manual, shadow, or automatic.",
            )
        if annotation_task == "segment" and auto_correction_mode == "automatic":
            raise HTTPException(
                status_code=422,
                detail="Automatic polygon replacement is disabled. Use Suggestions only or Human decisions only.",
            )
        with annotation_qa_jobs_lock:
            active = next(
                (
                    job
                    for job in annotation_qa_jobs.values()
                    if job.get("status") in {"queued", "running", "stopping"}
                ),
                None,
            )
        if active is not None:
            raise HTTPException(status_code=409, detail="Annotation QA is already running.")

        yaml_path, _dataset_root, _payload = prepared_dataset_yaml(request.dataset_yaml)
        try:
            model = normalize_annotation_qa_model_name(request.model)
            model_status = annotation_qa_model_status(model)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not model_status["available"]:
            raise HTTPException(status_code=409, detail=model_status["reason"])
        if auto_correction_mode == "automatic" and not model_status["automatic_allowed"]:
            raise HTTPException(
                status_code=422,
                detail="SAM 3 automatic correction is disabled until its QA thresholds are calibrated. Use Suggestions only or Human decisions only.",
            )
        job_id = annotation_qa_job_id()
        run_dir = (ANNOTATION_QA_ROOT / job_id).resolve()
        ensure_annotation_qa_path(run_dir)
        stop_event = threading.Event()
        request_payload = {
            "dataset_yaml": str(yaml_path),
            "task": annotation_task,
            "model": model,
            "model_label": model_status["label"],
            "scope": request.scope,
            "preset": request.preset,
            "box_tolerance_percent": request.box_tolerance_percent,
            "sam_max_difference_percent": request.sam_max_difference_percent,
            "auto_correction_mode": auto_correction_mode,
            "sam_prompt_expansion_percent": request.sam_prompt_expansion_percent,
            "sam_prompt_jitter_percent": request.sam_prompt_jitter_percent,
            "sam_stability_bbox_iou_min": request.sam_stability_bbox_iou_min,
            "sam_stability_edge_percent_max": request.sam_stability_edge_percent_max,
            "sam_auto_quality_min": request.sam_auto_quality_min,
            "sam_auto_yolo_iou_min": request.sam_auto_yolo_iou_min,
            "sam_auto_center_shift_max": request.sam_auto_center_shift_max,
            "sam_auto_neighbor_iou_max": request.sam_auto_neighbor_iou_max,
            "auto_audit_percent": request.auto_audit_percent,
            "max_images": request.max_images,
            "max_side": request.max_side,
        }
        job = {
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
            "percent": 0,
            "detail": "Annotation QA queued.",
            "run_dir": str(run_dir),
            "dataset_yaml": str(yaml_path),
            "annotation_task": annotation_task,
            "model": model,
            "model_label": model_status["label"],
            "scope": request.scope,
            "preset": request.preset,
            "box_tolerance_percent": request.box_tolerance_percent,
            "sam_max_difference_percent": request.sam_max_difference_percent,
            "auto_correction_mode": auto_correction_mode,
            "sam_prompt_expansion_percent": request.sam_prompt_expansion_percent,
            "sam_prompt_jitter_percent": request.sam_prompt_jitter_percent,
            "sam_stability_bbox_iou_min": request.sam_stability_bbox_iou_min,
            "sam_stability_edge_percent_max": request.sam_stability_edge_percent_max,
            "sam_auto_quality_min": request.sam_auto_quality_min,
            "sam_auto_yolo_iou_min": request.sam_auto_yolo_iou_min,
            "sam_auto_center_shift_max": request.sam_auto_center_shift_max,
            "sam_auto_neighbor_iou_max": request.sam_auto_neighbor_iou_max,
            "auto_audit_percent": request.auto_audit_percent,
            "images_scanned": 0,
            "total_images": 0,
            "labels_checked": 0,
            "issues": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "summary": {},
            "report_available": False,
            "stop_event": stop_event,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        with annotation_qa_jobs_lock:
            annotation_qa_jobs[job_id] = job
        persist_job(job_id, "annotation_qa", "queued", annotation_qa_payload(job), force=True)
        thread = threading.Thread(
            target=run_annotation_qa_job,
            args=(job_id, request_payload, stop_event),
            daemon=True,
        )
        job["thread"] = thread
        thread.start()
        return annotation_qa_payload(job)


    @router.get("/api/annotation-qa/status/{job_id}")
    def annotation_qa_status(job_id: str):
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
            raise HTTPException(status_code=404, detail="Annotation QA job not found.")
        with annotation_qa_jobs_lock:
            job = annotation_qa_jobs.get(job_id)
            if job is not None:
                return annotation_qa_payload(dict(job))
        job = annotation_qa_job_from_disk(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Annotation QA job not found.")
        with annotation_qa_jobs_lock:
            annotation_qa_jobs[job_id] = dict(job)
        return annotation_qa_payload(job)


    @router.post("/api/annotation-qa/stop/{job_id}")
    def stop_annotation_qa(job_id: str):
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
            raise HTTPException(status_code=404, detail="Annotation QA job not found.")
        with annotation_qa_jobs_lock:
            job = annotation_qa_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Annotation QA job not found.")
            event = job.get("stop_event")
            if isinstance(event, threading.Event):
                event.set()
            job["status"] = "stopping"
            job["stage"] = "stopping"
            job["detail"] = "Stopping annotation QA..."
            job["updated_at"] = time.time()
            return annotation_qa_payload(dict(job))


    @router.get("/api/annotation-qa/results/{job_id}")
    def annotation_qa_results(job_id: str):
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
            raise HTTPException(status_code=404, detail="Annotation QA results not found.")
        run_dir = (ANNOTATION_QA_ROOT / job_id).resolve()
        ensure_annotation_qa_path(run_dir)
        report_path = run_dir / ANNOTATION_QA_REPORT_JSON_FILE
        if not report_path.is_file():
            raise HTTPException(status_code=404, detail="Annotation QA results are not available yet.")
        return FileResponse(report_path, media_type="application/json")


    @router.get("/api/annotation-qa/preview/{job_id}/{preview_name}")
    def annotation_qa_preview(job_id: str, preview_name: str):
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
            raise HTTPException(status_code=404, detail="Annotation QA preview not found.")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.jpg", preview_name):
            raise HTTPException(status_code=404, detail="Annotation QA preview not found.")
        preview_path = (ANNOTATION_QA_ROOT / job_id / "previews" / preview_name).resolve()
        ensure_annotation_qa_path(preview_path)
        if not preview_path.is_file():
            raise HTTPException(status_code=404, detail="Annotation QA preview not found.")
        return FileResponse(preview_path, media_type="image/jpeg")


    @router.get("/api/annotation-qa/download/{job_id}/{artifact}")
    def annotation_qa_download(job_id: str, artifact: str):
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
            raise HTTPException(status_code=404, detail="Annotation QA report not found.")
        filenames = {
            "json": ANNOTATION_QA_REPORT_JSON_FILE,
            "csv": ANNOTATION_QA_REPORT_CSV_FILE,
            "summary": ANNOTATION_QA_SUMMARY_FILE,
        }
        filename = filenames.get(artifact)
        if filename is None:
            raise HTTPException(status_code=404, detail="Annotation QA report not found.")
        report_path = (ANNOTATION_QA_ROOT / job_id / filename).resolve()
        ensure_annotation_qa_path(report_path)
        if not report_path.is_file():
            raise HTTPException(status_code=404, detail="Annotation QA report not found.")
        media_type = "text/csv" if artifact == "csv" else "application/json"
        return FileResponse(report_path, media_type=media_type, filename=filename)


    @router.post("/api/annotation-qa/mark/{job_id}")
    def mark_annotation_qa_issue(job_id: str, request: AnnotationQaMarkRequest):
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
            raise HTTPException(status_code=404, detail="Annotation QA job not found.")
        allowed = {"unreviewed", "fix_accepted", "accepted", "false_positive", "needs_fix", "ignored"}
        status = str(request.status or "").strip()
        if status not in allowed:
            raise HTTPException(status_code=400, detail="Unknown annotation QA review status.")
        run_dir = (ANNOTATION_QA_ROOT / job_id).resolve()
        ensure_annotation_qa_path(run_dir)
        report_path = run_dir / ANNOTATION_QA_REPORT_JSON_FILE
        report = read_json_object(report_path)
        issues = report.get("issues")
        if not isinstance(issues, list):
            raise HTTPException(status_code=404, detail="Annotation QA results are not available.")
        matched = False
        for issue in issues:
            if isinstance(issue, dict) and issue.get("issue_id") == request.issue_id:
                issue["review_status"] = status
                if issue.get("audit_required"):
                    issue["audit_status"] = "passed" if status == "fix_accepted" else "failed"
                if status != "fix_accepted":
                    issue["accepted_fix"] = ""
                    issue["accepted_class_id"] = None
                    issue["accepted_class_name"] = ""
                    issue["accepted_fix_source"] = ""
                matched = True
                break
        if not matched:
            raise HTTPException(status_code=404, detail="Annotation QA issue not found.")
        write_annotation_qa_report(run_dir, issues, report.get("summary") or read_json_object(run_dir / ANNOTATION_QA_SUMMARY_FILE))
        return {"issue_id": request.issue_id, "status": status}


    @router.post("/api/annotation-qa/fix/{job_id}")
    def accept_annotation_qa_fix(job_id: str, request: AnnotationQaFixRequest):
        run_dir = annotation_qa_run_dir(job_id)
        issue = set_annotation_qa_issue_fix(run_dir, request.issue_id, request.fix, request.class_id)
        return {
            "issue_id": request.issue_id,
            "accepted_fix": issue.get("accepted_fix", ""),
            "accepted_fix_source": issue.get("accepted_fix_source", ""),
            "accepted_class_id": issue.get("accepted_class_id"),
            "accepted_class_name": issue.get("accepted_class_name", ""),
            "review_status": issue.get("review_status", "unreviewed"),
            "audit_status": issue.get("audit_status", "not_required"),
        }


    @router.post("/api/annotation-qa/apply/{job_id}")
    def apply_annotation_qa_corrections(job_id: str):
        if deps.training_active():
            raise HTTPException(status_code=409, detail="Stop training before applying annotation fixes.")
        if deps.test_active():
            raise HTTPException(status_code=409, detail="Stop model testing before applying annotation fixes.")
        return apply_annotation_qa_fixes(job_id)


    @router.post("/api/annotation-qa/roboflow/preview/{job_id}")
    def preview_annotation_qa_roboflow_sync(job_id: str, request: AnnotationQaRoboflowRequest):
        api_key = request.api_key or os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="A Roboflow private API key is required to preview synchronization.")
        try:
            return build_annotation_qa_roboflow_preview(job_id, str(api_key))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Roboflow synchronization preview failed: {redact_secret(exc, str(api_key))}") from exc


    @router.post("/api/annotation-qa/roboflow/publish/{job_id}")
    def publish_annotation_qa_roboflow_sync(job_id: str, request: AnnotationQaRoboflowRequest):
        if not request.confirmed:
            raise HTTPException(status_code=400, detail="Confirm the bound Roboflow project before publishing.")
        api_key = request.api_key or os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="A Roboflow private API key is required to publish corrections.")
        try:
            return publish_annotation_qa_to_roboflow(job_id, str(api_key), request.preview_id)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Roboflow annotation publishing failed: {redact_secret(exc, str(api_key))}") from exc


    @router.get("/api/annotation-qa/roboflow/status/{job_id}")
    def annotation_qa_roboflow_sync_status(job_id: str):
        run_dir = annotation_qa_run_dir(job_id)
        audit = read_json_object(run_dir / ROBOFLOW_SYNC_LOG_FILE)
        if audit:
            return audit
        preview = read_json_object(run_dir / ROBOFLOW_SYNC_PREVIEW_FILE)
        if preview:
            return preview
        raise HTTPException(status_code=404, detail="No Roboflow synchronization preview is available.")

    return router
