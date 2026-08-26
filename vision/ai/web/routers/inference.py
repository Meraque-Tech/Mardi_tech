"""Inference upload, execution, preview, streaming, and result routes."""

from __future__ import annotations

import functools
import mimetypes
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from ..common.uploads import UploadLimits, UploadValidationError, validate_upload_totals
from ..schemas import WebRTCOffer


@dataclass(frozen=True)
class InferenceRouterDependencies:
    job_root: Path
    upload_root: Path
    weight_extensions: set[str]
    timezone: object
    upload_limits: UploadLimits
    jobs: dict[str, dict]
    jobs_lock: object
    peers: dict[str, set]
    peers_lock: object
    compute_start_lock: object
    ensure_compute_available: Callable[[str], None]
    ensure_dirs: Callable
    ensure_inference_path: Callable[[Path], None]
    family_for_weight_path: Callable[[Path], str]
    inference_job_from_disk: Callable
    inference_job_payload: Callable[[dict], dict]
    inference_media_type: Callable[[Path], str]
    inference_mjpeg_stream: Callable
    latest_inference_webrtc_frame: Callable
    persist_job: Callable
    relative_to_repo: Callable[[Path], str]
    resolve_inference_weight_path: Callable[[str], Path]
    run_inference_job: Callable
    safe_upload_path: Callable[[str], Path]
    stop_inference_process: Callable[[str], dict]
    upload_validation_http_error: Callable


def create_inference_router(deps: InferenceRouterDependencies) -> APIRouter:
    router = APIRouter()
    INFERENCE_JOB_ROOT = deps.job_root
    INFERENCE_UPLOAD_ROOT = deps.upload_root
    INFERENCE_WEIGHT_EXTENSIONS = deps.weight_extensions
    MYT = deps.timezone
    inference_jobs = deps.jobs
    inference_jobs_lock = deps.jobs_lock
    inference_peers = deps.peers
    inference_peers_lock = deps.peers_lock
    ensure_compute_available = deps.ensure_compute_available
    ensure_dirs = deps.ensure_dirs
    ensure_inference_path = deps.ensure_inference_path
    family_for_weight_path = deps.family_for_weight_path
    inference_job_from_disk = deps.inference_job_from_disk
    inference_job_payload = deps.inference_job_payload
    inference_media_type = deps.inference_media_type
    inference_mjpeg_stream = deps.inference_mjpeg_stream
    latest_inference_webrtc_frame = deps.latest_inference_webrtc_frame
    persist_job = deps.persist_job
    relative_to_repo = deps.relative_to_repo
    resolve_inference_weight_path = deps.resolve_inference_weight_path
    run_inference_job = deps.run_inference_job
    safe_upload_path = deps.safe_upload_path
    stop_inference_process = deps.stop_inference_process

    def serialized_compute_start(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with deps.compute_start_lock:
                return function(*args, **kwargs)
        return wrapper

    def save_upload(upload, target, progress_callback, stage, detail_prefix):
        position = upload.file.tell()
        upload.file.seek(0, os.SEEK_END)
        size = max(0, int(upload.file.tell()))
        upload.file.seek(position)
        try:
            validate_upload_totals([size], deps.upload_limits)
        except UploadValidationError as exc:
            raise deps.upload_validation_http_error(exc) from exc
        copied = 0
        upload.file.seek(0)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as output:
            while chunk := upload.file.read(1024 * 1024):
                output.write(chunk)
                copied += len(chunk)
                progress_callback(stage, copied, size, f"{detail_prefix}: {copied} of {size} bytes")
        return copied

    @router.post("/api/inference/run")
    @serialized_compute_start
    def run_inference(
        weight_source: str = Form("selected"),
        weight_path: str = Form(""),
        weight_family: str = Form("auto"),
        convert_to_onnx: bool = Form(False),
        imgsz: int = Form(512),
        conf: float = Form(0.25),
        iou: float = Form(0.45),
        vid_stride: int = Form(1),
        show_boxes: bool = Form(True),
        show_labels: bool = Form(True),
        show_conf: bool = Form(True),
        show_masks: bool = Form(True),
        weight_file: Optional[UploadFile] = File(None),
        media_file: UploadFile = File(...),
    ):
        ensure_dirs()
        ensure_compute_available("inference")
        if imgsz < 32:
            raise HTTPException(status_code=400, detail="Image size must be at least 32.")
        if not 0 <= conf <= 1 or not 0 <= iou <= 1:
            raise HTTPException(status_code=400, detail="Confidence and IoU thresholds must be between 0 and 1.")
        if vid_stride < 1:
            raise HTTPException(status_code=400, detail="Video frame stride must be at least 1.")

        job_id = datetime.now(MYT).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        job_dir = (INFERENCE_JOB_ROOT / job_id).resolve()
        ensure_inference_path(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)

        if weight_source == "selected":
            weights_path = resolve_inference_weight_path(weight_path)
            inference_family = family_for_weight_path(weights_path)
            weights_label = relative_to_repo(weights_path)
            convert_to_onnx = False
        elif weight_source == "upload":
            if weight_file is None or not weight_file.filename:
                raise HTTPException(status_code=400, detail="Choose a .pt or .onnx weights file to upload.")
            weight_suffix = Path(weight_file.filename).suffix.lower()
            if weight_suffix not in INFERENCE_WEIGHT_EXTENSIONS:
                raise HTTPException(status_code=400, detail="Uploaded weights must be a .pt or .onnx file.")
            normalized_family = str(weight_family or "ultralytics").strip().lower()
            if normalized_family == "auto":
                normalized_family = "ultralytics"
            if normalized_family not in {"ultralytics", "rfdetr", "dfine"}:
                raise HTTPException(status_code=400, detail="Unknown inference weights backend.")
            if normalized_family in {"dfine", "rfdetr"} and weight_suffix != ".pt":
                raise HTTPException(status_code=400, detail="Uploaded DETR-family weights must be a .pt file.")
            if convert_to_onnx and weight_suffix != ".pt":
                convert_to_onnx = False
            if normalized_family in {"dfine", "rfdetr"}:
                convert_to_onnx = False
            weights_path = (INFERENCE_UPLOAD_ROOT / job_id / Path(weight_file.filename).name).resolve()
            ensure_inference_path(weights_path)
            save_upload(weight_file, weights_path, lambda *_args: None, "saving", "Saving inference weights")
            weights_label = f"uploaded {weights_path.name}{' -> ONNX' if convert_to_onnx else ''}"
            inference_family = normalized_family
        else:
            raise HTTPException(status_code=400, detail="Unknown inference weights source.")

        if inference_family == "dfine":
            raise HTTPException(
                status_code=400,
                detail="D-FINE inference is not available yet. Choose Ultralytics or RF-DETR weights.",
            )

        if not media_file.filename:
            raise HTTPException(status_code=400, detail="Choose an image or video file for inference.")
        media_name = safe_upload_path(media_file.filename).name
        input_path = (job_dir / media_name).resolve()
        ensure_inference_path(input_path)
        save_upload(media_file, input_path, lambda *_args: None, "saving", "Saving inference media")
        media_kind = inference_media_type(input_path)
        output_path = job_dir / ("annotated.jpg" if media_kind == "image" else "annotated.mp4")
        result_path = job_dir / "result.json"
        progress_path = job_dir / "progress.json"
        preview_path = job_dir / "preview.jpg"
        inference_device = os.getenv("INFERENCE_DEVICE") or os.getenv("TRAINING_DEVICE") or ""
        inference_request = {
            "weights_path": weights_path,
            "input_path": input_path,
            "output_path": output_path,
            "json_path": result_path,
            "progress_path": progress_path,
            "preview_path": preview_path,
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "device": inference_device,
            "vid_stride": vid_stride,
            "show_boxes": show_boxes,
            "show_labels": show_labels,
            "show_conf": show_conf,
            "show_masks": show_masks,
            "convert_to_onnx": convert_to_onnx,
            "family": inference_family,
        }
        stop_event = threading.Event()
        preview_condition = threading.Condition()

        job_payload = {
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
            "percent": 0,
            "detail": "Inference job queued.",
            "frames": 0,
            "total_frames": 0,
            "detections": 0,
            "classifications": 0,
            "task": "",
            "elapsed_ms": 0,
            "media_type": media_kind,
            "weights": weights_label,
            "preview_available": False,
            "preview_fps": 0,
            "stoppable": False,
            "stop_event": stop_event,
            "preview_condition": preview_condition,
            "preview_frame": None,
            "preview_frame_times": [],
            "preview_updated_at": 0,
            "webrtc_frame": None,
            "webrtc_frame_seq": 0,
            "webrtc_updated_at": 0,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        with inference_jobs_lock:
            inference_jobs[job_id] = job_payload
        persist_job(job_id, "inference", "queued", inference_job_payload(job_payload), force=True)

        thread = threading.Thread(
            target=run_inference_job,
            args=(job_id, inference_request, result_path, output_path, preview_path, stop_event),
            daemon=True,
        )
        thread.start()
        return inference_job_payload(job_payload)


    @router.get("/api/inference/status/{job_id}")
    def inference_status(job_id: str):
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
            raise HTTPException(status_code=404, detail="Inference job not found.")
        with inference_jobs_lock:
            job = inference_jobs.get(job_id)
            if job is not None:
                return inference_job_payload(dict(job))
        job = inference_job_from_disk(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Inference job not found.")
        with inference_jobs_lock:
            inference_jobs[job_id] = dict(job)
        return inference_job_payload(job)


    @router.post("/api/inference/stop/{job_id}")
    def stop_inference(job_id: str):
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
            raise HTTPException(status_code=404, detail="Inference job not found.")
        return stop_inference_process(job_id)


    @router.get("/api/inference/preview/{job_id}")
    def inference_preview(job_id: str):
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
            raise HTTPException(status_code=404, detail="Inference preview not found.")
        job_dir = (INFERENCE_JOB_ROOT / job_id).resolve()
        ensure_inference_path(job_dir)
        path = job_dir / "preview.jpg"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Inference preview not available yet.")
        return FileResponse(path, media_type="image/jpeg")


    @router.get("/api/inference/stream/{job_id}")
    def inference_stream(job_id: str):
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
            raise HTTPException(status_code=404, detail="Inference stream not found.")
        with inference_jobs_lock:
            if job_id not in inference_jobs:
                raise HTTPException(status_code=404, detail="Inference stream not available.")
        return StreamingResponse(
            inference_mjpeg_stream(job_id),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )


    @router.post("/api/inference/webrtc/{job_id}/offer")
    async def inference_webrtc_offer(job_id: str, offer: WebRTCOffer):
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
            raise HTTPException(status_code=404, detail="Inference WebRTC stream not found.")
        with inference_jobs_lock:
            if job_id not in inference_jobs:
                raise HTTPException(status_code=404, detail="Inference WebRTC stream not available.")

        try:
            import numpy as np
            from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
            from aiortc.mediastreams import MediaStreamError
            from av import VideoFrame
        except ModuleNotFoundError as exc:
            missing = exc.name or "aiortc"
            raise HTTPException(
                status_code=501,
                detail=f"WebRTC live preview requires {missing}. Rebuild the container after updating requirements.txt.",
            ) from exc

        class InferenceVideoTrack(VideoStreamTrack):
            kind = "video"

            def __init__(self, stream_job_id: str):
                super().__init__()
                self.stream_job_id = stream_job_id
                self.last_frame = None

            async def recv(self):
                pts, time_base = await self.next_timestamp()
                frame, status = latest_inference_webrtc_frame(self.stream_job_id)
                if status in {"complete", "failed", "stopped", "missing"}:
                    raise MediaStreamError
                if frame is not None:
                    self.last_frame = frame
                elif self.last_frame is not None:
                    frame = self.last_frame
                else:
                    frame = np.zeros((360, 640, 3), dtype=np.uint8)

                video_frame = VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="bgr24")
                video_frame.pts = pts
                video_frame.time_base = time_base
                return video_frame

        peer = RTCPeerConnection()
        peer.addTrack(InferenceVideoTrack(job_id))
        with inference_peers_lock:
            inference_peers.setdefault(job_id, set()).add(peer)

        @peer.on("connectionstatechange")
        async def on_connectionstatechange():
            if peer.connectionState in {"failed", "disconnected", "closed"}:
                with inference_peers_lock:
                    peers = inference_peers.get(job_id)
                    if peers is not None:
                        peers.discard(peer)
                        if not peers:
                            inference_peers.pop(job_id, None)
                if peer.connectionState != "closed":
                    await peer.close()

        try:
            remote_offer = RTCSessionDescription(sdp=offer.sdp, type=offer.type)
            await peer.setRemoteDescription(remote_offer)
            answer = await peer.createAnswer()
            await peer.setLocalDescription(answer)
        except Exception as exc:
            with inference_peers_lock:
                peers = inference_peers.get(job_id)
                if peers is not None:
                    peers.discard(peer)
            await peer.close()
            raise HTTPException(status_code=400, detail=f"Could not create WebRTC stream: {exc}") from exc

        return {
            "sdp": peer.localDescription.sdp,
            "type": peer.localDescription.type,
        }


    @router.get("/api/inference/result/{job_id}")
    def inference_result(job_id: str, download: bool = False):
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", job_id):
            raise HTTPException(status_code=404, detail="Inference result not found.")
        job_dir = (INFERENCE_JOB_ROOT / job_id).resolve()
        ensure_inference_path(job_dir)
        candidates = [job_dir / "annotated.jpg", job_dir / "annotated.mp4"]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise HTTPException(status_code=404, detail="Inference result not found.")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        filename = f"inference_{job_id}{path.suffix}" if download else path.name
        return FileResponse(path, media_type=media_type, filename=filename)

    return router
