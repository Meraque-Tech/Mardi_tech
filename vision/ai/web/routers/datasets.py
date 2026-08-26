"""Dataset upload, preparation, download, and Roboflow HTTP routes."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..common.uploads import (
    UploadLimits,
    UploadValidationError,
    extract_zip_safely,
    replace_directory,
    safe_leaf_filename,
    validate_upload_totals,
)
from ..roboflow_sync import (
    build_roboflow_provenance,
    redact_secret,
    roboflow_search_images,
    write_roboflow_provenance,
)
from ..schemas import DatasetDownloadRequest, RoboflowRequest, SplitConfig
from ..services.datasets import (
    build_annotated_dataset_archive,
    build_dataset_archive,
    cleanup_expired_dataset_downloads,
    dataset_response,
    prepare_dataset,
    prepare_roboflow_download,
    prepared_dataset_yaml,
)

DatasetProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class DatasetRouterDependencies:
    data_root: Path
    summary_filename: str
    upload_limits: UploadLimits
    preparation_jobs: dict[str, dict]
    preparation_lock: object
    downloads: dict[str, dict]
    downloads_lock: object
    resource_coordinator: object
    acquire_resource: Callable
    clean_name: Callable[[str, str], str]
    dataset_progress_callback: Callable
    safe_upload_path: Callable[[str], Path]
    update_dataset_preparation: Callable
    upload_validation_http_error: Callable
    write_json_object: Callable[[Path, dict], None]


def create_dataset_router(deps: DatasetRouterDependencies) -> APIRouter:
    router = APIRouter()
    DATA_ROOT = deps.data_root
    DATASET_SUMMARY_FILE = deps.summary_filename
    UPLOAD_LIMITS = deps.upload_limits
    dataset_preparation_jobs = deps.preparation_jobs
    dataset_preparation_lock = deps.preparation_lock
    dataset_downloads = deps.downloads
    dataset_download_lock = deps.downloads_lock
    resource_coordinator = deps.resource_coordinator
    acquire_resource = deps.acquire_resource
    clean_name = deps.clean_name
    dataset_progress_callback = deps.dataset_progress_callback
    safe_upload_path = deps.safe_upload_path
    update_dataset_preparation = deps.update_dataset_preparation
    upload_validation_http_error = deps.upload_validation_http_error
    write_json_object = deps.write_json_object

    @router.get("/api/dataset/preparation/status")
    def dataset_preparation_status(job_id: str):
        with dataset_preparation_lock:
            progress = dataset_preparation_jobs.get(job_id)
            if progress is None:
                raise HTTPException(status_code=404, detail="Dataset preparation job not found.")
            return dict(progress)


    @router.post("/api/dataset/download/prepare")
    def prepare_dataset_download(request: DatasetDownloadRequest):
        cleanup_expired_dataset_downloads()
        archive_path, filename = build_dataset_archive(request.dataset_yaml)
        download_id = uuid.uuid4().hex
        with dataset_download_lock:
            dataset_downloads[download_id] = {
                "path": str(archive_path),
                "filename": filename,
                "created_at": time.time(),
            }
        return {
            "download_url": f"/api/dataset/download/{download_id}",
            "filename": filename,
            "size": archive_path.stat().st_size,
        }


    @router.post("/api/dataset/download/annotated/prepare")
    def prepare_annotated_dataset_download(request: DatasetDownloadRequest):
        cleanup_expired_dataset_downloads()
        archive_path, filename = build_annotated_dataset_archive(request.dataset_yaml)
        download_id = uuid.uuid4().hex
        with dataset_download_lock:
            dataset_downloads[download_id] = {
                "path": str(archive_path),
                "filename": filename,
                "created_at": time.time(),
            }
        return {
            "download_url": f"/api/dataset/download/{download_id}",
            "filename": filename,
            "size": archive_path.stat().st_size,
        }


    @router.get("/api/dataset/download/{download_id}")
    def download_prepared_dataset(download_id: str, background_tasks: BackgroundTasks):
        with dataset_download_lock:
            download = dataset_downloads.pop(download_id, None)
        if download is None:
            raise HTTPException(status_code=404, detail="Dataset download is unavailable or has expired.")

        archive_path = Path(download["path"])
        if not archive_path.is_file():
            raise HTTPException(status_code=404, detail="Prepared dataset ZIP was not found.")
        background_tasks.add_task(archive_path.unlink, missing_ok=True)
        return FileResponse(
            archive_path,
            media_type="application/zip",
            filename=download["filename"],
        )


    def upload_size(upload: UploadFile) -> int:
        position = upload.file.tell()
        upload.file.seek(0, os.SEEK_END)
        size = upload.file.tell()
        upload.file.seek(position)
        return max(0, int(size))


    def save_upload(
        upload: UploadFile,
        target: Path,
        progress_callback: DatasetProgressCallback,
        stage: str,
        detail_prefix: str,
        offset: int = 0,
        total: Optional[int] = None,
    ) -> int:
        size = upload_size(upload)
        try:
            validate_upload_totals([size], UPLOAD_LIMITS)
        except UploadValidationError as exc:
            raise upload_validation_http_error(exc) from exc
        work_total = total if total is not None else size
        copied = 0
        upload.file.seek(0)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as output:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                copied += len(chunk)
                progress_callback(
                    stage,
                    offset + copied,
                    work_total,
                    f"{detail_prefix}: {offset + copied} of {work_total} bytes",
                )
        return copied


    def extract_zip_with_progress(
        zip_path: Path,
        extract_dir: Path,
        progress_callback: DatasetProgressCallback,
        stage: str = "extracting",
    ):
        staging_dir = extract_dir.with_name(f".{extract_dir.name}.extract-{uuid.uuid4().hex}")

        def report(index: int, count: int, extracted: int, total: int):
            work_total = total or count
            current = extracted if total else index
            progress_callback(
                stage,
                current,
                work_total,
                f"Extracting {index} of {count} ZIP entries",
            )

        try:
            progress_callback(stage, 0, 1, "Inspecting ZIP safety and size limits")
            extract_zip_safely(zip_path, staging_dir, UPLOAD_LIMITS, progress=report)
            replace_directory(staging_dir, extract_dir)
        except UploadValidationError as exc:
            raise upload_validation_http_error(exc) from exc
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)


    class RoboflowDirectDownloadUnavailable(Exception):
        """Raised when the installed SDK cannot expose its export download flow."""


    def normalized_roboflow_percent(value) -> Optional[float]:
        if value is None:
            return None
        try:
            percent = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(percent):
            return None
        if 0 <= percent <= 1:
            percent *= 100
        return min(100.0, max(0.0, percent))


    def report_roboflow_server_progress(
        progress_callback: DatasetProgressCallback,
        stage: str,
        progress,
        indeterminate_detail: str,
        determinate_prefix: str,
    ):
        percent = normalized_roboflow_percent(progress)
        if percent is None:
            progress_callback(stage, 0, 0, indeterminate_detail)
            return
        progress_callback(
            stage,
            round(percent * 10),
            1000,
            f"{determinate_prefix}: {percent:.1f}%",
        )


    def wait_for_roboflow_export(
        rfapi,
        api_key: str,
        workspace: str,
        project_name: str,
        version: str,
        dataset_format: str,
        progress_callback: DatasetProgressCallback,
        timeout_seconds: int = 1800,
    ) -> str:
        deadline = time.monotonic() + timeout_seconds
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Roboflow version preparation timed out.")
            try:
                version_response = rfapi.get_version(
                    api_key=api_key,
                    workspace_url=workspace,
                    project_url=project_name,
                    version=version,
                    nocache=True,
                )
            except TypeError as exc:
                raise RoboflowDirectDownloadUnavailable from exc
            version_info = version_response.get("version", {})
            generating = bool(
                version_info.get("generating")
                or version_info.get("images", 0) == 0
            )
            if not generating:
                break
            report_roboflow_server_progress(
                progress_callback,
                "preparing_roboflow_version",
                version_info.get("progress"),
                "Roboflow is preparing the dataset version.",
                "Preparing Roboflow dataset version",
            )
            time.sleep(5)

        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Roboflow export preparation timed out.")
            try:
                export_info = rfapi.get_version_export(
                    api_key=api_key,
                    workspace_url=workspace,
                    project_url=project_name,
                    version=version,
                    format=dataset_format,
                )
            except TypeError as exc:
                raise RoboflowDirectDownloadUnavailable from exc

            export = export_info.get("export")
            if isinstance(export, dict) and export.get("link"):
                progress_callback(
                    "preparing_roboflow_export",
                    1000,
                    1000,
                    "Roboflow export is ready.",
                )
                return str(export["link"])
            if export_info.get("ready") is False:
                report_roboflow_server_progress(
                    progress_callback,
                    "preparing_roboflow_export",
                    export_info.get("progress"),
                    "Roboflow is generating the YOLOv8 export.",
                    "Generating YOLOv8 export",
                )
                time.sleep(1)
                continue
            response_keys = ", ".join(sorted(str(key) for key in export_info)) or "none"
            raise RuntimeError(
                f"Unexpected Roboflow export response fields: {response_keys}."
            )


    def readable_byte_count(value: int) -> str:
        amount = float(max(0, value))
        units = ("B", "KB", "MB", "GB", "TB")
        for unit in units:
            if amount < 1024 or unit == units[-1]:
                digits = 0 if unit == "B" else 1
                return f"{amount:.{digits}f} {unit}"
            amount /= 1024
        return f"{amount:.1f} TB"


    def stream_roboflow_export(
        link: str,
        zip_path: Path,
        progress_callback: DatasetProgressCallback,
    ):
        try:
            import requests
        except ImportError as exc:
            raise RoboflowDirectDownloadUnavailable from exc

        try:
            response_context = requests.get(link, stream=True, timeout=(30, 300))
        except TypeError as exc:
            raise RoboflowDirectDownloadUnavailable from exc

        with response_context as response:
            response.raise_for_status()
            try:
                total_bytes = max(0, int(response.headers.get("content-length", 0)))
            except (TypeError, ValueError):
                total_bytes = 0
            if total_bytes > UPLOAD_LIMITS.max_file_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Roboflow export exceeds the configured "
                        f"{UPLOAD_LIMITS.max_file_bytes}-byte download limit."
                    ),
                )
            downloaded = 0
            progress_callback(
                "downloading_roboflow",
                0,
                total_bytes,
                (
                    f"Downloading Roboflow dataset: 0 of {readable_byte_count(total_bytes)}"
                    if total_bytes
                    else "Downloading Roboflow dataset; total size is unavailable."
                ),
            )
            with zip_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    output.write(chunk)
                    downloaded += len(chunk)
                    if downloaded > UPLOAD_LIMITS.max_file_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail="Roboflow export exceeded the configured download limit.",
                        )
                    detail = f"Downloaded {readable_byte_count(downloaded)}"
                    if total_bytes:
                        detail += f" of {readable_byte_count(total_bytes)}"
                    progress_callback(
                        "downloading_roboflow",
                        downloaded,
                        total_bytes,
                        detail,
                    )


    def download_roboflow_dataset_with_progress(
        version_obj,
        api_key: str,
        workspace: str,
        project_name: str,
        version: str,
        dataset_format: str,
        download_dir: Path,
        progress_callback: DatasetProgressCallback,
    ) -> Path:
        try:
            from roboflow.adapters import rfapi
        except ImportError as exc:
            raise RoboflowDirectDownloadUnavailable from exc

        required_attributes = ("get_version", "get_version_export")
        if any(not hasattr(rfapi, attribute) for attribute in required_attributes):
            raise RoboflowDirectDownloadUnavailable

        resolved_workspace = str(getattr(version_obj, "workspace", workspace) or workspace)
        resolved_project = str(getattr(version_obj, "project", project_name) or project_name)
        link = wait_for_roboflow_export(
            rfapi,
            api_key,
            resolved_workspace,
            resolved_project,
            str(version_obj.version if hasattr(version_obj, "version") else version),
            dataset_format,
            progress_callback,
        )
        zip_path = download_dir / "roboflow.zip"
        stream_roboflow_export(link, zip_path, progress_callback)
        extract_zip_with_progress(
            zip_path,
            download_dir,
            progress_callback,
            stage="extracting_roboflow",
        )
        zip_path.unlink(missing_ok=True)
        return download_dir


    def mark_dataset_preparation_failed(job_id: str, exc: Exception):
        detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
        update_dataset_preparation(job_id, "failed", 0, 0, str(detail), status="failed")


    @router.post("/api/dataset/upload")
    def upload_dataset(
        file: UploadFile = File(...),
        classes: str = Form(...),
        train: int = Form(70),
        val: int = Form(15),
        test: int = Form(15),
        name: str = Form("dataset"),
        force_split: bool = Form(False),
        job_id: str = Form(""),
    ):
        if not file.filename or not file.filename.lower().endswith(".zip"):
            raise HTTPException(status_code=400, detail="Please upload a ZIP file.")

        try:
            class_names = json.loads(classes)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Classes must be valid JSON.") from exc
        split = SplitConfig(train=train, val=val, test=test)
        clean = clean_name(name, "uploaded_dataset")
        upload_dir = DATA_ROOT / "uploads" / clean
        extract_dir = DATA_ROOT / "extracted" / clean
        progress_callback = dataset_progress_callback(job_id)
        workspace_owner = job_id or uuid.uuid4().hex
        staging_upload_dir = upload_dir.with_name(f".{upload_dir.name}.upload-{uuid.uuid4().hex}")
        acquire_resource(f"dataset:{clean}", workspace_owner, f"dataset preparation for {clean}")
        try:
            zip_size = upload_size(file)
            try:
                validate_upload_totals([zip_size], UPLOAD_LIMITS)
                upload_filename = safe_leaf_filename(file.filename)
            except UploadValidationError as exc:
                raise upload_validation_http_error(exc) from exc
            progress_callback("saving", 0, zip_size, "Preparing the upload destination")
            staging_upload_dir.mkdir(parents=True, exist_ok=True)

            zip_path = staging_upload_dir / upload_filename
            progress_callback("saving", 0, zip_size, f"Saving uploaded ZIP: 0 of {zip_size} bytes")
            save_upload(file, zip_path, progress_callback, "saving", "Saving uploaded ZIP")
            extract_zip_with_progress(zip_path, extract_dir, progress_callback)
            replace_directory(staging_upload_dir, upload_dir)

            yaml_path = prepare_dataset(
                source=extract_dir,
                name=clean,
                classes=class_names,
                split=split,
                force_split=force_split,
                progress_callback=progress_callback,
            )
            response = dataset_response(yaml_path, "Uploaded dataset is ready.", progress_callback)
            update_dataset_preparation(job_id, "complete", 1, 1, "Dataset preparation complete.", status="complete")
            return response
        except zipfile.BadZipFile as exc:
            mark_dataset_preparation_failed(job_id, exc)
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP.") from exc
        except Exception as exc:
            mark_dataset_preparation_failed(job_id, exc)
            raise
        finally:
            if staging_upload_dir.exists():
                shutil.rmtree(staging_upload_dir, ignore_errors=True)
            resource_coordinator.release(f"dataset:{clean}", workspace_owner)


    @router.post("/api/dataset/folder")
    def upload_folder_dataset(
        files: list[UploadFile] = File(...),
        classes: str = Form(...),
        train: int = Form(70),
        val: int = Form(15),
        test: int = Form(15),
        name: str = Form("dataset"),
        force_split: bool = Form(False),
        job_id: str = Form(""),
    ):
        if not files:
            raise HTTPException(status_code=400, detail="Choose a dataset folder first.")

        try:
            class_names = json.loads(classes)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Classes must be valid JSON.") from exc

        split = SplitConfig(train=train, val=val, test=test)
        clean = clean_name(name, "uploaded_folder_dataset")
        upload_dir = DATA_ROOT / "folder_uploads" / clean
        progress_callback = dataset_progress_callback(job_id)
        workspace_owner = job_id or uuid.uuid4().hex
        staging_upload_dir = upload_dir.with_name(f".{upload_dir.name}.upload-{uuid.uuid4().hex}")
        acquire_resource(f"dataset:{clean}", workspace_owner, f"dataset preparation for {clean}")
        try:
            sizes = [upload_size(upload) for upload in files]
            try:
                total_bytes = validate_upload_totals(sizes, UPLOAD_LIMITS, folder=True)
            except UploadValidationError as exc:
                raise upload_validation_http_error(exc) from exc
            progress_callback("saving", 0, total_bytes, "Preparing the upload destination")
            staging_upload_dir.mkdir(parents=True, exist_ok=True)
            upload_root = staging_upload_dir.resolve()
            saved_bytes = 0
            saved_count = 0
            progress_callback("saving", 0, total_bytes, f"Saving 0 of {len(files)} files")

            for upload in files:
                relative_path = safe_upload_path(upload.filename or "")
                target = (staging_upload_dir / relative_path).resolve()
                try:
                    target.relative_to(upload_root)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=f"Invalid upload path: {upload.filename}") from exc

                saved_bytes += save_upload(
                    upload,
                    target,
                    progress_callback,
                    "saving",
                    f"Saving file {saved_count + 1} of {len(files)}",
                    offset=saved_bytes,
                    total=total_bytes,
                )
                saved_count += 1

            if saved_count == 0:
                raise HTTPException(status_code=400, detail="No files were uploaded.")

            replace_directory(staging_upload_dir, upload_dir)

            yaml_path = prepare_dataset(
                source=upload_dir,
                name=clean,
                classes=class_names,
                split=split,
                force_split=force_split,
                progress_callback=progress_callback,
            )
            response = dataset_response(yaml_path, "Uploaded folder dataset is ready.", progress_callback)
            update_dataset_preparation(job_id, "complete", 1, 1, "Dataset preparation complete.", status="complete")
            return response
        except Exception as exc:
            mark_dataset_preparation_failed(job_id, exc)
            raise
        finally:
            if staging_upload_dir.exists():
                shutil.rmtree(staging_upload_dir, ignore_errors=True)
            resource_coordinator.release(f"dataset:{clean}", workspace_owner)


    @router.post("/api/dataset/roboflow")
    def roboflow_dataset(request: RoboflowRequest):
        api_key = request.api_key or os.getenv("ROBOFLOW_API_KEY")
        workspace = request.workspace or os.getenv("ROBOFLOW_WORKSPACE")
        project_name = request.project or os.getenv("ROBOFLOW_PROJECT")
        version = request.version or os.getenv("ROBOFLOW_VERSION")
        dataset_format = "yolov8"

        missing = [
            name for name, value in {
                "api_key": api_key,
                "workspace": workspace,
                "project": project_name,
                "version": version,
            }.items()
            if not value
        ]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing Roboflow fields: {', '.join(missing)}")

        try:
            from roboflow import Roboflow
        except ImportError as exc:
            raise HTTPException(
                status_code=500,
                detail="The roboflow package is not installed. Run: pip install roboflow",
            ) from exc

        clean = clean_name(request.name, "dataset")
        download_dir = DATA_ROOT / "roboflow" / clean
        progress_callback = dataset_progress_callback(request.job_id)
        workspace_owner = request.job_id or uuid.uuid4().hex
        acquire_resource(f"dataset:{clean}", workspace_owner, f"Roboflow preparation for {clean}")
        progress_callback(
            "preparing_roboflow_version",
            0,
            0,
            "Connecting to Roboflow and checking the dataset version.",
        )

        try:
            if download_dir.exists():
                shutil.rmtree(download_dir)
            download_dir.mkdir(parents=True, exist_ok=True)
            rf = Roboflow(api_key=api_key)
            project = rf.workspace(workspace).project(project_name)
            version_obj = project.version(int(version))
            try:
                dataset_root = download_roboflow_dataset_with_progress(
                    version_obj,
                    str(api_key),
                    str(workspace),
                    str(project_name),
                    str(version),
                    dataset_format,
                    download_dir,
                    progress_callback,
                )
            except RoboflowDirectDownloadUnavailable:
                progress_callback(
                    "fetching_roboflow",
                    0,
                    0,
                    "The installed Roboflow SDK does not expose download progress; using its standard downloader.",
                )
                try:
                    dataset = version_obj.download(
                        dataset_format,
                        location=str(download_dir),
                        overwrite=True,
                    )
                except TypeError:
                    dataset = version_obj.download(dataset_format, location=str(download_dir))
                dataset_root = Path(getattr(dataset, "location", download_dir))
        except Exception as exc:
            shutil.rmtree(download_dir, ignore_errors=True)
            error = HTTPException(status_code=502, detail=f"Roboflow download failed: {exc}")
            mark_dataset_preparation_failed(request.job_id, error)
            raise error from exc

        try:
            if request.force_split:
                split = SplitConfig(train=request.train, val=request.val, test=request.test)
                yaml_path = prepare_dataset(
                    source=dataset_root,
                    name=clean,
                    classes=request.classes,
                    split=split,
                    force_split=True,
                    progress_callback=progress_callback,
                )
                message = (
                    "Roboflow dataset is ready. The downloaded split was rebuilt locally "
                    f"to {request.train}/{request.val}/{request.test}; the Roboflow version "
                    "was not modified."
                )
            else:
                progress_callback(
                    "validating_dataset",
                    0,
                    0,
                    "Validating the downloaded dataset paths and split folders.",
                )
                split = SplitConfig(train=request.train, val=request.val, test=request.test)
                yaml_path, preparation_action = prepare_roboflow_download(
                    dataset_root,
                    clean,
                    request.classes,
                    split,
                    progress_callback,
                )
                if preparation_action == "normalized":
                    progress_callback(
                        "normalizing_paths",
                        1,
                        1,
                        "Normalized invalid export paths to the detected split folders.",
                    )
                elif preparation_action == "rebuilt":
                    progress_callback(
                        "rebuilding_split",
                        1,
                        1,
                        "Rebuilt a local train/validation/test split from the usable downloaded images.",
                    )
                if preparation_action == "normalized":
                    message = (
                        "Roboflow dataset is ready. Invalid export paths were normalized "
                        "to the detected train/validation/test folders."
                    )
                elif preparation_action == "rebuilt":
                    message = (
                        "Roboflow dataset is ready. The downloaded export had invalid split paths, "
                        f"so a local {request.train}/{request.val}/{request.test} split was rebuilt "
                        "from the usable images."
                    )
                else:
                    message = "Roboflow dataset is ready."

            response = dataset_response(
                yaml_path,
                message,
                progress_callback,
                source_metadata=(
                    {
                        "augmentation": getattr(version_obj, "augmentation", None),
                        "images": getattr(version_obj, "images", None),
                        "splits": getattr(version_obj, "splits", None),
                    }
                    if not request.force_split
                    else None
                ),
            )
            try:
                _prepared_yaml, prepared_root, prepared_payload = prepared_dataset_yaml(str(yaml_path))
                progress_callback(
                    "binding_roboflow_images",
                    0,
                    0,
                    "Binding local files to their source Roboflow image IDs.",
                )
                mapping_error = ""
                try:
                    remote_records = roboflow_search_images(
                        str(api_key), str(workspace), str(project_name)
                    )
                except Exception as exc:
                    remote_records = []
                    mapping_error = redact_secret(exc, str(api_key))
                provenance = build_roboflow_provenance(
                    dataset_root=prepared_root,
                    yaml_payload=prepared_payload,
                    class_names=response.get("classes") or [],
                    workspace=str(workspace),
                    project=str(project_name),
                    version=str(version),
                    remote_records=remote_records,
                )
                if mapping_error:
                    provenance["mapping_error"] = mapping_error
                write_roboflow_provenance(prepared_root, provenance)
                response["summary"]["roboflow_sync"] = {
                    "workspace": provenance["workspace"],
                    "project": provenance["project"],
                    "source_version": provenance["source_version"],
                    **provenance["mapping"],
                    "available": provenance["mapping"]["eligible"] > 0,
                    "error": mapping_error,
                }
                write_json_object(prepared_root / DATASET_SUMMARY_FILE, response["summary"])
            except Exception as exc:
                response["summary"]["roboflow_sync"] = {
                    "workspace": str(workspace),
                    "project": str(project_name),
                    "source_version": str(version),
                    "available": False,
                    "error": f"Roboflow image binding failed: {exc}",
                }
            update_dataset_preparation(
                request.job_id,
                "complete",
                1,
                1,
                "Dataset preparation complete.",
                status="complete",
            )
            return response
        except Exception as exc:
            mark_dataset_preparation_failed(request.job_id, exc)
            raise
        finally:
            resource_coordinator.release(f"dataset:{clean}", workspace_owner)

    return router
