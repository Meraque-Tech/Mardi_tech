"""Strict Roboflow provenance and annotation synchronization helpers."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote


ROBOFLOW_PROVENANCE_FILE = ".roboflow_provenance.json"
ROBOFLOW_SYNC_PREVIEW_FILE = "roboflow_sync_preview.json"
ROBOFLOW_SYNC_LOG_FILE = "roboflow_sync_log.json"


def redact_secret(value: Any, secret: str) -> str:
    text = str(value)
    return text.replace(str(secret), "[redacted]") if secret else text


def _json_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_roboflow_filename(name: str) -> str:
    """Return a conservative key shared by common Roboflow export filenames."""
    filename = Path(str(name or "")).name.lower()
    stem = Path(filename).stem
    stem = re.sub(r"\.rf\.[a-z0-9_-]+$", "", stem)
    stem = re.sub(r"_(?:jpg|jpeg|png|bmp|webp)$", "", stem)
    return re.sub(r"[^a-z0-9]+", "-", stem).strip("-")


def canonical_local_annotation(label_path: Path, class_names: list[str]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    if not label_path.is_file():
        return rows
    for raw_line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw_line.strip().split()
        if len(fields) != 5:
            continue
        try:
            class_id = int(float(fields[0]))
            values = [round(float(value), 6) for value in fields[1:]]
        except (TypeError, ValueError):
            continue
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
        rows.append([class_name, *values])
    return sorted(rows, key=lambda row: (str(row[0]), *row[1:]))


def canonical_remote_annotation(image_payload: dict, class_names: list[str]) -> list[list[Any]]:
    image = image_payload.get("image") if isinstance(image_payload.get("image"), dict) else image_payload
    annotation = image.get("annotation") if isinstance(image, dict) else None
    if not isinstance(annotation, dict):
        return []
    try:
        width = float(annotation.get("width") or 0)
        height = float(annotation.get("height") or 0)
    except (TypeError, ValueError):
        return []
    if width <= 0 or height <= 0:
        return []
    rows: list[list[Any]] = []
    for box in annotation.get("boxes") or []:
        if not isinstance(box, dict):
            continue
        try:
            values = [
                round(float(box["x"]) / width, 6),
                round(float(box["y"]) / height, 6),
                round(float(box["width"]) / width, 6),
                round(float(box["height"]) / height, 6),
            ]
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        label = str(box.get("label") or "").strip()
        if not label:
            continue
        rows.append([label, *values])
    return sorted(rows, key=lambda row: (str(row[0]), *row[1:]))


def annotation_digest(annotation: list[list[Any]]) -> str:
    return _json_digest(annotation)


def label_relative_for_image(image_relative: str) -> Path:
    parts = list(Path(image_relative).parts)
    try:
        parts[parts.index("images")] = "labels"
    except ValueError:
        return Path("labels") / f"{Path(image_relative).stem}.txt"
    return Path(*parts).with_suffix(".txt")


def dataset_image_paths(dataset_root: Path, yaml_payload: dict) -> list[Path]:
    found: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        values = yaml_payload.get(split)
        if not values:
            continue
        for value in values if isinstance(values, list) else [values]:
            folder = Path(str(value)).expanduser()
            if not folder.is_absolute():
                folder = dataset_root / folder
            if not folder.is_dir():
                continue
            for image_path in folder.rglob("*"):
                if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    found[str(image_path.resolve())] = image_path.resolve()
    return sorted(found.values(), key=lambda path: str(path).lower())


def build_roboflow_provenance(
    dataset_root: Path,
    yaml_payload: dict,
    class_names: list[str],
    workspace: str,
    project: str,
    version: str,
    remote_records: list[dict],
) -> dict:
    """Bind local source images to unique Roboflow image IDs without guessing."""
    exact: dict[str, list[dict]] = {}
    normalized: dict[str, list[dict]] = {}
    for record in remote_records:
        if not isinstance(record, dict) or not record.get("id") or not record.get("name"):
            continue
        exact.setdefault(Path(str(record["name"])).name.lower(), []).append(record)
        normalized.setdefault(normalize_roboflow_filename(str(record["name"])), []).append(record)

    entries: dict[str, dict] = {}
    claims: dict[str, list[str]] = {}
    for image_path in dataset_image_paths(dataset_root, yaml_payload):
        relative = image_path.relative_to(dataset_root).as_posix()
        candidates = exact.get(image_path.name.lower(), [])
        method = "exact_filename"
        if len(candidates) != 1:
            candidates = normalized.get(normalize_roboflow_filename(image_path.name), [])
            method = "normalized_filename"
        label_path = dataset_root / label_relative_for_image(relative)
        baseline = canonical_local_annotation(label_path, class_names)
        entry = {
            "local_image": relative,
            "local_label": label_relative_for_image(relative).as_posix(),
            "baseline_annotation": baseline,
            "baseline_digest": annotation_digest(baseline),
            "eligible": len(candidates) == 1,
            "match_method": method if len(candidates) == 1 else "",
            "roboflow_image_id": str(candidates[0]["id"]) if len(candidates) == 1 else "",
            "roboflow_image_name": str(candidates[0]["name"]) if len(candidates) == 1 else "",
            "reason": "" if len(candidates) == 1 else (
                "No unique Roboflow image match was found."
                if not candidates
                else "Multiple Roboflow images matched this filename."
            ),
        }
        entries[relative] = entry
        if entry["roboflow_image_id"]:
            claims.setdefault(entry["roboflow_image_id"], []).append(relative)

    for image_id, relative_paths in claims.items():
        if len(relative_paths) <= 1:
            continue
        exact_sources = [
            relative for relative in relative_paths
            if entries[relative].get("match_method") == "exact_filename"
        ]
        if len(exact_sources) == 1:
            for relative in relative_paths:
                if relative == exact_sources[0]:
                    continue
                entries[relative].update({
                    "eligible": False,
                    "reason": "This file is a generated derivative of an exact-matched source image.",
                })
            continue
        for relative in relative_paths:
            entries[relative].update({
                "eligible": False,
                "reason": "Multiple local files map to this source image; it may be an augmented derivative.",
            })

    eligible = sum(1 for entry in entries.values() if entry["eligible"])
    return {
        "schema_version": 1,
        "provider": "roboflow",
        "workspace": str(workspace),
        "project": str(project),
        "source_version": str(version),
        "class_names": list(class_names),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "images": entries,
        "mapping": {
            "total": len(entries),
            "eligible": eligible,
            "ineligible": len(entries) - eligible,
        },
    }


def load_roboflow_provenance(dataset_root: Path) -> dict:
    try:
        payload = json.loads((dataset_root / ROBOFLOW_PROVENANCE_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_roboflow_provenance(dataset_root: Path, payload: dict) -> None:
    (dataset_root / ROBOFLOW_PROVENANCE_FILE).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _request_json(response: Any, action: str) -> dict:
    if not getattr(response, "ok", False):
        detail = ""
        try:
            body = response.json()
            detail = str(body.get("error") or body.get("message") or body)
        except Exception:
            detail = str(getattr(response, "text", ""))[:500]
        raise RuntimeError(f"Roboflow {action} failed ({getattr(response, 'status_code', 'unknown')}): {detail}")
    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(f"Roboflow {action} returned invalid JSON.") from exc
    return payload if isinstance(payload, dict) else {}


def roboflow_search_images(
    api_key: str,
    workspace: str,
    project: str,
    request_post: Optional[Callable[..., Any]] = None,
) -> list[dict]:
    if request_post is None:
        import requests
        request_post = requests.post
    url = f"https://api.roboflow.com/{quote(workspace, safe='')}/{quote(project, safe='')}/search"
    offset = 0
    records: list[dict] = []
    while True:
        response = request_post(
            url,
            params={"api_key": api_key},
            json={
                "in_dataset": True,
                "offset": offset,
                "limit": 250,
                "fields": ["id", "name", "annotations", "split"],
            },
            timeout=(15, 60),
        )
        payload = _request_json(response, "image search")
        page = payload.get("results") if isinstance(payload.get("results"), list) else []
        records.extend(item for item in page if isinstance(item, dict))
        offset += len(page)
        total = int(payload.get("total") or len(records))
        if not page or offset >= total:
            return records


def roboflow_image_details(
    api_key: str,
    workspace: str,
    project: str,
    image_id: str,
    request_get: Optional[Callable[..., Any]] = None,
) -> dict:
    if request_get is None:
        import requests
        request_get = requests.get
    url = (
        f"https://api.roboflow.com/{quote(workspace, safe='')}/"
        f"{quote(project, safe='')}/images/{quote(image_id, safe='')}"
    )
    response = request_get(url, params={"api_key": api_key}, timeout=(15, 60))
    return _request_json(response, "image lookup")


def roboflow_upload_annotation(
    api_key: str,
    project: str,
    image_id: str,
    annotation_text: str,
    class_names: list[str],
    annotation_name: str,
    request_post: Optional[Callable[..., Any]] = None,
) -> dict:
    if request_post is None:
        import requests
        request_post = requests.post
    url = (
        f"https://api.roboflow.com/dataset/{quote(project, safe='')}/"
        f"annotate/{quote(image_id, safe='')}"
    )
    response = request_post(
        url,
        params={
            "api_key": api_key,
            "name": annotation_name,
            "overwrite": "true",
        },
        json={
            "annotationFile": annotation_text,
            "labelmap": {str(index): name for index, name in enumerate(class_names)},
        },
        timeout=(15, 90),
    )
    return _request_json(response, "annotation upload")


def sync_preview_digest(target: dict, items: list[dict]) -> str:
    stable_items = [
        {
            "local_image": item.get("local_image"),
            "roboflow_image_id": item.get("roboflow_image_id"),
            "baseline_digest": item.get("baseline_digest"),
            "remote_digest": item.get("remote_digest"),
            "corrected_digest": item.get("corrected_digest"),
            "status": item.get("status"),
        }
        for item in items
    ]
    return _json_digest({"target": target, "items": stable_items})
