"""Upload limits and safe ZIP extraction independent of FastAPI."""

from __future__ import annotations

import os
import shutil
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable


class UploadValidationError(ValueError):
    """Raised when an upload violates a configured safety limit."""


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _positive_env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


@dataclass(frozen=True)
class UploadLimits:
    max_file_bytes: int = 10 * 1024**3
    max_folder_bytes: int = 25 * 1024**3
    max_folder_files: int = 100_000
    max_zip_entries: int = 100_000
    max_zip_uncompressed_bytes: int = 50 * 1024**3
    max_zip_compression_ratio: float = 500.0

    @classmethod
    def from_environment(cls) -> "UploadLimits":
        return cls(
            max_file_bytes=_positive_env_int("WEB_MAX_UPLOAD_BYTES", cls.max_file_bytes),
            max_folder_bytes=_positive_env_int("WEB_MAX_FOLDER_UPLOAD_BYTES", cls.max_folder_bytes),
            max_folder_files=_positive_env_int("WEB_MAX_FOLDER_FILES", cls.max_folder_files),
            max_zip_entries=_positive_env_int("WEB_MAX_ZIP_ENTRIES", cls.max_zip_entries),
            max_zip_uncompressed_bytes=_positive_env_int(
                "WEB_MAX_ZIP_UNCOMPRESSED_BYTES", cls.max_zip_uncompressed_bytes
            ),
            max_zip_compression_ratio=_positive_env_float(
                "WEB_MAX_ZIP_COMPRESSION_RATIO", cls.max_zip_compression_ratio
            ),
        )


def safe_leaf_filename(filename: str, fallback: str = "upload") -> str:
    """Return a portable leaf name and reject traversal-like empty names."""
    normalized = str(filename or "").replace("\\", "/").replace("\x00", "")
    name = normalized.rsplit("/", 1)[-1].strip()
    if not name or name in {".", ".."}:
        raise UploadValidationError("The uploaded file has an invalid filename.")
    return name or fallback


def validate_upload_totals(sizes: list[int], limits: UploadLimits, folder: bool = False) -> int:
    if folder and len(sizes) > limits.max_folder_files:
        raise UploadValidationError(
            f"Folder upload contains {len(sizes)} files; the limit is {limits.max_folder_files}."
        )
    if any(size < 0 for size in sizes):
        raise UploadValidationError("Upload size could not be determined safely.")
    if any(size > limits.max_file_bytes for size in sizes):
        raise UploadValidationError(
            f"An uploaded file exceeds the {limits.max_file_bytes}-byte per-file limit."
        )
    total = sum(sizes)
    total_limit = limits.max_folder_bytes if folder else limits.max_file_bytes
    if total > total_limit:
        raise UploadValidationError(f"Upload exceeds the configured {total_limit}-byte limit.")
    return total


def _safe_member_path(name: str) -> Path:
    normalized = name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if (
        not normalized
        or "\x00" in normalized
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or (pure.parts and ":" in pure.parts[0])
    ):
        raise UploadValidationError(f"ZIP contains an unsafe member path: {name!r}")
    return Path(*pure.parts)


def inspect_zip(archive: zipfile.ZipFile, limits: UploadLimits) -> tuple[list[tuple[zipfile.ZipInfo, Path]], int]:
    entries = archive.infolist()
    if len(entries) > limits.max_zip_entries:
        raise UploadValidationError(
            f"ZIP contains {len(entries)} entries; the limit is {limits.max_zip_entries}."
        )

    inspected: list[tuple[zipfile.ZipInfo, Path]] = []
    total_uncompressed = 0
    seen: set[Path] = set()
    for entry in entries:
        relative = _safe_member_path(entry.filename)
        if relative in seen:
            raise UploadValidationError(f"ZIP contains a duplicate member: {entry.filename}")
        seen.add(relative)
        mode = entry.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise UploadValidationError(f"ZIP symbolic links are not accepted: {entry.filename}")
        if entry.flag_bits & 0x1:
            raise UploadValidationError(f"Encrypted ZIP entries are not accepted: {entry.filename}")
        total_uncompressed += max(0, int(entry.file_size))
        if total_uncompressed > limits.max_zip_uncompressed_bytes:
            raise UploadValidationError(
                "ZIP uncompressed content exceeds the configured "
                f"{limits.max_zip_uncompressed_bytes}-byte limit."
            )
        compressed = max(1, int(entry.compress_size))
        ratio = int(entry.file_size) / compressed
        if ratio > limits.max_zip_compression_ratio:
            raise UploadValidationError(
                f"ZIP entry {entry.filename!r} exceeds the allowed compression ratio."
            )
        inspected.append((entry, relative))
    return inspected, total_uncompressed


def extract_zip_safely(
    zip_path: Path,
    destination: Path,
    limits: UploadLimits,
    progress: Callable[[int, int, int, int], None] | None = None,
) -> None:
    """Extract regular files while enforcing paths and actual byte limits."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        entries, declared_total = inspect_zip(archive, limits)
        extracted = 0
        for index, (entry, relative) in enumerate(entries, start=1):
            target = (destination / relative).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise UploadValidationError(f"ZIP member escapes its destination: {entry.filename}") from exc
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry, "r") as source, target.open("wb") as output:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        extracted += len(chunk)
                        if extracted > limits.max_zip_uncompressed_bytes:
                            raise UploadValidationError("ZIP expanded beyond its configured size limit.")
                        output.write(chunk)
            if progress is not None:
                progress(index, len(entries), extracted, declared_total)


def replace_directory(source: Path, destination: Path) -> None:
    """Publish a prepared directory only after its replacement is complete."""
    backup = destination.with_name(f".{destination.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.replace(backup)
    try:
        source.replace(destination)
    except Exception:
        if backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)
