"""Efficient filesystem helpers used by the web runtime."""

from __future__ import annotations

from pathlib import Path


def read_text_tail(path: Path, max_chars: int = 20_000, encoding: str = "utf-8") -> str:
    """Read at most the trailing ``max_chars`` without loading the whole file."""
    if max_chars <= 0 or not path.is_file():
        return ""

    # UTF-8 characters occupy at most four bytes. Reading one extra byte window
    # also gives the decoder room when the seek lands in a multibyte character.
    byte_window = max_chars * 4 + 4
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - byte_window))
            data = handle.read()
    except OSError:
        return ""
    return data.decode(encoding, errors="replace")[-max_chars:]


def file_signature(path: Path) -> tuple[int, int]:
    """Return a cheap cache signature, or a zero signature for missing files."""
    try:
        stat = path.stat()
    except OSError:
        return (0, 0)
    return (stat.st_mtime_ns, stat.st_size)
