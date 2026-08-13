"""Persistent, paginated storage for Annotation QA findings."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional


DATABASE_FILE = "issues.db"
SCHEMA_VERSION = 1

RESOLVED_SQL = """(
    review_status IN ('fix_accepted', 'accepted', 'false_positive', 'ignored')
    AND NOT (audit_required = 1 AND audit_status = 'pending')
)"""
UNRESOLVED_SQL = f"(NOT {RESOLVED_SQL} AND review_status != 'needs_fix')"


def database_path(run_dir: Path) -> Path:
    return run_dir / DATABASE_FILE


def _connect(run_dir: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path(run_dir), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def _safe_sam(issue: dict) -> int:
    metrics = issue.get("metrics") if isinstance(issue.get("metrics"), dict) else {}
    edges = metrics.get("edge_differences") if isinstance(metrics.get("edge_differences"), dict) else {}
    stability = metrics.get("prompt_stability") if isinstance(metrics.get("prompt_stability"), dict) else {}
    recommended = issue.get("recommended_bbox")
    return int(
        issue.get("auto_fix_eligible") is True
        and issue.get("quality_gate_passed") is True
        and issue.get("difference_band") == "reviewable"
        and issue.get("fix_type") == "replace_box"
        and isinstance(recommended, list)
        and len(recommended) == 4
        and stability.get("passed", True) is not False
        and edges.get("within_tolerance") is False
        and edges.get("within_max_difference") is True
    )


def _values(issue: dict, sequence: int) -> tuple:
    return (
        str(issue.get("issue_id") or ""),
        int(sequence),
        str(issue.get("image") or ""),
        str(issue.get("image_name") or ""),
        str(issue.get("split") or ""),
        issue.get("class_id"),
        str(issue.get("class_name") or ""),
        str(issue.get("issue_type") or ""),
        str(issue.get("severity") or "low"),
        float(issue.get("score") or 0.0),
        str(issue.get("review_status") or "unreviewed"),
        str(issue.get("accepted_fix") or ""),
        issue.get("accepted_class_id"),
        str(issue.get("accepted_class_name") or ""),
        int(bool(issue.get("audit_required"))),
        str(issue.get("audit_status") or "not_required"),
        str(issue.get("difference_band") or ""),
        _safe_sam(issue),
        int(bool(issue.get("applied"))),
        json.dumps(issue, separators=(",", ":"), ensure_ascii=False),
    )


INSERT_SQL = """
INSERT OR REPLACE INTO issues (
    issue_id, sequence, image, image_name, split, class_id, class_name,
    issue_type, severity, score, review_status, accepted_fix,
    accepted_class_id, accepted_class_name, audit_required, audit_status,
    difference_band, safe_sam, applied, payload
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def initialize_store(run_dir: Path, *, replace: bool = False) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = database_path(run_dir)
    if replace and path.exists():
        path.unlink()
    with _connect(run_dir) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS issues (
                issue_id TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL,
                image TEXT NOT NULL,
                image_name TEXT NOT NULL,
                split TEXT NOT NULL,
                class_id INTEGER,
                class_name TEXT NOT NULL,
                issue_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                score REAL NOT NULL,
                review_status TEXT NOT NULL,
                accepted_fix TEXT NOT NULL,
                accepted_class_id INTEGER,
                accepted_class_name TEXT NOT NULL,
                audit_required INTEGER NOT NULL,
                audit_status TEXT NOT NULL,
                difference_band TEXT NOT NULL,
                safe_sam INTEGER NOT NULL,
                applied INTEGER NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_qa_issues_sequence ON issues(sequence);
            CREATE INDEX IF NOT EXISTS idx_qa_issues_review ON issues(review_status, audit_required, audit_status);
            CREATE INDEX IF NOT EXISTS idx_qa_issues_queue ON issues(difference_band, safe_sam, severity);
            CREATE INDEX IF NOT EXISTS idx_qa_issues_filters ON issues(split, class_name, issue_type);
            CREATE INDEX IF NOT EXISTS idx_qa_issues_image_name ON issues(image_name);
            """
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )


class IssueWriter:
    """Buffered single-job writer used by the scanner."""

    def __init__(self, run_dir: Path, batch_size: int = 200):
        initialize_store(run_dir, replace=True)
        self.connection = _connect(run_dir)
        self.batch_size = max(1, int(batch_size))
        self.pending: list[tuple] = []
        self.count = 0

    def append(self, issue: dict) -> None:
        self.count += 1
        self.pending.append(_values(issue, self.count))
        if len(self.pending) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        self.connection.executemany(INSERT_SQL, self.pending)
        self.connection.commit()
        self.pending.clear()

    def close(self) -> None:
        self.flush()
        self.connection.close()


def store_exists(run_dir: Path) -> bool:
    return database_path(run_dir).is_file()


def import_issues(run_dir: Path, issues: Iterable[dict], *, replace: bool = True) -> int:
    initialize_store(run_dir, replace=replace)
    connection = _connect(run_dir)
    batch: list[tuple] = []
    count = 0
    try:
        for count, issue in enumerate(issues, start=1):
            if not isinstance(issue, dict) or not issue.get("issue_id"):
                continue
            batch.append(_values(issue, count))
            if len(batch) >= 500:
                connection.executemany(INSERT_SQL, batch)
                connection.commit()
                batch.clear()
        if batch:
            connection.executemany(INSERT_SQL, batch)
            connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    return count


def iter_legacy_issues(report_path: Path) -> Iterator[dict]:
    """Incrementally decode the issues array without loading a legacy report into memory."""
    decoder = json.JSONDecoder()
    marker_pattern = re.compile(r'"issues"\s*:\s*\[')
    buffer = ""
    started = False
    eof = False
    with report_path.open("r", encoding="utf-8") as file:
        while True:
            if not eof and len(buffer) < 1024 * 1024:
                chunk = file.read(1024 * 1024)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            if not started:
                marker_match = marker_pattern.search(buffer)
                if marker_match is None:
                    if eof:
                        raise ValueError("Legacy report has no issues array.")
                    buffer = buffer[-64:]
                    continue
                buffer = buffer[marker_match.end():]
                started = True
            buffer = buffer.lstrip(" \t\r\n,")
            if buffer.startswith("]"):
                return
            if not buffer and eof:
                raise ValueError("Legacy report ended before the issues array was complete.")
            try:
                issue, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = file.read(1024 * 1024)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
                continue
            buffer = buffer[end:]
            if isinstance(issue, dict):
                yield issue


def _decode(row: sqlite3.Row) -> dict:
    payload = json.loads(row["payload"])
    return payload if isinstance(payload, dict) else {}


def iter_issues(run_dir: Path, where: str = "", params: tuple = ()) -> Iterator[dict]:
    connection = _connect(run_dir)
    sql = "SELECT payload FROM issues"
    if where:
        sql += f" WHERE {where}"
    sql += " ORDER BY sequence"
    try:
        for row in connection.execute(sql, params):
            yield _decode(row)
    finally:
        connection.close()


def issue_count(run_dir: Path, where: str = "", params: tuple = ()) -> int:
    with _connect(run_dir) as connection:
        sql = "SELECT COUNT(*) FROM issues"
        if where:
            sql += f" WHERE {where}"
        return int(connection.execute(sql, params).fetchone()[0])


def get_issue(run_dir: Path, issue_id: str) -> Optional[dict]:
    with _connect(run_dir) as connection:
        row = connection.execute("SELECT payload FROM issues WHERE issue_id = ?", (issue_id,)).fetchone()
    return _decode(row) if row else None


def update_issue(run_dir: Path, issue_id: str, updater: Callable[[dict], None]) -> Optional[dict]:
    connection = _connect(run_dir)
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT sequence, payload FROM issues WHERE issue_id = ?", (issue_id,)
        ).fetchone()
        if row is None:
            connection.rollback()
            return None
        issue = _decode(row)
        updater(issue)
        connection.execute(INSERT_SQL, _values(issue, int(row["sequence"])))
        connection.commit()
        return issue
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def update_issues(run_dir: Path, issues: Iterable[dict]) -> None:
    connection = _connect(run_dir)
    try:
        batch = []
        for issue in issues:
            row = connection.execute(
                "SELECT sequence FROM issues WHERE issue_id = ?", (issue.get("issue_id"),)
            ).fetchone()
            if row:
                batch.append(_values(issue, int(row["sequence"])))
            if len(batch) >= 200:
                connection.executemany(INSERT_SQL, batch)
                connection.commit()
                batch.clear()
        if batch:
            connection.executemany(INSERT_SQL, batch)
            connection.commit()
    finally:
        connection.close()


def queue_condition(queue: str) -> str:
    if queue == "needs_review":
        return UNRESOLVED_SQL
    if queue == "audits":
        return "(audit_required = 1 AND audit_status = 'pending')"
    if queue == "safe_suggestions":
        return f"({UNRESOLVED_SQL} AND safe_sam = 1)"
    if queue == "manual":
        return f"(review_status = 'needs_fix' OR (NOT {RESOLVED_SQL} AND difference_band = 'large_disagreement'))"
    if queue == "resolved":
        return RESOLVED_SQL
    return "1 = 1"


def queue_counts(run_dir: Path) -> dict:
    queues = ("needs_review", "audits", "safe_suggestions", "manual", "resolved", "all")
    counts = {name: issue_count(run_dir, queue_condition(name)) for name in queues}
    counts["reviewed"] = counts["all"] - counts["needs_review"]
    counts["high_priority"] = issue_count(
        run_dir, f"({queue_condition('needs_review')}) AND severity = 'high'"
    )
    counts["accepted_box"] = issue_count(run_dir, "accepted_fix = 'sam_box'")
    counts["accepted_class"] = issue_count(run_dir, "accepted_class_id IS NOT NULL")
    counts["accepted_total"] = issue_count(
        run_dir, "accepted_fix = 'sam_box' OR accepted_class_id IS NOT NULL"
    )
    return counts


def filter_options(run_dir: Path) -> dict:
    columns = {"splits": "split", "classes": "class_name", "issue_types": "issue_type"}
    result = {}
    with _connect(run_dir) as connection:
        for key, column in columns.items():
            rows = connection.execute(
                f"SELECT DISTINCT {column} FROM issues WHERE {column} != '' ORDER BY {column}"
            )
            result[key] = [str(row[0]) for row in rows]
    return result


def query_issues(
    run_dir: Path,
    *,
    queue: str = "needs_review",
    page: int = 1,
    page_size: int = 50,
    search: str = "",
    severity: str = "",
    split: str = "",
    class_name: str = "",
    issue_type: str = "",
) -> dict:
    conditions = [queue_condition(queue)]
    params: list[object] = []
    for column, value in (
        ("severity", severity), ("split", split), ("class_name", class_name), ("issue_type", issue_type),
    ):
        if value:
            conditions.append(f"{column} = ?")
            params.append(value)
    if search:
        conditions.append("LOWER(image_name) LIKE ?")
        params.append(f"%{search.lower()}%")
    where = " AND ".join(f"({condition})" for condition in conditions)
    page_size = min(200, max(1, int(page_size)))
    page = max(1, int(page))
    with _connect(run_dir) as connection:
        total = int(connection.execute(f"SELECT COUNT(*) FROM issues WHERE {where}", params).fetchone()[0])
        pages = max(1, (total + page_size - 1) // page_size)
        page = min(page, pages)
        offset = (page - 1) * page_size
        rows = connection.execute(
            f"""SELECT payload FROM issues WHERE {where}
            ORDER BY CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                     score DESC, image_name, sequence
            LIMIT ? OFFSET ?""",
            (*params, page_size, offset),
        ).fetchall()
    return {
        "items": [_decode(row) for row in rows],
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": pages,
    }
