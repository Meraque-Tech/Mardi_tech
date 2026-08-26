"""Thread-safe resource and named-workspace coordination."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path


class ResourceBusyError(RuntimeError):
    def __init__(self, resource: str, active_operation: str):
        self.resource = resource
        self.active_operation = active_operation
        super().__init__(f"{resource} is already in use by {active_operation}.")


@dataclass(frozen=True)
class ResourceLease:
    resource: str
    owner: str
    operation: str
    acquired_at: float


class ResourceCoordinator:
    """Provide atomic leases for process-local scarce resources."""

    def __init__(self):
        self._leases: dict[str, ResourceLease] = {}
        self._lock = threading.RLock()

    def acquire(self, resource: str, owner: str, operation: str) -> ResourceLease:
        with self._lock:
            active = self._leases.get(resource)
            if active is not None and active.owner != owner:
                raise ResourceBusyError(resource, active.operation)
            lease = active or ResourceLease(resource, owner, operation, time.time())
            self._leases[resource] = lease
            return lease

    def release(self, resource: str, owner: str) -> bool:
        with self._lock:
            active = self._leases.get(resource)
            if active is None or active.owner != owner:
                return False
            self._leases.pop(resource, None)
            return True

    def active(self, resource: str) -> ResourceLease | None:
        with self._lock:
            return self._leases.get(resource)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [lease.__dict__.copy() for lease in self._leases.values()]


class PersistentJobStore:
    """Durable, low-frequency job metadata for restart recovery and history."""

    ACTIVE_STATES = {"queued", "starting", "running", "stopping"}

    def __init__(self, path: Path, write_interval_seconds: float = 1.0):
        self.path = path
        self.write_interval_seconds = max(0.0, float(write_interval_seconds))
        self._lock = threading.RLock()
        self._records: dict[str, dict] | None = None
        self._last_writes: dict[str, float] = {}

    def _load_locked(self) -> dict[str, dict]:
        if self._records is not None:
            return self._records
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            payload = {}
        self._records = payload if isinstance(payload, dict) else {}
        return self._records

    def _write_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(json.dumps(self._records or {}, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)

    def update(
        self,
        job_id: str,
        workflow: str,
        status: str,
        details: dict | None = None,
        force: bool = False,
    ) -> dict:
        now = time.time()
        with self._lock:
            records = self._load_locked()
            existing = dict(records.get(job_id) or {})
            record = {
                **existing,
                "job_id": job_id,
                "workflow": workflow,
                "status": status,
                "created_at": existing.get("created_at", now),
                "updated_at": now,
                "details": details or existing.get("details") or {},
            }
            records[job_id] = record
            terminal = status not in self.ACTIVE_STATES
            last_write = self._last_writes.get(job_id, 0.0)
            if force or terminal or now - last_write >= self.write_interval_seconds:
                self._write_locked()
                self._last_writes[job_id] = now
            return dict(record)

    def recover_interrupted(self) -> int:
        recovered = 0
        now = time.time()
        with self._lock:
            records = self._load_locked()
            for job_id, record in list(records.items()):
                if record.get("status") not in self.ACTIVE_STATES:
                    continue
                records[job_id] = {
                    **record,
                    "status": "interrupted",
                    "updated_at": now,
                    "interrupted_reason": "The web process restarted before this job reached a terminal state.",
                }
                recovered += 1
            if recovered:
                self._write_locked()
        return recovered

    def list(self, limit: int = 100) -> list[dict]:
        with self._lock:
            records = [dict(record) for record in self._load_locked().values()]
        records.sort(key=lambda record: float(record.get("updated_at") or 0), reverse=True)
        return records[: max(1, int(limit))]
