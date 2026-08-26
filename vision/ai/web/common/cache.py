"""Small thread-safe caches for filesystem-derived API responses."""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Hashable
from typing import Any


class SignatureCache:
    """Cache JSON-like values until the caller-provided signature changes."""

    def __init__(self, max_items: int = 128):
        self.max_items = max(1, int(max_items))
        self._items: dict[Hashable, tuple[Hashable, Any, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: Hashable, signature: Hashable):
        with self._lock:
            item = self._items.get(key)
            if item is None or item[0] != signature:
                return None
            self._items[key] = (item[0], item[1], time.monotonic())
            return copy.deepcopy(item[1])

    def put(self, key: Hashable, signature: Hashable, value):
        with self._lock:
            self._items[key] = (signature, copy.deepcopy(value), time.monotonic())
            while len(self._items) > self.max_items:
                oldest = min(self._items, key=lambda item_key: self._items[item_key][2])
                self._items.pop(oldest, None)
        return value

    def clear(self):
        with self._lock:
            self._items.clear()


class TimedCache:
    """Cache one JSON-like value for a short period."""

    def __init__(self, ttl_seconds: float):
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self._value = None
        self._stored_at = 0.0
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            if self._value is None or time.monotonic() - self._stored_at >= self.ttl_seconds:
                return None
            return copy.deepcopy(self._value)

    def put(self, value):
        with self._lock:
            self._value = copy.deepcopy(value)
            self._stored_at = time.monotonic()
        return value

    def clear(self):
        with self._lock:
            self._value = None
            self._stored_at = 0.0
