"""Bounded cache for heavyweight inference model objects."""

from __future__ import annotations

import gc
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable


def release_model_entry(entry: dict) -> None:
    model = entry.get("model")
    candidates = [model, getattr(model, "model", None)]
    for candidate in candidates:
        move = getattr(candidate, "to", None)
        if callable(move):
            try:
                move("cpu")
                break
            except Exception:
                continue
    entry.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class BoundedModelCache:
    """LRU cache with explicit cleanup of evicted model objects."""

    def __init__(self, max_items: int = 2, cleanup: Callable[[dict], None] = release_model_entry):
        self.max_items = max(1, int(max_items))
        self.cleanup = cleanup
        self.entries: OrderedDict[Hashable, dict] = OrderedDict()
        self.lock = threading.Lock()

    def get_or_create(self, key: Hashable, factory: Callable[[], dict]) -> tuple[dict, bool]:
        evicted: list[dict] = []
        with self.lock:
            entry = self.entries.get(key)
            if entry is not None:
                self.entries.move_to_end(key)
                return entry, True
            entry = factory()
            self.entries[key] = entry
            while len(self.entries) > self.max_items:
                _old_key, old_entry = self.entries.popitem(last=False)
                evicted.append(old_entry)
        for old_entry in evicted:
            self.cleanup(old_entry)
        return entry, False

    def clear(self) -> None:
        with self.lock:
            entries = list(self.entries.values())
            self.entries.clear()
        for entry in entries:
            self.cleanup(entry)
