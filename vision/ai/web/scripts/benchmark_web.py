#!/usr/bin/env python3
"""Measure read-only web API latency before and after runtime changes."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


DEFAULT_ENDPOINTS = (
    "/api/train/status",
    "/api/test/status",
    "/api/train/sessions",
    "/api/inference/storage",
    "/api/jobs?limit=20",
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def measure(base_url: str, endpoint: str, iterations: int) -> dict:
    durations = []
    response_bytes = 0
    for _ in range(iterations):
        started = time.perf_counter()
        with urllib.request.urlopen(f"{base_url.rstrip('/')}{endpoint}", timeout=30) as response:
            payload = response.read()
        durations.append((time.perf_counter() - started) * 1000)
        response_bytes = len(payload)
    return {
        "endpoint": endpoint,
        "iterations": iterations,
        "mean_ms": round(statistics.fmean(durations), 2),
        "p50_ms": round(percentile(durations, 0.50), 2),
        "p95_ms": round(percentile(durations, 0.95), 2),
        "response_bytes": response_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--endpoint", action="append", dest="endpoints")
    args = parser.parse_args()
    endpoints = tuple(args.endpoints or DEFAULT_ENDPOINTS)
    results = [measure(args.base_url, endpoint, max(1, args.iterations)) for endpoint in endpoints]
    print(json.dumps({"base_url": args.base_url, "results": results}, indent=2))


if __name__ == "__main__":
    main()
