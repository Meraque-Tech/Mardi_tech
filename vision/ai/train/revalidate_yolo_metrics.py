#!/usr/bin/env python3
"""Revalidate an existing Ultralytics checkpoint and refresh web metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

try:
    from .yolo_metrics import build_yolo_metric_families
except ImportError:
    from yolo_metrics import build_yolo_metric_families


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Existing Ultralytics training run directory.")
    parser.add_argument("--data", default=None, help="Dataset YAML override. Defaults to args.yaml data.")
    parser.add_argument("--device", default=None, help="Validation device override.")
    parser.add_argument("--batch", type=int, default=1, help="Validation batch size.")
    parser.add_argument("--workers", type=int, default=2, help="Validation dataloader workers.")
    return parser.parse_args()


def read_yaml(path: Path) -> dict:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc
    return payload if isinstance(payload, dict) else {}


def read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def inferred_task(config: dict, metrics) -> str:
    configured = str(config.get("task") or "").lower()
    if configured in {"detect", "segment"}:
        return configured
    model = str(config.get("model") or "").lower()
    return "segment" if getattr(metrics, "seg", None) is not None or "-seg" in model else "detect"


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    weights = run_dir / "weights" / "best.pt"
    config_path = run_dir / "args.yaml"
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory does not exist: {run_dir}")
    if not weights.is_file():
        raise SystemExit(f"Checkpoint does not exist: {weights}")
    config = read_yaml(config_path)
    data = Path(str(args.data or config.get("data") or "")).expanduser()
    if not data.is_file():
        raise SystemExit(f"Dataset YAML does not exist: {data}")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        split="val",
        imgsz=int(config.get("imgsz") or 640),
        batch=args.batch,
        workers=args.workers,
        device=args.device or config.get("device") or None,
        plots=False,
        verbose=False,
    )
    payload = build_yolo_metric_families(metrics, inferred_task(config, metrics))
    for overall in payload.get("overall_by_type", {}).values():
        if overall:
            overall["source"] = "best_checkpoint_validation"
    if payload.get("overall"):
        payload["overall"]["source"] = "best_checkpoint_validation"
    payload.update({
        "per_class_source": "best_checkpoint_validation",
        "per_class_note": (
            f"Per-class {payload.get('primary_metric_type', 'box')} metrics were recalculated by validating best.pt."
        ),
        "best_checkpoint": str(weights),
    })
    if payload.get("metric_warnings"):
        payload["per_class_note"] += " " + " ".join(payload["metric_warnings"])

    output_path = run_dir / "web_metrics.json"
    existing = read_json(output_path)
    existing.update(payload)
    output_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"Ultralytics metric summary keys: {payload.get('metric_summary_keys', [])}")
    print(f"Refreshed metrics: {output_path}")


if __name__ == "__main__":
    main()
