"""D-FINE inference adapter for the training web UI."""

from __future__ import annotations


def run_dfine_inference(*_args, **_kwargs):
    raise RuntimeError(
        "D-FINE inference routing is configured, but the D-FINE inference adapter is not implemented yet. "
        "Use YOLO/RF-DETR inference for now, or add an adapter around the official D-FINE torch inference entrypoint."
    )
