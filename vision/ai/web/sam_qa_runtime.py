"""Memory-aware SAM runtime used by Annotation QA."""

from __future__ import annotations

import gc
from contextlib import nullcontext
from typing import Optional


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "cuda" in text and "out of memory" in text


def extract_prompt_masks(results, prompt_count: int, allow_single_positional: bool = False):
    """Map returned masks to prompts, failing closed when mapping is ambiguous."""
    if not results:
        return [None] * prompt_count
    result = results[0]
    masks = getattr(result, "masks", None)
    data = getattr(masks, "data", None)
    if data is None:
        return [None] * prompt_count
    try:
        mask_data = [item.detach().cpu().numpy() for item in data]
    except AttributeError:
        mask_data = list(data)

    boxes = getattr(result, "boxes", None)
    prompt_indices = getattr(boxes, "cls", None)
    confidences = getattr(boxes, "conf", None)
    if allow_single_positional and prompt_count == 1 and len(mask_data) == 1:
        confidence = None
        if confidences is not None and len(confidences):
            try:
                confidence = float(confidences[0].item())
            except (AttributeError, TypeError, ValueError):
                try:
                    confidence = float(confidences[0])
                except (TypeError, ValueError):
                    pass
        return [{"mask": mask_data[0], "confidence": confidence, "prompt_index": 0}]
    if prompt_indices is None or len(prompt_indices) < len(mask_data):
        return None

    mapped = [None] * prompt_count
    for result_index, mask in enumerate(mask_data):
        try:
            prompt_index = int(prompt_indices[result_index].item())
        except (AttributeError, TypeError, ValueError):
            try:
                prompt_index = int(prompt_indices[result_index])
            except (TypeError, ValueError):
                return None
        if prompt_index < 0 or prompt_index >= prompt_count or mapped[prompt_index] is not None:
            return None
        confidence = None
        if confidences is not None and result_index < len(confidences):
            try:
                confidence = float(confidences[result_index].item())
            except (AttributeError, TypeError, ValueError):
                try:
                    confidence = float(confidences[result_index])
                except (TypeError, ValueError):
                    pass
        mapped[prompt_index] = {
            "mask": mask,
            "confidence": confidence,
            "prompt_index": prompt_index,
        }
    return mapped


class SamQaRuntime:
    """One-job SAM runtime with one-image feature reuse and adaptive prompt chunks."""

    def __init__(self, model_config: dict, device: str = "", requested_max_side: int = 1280):
        self.config = dict(model_config)
        self.backend = str(self.config.get("backend") or "sam2")
        self.model_path = str(self.config["path"])
        self.device_requested = str(device or "")
        self.max_side = min(int(requested_max_side), int(self.config.get("max_side") or requested_max_side))
        self.initial_chunk_size = int(self.config.get("prompt_chunk") or 64)
        self.chunk_size = self.initial_chunk_size
        self.model = None
        self.predictor = None
        self.source = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.precision = "fp32"
        self.device = self.device_requested
        self.feature_reuse = False
        self.prompt_mapping_fallbacks = 0
        self.oom_retries = 0
        self.prompts_processed = 0
        self.images_encoded = 0
        self.images_resized = 0
        self.peak_vram_mb = 0.0
        self.last_inference_shape = None
        self._torch = None

    def load(self):
        try:
            import torch
            from ultralytics import SAM
        except Exception as exc:
            raise RuntimeError("Ultralytics SAM and PyTorch are required for Annotation QA.") from exc
        self._torch = torch
        if not self.device:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        uses_cuda = self.device != "cpu" and torch.cuda.is_available()
        if uses_cuda:
            self.precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"

        if self.backend == "sam3":
            try:
                from ultralytics.models.sam.predict import SAM3Predictor

                overrides = {
                    "conf": 0.0,
                    "task": "segment",
                    "mode": "predict",
                    "model": self.model_path,
                    "verbose": False,
                    "imgsz": self.max_side,
                }
                if self.device_requested:
                    overrides["device"] = self.device_requested
                self.predictor = SAM3Predictor(overrides=overrides)
                self.feature_reuse = True
            except ImportError as exc:
                raise RuntimeError(
                    "This Ultralytics installation does not expose SAM3Predictor. Version 8.3.237 or newer is required."
                ) from exc
        else:
            self.model = SAM(self.model_path)
        if uses_cuda:
            torch.cuda.reset_peak_memory_stats()
            self._update_peak_vram()
        return self

    def _precision_context(self):
        torch = self._torch
        if torch is None or self.precision == "fp32" or not torch.cuda.is_available():
            return nullcontext()
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _inference_context(self):
        return self._torch.inference_mode() if self._torch is not None else nullcontext()

    def _update_peak_vram(self):
        torch = self._torch
        if torch is None or not torch.cuda.is_available():
            return
        self.peak_vram_mb = max(
            self.peak_vram_mb,
            float(torch.cuda.max_memory_allocated()) / (1024 * 1024),
        )

    def _clear_cuda(self):
        torch = self._torch
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def set_image(self, image):
        import cv2

        if image is None or not getattr(image, "shape", None):
            raise RuntimeError("SAM QA received an invalid image.")
        height, width = image.shape[:2]
        scale = min(1.0, self.max_side / max(height, width))
        if scale < 1.0:
            resized_width = max(1, int(round(width * scale)))
            resized_height = max(1, int(round(height * scale)))
            self.source = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
            self.images_resized += 1
        else:
            self.source = image
        inference_height, inference_width = self.source.shape[:2]
        self.scale_x = inference_width / width
        self.scale_y = inference_height / height
        self.last_inference_shape = [inference_height, inference_width]
        self.images_encoded += 1
        if self.backend == "sam3":
            try:
                with self._inference_context(), self._precision_context():
                    self.predictor.set_image(self.source)
            except TypeError:
                # Some Ultralytics releases do not accept autocast around set_image.
                with self._inference_context():
                    self.predictor.set_image(self.source)
            self._update_peak_vram()

    def _scaled_box(self, box):
        x1, y1, x2, y2 = box
        return [
            int(round(x1 * self.scale_x)),
            int(round(y1 * self.scale_y)),
            int(round(x2 * self.scale_x)),
            int(round(y2 * self.scale_y)),
        ]

    def _predict_chunk(self, boxes, allow_single_positional: bool = False):
        scaled = [self._scaled_box(box) for box in boxes]
        with self._inference_context(), self._precision_context():
            if self.backend == "sam3":
                results = self.predictor(bboxes=scaled)
            else:
                kwargs = {
                    "source": self.source,
                    "bboxes": scaled,
                    "conf": 0.0,
                    "verbose": False,
                }
                if self.device_requested:
                    kwargs["device"] = self.device_requested
                results = self.model.predict(**kwargs)
        self._update_peak_vram()
        return extract_prompt_masks(results, len(boxes), allow_single_positional)

    def _predict_singles(self, boxes, stop_event=None):
        mapped = []
        for box in boxes:
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("Annotation QA was stopped.")
            result = self._predict_chunk([box], allow_single_positional=True)
            if result is None:
                return None
            mapped.append(result[0])
        return mapped

    def predict_prompts(self, boxes, stop_event=None):
        if not boxes:
            return []
        output = []
        offset = 0
        while offset < len(boxes):
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("Annotation QA was stopped.")
            size = min(self.chunk_size, len(boxes) - offset)
            chunk = boxes[offset:offset + size]
            try:
                mapped = self._predict_chunk(chunk, allow_single_positional=size == 1 and self.backend == "sam3")
                if mapped is None and self.backend == "sam3" and size > 1:
                    self.prompt_mapping_fallbacks += 1
                    mapped = self._predict_singles(chunk, stop_event)
                if mapped is None:
                    return None
            except Exception as exc:
                if not is_cuda_oom(exc):
                    raise
                self.oom_retries += 1
                self._clear_cuda()
                if size <= 1:
                    raise RuntimeError(
                        f"SAM QA ran out of VRAM with one prompt at {self.last_inference_shape}. Reduce the maximum image side."
                    ) from exc
                self.chunk_size = max(1, size // 2)
                continue
            output.extend(mapped)
            offset += size
            self.prompts_processed += size
        return output

    def reset_image(self):
        if self.backend == "sam3" and self.predictor is not None:
            reset = getattr(self.predictor, "reset_image", None)
            if callable(reset):
                reset()
        self.source = None

    def stats(self):
        return {
            "backend": self.backend,
            "device": self.device,
            "precision": self.precision,
            "max_side": self.max_side,
            "initial_prompt_chunk": self.initial_chunk_size,
            "final_prompt_chunk": self.chunk_size,
            "feature_reuse": self.feature_reuse,
            "prompt_mapping_fallbacks": self.prompt_mapping_fallbacks,
            "oom_retries": self.oom_retries,
            "prompts_processed": self.prompts_processed,
            "images_encoded": self.images_encoded,
            "images_resized": self.images_resized,
            "last_inference_shape": self.last_inference_shape,
            "peak_vram_mb": round(self.peak_vram_mb, 1),
        }

    def close(self):
        try:
            self.reset_image()
        except Exception:
            pass
        self.predictor = None
        self.model = None
        gc.collect()
        self._clear_cuda()
