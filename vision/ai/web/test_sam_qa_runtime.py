"""Tests for memory-aware SAM Annotation QA inference helpers."""

import threading
import unittest

import numpy as np

from vision.ai.web.sam_qa_runtime import SamQaRuntime, extract_prompt_masks, is_cuda_oom


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeMasks:
    def __init__(self, count):
        self.data = [np.full((4, 4), index, dtype=np.uint8) for index in range(count)]


class FakeBoxes:
    def __init__(self, classes=None, confidences=None):
        self.cls = None if classes is None else [Scalar(value) for value in classes]
        self.conf = None if confidences is None else [Scalar(value) for value in confidences]


class FakeResult:
    def __init__(self, count, classes=None, confidences=None):
        self.masks = FakeMasks(count)
        self.boxes = FakeBoxes(classes, confidences)


class AdaptiveRuntime(SamQaRuntime):
    def __init__(self):
        super().__init__({"path": "fake.pt", "backend": "sam3", "max_side": 1008, "prompt_chunk": 8})
        self.failures = 0

    def _predict_chunk(self, boxes, allow_single_positional=False):
        if len(boxes) > 2:
            self.failures += 1
            raise RuntimeError("CUDA out of memory")
        return [{"mask": np.ones((2, 2), dtype=np.uint8), "confidence": 1.0, "prompt_index": i} for i in range(len(boxes))]


class SamQaRuntimeTests(unittest.TestCase):
    def test_prompt_indices_restore_original_order(self):
        result = FakeResult(2, classes=[1, 0], confidences=[0.8, 0.9])
        mapped = extract_prompt_masks([result], 2)
        self.assertEqual(mapped[0]["confidence"], 0.9)
        self.assertEqual(mapped[1]["confidence"], 0.8)

    def test_ambiguous_batch_mapping_fails_closed(self):
        result = FakeResult(2, classes=[0, 0], confidences=[0.8, 0.9])
        self.assertIsNone(extract_prompt_masks([result], 2))

    def test_single_prompt_can_use_explicit_positional_mapping(self):
        result = FakeResult(1, classes=None, confidences=[0.75])
        self.assertIsNone(extract_prompt_masks([result], 1))
        mapped = extract_prompt_masks([result], 1, allow_single_positional=True)
        self.assertEqual(mapped[0]["prompt_index"], 0)
        self.assertEqual(mapped[0]["confidence"], 0.75)

    def test_image_is_downscaled_to_the_backend_limit(self):
        runtime = SamQaRuntime({"path": "fake.pt", "backend": "sam2", "max_side": 1024, "prompt_chunk": 8})
        runtime.set_image(np.zeros((1200, 2400, 3), dtype=np.uint8))
        self.assertEqual(runtime.source.shape[:2], (512, 1024))
        self.assertEqual(runtime._scaled_box((0, 0, 2400, 1200)), [0, 0, 1024, 512])
        self.assertEqual(runtime.images_resized, 1)

    def test_cuda_oom_halves_prompt_chunk_and_retries(self):
        runtime = AdaptiveRuntime()
        results = runtime.predict_prompts([(0, 0, 1, 1)] * 8, threading.Event())
        self.assertEqual(len(results), 8)
        self.assertEqual(runtime.chunk_size, 2)
        self.assertEqual(runtime.oom_retries, 2)
        self.assertEqual(runtime.failures, 2)

    def test_cuda_oom_detection_does_not_hide_other_errors(self):
        self.assertTrue(is_cuda_oom(RuntimeError("CUDA out of memory")))
        self.assertFalse(is_cuda_oom(RuntimeError("invalid checkpoint")))


if __name__ == "__main__":
    unittest.main()
