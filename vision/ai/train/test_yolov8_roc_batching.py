"""Regression tests for bounded YOLO ROC-AUC prediction batches."""

from pathlib import Path

import pytest

from vision.ai.train.train_yolov8 import iter_batched_predictions


class RecordingPredictor:
    def __init__(self):
        self.calls = []

    def predict(self, *, source, **kwargs):
        self.calls.append((list(source), dict(kwargs)))
        return [f"result:{path}" for path in source]


def test_predictions_are_manually_batched_and_keep_input_order():
    predictor = RecordingPredictor()
    image_paths = [Path(f"/dataset/image-{index:04d}.jpg") for index in range(1018)]

    predictions = list(
        iter_batched_predictions(
            predictor,
            image_paths,
            imgsz=512,
            conf=0.001,
        )
    )

    assert len(predictor.calls) == len(image_paths)
    assert max(len(source) for source, _kwargs in predictor.calls) == 1
    assert [path for path, _result in predictions] == image_paths
    assert [result for _path, result in predictions] == [
        f"result:{path}" for path in image_paths
    ]
    assert all(kwargs["stream"] is False for _source, kwargs in predictor.calls)
    assert all("batch" not in kwargs for _source, kwargs in predictor.calls)


def test_prediction_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        list(iter_batched_predictions(RecordingPredictor(), [], batch_size=0))


def test_unexpected_result_count_fails_instead_of_dropping_images():
    class MissingResultPredictor:
        def predict(self, *, source, **kwargs):
            return []

    with pytest.raises(RuntimeError, match="expected 1, got 0"):
        list(
            iter_batched_predictions(
                MissingResultPredictor(),
                [Path("/dataset/image.jpg")],
            )
        )
