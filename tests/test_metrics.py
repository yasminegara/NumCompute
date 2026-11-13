import numpy as np
import pytest

from numcompute_stream.metrics import (
    accuracy,
    confusion_matrix,
    f1,
    mse,
    precision,
    recall,
    StreamingAccuracy,
    StreamingMSE
)


def test_accuracy():
    assert accuracy([1, 0, 1, 1], [1, 1, 1, 0]) == 0.5


def test_confusion_matrix():
    result = confusion_matrix([0, 0, 1, 1], [0, 1, 1, 0])
    assert np.array_equal(result, [[1, 1], [1, 1]])


def test_classification_metrics():
    true = [1, 1, 1, 0]
    pred = [1, 1, 0, 1]
    assert precision(true, pred) == 2 / 3
    assert recall(true, pred) == 2 / 3
    assert f1(true, pred) == 2 / 3


def test_mse():
    assert mse([1, 2, 3], [1, 4, 2]) == 5 / 3


def test_metrics_reject_different_lengths():
    with pytest.raises(ValueError):
        accuracy([1, 2], [1])


def test_metrics_empty_inputs():
    with pytest.raises(ValueError):
        accuracy([], [])
    with pytest.raises(ValueError):
        mse([], [])


def test_metrics_all_predictions_equal():
    assert accuracy([1, 1, 1], [1, 1, 1]) == 1.0
    assert confusion_matrix([1, 1, 1], [1, 1, 1]).tolist() == [[3]]

def test_streaming_accuracy():
    """Test incremental streaming accuracy updates and state reset across chunks."""
    metric = StreamingAccuracy()
    metric.update([0, 1], [0, 0])
    metric.update([1, 1], [1, 1])
    assert metric.total_count == 4
    assert metric.correct_count == 3
    assert np.isclose(metric.get(), 0.75)
    
    metric.reset()
    assert metric.total_count == 0
    assert metric.get() == 0.0


def test_streaming_mse():
    """Test incremental streaming MSE updates and state reset across chunks."""
    metric = StreamingMSE()
    metric.update([1.0, 2.0], [1.0, 2.0])
    metric.update([3.0, 4.0], [4.0, 3.0])
    assert metric.total_count == 4
    assert np.isclose(metric.get(), 0.5)
    
    metric.reset()
    assert metric.total_count == 0
    assert metric.get() == 0.0