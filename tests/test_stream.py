"""Unit tests for the stream orchestration module (NumCompute framework).
Covers prequential evaluation workflows, empty streams, empty chunks,
regression metrics, and graceful handling of unfitted models.
"""

import numpy as np
import pytest
from numcompute_stream.stream import StreamEvaluator
from numcompute_stream.metrics import StreamingAccuracy, StreamingMSE


class DummyStreamingClassifier:
    """Dummy classifier model for streaming tests."""
    
    def __init__(self, fitted=True):
        """Initialize classifier with fitted state."""
        self.fitted = fitted

    def partial_fit(self, X, y):
        """Incrementally fit the classifier."""
        self.fitted = True
        return self

    def predict(self, X):
        """Predict labels for input samples."""
        if not self.fitted:
            raise ValueError("Model not fitted yet!")
        return np.zeros(len(X), dtype=int)


class DummyStreamingRegressor:
    """Dummy regressor model for streaming tests."""
    
    def __init__(self, fitted=True):
        """Initialize regressor with fitted state."""
        self.fitted = fitted

    def partial_fit(self, X, y):
        """Incrementally fit the regressor."""
        self.fitted = True
        return self

    def predict(self, X):
        """Predict numeric values for input samples."""
        return np.ones(len(X), dtype=float) * 2.0


def test_stream_evaluator_workflow():
    """Test standard prequential evaluation workflow with classification."""
    model = DummyStreamingClassifier(fitted=True)
    metric = StreamingAccuracy()
    evaluator = StreamEvaluator(model, metric)

    stream_data = [
        (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0, 1])),
        (np.array([[5.0, 6.0], [7.0, 8.0]]), np.array([1, 0]))
    ]

    history = evaluator.evaluate_stream(stream_data)
    
    assert isinstance(history, list)
    assert len(history) == 2
    assert all(0.0 <= score <= 1.0 for score in history)


def test_stream_evaluator_empty_stream():
    """Test behavior when an empty stream is provided."""
    model = DummyStreamingClassifier()
    metric = StreamingAccuracy()
    evaluator = StreamEvaluator(model, metric)

    history = evaluator.evaluate_stream([])
    assert history == []


def test_stream_evaluator_empty_chunks():
    """Test behavior when stream contains empty or zero-sized chunks."""
    model = DummyStreamingClassifier()
    metric = StreamingAccuracy()
    evaluator = StreamEvaluator(model, metric)

    stream_data = [
        (np.array([]).reshape(0, 2), np.array([])),
        (np.array([[1.0, 2.0]]), np.array([0]))
    ]

    history = evaluator.evaluate_stream(stream_data)
    assert isinstance(history, list)


def test_stream_evaluator_regression_mse():
    """Test prequential evaluation workflow with regression and StreamingMSE."""
    model = DummyStreamingRegressor(fitted=True)
    metric = StreamingMSE()
    evaluator = StreamEvaluator(model, metric)

    stream_data = [
        (np.array([[1.0], [2.0]]), np.array([2.0, 2.0])),
        (np.array([[3.0], [4.0]]), np.array([2.0, 2.0]))
    ]

    history = evaluator.evaluate_stream(stream_data)
    assert isinstance(history, list)
    assert len(history) == 2


def test_stream_evaluator_unfitted_model_handling():
    """Test that evaluator gracefully handles prediction failure on first unfitted chunk."""
    model = DummyStreamingClassifier(fitted=False)
    metric = StreamingAccuracy()
    evaluator = StreamEvaluator(model, metric)

    stream_data = [
        (np.array([[1.0, 2.0]]), np.array([0])),
        (np.array([[3.0, 4.0]]), np.array([1]))
    ]

    history = evaluator.evaluate_stream(stream_data)
    assert isinstance(history, list)
    assert len(history) == 1
    