"""Unit tests for the pipeline module (NumCompute framework)."""
import numpy as np
import pytest
from numcompute_stream.pipeline import Pipeline, Transformer, Estimator
from numcompute_stream.preprocessing import StandardScaler

class DummyMultiplier(Transformer):
    def __init__(self, factor=2.0):
        self.factor = factor
    def transform(self, X):
        return np.asarray(X) * self.factor

class DummyEstimator(Estimator):
    def fit(self, X, y=None):
        self.coef_ = 1.0
        return self
    def predict(self, X):
        return np.asarray(X) * 2

def test_pipeline_transform():
    X = [[1.0, 2.0], [3.0, 4.0]]
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('mult', DummyMultiplier(10.0))
    ])
    Xt = pipe.fit_transform(X)
    assert Xt.shape == (2, 2)

def test_pipeline_predict():
    X = [1, 2, 3]
    pipe = Pipeline([
        ('model', DummyEstimator())
    ])
    preds = pipe.fit(X).predict(X)
    assert preds.tolist() == [2, 4, 6]
    
def test_full_pipeline_workflow():
    """Test an end-to-end pipeline combining SimpleImputer and MinMaxScaler."""
    X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    from numcompute_stream.preprocessing import SimpleImputer, MinMaxScaler
    
    pipe = Pipeline([
        ('impute', SimpleImputer(strategy="mean")),
        ('scale', MinMaxScaler(feature_range=(0.0, 1.0)))
    ])
    
    Xt = pipe.fit_transform(X)
    assert Xt.shape == (3, 2)
    assert not np.isnan(Xt).any()
    assert np.allclose(Xt.min(), 0.0)
    assert np.allclose(Xt.max(), 1.0)
    
class DummyStreamingEstimator(Estimator):
    def __init__(self):
        self.partial_fitted_count = 0
    def partial_fit(self, X, y=None):
        self.partial_fitted_count += 1
        return self
    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def test_pipeline_partial_fit_workflow():
    """Test incremental streaming training via pipeline partial_fit."""
    from numcompute_stream.preprocessing import StandardScaler
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', DummyStreamingEstimator())
    ])
    
    chunk1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    y1 = np.array([0, 1])
    
    chunk2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    y2 = np.array([1, 0])
    
    pipe.partial_fit(chunk1, y1)
    pipe.partial_fit(chunk2, y2)
    
    assert pipe.steps[-1][1].partial_fitted_count == 2
    assert pipe.steps[0][1].n_samples_seen_ == 4


def test_pipeline_invalid_steps_type():
    """Test that Pipeline raises TypeError when steps format is invalid."""
    with pytest.raises(TypeError):
        Pipeline("not_a_list")


def test_pipeline_invalid_transform_step():
    """Test that Pipeline raises ValueError when an intermediate step lacks transform."""
    pipe = Pipeline([
        ('bad_step', Estimator()),
        ('model', DummyEstimator())
    ])
    with pytest.raises(ValueError):
        pipe.transform([[1, 2]])