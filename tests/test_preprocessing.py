"""Unit tests for the preprocessing module (NumCompute framework).

This file checks the correctness of data preprocessing tranformers: StandardScaler, 
MinMaxScaler, OneHotEncoder and SimpleImputer. It ensures proper behavior with fitting, tranforming,
handling and imputing NaNs and raising appropriate exceptions (ValueError & TypeError).

Covered Test Cases:
- StandardScaler: mean centering and unit variance scaling.
- MinMaxScaler: scaling features to specified ranges, like (0.0, 1.0).
- OneHotEncoder: categorical encoding to binary metrics.
- SimpleImputer: missing value imputation using mean, median or constant strategies.
- Streaming partial_fit: incremental updates over multiple data chunks.
- Exception handling: for unfitted estimators and invalid types.
"""
import numpy as np
import pytest
from numcompute_stream.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, SimpleImputer

def test_standard_scaler():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    standard_scaler = StandardScaler()
    scaled_X = standard_scaler.fit_transform(X)
    
    assert np.allclose(scaled_X.mean(axis=0), 0.0)
    assert np.allclose(scaled_X.std(axis=0), 1.0)
    
def test_minmax_scaler():
    X = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    minmax_scaler = MinMaxScaler()
    scaled_X = minmax_scaler.fit_transform(X)
    
    assert np.allclose(scaled_X.min(axis=0), 0.0)
    assert np.allclose(scaled_X.max(axis=0), 1.0)
    
def test_onehot_encoder():
    X = np.array([[2, 2], [1, 3], [2, 3]])
    encoder = OneHotEncoder()
    encoded_X = encoder.fit_transform(X)
    assert encoded_X.shape == (3, 4)
    assert np.all(encoded_X.sum(axis=1) == 2.0)
    
def test_simple_imputer():
    X = np.array([[1.0, np.nan], [3.0, 2.0], [2.0, 4.0]])
    imputer = SimpleImputer(strategy="mean")
    imputed_X = imputer.fit_transform(X)
    
    assert not np.isnan(imputed_X).any()
    assert imputed_X[0, 1] == 3.0

def test_preprocessing_not_fitted_error():
    scaler = StandardScaler()
    with pytest.raises(ValueError):
        scaler.transform([1, 2])
    
def test_preprocessing_constant_column():
    """Test StandardScaler behavior with a constant feature column (zero variance)."""
    X_const = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X_const)
    assert not np.isnan(scaled).any()
    assert np.allclose(scaled[:, 0], 0.0)

def test_simple_imputer_median():
    """Test SimpleImputer using the median strategy with NaN values."""
    X_nan = np.array([[1.0, np.nan], [2.0, 2.0], [10.0, 3.0]])
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(X_nan)
    assert not np.isnan(imputed).any()
    assert imputed[0, 1] == 2.5
    
def test_standard_scaler_partial_fit():
    """Test incremental updates of StandardScaler using partial_fit over multiple chunks."""
    scaler = StandardScaler()
    
    chunk1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    chunk2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    
    scaler.partial_fit(chunk1)
    scaler.partial_fit(chunk2)
    
    assert scaler.n_samples_seen_ == 4
    assert scaler.mean_ is not None
    assert scaler.scale_ is not None
    
    transformed = scaler.transform(chunk1)
    assert transformed.shape == chunk1.shape


def test_onehot_encoder_partial_fit():
    """Test incremental category expansion in OneHotEncoder across streaming chunks."""
    encoder = OneHotEncoder()
    
    chunk1 = np.array([[1], [2]])
    chunk2 = np.array([[3], [1]]) # Introduces a new category '3'
    
    encoder.partial_fit(chunk1)
    encoder.partial_fit(chunk2)
    
    assert len(encoder.categories_[0]) == 3
    
    encoded = encoder.transform(np.array([[3], [2]]))
    assert encoded.shape == (2, 3)
    
def test_minmax_scaler_partial_fit():
    """Test incremental updates of MinMaxScaler across multiple streaming chunks."""
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    
    chunk1 = np.array([[0.0, 10.0], [5.0, 20.0]])
    chunk2 = np.array([[-5.0, 5.0], [10.0, 30.0]])
    
    scaler.partial_fit(chunk1)
    scaler.partial_fit(chunk2)
    
    transformed = scaler.transform(chunk2)
    assert np.all(transformed >= 0.0)
    assert np.all(transformed <= 1.0)


def test_simple_imputer_partial_fit():
    """Test incremental imputation updates across streaming chunks with NaNs."""
    imputer = SimpleImputer(strategy="mean")
    
    chunk1 = np.array([[1.0, np.nan], [3.0, 2.0]])
    chunk2 = np.array([[np.nan, 4.0], [5.0, 6.0]])
    
    imputer.partial_fit(chunk1)
    imputer.partial_fit(chunk2)
    
    imputed = imputer.transform(chunk2)
    assert not np.isnan(imputed).any()