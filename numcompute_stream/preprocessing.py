"""Data preprocessing transformers for NumCompute framework:
Includes StandardScaler, MinMaxScaler, OneHotEncoder, and SimpleImputer.
All transformations are fully vectorized using Numpy and support streaming via .partial_fit().
"""
import numpy as np
__all__ = ["StandardScaler", "MinMaxScaler", "OneHotEncoder", "SimpleImputer"]

def to_numeric(a, data="data"):
    """Convert input into a numeric ndarray and raise TypeError otherwise"""
    arr = np.asarray(a)
    if not (np.issubdtype(arr.dtype, np.number) or arr.dtype == bool):
        raise TypeError(f"{data} must be numeric, got {arr.dtype} instead.")
    return arr

class StandardScaler:
    """
    Standardize features by substracting the mean and scaling to unit variance.
    
    Complexity:
    -----------
    ** Fit / Partial_Fit:
        - Time : 0(n_samples * n_features)
        - Space : 0(n_features)
    ** Transform:
        - Time : 0(n_samples * n_features)
        - Space : 0(n_samples * n_features)
    """
    
    def __init__(self):
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_samples_seen_ = 0
        
    def partial_fit(self, X, y=None):
        """Incrementally compute mean and variance to be used later for scaling.
        ____________
        Parameters:
            X : array_like, shape (n_samples, n_features) -> input chunk data to update mean and variance.
        ------------
        Returns :
            self: object -> Partially fitted scaler.
        """
        arr = to_numeric(X, "X")
        n_samples = arr.shape[0]
        if n_samples == 0:
            return self
            
        chunk_mean = np.mean(arr, axis=0)
        chunk_var = np.var(arr, axis=0)
        
        if self.n_samples_seen_ == 0:
            self.mean_ = chunk_mean
            self.var_ = chunk_var
            self.n_samples_seen_ = n_samples
        else:
            n1 = self.n_samples_seen_
            n2 = n_samples
            n_new = n1 + n2
            
            delta = chunk_mean - self.mean_
            new_mean = self.mean_ + delta * (n2 / n_new)
            
            # M2 aggregation for variance (Welford's parallel update)
            m2_1 = self.var_ * n1
            m2_2 = chunk_var * n2
            new_m2 = m2_1 + m2_2 + (delta ** 2) * (n1 * n2 / n_new)
            new_var = new_m2 / n_new
            
            self.mean_ = new_mean
            self.var_ = new_var
            self.n_samples_seen_ = n_new
            
        std = np.sqrt(self.var_)
        # Avoid zero division when std is 0
        self.scale_ = np.where(std == 0.0, 1.0, std)
        return self

    def fit(self, X, y=None):
        """Compute mean and standard deviation to be used later for scaling.
        ____________
        Parameters:
            X : array_like, shape (n_samples, n_features) -> input data to compute mean and standard deviation.
        ------------
        Returns :
            self: object -> Fitted scaler.
        """
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_samples_seen_ = 0
        return self.partial_fit(X, y)
    
    def transform(self, X):
        """Perform data standardization by centering and scaling it.
        Parameters:
            X : array_like, shape (n_samples, n_features) -> input data to compute mean and standard deviation.
            ------------
        Returns :
            self: object -> Scaled output."""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("This StandardScaler instance is not fitted yet!")
        arr = to_numeric(X, "X")
        return (arr - self.mean_) / self.scale_
        
    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)
        
        
class MinMaxScaler:
    """Transform data by scaling each feature to a given range.
    __________
    Complexity:
    ** Fit & Transform:
        - Time : 0(n_samples * n_features)
        - Space : 0(n_features)
    """
    def __init__(self, feature_range=(0.0, 1.0)):
        self.min_ = None
        self.scale_ = None
        self.feature_range = feature_range
        self._min_seen = None
        self._max_seen = None
        
    def partial_fit(self, X, y=None):
        """Incrementally compute the minimum and maximum to be used later for scaling.
        ____________
        Parameters:
            X : array_like, shape (n_samples, n_features) -> input chunk data to update min and max.
        ------------
        Returns :
            self: object -> Partially fitted scaler.
        """
        arr = to_numeric(X, "X")
        chunk_min = np.min(arr, axis=0)
        chunk_max = np.max(arr, axis=0)
        
        if self._min_seen is None:
            self._min_seen = chunk_min
            self._max_seen = chunk_max
        else:
            self._min_seen = np.minimum(self._min_seen, chunk_min)
            self._max_seen = np.maximum(self._max_seen, chunk_max)
            
        data_range = self._max_seen - self._min_seen
        data_range = np.where(data_range == 0.0, 1.0, data_range)
        
        feature_min, feature_max = self.feature_range
        feature_range = feature_max - feature_min
        
        self.scale_ = feature_range / data_range
        self.min_ = feature_min - self._min_seen * self.scale_
        return self

    def fit(self, X, y=None):
        """Compute the minimum and maximum to be used later for scaling."""
        self._min_seen = None
        self._max_seen = None
        return self.partial_fit(X, y)
    
    def transform(self, X):
        if self.scale_ is None or self.min_ is None:
            raise ValueError("This MinMaxScaler instance should be fitted before!")
        arr = to_numeric(X, "X")
        return arr * self.scale_ + self.min_
    
    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)
    
class OneHotEncoder:
    """Encode categorical features as a binary matrix by creating a new binary column for each unique category.
    __________
    Complexity:
    ** Fit & Transform:
        - Time : 0(n_samples * n_features)
        - Space : 0(n_samples * n_unique_categories)
    """
    
    def __init__(self):
        self.categories_ = []
        
    def partial_fit(self, X, y=None):
        """Incrementally fit a OneHotEncoder to X by expanding known categories.
        ____________
        Parameters:
            X : array_like, shape (n_samples, n_features) or (n_samples,) -> input chunk data.
        ------------
        Returns :
            self: object -> Partially fitted encoder.
        """
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr[:, None]
            
        if not self.categories_:
            self.categories_ = [np.unique(arr[:, i]) for i in range(arr.shape[1])]
        else:
            new_categories = []
            for i in range(arr.shape[1]):
                combined = np.union1d(self.categories_[i], np.unique(arr[:, i]))
                new_categories.append(combined)
            self.categories_ = new_categories
        return self

    def fit(self, X, y=None):
        """Fit a OneHotEncoder to X."""
        self.categories_ = []
        return self.partial_fit(X, y)
    
    def transform(self, X):
        """Transform X using one-hot encoding
        """
        if not self.categories_:
            raise ValueError("OneHotEncoder class instance is not fitted yet!")
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr[:, None]
        #Initiate encoded matrix
        encoded_cols = []
        for i, catgs in enumerate(self.categories_):
            col = arr[:, i]
            
            #Create a boolean matrix for broadcasting
            encoded = (col[:, None] == catgs).astype(float)
            encoded_cols.append(encoded)
        return np.hstack(encoded_cols)
    
    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)
    
class SimpleImputer:
    """Imputation transformer for completing missing values (NaN).
    Complexity:
    ** Fit & Transform:
        - Time : 0(n_samples * n_features)
        - Space : 0(n_features)
    """
    
    def __init__(self, strategy="mean", fill_value=None):
        if strategy not in ["mean", "median", "constant"]:
            raise ValueError(f"Unrecognized input strategy: {strategy}.")
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None
    
    def partial_fit(self, X, y=None):
        """Incrementally compute imputation statistics.
        ____________
        Parameters:
            X : array_like, shape (n_samples, n_features) -> input chunk data.
        ------------
        Returns :
            self: object -> Partially fitted imputer.
        """
        arr = to_numeric(X, "X")
        if self.strategy == "constant":
            if self.fill_value is None:
                raise ValueError("fill_value must be specified as strategy is 'constant'")
            self.statistics_ = np.full(arr.shape[1], self.fill_value)
            return self
            
        chunk_stat = np.nanmean(arr, axis=0) if self.strategy == "mean" else np.nanmedian(arr, axis=0)
        
        if self.statistics_ is None:
            self.statistics_ = chunk_stat
        else:
            chunk_stat = np.where(np.isnan(chunk_stat), self.statistics_, chunk_stat)
            self.statistics_ = (self.statistics_ + chunk_stat) / 2.0
        return self

    def fit(self, X, y=None):
        """Compute imputation statistics."""
        self.statistics_ = None
        return self.partial_fit(X, y)
    
    def transform(self, X):
        """Impute missing values in X."""
        if self.statistics_ is None:
            raise ValueError("SimpleImputer class instance is not fitted yet!")
        arr = to_numeric(X, "X").copy()
        mask = np.isnan(arr)
        if mask.any():
            rows, cols = np.nonzero(mask)
            fallback_stats = np.where(np.isnan(self.statistics_), 0.0, self.statistics_)
            arr[rows, cols] = fallback_stats[cols]
        return arr
    
    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)