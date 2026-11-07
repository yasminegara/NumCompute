"""Basic descriptive statistics."""

import numpy as np

__all__ = [
    "mean",
    "median",
    "std",
    "minimum",
    "maximum",
    "histogram",
    "quantiles",
]


def _as_numeric(data):
    """Convert data to a numeric NumPy array."""
    arr = np.asarray(data)
    if not (np.issubdtype(arr.dtype, np.number) or arr.dtype == bool):
        raise TypeError(f"data must be numeric, got dtype {arr.dtype}")
    return arr


def mean(data, axis=None):
    """Calculate the mean, ignoring NaN values.

    Parameters
    ----------
    data : array_like
        Numeric input data.
    axis : int or None, default=None
        Axis along which to calculate the mean. ``None`` uses all values.

    Returns
    -------
    float or ndarray
        Mean value, or means along the selected axis.
    """
    return np.nanmean(_as_numeric(data), axis=axis)


def median(data, axis=None):
    """Calculate the median, ignoring NaN values.

    Parameters
    ----------
    data : array_like
        Numeric input data.
    axis : int or None, default=None
        Axis along which to calculate the median.

    Returns
    -------
    float or ndarray
        Median value, or medians along the selected axis.
    """
    return np.nanmedian(_as_numeric(data), axis=axis)


def std(data, axis=None, ddof=0):
    """Calculate the standard deviation, ignoring NaN values.

    Parameters
    ----------
    data : array_like
        Numeric input data.
    axis : int or None, default=None
        Axis along which to calculate the standard deviation.
    ddof : int, default=0
        Delta degrees of freedom used in the calculation.

    Returns
    -------
    float or ndarray
        Standard deviation value or axis-wise standard deviations.
    """
    return np.nanstd(_as_numeric(data), axis=axis, ddof=ddof)


def minimum(data, axis=None):
    """Find the minimum, ignoring NaN values.

    Parameters
    ----------
    data : array_like
        Numeric input data.
    axis : int or None, default=None
        Axis along which to find the minimum.

    Returns
    -------
    scalar or ndarray
        Minimum value, or minima along the selected axis.
    """
    return np.nanmin(_as_numeric(data), axis=axis)


def maximum(data, axis=None):
    """Find the maximum, ignoring NaN values.

    Parameters
    ----------
    data : array_like
        Numeric input data.
    axis : int or None, default=None
        Axis along which to find the maximum.

    Returns
    -------
    scalar or ndarray
        Maximum value, or maxima along the selected axis.
    """
    return np.nanmax(_as_numeric(data), axis=axis)


def histogram(data, bins=10, range=None):
    """Create a histogram after removing NaN values.

    Parameters
    ----------
    data : array_like
        Numeric input data.
    bins : int or array_like, default=10
        Number of bins or the bin edges.
    range : tuple or None, default=None
        Lower and upper range of the bins.

    Returns
    -------
    counts : ndarray
        Number of values in each bin.
    edges : ndarray
        Bin boundaries, with length ``len(counts) + 1``.
    """
    arr = _as_numeric(data)
    valid = arr[~np.isnan(arr)]
    return np.histogram(valid, bins=bins, range=range)


def quantiles(data, q, axis=None):
    """Calculate quantiles between 0 and 1, ignoring NaN values.

    Parameters
    ----------
    data : array_like
        Numeric input data.
    q : float or array_like
        Quantile values in the interval ``[0, 1]``.
    axis : int or None, default=None
        Axis along which to calculate quantiles.

    Returns
    -------
    float or ndarray
        Quantile value(s). For multiple ``q`` values, the q dimension is
        returned first.

    Raises
    ------
    ValueError
        If a quantile is outside ``[0, 1]`` or is NaN.
    """
    arr = _as_numeric(data)
    q_arr = np.asarray(q, dtype=float)
    if np.isnan(q_arr).any() or np.any((q_arr < 0) | (q_arr > 1)):
        raise ValueError("q must be in the interval [0, 1]")
    return np.nanquantile(arr, q_arr, axis=axis)
