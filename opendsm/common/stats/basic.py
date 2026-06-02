#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2024 OpenEEmeter contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

from typing import Literal, Optional, Union

import numba
import numpy as np

from scipy.special import (
    stdtrit,  # faster than using t.ppf
    erfinv,  # faster than using norm.ppf
)

from opendsm.common.utils import to_np_array



# Constant to convert MAD to std deviation for normal distribution
# Equivalent to 1 / norm_dist.ppf(0.75)
MAD_k = 1 / (erfinv(2 * 0.75 - 1) * np.sqrt(2))


def t_stat(alpha: float, dof: float, tail: Union[int, str] = 2) -> float:
    """Calculate the t-statistic for a given number of degrees of freedom.

    Args:
        alpha: Significance level
        dof: Degrees of freedom
        tail: Type of tail test - 1/"one" for one-tailed, 2/"two" for two-tailed

    Returns:
        Calculated t-statistic value

    Raises:
        ValueError: If tail parameter is invalid
    """
    if (tail == "one") or (tail == 1):
        perc = np.asarray(1 - alpha)
    elif (tail == "two") or (tail == 2):
        perc = np.asarray(1 - alpha / 2)
    else:
        raise ValueError(f"Invalid tail parameter: {tail}. Must be 1/'one' or 2/'two'")

    return stdtrit(dof, perc)


def z_stat(alpha: float, tail: Union[int, str] = 2) -> float:
    """Calculate the z-statistic for hypothesis testing.

    Args:
        alpha: Significance level
        tail: Type of tail test - 1/"one" for one-tailed, 2/"two" for two-tailed

    Returns:
        Calculated z-statistic value

    Raises:
        ValueError: If tail parameter is invalid
    """
    if (tail == "one") or (tail == 1):
        perc = np.asarray(1 - alpha)
    elif (tail == "two") or (tail == 2):
        perc = np.asarray(1 - alpha / 2)
    else:
        raise ValueError(f"Invalid tail parameter: {tail}. Must be 1/'one' or 2/'two'")

    return erfinv(2 * perc - 1) * np.sqrt(2)


def unc_factor(
    n: int, interval: Literal["PI", "CI"] = "PI", alpha: float = 0.10
) -> float:
    """Calculate uncertainty factor for confidence or prediction intervals.

    Args:
        n: Sample size
        interval: Interval type - "CI" for Confidence Interval or "PI" for Prediction Interval
        alpha: Significance level

    Returns:
        Uncertainty factor value

    Raises:
        ValueError: If interval type is invalid
    """
    if interval == "CI":
        return t_stat(alpha, n - 1) / np.sqrt(n)
    elif interval == "PI":
        return t_stat(alpha, n - 1) * (1 + 1 / np.sqrt(n))
    else:
        raise ValueError(f"Invalid interval: {interval}. Must be 'CI' or 'PI'")


@numba.jit(nopython=True, cache=True)
def weighted_std(
    x: np.ndarray,
    w: np.ndarray,
    mean: Optional[float] = None,
    w_sum_err: float = 1e-6,
) -> float:
    """Calculate weighted standard deviation with optional normalization.

    Args:
        x: Input data array
        w: Weights for each data point
        mean: Pre-computed mean (if None, calculated from weighted data)
        w_sum_err: Tolerance for weight normalization check

    Returns:
        Weighted standard deviation
    """
    n = float(len(x))

    w_sum = np.sum(w)
    if w_sum < 1 - w_sum_err or w_sum > 1 + w_sum_err:
        w /= w_sum

    if mean is None:
        mean = np.sum(w * x)

    var = np.sum(w * np.power((x - mean), 2)) / (1 - 1 / n)

    return np.sqrt(var)


def fast_std(
    x: np.ndarray,
    weights: Optional[Union[np.ndarray, float, int]] = None,
    mean: Optional[float] = None,
) -> float:
    """Calculate standard deviation (weighted or unweighted) efficiently.

    Automatically determines whether to use weighted or unweighted calculation
    based on the weights parameter.

    Args:
        x: Input data array
        weights: Optional weights (array, scalar, or None for unweighted)
        mean: Pre-computed mean (if None, calculated from data)

    Returns:
        Standard deviation value
    """
    if isinstance(weights, (int, float)):
        weights = np.array([weights])

    if weights is None or len(weights) == 1 or np.allclose(weights - weights[0], 0):
        if mean is None:
            return np.std(x)
        else:
            n = float(len(x))
            var = np.sum(np.power((x - mean), 2)) / n
            return np.sqrt(var)
    else:
        if mean is None:
            mean = np.average(x, weights=weights)

        return weighted_std(x, weights, mean)


@numba.jit(nopython=True, cache=True)
def _weighted_quantile(
    values: np.ndarray,
    quantiles: np.ndarray,
    weights: Optional[np.ndarray] = None,
    values_presorted: bool = False,
    old_style: bool = False,
) -> np.ndarray:
    """Calculate weighted quantiles (numba-optimized internal implementation).

    Similar to numpy.percentile but supports weighted observations.
    Reference: https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy

    Args:
        values: Input data array
        quantiles: Array of quantiles to compute (must be in [0, 1])
        weights: Optional weights for each value (same length as values)
        values_presorted: If True, assumes values are already sorted
        old_style: If True, uses numpy.quantile-compatible output

    Returns:
        Array of computed quantiles

    Raises:
        ValueError: If quantiles are not in [0, 1]
    """
    for q in quantiles:
        if not 0 <= q <= 1:
            raise ValueError("quantiles should be in [0, 1]")

    finite_idx = np.where(np.isfinite(values))
    values = values[finite_idx]

    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = weights[finite_idx]

    if not values_presorted:
        sorted_idx = np.argsort(values)
        values = values[sorted_idx]
        weights = weights[sorted_idx]

    res = np.cumsum(weights) - 0.5 * weights
    if old_style:  # To be convenient with numpy.quantile
        res -= res[0]
        res /= res[-1]
    else:
        res /= np.sum(weights)

    return np.interp(quantiles, res, values)


def weighted_quantile(
    values: Union[np.ndarray, list],
    quantiles: Union[np.ndarray, list, float],
    weights: Optional[Union[np.ndarray, list]] = None,
    values_presorted: bool = False,
    old_style: bool = False,
) -> np.ndarray:
    """Calculate weighted quantiles with input validation.

    Public wrapper for _weighted_quantile that handles input conversion
    and provides better error messages.

    Args:
        values: Input data (array-like)
        quantiles: Quantiles to compute (array-like or scalar, in [0, 1])
        weights: Optional weights (array-like)
        values_presorted: If True, assumes values are already sorted
        old_style: If True, uses numpy.quantile-compatible output

    Returns:
        Array of computed quantiles

    Raises:
        Exception: If weighted quantile calculation fails
    """
    values = to_np_array(values)
    quantiles = to_np_array(quantiles)

    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = to_np_array(weights)

    try:
        res = _weighted_quantile(values, quantiles, weights, values_presorted, old_style)
    except Exception as e:
        print("Error in weighted_quantile:")
        print(f"  values shape: {values.shape}, dtype: {values.dtype}")
        print(f"  quantiles: {quantiles}")
        print(f"  weights shape: {weights.shape}, dtype: {weights.dtype}")
        raise Exception(f"Error in weighted_quantile: {str(e)}") from e

    return res


@numba.jit(nopython=True, cache=True)
def _median_absolute_deviation(
    x: np.ndarray,
    median: Optional[float] = None,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Calculate Median Absolute Deviation (numba-optimized internal implementation).

    Computes MAD scaled to match standard deviation of normal distribution.
    Supports both weighted and unweighted calculations. Only handles 1D arrays.

    Args:
        x: Input data array (1D)
        median: Pre-computed median (if None, calculated from data)
        weights: Optional weights for weighted MAD calculation

    Returns:
        MAD value scaled to match standard deviation units
    """
    mu = median
    if weights is None:
        if mu is None:
            mu = np.median(x)

        sigma = np.median(np.abs(x - mu))

    else:
        if mu is None:
            mu = _weighted_quantile(x, np.array([0.5]), weights=weights, values_presorted=False)[0]

        sigma = _weighted_quantile(
            np.abs(x - mu), np.array([0.5]), weights=weights, values_presorted=False
        )[0]

    return sigma * MAD_k


def median_absolute_deviation(
    x: Union[np.ndarray, list],
    median: Optional[float] = None,
    weights: Optional[Union[np.ndarray, list]] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """Calculate Median Absolute Deviation (MAD) scaled to standard deviation.

    Public wrapper that handles input conversion. Supports both weighted
    and unweighted calculations.

    Args:
        x: Input data (array-like)
        median: Pre-computed median (if None, calculated from data)
        weights: Optional weights for weighted MAD calculation
        axis: Axis along which to compute MAD (None for flattened array)

    Returns:
        MAD value scaled to match standard deviation units
    """
    x = to_np_array(x)

    if weights is not None:
        weights = to_np_array(weights)

    if axis is None:
        # Flatten array for 1D calculation
        x_flat = x.ravel()
        weights_flat = weights.ravel() if weights is not None else None
        return _median_absolute_deviation(x_flat, median=median, weights=weights_flat)
    else:
        # Apply along specified axis
        def mad_1d(x_slice):
            return _median_absolute_deviation(x_slice, median=None, weights=None)

        return np.apply_along_axis(mad_1d, axis, x)