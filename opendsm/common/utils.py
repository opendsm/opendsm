#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2014-2025 OpenDSM contributors
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numba
import numpy as np
import pandas as pd
from numba.extending import overload
from scipy.stats import norm as norm_dist
from scipy.stats import t as t_dist

from pandas.api.types import is_datetime64_any_dtype as is_pd_datetime

MIN_POS_SYSTEM_VALUE = (np.finfo(float).tiny * (1e20)) ** (1 / 2)
MAX_POS_SYSTEM_VALUE = (np.finfo(float).max * (1e-20)) ** (1 / 2)
LN_MIN_POS_SYSTEM_VALUE = np.log(MIN_POS_SYSTEM_VALUE)
LN_MAX_POS_SYSTEM_VALUE = np.log(MAX_POS_SYSTEM_VALUE)


@overload(np.clip)
def np_clip(a, a_min, a_max):
    """
    This function applies a clip operation on the input array 'a' using the provided minimum and maximum values.
    The clip operation ensures that all elements in 'a' are within the range [a_min, a_max].
    If an element in 'a' is less than 'a_min', it is replaced with 'a_min'.
    If an element in 'a' is greater than 'a_max', it is replaced with 'a_max'.
    NaN values in 'a' are preserved as NaN.

    Parameters:
    a (numpy array): The input array to be clipped.
    a_min (float): The minimum value for the clip operation.
    a_max (float): The maximum value for the clip operation.

    Returns:
    numpy array: The clipped array.
    """

    @numba.vectorize
    def _clip(a, a_min, a_max):
        """
        This is a vectorized implementation of the clip function.
        It applies the clip operation on each element of the input array 'a'.

        Parameters:
        a (float): The input value to be clipped.
        a_min (float): The minimum value for the clip operation.
        a_max (float): The maximum value for the clip operation.

        Returns:
        float: The clipped value.
        """

        if np.isnan(a):
            return np.nan
        elif a < a_min:
            return a_min
        elif a > a_max:
            return a_max
        else:
            return a

    def clip_impl(a, a_min, a_max):
        """
        This is a numba implementation of the clip function.
        It applies the clip operation on the input array 'a' using the provided minimum and maximum values.

        Parameters:
        a (numpy array): The input array to be clipped.
        a_min (float): The minimum value for the clip operation.
        a_max (float): The maximum value for the clip operation.

        Returns:
        numpy array: The clipped array.
        """

        return _clip(a, a_min, a_max)

    return clip_impl

def safe_divide(num, den, min_denominator=1e-3, return_all=True):
    """
    Safely divide numerator by denominator, returning np.nan for invalid cases.
    Works on scalars or numpy arrays.

    Parameters:
    numerator: scalar or array-like
    denominator: scalar or array-like
    min_denominator: float, minimum allowed denominator

    Returns:
    result: scalar or numpy array, or np.nan where division is unsafe
    """
    min_den = min_denominator

    input_num_type = type(num)
    input_den_type = type(den)

    num = np.asarray(num)
    den = np.asarray(den)

    # Create mask for invalid denominators
    invalid_mask = (den == 0) | ((den <= min_den) & (num > 10 * min_den))

    # Prepare result array
    # Determine the maximum shape that can broadcast num and den
    result_shape = np.broadcast(num, den).shape
    result = np.empty(result_shape, dtype=np.float64)

    # Where valid, perform division
    valid_mask = ~invalid_mask
    # Use numpy errstate to suppress divide by zero warnings and replace with nan
    with np.errstate(divide='ignore', invalid='ignore'):
        if num.ndim > 0 and den.ndim > 0:
            temp_result = np.divide(num[valid_mask], den[valid_mask])
            temp_result[np.isinf(temp_result)] = np.nan
            result[valid_mask] = temp_result
        elif num.ndim > 0 and den.ndim == 0:
            temp_result = np.divide(num[valid_mask], den)
            temp_result[np.isinf(temp_result)] = np.nan
            result[valid_mask] = temp_result
        elif num.ndim == 0 and den.ndim > 0:
            temp_result = np.divide(num, den[valid_mask])
            temp_result[np.isinf(temp_result)] = np.nan
            result[valid_mask] = temp_result
        else:
            temp_result = np.divide(num, den)
            if np.isinf(temp_result):
                temp_result = np.nan
            result[valid_mask] = temp_result

    # Where invalid, set to np.nan
    result[invalid_mask] = np.nan

    # replace any non-finite values with np.nan
    result = np.where(np.isfinite(result), result, np.nan)

    # If input was scalar, return scalar
    if input_num_type not in (np.ndarray, list, pd.Series) and input_den_type not in (np.ndarray, list, pd.Series):
        if invalid_mask:
            return np.nan
        else:
            return float(result)

    if return_all:
        return result
    else:
        return result[~invalid_mask]


def OoM(x, method="round"):
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    return OoM_numba(x, method=method)


@numba.jit(nopython=True, cache=True)
def OoM_numba(x, method="round"):
    """
    This function calculates the order of magnitude (OoM) of each element in the input array 'x' using the specified method.

    Parameters:
    x (numpy array): The input array for which the OoM is to be calculated.
    method (str): The method to be used for calculating the OoM. It can be one of the following:
                  "round" - round to the nearest integer (default)
                  "floor" - round down to the nearest integer
                  "ceil" - round up to the nearest integer
                  "exact" - return the exact OoM without rounding

    Returns:
    x_OoM (numpy array): The array of the same shape as 'x' containing the OoM of each element in 'x'.
    """

    x_OoM = np.empty_like(x)
    for i, xi in enumerate(x):
        if xi == 0.0:
            x_OoM[i] = 1.0

        elif method.lower() == "floor":
            x_OoM[i] = np.floor(np.log10(np.abs(xi)))

        elif method.lower() == "ceil":
            x_OoM[i] = np.ceil(np.log10(np.abs(xi)))

        elif method.lower() == "round":
            x_OoM[i] = np.round(np.log10(np.abs(xi)))

        else:  # "exact"
            x_OoM[i] = np.log10(np.abs(xi))

    return x_OoM


def RoundToSigFigs(x, p):
    """
    This function rounds the input array 'x' to 'p' significant figures.

    Parameters:
    x (numpy.ndarray): The input array to be rounded.
    p (int): The number of significant figures to round to.

    Returns:
    numpy.ndarray: The rounded array.
    """

    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - OoM(x_positive))
    return np.round(x * mags) / mags


def t_stat(alpha, n, tail=2):
    """
    Calculate the t-statistic for a given alpha level, sample size, and tail type.

    Parameters:
    alpha (float): The significance level.
    n (int): The sample size.
    tail (int or str): The type of tail test. Can be 1 or "one" for one-tailed test,
                       and 2 or "two" for two-tailed test. Default is 2.

    Returns:
    float: The calculated t-statistic.
    """

    degrees_of_freedom = n - 1
    if tail == "one" or tail == 1:
        perc = 1 - alpha
    elif tail == "two" or tail == 2:
        perc = 1 - alpha / 2

    return t_dist.ppf(perc, degrees_of_freedom, 0, 1)


def unc_factor(n, interval="PI", alpha=0.10):
    """
    Calculates the uncertainty factor for a given sample size, confidence interval type, and significance level.

    Parameters:
    n (int): The sample size.
    interval (str, optional): The type of confidence interval. Defaults to "PI" (Prediction Interval).
    alpha (float, optional): The significance level. Defaults to 0.10.

    Returns:
    float: The uncertainty factor.
    """

    if interval == "CI":
        return t_stat(alpha, n) / np.sqrt(n)

    if interval == "PI":
        return t_stat(alpha, n) * (1 + 1 / np.sqrt(n))


# Conversion factor from MAD to std for normal distribution
MAD_k = 1 / norm_dist.ppf(0.75)


def median_absolute_deviation(x, median=None, axis=None):
    """
    This function calculates the Median Absolute Deviation (MAD) of a given array.

    Parameters:
    x (numpy array): The input array for which the MAD is to be calculated.
    median (float, optional): The median value. If None, the median is calculated from the input array. Defaults to None.

    Returns:
    float: The calculated MAD of the input array.
    """

    mu = median
    if mu is None:
        mu = np.median(x, axis=axis)

    sigma = np.median(np.abs(x - mu), axis=axis) * MAD_k

    return sigma


@numba.jit(nopython=True, cache=True)
def weighted_std(x, w, mean=None, w_sum_err=1e-6):
    """
    Calculate the weighted standard deviation of a given array.

    Parameters:
    x (numpy.ndarray): The input array.
    w (numpy.ndarray): The weights for the input array.
    mean (float, optional): The mean value. If None, the mean is calculated from the input array. Defaults to None.
    w_sum_err (float, optional): The error tolerance for the sum of weights. Defaults to 1e-6.

    Returns:
    float: The calculated weighted standard deviation.
    """

    n = float(len(x))

    w_sum = np.sum(w)
    if w_sum < 1 - w_sum_err or w_sum > 1 + w_sum_err:
        w /= w_sum

    if mean is None:
        mean = np.sum(w * x)

    var = np.sum(w * np.power((x - mean), 2)) / (1 - 1 / n)

    return np.sqrt(var)


def fast_std(x, weights=None, mean=None):
    """
    Function to calculate the approximate standard deviation quickly of a given array.
    This function can handle both weighted and unweighted calculations.

    Parameters:
    x (numpy.ndarray): The input array for which the standard deviation is to be calculated.
    weights (numpy.ndarray, optional): An array of weights for the input array. Defaults to None.
    mean (float, optional): The mean of the input array. If not provided, it will be calculated. Defaults to None.

    Returns:
    float: The calculated standard deviation.
    """

    if isinstance(weights, int) or isinstance(weights, float):
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