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


def to_np_array(x):
    """
    This function converts the input value 'x' to a numpy array.

    Parameters:
    x [int, float, array]: The input value to be converted to a numpy array.

    Returns:
    numpy array: The converted numpy array.
    """
    if x is None:
        return None

    if not hasattr(x, "__len__"):
        x = [x]

    if not isinstance(x, np.ndarray):
        x = np.array(x)

    # if ndim is 0 then convert to 1D array
    if x.ndim == 0:
        x = np.array([x])

    return np.array(x)


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


def sigmoid(x, x0=0, k=1):
    # https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth
    
    def _positive_sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def _negative_sigmoid(x):
        # Cache exp so you won't have to calculate it twice
        exp = np.exp(x)

        return exp / (exp + 1)

    x = (x - x0) / k

    positive = x >= 0
    # Boolean array inversion is faster than another comparison
    negative = ~positive

    # empty contains junk hence will be faster to allocate
    # Zeros has to zero-out the array after allocation, no need for that
    # See comment to the answer when it comes to dtype
    res = np.empty_like(x, dtype=float)
    res[positive] = _positive_sigmoid(x[positive])
    res[negative] = _negative_sigmoid(x[negative])

    return res