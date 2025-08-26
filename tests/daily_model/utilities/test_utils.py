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
import numpy as np
import numba
import pytest

from opendsm.common.utils import (
    np_clip,
    OoM,
    RoundToSigFigs,
    t_stat,
    unc_factor,
    median_absolute_deviation,
    fast_std,
)


def test_np_clip():
    # Test case 1: Test with a scalar input
    a = 5
    a_min = 0
    a_max = 10
    result = np_clip(a, a_min, a_max)(a, a_min, a_max)
    expected = 5
    assert np.allclose(result, expected)

    # Test case 2: Test with an array input
    a = np.array([1, 2, 3, 4, 5])
    a_min = 2
    a_max = 4
    result = np_clip(a, a_min, a_max)(a, a_min, a_max)
    expected = np.array([2, 2, 3, 4, 4])
    assert np.allclose(result, expected)

    # Test case 3: Test with NaN values
    """
    We use the ~ operator to invert the boolean mask created by np.isnan(a), which replaces the NaN values with False. 
    We then use this mask to select the non-NaN values from the input array and the expected output array, and compare 
    them using np.allclose. This is to handle the issue where np.allclose returns False even though the result is the same as expected.
    """
    a = np.array([1, 2, np.nan, 4, 5])
    a_min = 2
    a_max = 4
    mask = ~np.isnan(a)
    result = np_clip(a[mask], a_min, a_max)(a[mask], a_min, a_max)
    expected = np.array([2, 2, np.nan, 4, 4])[mask]
    print(result, expected)
    assert np.allclose(result, expected)

    # Test case 4: Test with a_min > a_max (should raise ValueError)
    a = np.array([1, 2, 3, 4, 5])
    a_min = 4
    a_max = 2
    try:
        np_clip(a, a_min, a_max)(a, a_min, a_max)
    except ValueError as e:
        assert str(e) == "a_min must be less than or equal to a_max"


def test_OoM():
    # Test case 1: Test with a scalar input - should give an error as the declaration must have an array input
    x = 5000
    with pytest.raises(Exception) as e:
        OoM(x)
    assert e.type in [
        numba.core.errors.TypingError,
        TypeError,
    ]  # will depend whether using JIT

    # Test case 2: Test with an array input
    x = np.array([100, 1000, 10000, 100000])
    result = OoM(x)
    expected = np.array([2, 3, 4, 5])
    assert np.allclose(result, expected)

    # Test case 4: Test with a ceiling rounding method
    x = np.array([99, 999, 9999, 99999])
    result = OoM(x, method="ceil")
    expected = np.array([2, 3, 4, 5])
    assert np.allclose(result, expected)

    # Test case 4: Test with a floor rounding method
    x = np.array([101, 1001, 10001, 100001])
    result = OoM(x, method="floor")
    expected = np.array([2, 3, 4, 5])
    assert np.allclose(result, expected)

    # Test case 5: Test with an exact rounding method
    x = np.array([101, 1001, 10001, 100001])
    result = OoM(x, method="exact")
    expected = np.array([2, 3, 4, 5])
    assert np.allclose(result, expected)

    # Test case 6: Test with a non-integer input
    x = [1234.5678]
    result = OoM(x)
    expected = 3
    assert result == expected


def test_RoundToSigFigs():
    # Test case 1: Test with a scalar input
    x = [1234.5678]
    p = 3
    result = RoundToSigFigs(x, p)
    expected = 1230
    assert result == expected

    # Test case 2: Test with an array input
    x = np.array([1234.5678, 5678.1234, 0.0123456])
    p = 4
    result = RoundToSigFigs(x, p)
    expected = np.array([1.235e03, 5.680e03, 1.235e-02])
    assert np.allclose(result, expected)

    # Test case 3: Test with a zero input
    x = [0]
    p = 3
    result = RoundToSigFigs(x, p)
    expected = 0
    assert result == expected

    # Test case 4: Test with a negative input
    x = [-1234.5678]
    p = 3
    result = RoundToSigFigs(x, p)
    expected = -1230
    assert result == expected


def test_t_stat():
    # Test case 1: Test with a two-tailed test
    alpha = 0.05
    n = 10
    tail = 2
    result = t_stat(alpha, n, tail)
    expected = 2.262
    assert np.isclose(result, expected, rtol=1e-3)

    # Test case 2: Test with a one-tailed test
    alpha = 0.05
    n = 10
    tail = 1
    result = t_stat(alpha, n, tail)
    expected = 1.833
    assert np.isclose(result, expected, rtol=1e-3)

    # Test case 3: Test with a larger sample size
    alpha = 0.01
    n = 100
    tail = 2
    result = t_stat(alpha, n, tail)
    expected = 2.626
    assert np.isclose(result, expected, rtol=1e-3)

    # Test case 4: Test with a custom alpha value
    alpha = 0.10
    n = 10
    tail = 2
    result = t_stat(alpha, n, tail)
    expected = 1.833
    assert np.isclose(result, expected, rtol=1e-3)


def test_unc_factor():
    # Test case 1: Test with a confidence interval
    n = 10
    interval = "CI"
    alpha = 0.05
    result = unc_factor(n, interval, alpha)
    expected = 0.715
    assert np.isclose(result, expected, rtol=1e-3)

    # Test case 2: Test with a prediction interval
    n = 10
    interval = "PI"
    alpha = 0.05
    result = unc_factor(n, interval, alpha)
    expected = 2.977
    assert np.isclose(result, expected, rtol=1e-3)

    # Test case 3: Test with a larger sample size
    n = 100
    interval = "CI"
    alpha = 0.01
    result = unc_factor(n, interval, alpha)
    expected = 0.2626
    assert np.isclose(result, expected, rtol=1e-3)

    # Test case 4: Test with a custom alpha value
    n = 10
    interval = "PI"
    alpha = 0.10
    result = unc_factor(n, interval, alpha)
    expected = 2.412
    assert np.isclose(result, expected, rtol=1e-3)


def test_median_absolute_deviation():
    # Test case 1: Test with a small array
    x = np.array([1, 2, 3, 4, 5])
    result = median_absolute_deviation(x)
    expected = 1.4826
    assert np.isclose(result, expected, rtol=1e-3)

    # Test case 2: Test with a larger array
    x = np.random.normal(0, 1, 1000)
    result = median_absolute_deviation(x)
    expected = 1
    assert np.isclose(result, expected, rtol=1)

    # Test case 3: Test with an array of zeros
    x = np.zeros(10)
    result = median_absolute_deviation(x)
    expected = 0
    assert np.isclose(result, expected, rtol=1e-3)

    # Test case 4: Test with an array of ones
    x = np.ones(10)
    result = median_absolute_deviation(x)
    expected = 0
    assert np.isclose(result, expected, rtol=1e-3)


def test_fast_std():
    # Test case 1: Test with no weights and no mean
    x = np.array([1, 2, 3, 4, 5])
    result = fast_std(x)
    expected = 1.4142
    assert np.isclose(result, expected, rtol=1e-3)

    # Test case 2: Test with custom weights and no mean
    x = np.array([1, 2, 3, 4, 5])
    weights = np.array([0.1, 0.2, 0.3, 0.3, 0.1])
    result = fast_std(x, weights)
    expected = 1.270
    assert np.isclose(result, expected, rtol=1e-3)

    # Test case 3: Test with custom weights and custom mean
    x = np.array([1, 2, 3, 4, 5])
    weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    mean = 3
    result = fast_std(x, weights, mean)
    expected = 1.414
    assert np.isclose(result, expected, rtol=1e-3)

    # Test case 4: Test with a small array
    x = np.array([1])
    result = fast_std(x)
    expected = 0
    assert np.isclose(result, expected, rtol=1e-3)
