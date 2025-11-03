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

from opendsm.common.stats.basic import (
    t_stat,
    unc_factor,
    median_absolute_deviation,
    fast_std,
)


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
