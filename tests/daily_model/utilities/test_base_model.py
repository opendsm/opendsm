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
import pytest
from scipy.stats import linregress, theilslopes

from opendsm.eemeter.models.daily.utilities.settings import DailySettings
from opendsm.eemeter.models.daily.utilities.base_model import (
    get_slope,
    linear_fit,
    get_smooth_coeffs,
    fix_identical_bnds,
    get_intercept,
)


@pytest.fixture
def get_settings():
    settings = DailySettings()
    return settings


def test_get_intercept():
    # Test case 1: alpha = 2, y has positive values
    y = np.array([1, 2, 3, 4, 5])
    assert get_intercept(y) == 3.0

    # Test case 2: alpha = 2, y has negative values
    y = np.array([-5, -4, -3, -2, -1])
    assert get_intercept(y) == -3.0

    # Test case 3: alpha = 1, y has positive values
    y = np.array([1, 2, 3, 4, 5])
    assert get_intercept(y, alpha=1) == 3.0

    # Test case 4: alpha = 1, y has negative values
    y = np.array([-5, -4, -3, -2, -1])
    assert get_intercept(y, alpha=1) == -3.0

    # Test case 5: alpha = 2, y has both positive and negative values
    y = np.array([-5, -4, -3, 2, 1])
    assert get_intercept(y) == -1.8

    # Test case 6: alpha = 1, y has both positive and negative values
    y = np.array([-5, -4, -3, 2, 1])
    assert get_intercept(y, alpha=1) == -3.0


def test_get_slope():
    # Test case 1: Test with alpha=2
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    x_bp = 3
    intercept = 0
    alpha = 2
    expected_slope = 2.0102
    assert np.isclose(
        get_slope(x, y, x_bp, intercept, alpha), expected_slope, atol=1e-3
    )

    # Test case 2: Test with alpha=1
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    x_bp = 3
    intercept = 0
    alpha = 1
    expected_slope = 2.125
    assert np.isclose(
        get_slope(x, y, x_bp, intercept, alpha), expected_slope, atol=1e-3
    )

    # Test case 3: Test with negative y values
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([-2, -4, -6, -8, -10])
    x_bp = 3
    intercept = 0
    alpha = 2
    expected_slope = -2.016
    assert np.isclose(
        get_slope(x, y, x_bp, intercept, alpha), expected_slope, atol=1e-3
    )

    # Test case 4: Test with non-zero intercept
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    x_bp = 3
    intercept = 1
    alpha = 2
    expected_slope = 2.0143
    assert np.isclose(
        get_slope(x, y, x_bp, intercept, alpha), expected_slope, atol=1e-3
    )


def test_linear_fit():
    # Test case 1: Test with alpha = 2
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    alpha = 2
    slope, intercept = linear_fit(x, y, alpha)
    res = linregress(x, y)
    assert slope == res.slope
    assert intercept == res.intercept

    # Test case 2: Test with alpha = 0.95
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    alpha = 0.95
    slope, intercept = linear_fit(x, y, alpha)
    res = theilslopes(y, x, alpha=0.95)
    assert slope == res[0]
    assert intercept == res[1]

    # Test case 3: Test with alpha = 2 and identical observations
    x = np.array([10, 10, 10, 10, 10])
    y = np.array([2, 4, 6, 8, 10])
    alpha = 2
    slope, intercept = linear_fit(x, y, alpha)
    with pytest.raises(ValueError):
        res = linregress(x, y)
    assert slope == 0
    assert intercept == x[0]


def test_get_smooth_coeffs():
    # Test case 1: Both pct_hdd_k and pct_cdd_k are less than min_pct_k
    coeffs = get_smooth_coeffs(10, 0.005, 20, 0.005, min_pct_k=0.01)
    assert np.allclose(coeffs, np.array([10, 0, 20, 0]))

    # Test case 2: pct_hdd_k is greater than min_pct_k and pct_cdd_k is less than min_pct_k
    coeffs = get_smooth_coeffs(10, 0.1, 20, 0.005, min_pct_k=0.01)
    assert np.allclose(coeffs, np.array([11, 1, 19.95, 0.05]))

    # Test case 3: pct_cdd_k is greater than min_pct_k and pct_hdd_k is less than min_pct_k
    coeffs = get_smooth_coeffs(10, 0.005, 20, 0.1, min_pct_k=0.01)
    assert np.allclose(coeffs, np.array([10.05, 0.05, 19, 1]))

    # Test case 4: pct_hdd_k and pct_cdd_k are both greater than min_pct_k and sum to less than or equal to 1
    coeffs = get_smooth_coeffs(10, 0.1, 20, 0.2, min_pct_k=0.01)
    assert np.allclose(coeffs, np.array([11, 1, 18, 2]))

    # Test case 5: pct_hdd_k and pct_cdd_k are both greater than min_pct_k and sum to greater than 1
    coeffs = get_smooth_coeffs(10, 0.5, 20, 0.6, min_pct_k=0.01)
    assert np.allclose(
        coeffs, np.array([14.54545455, 4.54545455, 14.54545455, 5.45454545])
    )

    # Test case 6: pct_match is 1.0, so the smoothed curve should converge at - or + inf
    coeffs = get_smooth_coeffs(10, 0.1, 20, 0.2, min_pct_k=0.01)
    assert np.allclose(coeffs, np.array([11, 1, 18, 2]))


def test_fix_identical_bnds():
    # Test case 1: No bounds are identical
    bnds = np.array([[1, 2], [3, 4], [5, 6]])
    assert np.array_equal(fix_identical_bnds(bnds), bnds)

    # Test case 2: One bound is identical
    bnds = np.array([[1, 2], [3, 3], [5, 6]])
    expected_output = np.array([[1, 2], [2, 4], [5, 6]])
    assert np.array_equal(fix_identical_bnds(bnds), expected_output)

    # Test case 3: Multiple bounds are identical
    bnds = np.array([[1, 2], [3, 3], [5, 5]])
    expected_output = np.array([[1, 2], [2, 4], [4, 6]])
    assert np.array_equal(fix_identical_bnds(bnds), expected_output)

    # Test case 4: All bounds are identical
    bnds = np.array([[1, 1], [1, 1], [1, 1]])
    expected_output = np.array([[0, 2], [0, 2], [0, 2]])
    assert np.array_equal(fix_identical_bnds(bnds), expected_output)
