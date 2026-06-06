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

from scipy import stats

from opendsm.common.stats.basic import (
    MAD_k,
    t_stat,
    z_stat,
    unc_factor,
    median_absolute_deviation,
    fast_std,
    weighted_std,
    weighted_quantile,
)



def test_t_stat():
    # Test case 1: Test with a two-tailed test
    alpha = 0.05
    dof = 9
    tail = 2
    result = t_stat(alpha, dof, tail)
    expected = 2.262
    assert np.isclose(result, expected, rtol=1e-3)

    # Test case 2: Test with a one-tailed test
    alpha = 0.05
    dof = 9
    tail = 1
    result = t_stat(alpha, dof, tail)
    expected = 1.833
    assert np.isclose(result, expected, rtol=1e-3)

    # Test case 3: Test with larger degrees of freedom
    alpha = 0.01
    dof = 99
    tail = 2
    result = t_stat(alpha, dof, tail)
    expected = 2.626
    assert np.isclose(result, expected, rtol=1e-3)

    # Test case 4: Test with a custom alpha value
    alpha = 0.10
    dof = 9
    tail = 2
    result = t_stat(alpha, dof, tail)
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


@pytest.mark.parametrize(
    "alpha,tail,scipy_perc",
    [(0.05, 2, 0.975), (0.05, 1, 0.95), (0.10, 2, 0.95)],
)
def test_z_stat_matches_scipy(alpha, tail, scipy_perc):
    """z_stat equals the standard-normal quantile for one- and two-tailed alpha."""
    assert z_stat(alpha, tail) == pytest.approx(stats.norm.ppf(scipy_perc))


@pytest.mark.parametrize("fn", [t_stat, z_stat])
def test_stat_invalid_tail_raises(fn):
    """An out-of-range tail argument raises a descriptive ValueError."""
    with pytest.raises(ValueError, match="tail"):
        if fn is t_stat:
            fn(0.05, 9, tail=3)
        else:
            fn(0.05, tail=3)


def test_unc_factor_invalid_interval_raises():
    """An interval other than 'CI'/'PI' raises ValueError."""
    with pytest.raises(ValueError, match="Invalid interval"):
        unc_factor(10, interval="ZZ")


def test_weighted_quantile_equal_weights_match_numpy_median():
    """With uniform weights the 0.5 quantile equals the numpy median."""
    x = np.arange(1.0, 11.0)

    result = weighted_quantile(x, 0.5)

    assert result[0] == pytest.approx(np.median(x))


def test_weighted_quantile_weights_shift_the_quantile():
    """Heavily weighting the low values pulls the weighted median downward."""
    x = np.arange(1.0, 11.0)
    weights = np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    weighted = weighted_quantile(x, 0.5, weights=weights)[0]

    assert weighted < np.median(x)
    assert weighted == pytest.approx(2.35)


def test_weighted_quantile_out_of_range_raises():
    """A quantile outside [0, 1] raises."""
    with pytest.raises(Exception, match="quantiles should be in"):
        weighted_quantile(np.arange(5.0), 1.5)


def test_weighted_std_applies_sample_size_correction():
    """weighted_std divides the weighted variance by (1 - 1/n), unlike np.std."""
    x = np.arange(1.0, 11.0)
    n = len(x)

    result = weighted_std(x.copy(), np.ones(n) / n)
    expected = np.sqrt(np.var(x) / (1 - 1 / n))

    assert result == pytest.approx(expected)
    assert result != pytest.approx(np.std(x))


def test_mad_recovers_sigma_for_normal():
    """Scaled MAD recovers the standard deviation of a normal sample."""
    x = np.random.default_rng(0).normal(0.0, 2.0, 5000)

    assert median_absolute_deviation(x) == pytest.approx(2.0, abs=0.1)


def test_mad_weighted_downweights_outlier():
    """Down-weighting a far point lowers the weighted MAD toward the inlier scale."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 100.0])
    full_weight = median_absolute_deviation(x, weights=np.ones(6))
    down_weight = median_absolute_deviation(x, weights=np.array([1, 1, 1, 1, 1, 1e-6]))

    assert down_weight < full_weight


def test_mad_along_axis_scales_with_row():
    """axis=1 computes a per-row MAD; scaling a row scales its MAD identically."""
    m = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]])

    result = median_absolute_deviation(m, axis=1)

    assert result[0] == pytest.approx(MAD_k)
    assert result[1] == pytest.approx(10 * MAD_k)


def test_mad_zero_when_majority_identical():
    """MAD is 0 when more than half the values equal the median."""
    x = np.array([5.0, 5.0, 5.0, 1.0, 9.0])

    assert median_absolute_deviation(x) == 0.0
