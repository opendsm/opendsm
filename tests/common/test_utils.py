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

from opendsm.common.utils import (
    np_clip,
    OoM,
    RoundToSigFigs,
    safe_divide,
    log_cosh,
    sigmoid,
    to_np_array,
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
    # Scalar inputs are wrapped via np.atleast_1d; return a 1-element array.
    # 5000 = 5 * 10^3 -> log10 ~ 3.7 -> rounds to 4 under method="round"
    assert np.array_equal(OoM(5000), np.array([4]))
    assert np.array_equal(OoM(101, method="floor"), np.array([2]))
    assert np.array_equal(OoM(99, method="ceil"), np.array([2]))

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

    # Test case 5: "exact" returns the unrounded log10, including for integer
    # input (output is float64, so it is not truncated to whole numbers).
    x = np.array([101, 1001, 10001, 100001])
    result = OoM(x, method="exact")
    expected = np.log10(x)
    assert np.allclose(result, expected)
    assert not np.allclose(result, np.array([2, 3, 4, 5]))

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


def test_safe_divide_scalar_valid_and_invalid():
    """A valid scalar division returns a float; a zero denominator returns NaN."""
    assert safe_divide(10, 2) == pytest.approx(5.0)
    assert isinstance(safe_divide(10, 2), float)
    assert np.isnan(safe_divide(10, 0))


def test_safe_divide_clamps_tiny_denominator():
    """A denominator <= min_denominator with a non-trivial numerator yields NaN."""
    assert np.isnan(safe_divide(10, 1e-4))


def test_safe_divide_array_masks_invalid_elementwise():
    """Element-wise division leaves NaN only where the denominator is invalid."""
    result = safe_divide(np.array([10.0, 10.0, 6.0]), np.array([2.0, 0.0, 3.0]))

    assert result[0] == pytest.approx(5.0)
    assert np.isnan(result[1])
    assert result[2] == pytest.approx(2.0)


def test_safe_divide_mixed_array_and_scalar():
    """Array-over-scalar and scalar-over-array broadcast and mask correctly."""
    assert np.allclose(safe_divide(np.array([10.0, 10.0]), 2), [5.0, 5.0])

    scalar_over_array = safe_divide(10, np.array([2.0, 0.0]))
    assert scalar_over_array[0] == pytest.approx(5.0)
    assert np.isnan(scalar_over_array[1])


def test_safe_divide_return_all_false_drops_invalid():
    """return_all=False returns only the valid quotients, dropping masked entries."""
    result = safe_divide(
        np.array([10.0, 10.0, 6.0]), np.array([2.0, 0.0, 3.0]), return_all=False
    )

    assert np.allclose(result, [5.0, 2.0])


def test_log_cosh_small_x_taylor_branch():
    """For small x, log(cosh(x)) ≈ x²/2 via the stable Taylor expansion."""
    assert log_cosh(0.0) == pytest.approx(0.0)
    assert log_cosh(1e-3) == pytest.approx((1e-3) ** 2 / 2, rel=1e-6)


def test_log_cosh_large_x_is_stable():
    """For large x, log(cosh(x)) ≈ |x| - ln(2) without overflow."""
    assert log_cosh(50.0) == pytest.approx(50.0 - np.log(2))


def test_log_cosh_matches_reference_array():
    """Array log_cosh matches the direct log(cosh(x)) computation."""
    x = np.array([0.0, 1.0, 2.0, -3.0])

    assert np.allclose(log_cosh(x), np.log(np.cosh(x)))


def test_sigmoid_midpoint_and_saturation():
    """sigmoid(0)=0.5 and saturates to 1 / 0 for large |x| without overflow."""
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(100.0) == pytest.approx(1.0)
    assert sigmoid(-100.0) == pytest.approx(0.0, abs=1e-40)


def test_sigmoid_callable_scale():
    """A callable k is evaluated per-point; a constant k=2 still gives 0.5 at x0."""
    x = np.array([0.0, 1.0, 2.0])

    result = sigmoid(x, k=lambda xx, x0: np.full_like(xx, 2.0))

    assert result[0] == pytest.approx(0.5)
    assert np.all(np.diff(result) > 0)


def test_sigmoid_callable_scale_rejects_nonpositive():
    """A callable k that returns a non-positive scale raises ValueError."""
    with pytest.raises(ValueError, match="non-negative and non-zero"):
        sigmoid(np.array([0.0, 1.0]), k=lambda xx, x0: np.zeros_like(xx))


def test_to_np_array_handles_scalar_list_array_none():
    """to_np_array wraps scalars/lists to 1-D arrays, passes arrays, keeps None."""
    assert np.array_equal(to_np_array(5), np.array([5]))
    assert np.array_equal(to_np_array([1, 2]), np.array([1, 2]))

    arr = np.array([1.0, 2.0])
    assert np.array_equal(to_np_array(arr), arr)
    assert to_np_array(None) is None


def test_to_np_array_wraps_zero_dim_array():
    """A 0-d array is promoted to a 1-element 1-D array."""
    result = to_np_array(np.array(5))

    assert result.ndim == 1
    assert np.array_equal(result, np.array([5]))


def test_np_clip_preserves_nan():
    """The clip overload leaves NaN untouched while clipping finite values."""
    a = np.array([np.nan, 1.0, 5.0])

    result = np_clip(a, 0, 3)(a, 0, 3)

    assert np.isnan(result[0])
    assert result[1] == pytest.approx(1.0)
    assert result[2] == pytest.approx(3.0)


def test_log_cosh_accepts_integer_dtype():
    """Integer input is promoted to float before the log-cosh computation."""
    result = log_cosh(np.array([0, 1, 2]))

    assert np.allclose(result, np.log(np.cosh([0.0, 1.0, 2.0])))


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_log_cosh_preserves_low_precision_dtype(dtype):
    """log_cosh runs in the input float precision and matches the reference."""
    x = np.array([0.0, 1.0, 2.0], dtype=dtype)

    result = log_cosh(x)

    assert result.dtype == dtype
    assert np.allclose(result, np.log(np.cosh(x.astype(np.float64))), atol=1e-2)
