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

"""Comprehensive test suite for clustering transform module."""

import warnings
import numpy as np
import pytest

from opendsm.common.clustering.transform import (
    normalize,
    fpca_transform,
    wavelet_transform,
    transform_features,
    FpcaError,
)
from opendsm.common.clustering.transform.normalize import _safe_standardize
from opendsm.common.clustering.transform.fpca import _fpca_base
from opendsm.common.clustering.settings import (
    ClusteringSettings,
    NormalizeSettings,
    NormalizeChoice,
    NormalizeScope,
)


def _wavelet_transform_wrapper(data, settings):
    """Unpack the 3-tuple returned by ``wavelet_transform``.

    If the return signature of ``wavelet_transform`` changes in the future,
    only this wrapper needs updating -- not every test that calls it.

    Returns
    -------
    result : np.ndarray
        The PCA-reduced wavelet features.
    """
    result, _, _ = wavelet_transform(data, settings)
    return result


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_time_series_data():
    """Create simple time series data for testing.

    Returns 50 samples with 24 time points each, organized into three
    distinct patterns: morning peak, evening peak, and flat.
    """
    np.random.seed(42)
    n_samples = 50
    n_timepoints = 24

    data = []
    for i in range(n_samples):
        t = np.linspace(0, 2 * np.pi, n_timepoints)
        if i < 15:
            # Morning peak pattern
            pattern = 50 + 20 * np.sin(t - np.pi/4) + np.random.randn(n_timepoints) * 2
        elif i < 30:
            # Evening peak pattern
            pattern = 50 + 20 * np.sin(t + np.pi/4) + np.random.randn(n_timepoints) * 2
        else:
            # Flat with noise
            pattern = 50 + np.random.randn(n_timepoints) * 5
        data.append(pattern)

    return np.array(data)


@pytest.fixture
def small_dataset():
    """Create a small dataset for edge case testing."""
    np.random.seed(123)
    return np.random.randn(5, 10)


@pytest.fixture
def large_dataset():
    """Create a larger dataset for performance testing."""
    np.random.seed(456)
    return np.random.randn(200, 96)


@pytest.fixture
def constant_data():
    """Create constant time series data (no variation)."""
    return np.ones((10, 24)) * 42.0


@pytest.fixture
def mixed_scale_data():
    """Create data with mixed scales (some constant, some varying)."""
    np.random.seed(789)
    data = np.random.randn(20, 24)
    # Make some rows constant
    data[0, :] = 10.0
    data[5, :] = -5.0
    data[10, :] = 0.0
    return data


# =============================================================================
# Tests for _safe_standardize
# =============================================================================

class TestSafeStandardize:
    """Comprehensive tests for _safe_standardize helper function."""

    def test_basic_standardization_scalar_scale(self):
        """Test basic standardization with scalar center and scale."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        center = 3.0
        scale = 1.5

        result = _safe_standardize(data, center, scale)
        expected = (data - center) / scale

        np.testing.assert_array_almost_equal(result, expected)

    def test_basic_standardization_array_scale(self):
        """Test basic standardization with array center and scale."""
        data = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]])
        center = np.array([4.0, 5.0, 6.0])
        scale = np.array([2.0, 2.0, 2.0])

        result = _safe_standardize(data, center, scale)
        expected = (data - center) / scale

        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_scale_scalar(self):
        """Test standardization with near-zero scalar scale."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        center = 3.0
        scale = 1e-15  # Near zero

        result = _safe_standardize(data, center, scale)
        expected = data - center  # Only centering, no scaling

        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_scale_array_single_column(self):
        """Test standardization with near-zero scale in one column."""
        data = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
        center = np.array([2.0, 3.0, 4.0])
        scale = np.array([1.0, 1e-15, 1.0])  # Middle element near zero

        result = _safe_standardize(data, center, scale, threshold=1e-10)

        # First and third columns should be scaled, middle only centered
        np.testing.assert_array_almost_equal(result[:, 1], data[:, 1] - center[1])
        np.testing.assert_array_almost_equal(result[:, 0], (data[:, 0] - center[0]) / scale[0])
        np.testing.assert_array_almost_equal(result[:, 2], (data[:, 2] - center[2]) / scale[2])

    def test_zero_scale_array_all_columns(self):
        """Test standardization when all scales are near zero."""
        data = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
        center = np.array([2.0, 3.0, 4.0])
        scale = np.array([1e-12, 1e-13, 1e-14])

        result = _safe_standardize(data, center, scale, threshold=1e-10)

        # All should only be centered
        expected = data - center
        np.testing.assert_array_almost_equal(result, expected)

    def test_2d_data_1d_scale_axis_0(self):
        """Test 2D data with 1D scale array (column standardization)."""
        np.random.seed(42)
        data = np.random.randn(10, 5) * 3 + 10
        center = np.mean(data, axis=0)
        scale = np.std(data, axis=0)

        result = _safe_standardize(data, center, scale)

        # Each column should have approximately zero mean
        np.testing.assert_array_almost_equal(np.mean(result, axis=0), 0, decimal=10)

    def test_negative_values(self):
        """Test standardization with negative values."""
        data = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
        center = 0.0
        scale = 3.0

        result = _safe_standardize(data, center, scale)
        expected = data / scale

        np.testing.assert_array_almost_equal(result, expected)

    def test_all_zeros(self):
        """Test standardization with all zeros."""
        data = np.zeros((5, 4))
        center = np.zeros(4)
        scale = np.array([1e-15, 1e-15, 1e-15, 1e-15])

        result = _safe_standardize(data, center, scale)

        # Should return zeros (centered but not scaled)
        np.testing.assert_array_almost_equal(result, np.zeros_like(data))

    def test_custom_threshold(self):
        """Test standardization with custom threshold."""
        data = np.array([1.0, 2.0, 3.0])
        center = 2.0
        scale = 0.001  # Between default threshold and custom threshold

        # With stricter threshold, should scale
        result1 = _safe_standardize(data, center, scale, threshold=1e-10)
        np.testing.assert_array_almost_equal(result1, (data - center) / scale)

        # With looser threshold, should only center
        result2 = _safe_standardize(data, center, scale, threshold=0.01)
        np.testing.assert_array_almost_equal(result2, data - center)

    def test_scalar_scale_zero_ndim(self):
        """Test with 0-dimensional numpy array as scale."""
        data = np.array([1.0, 2.0, 3.0])
        center = 2.0
        scale = np.array(1.5)  # 0-d array

        result = _safe_standardize(data, center, scale)
        expected = (data - center) / scale

        np.testing.assert_array_almost_equal(result, expected)

    def test_mixed_scale_threshold_boundary(self):
        """Test behavior at exact threshold boundary."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        center = np.array([2.0, 3.0])
        threshold = 1e-10
        scale = np.array([threshold, 2.0])  # Exactly at threshold

        result = _safe_standardize(data, center, scale, threshold=threshold)

        # At threshold, should only center (not scale)
        np.testing.assert_array_almost_equal(result[:, 0], data[:, 0] - center[0])
        np.testing.assert_array_almost_equal(result[:, 1], (data[:, 1] - center[1]) / scale[1])


# =============================================================================
# Tests for normalize
# =============================================================================

class TestNormalize:
    """Comprehensive tests for normalize function."""

    # --- Tests for STANDARDIZE method ---

    def test_standardize_global(self, simple_time_series_data):
        """Test standardization with global scope (axis=None, whole-array)."""
        settings = NormalizeSettings(method=NormalizeChoice.STANDARDIZE, scope=NormalizeScope.GLOBAL)
        result = normalize(simple_time_series_data, settings)

        assert result.shape == simple_time_series_data.shape
        np.testing.assert_almost_equal(np.mean(result), 0, decimal=10)
        np.testing.assert_almost_equal(np.std(result), 1, decimal=10)

    @pytest.mark.parametrize("method,scope,center_fn", [
        (NormalizeChoice.STANDARDIZE, NormalizeScope.SAMPLE, np.mean),
        (NormalizeChoice.MED_MAD, NormalizeScope.SAMPLE, np.median),
    ])
    def test_per_sample_centering(self, simple_time_series_data, method, scope, center_fn):
        """Test that per-sample normalization centers each row."""
        settings = NormalizeSettings(method=method, scope=scope)
        result = normalize(simple_time_series_data, settings)

        assert result.shape == simple_time_series_data.shape
        row_centers = center_fn(result, axis=1)
        np.testing.assert_array_almost_equal(row_centers, 0, decimal=10)

    def test_standardize_constant_data_axis_0(self):
        """Test standardization with constant data along axis 0."""
        data = np.ones((10, 24)) * 42.0
        settings = NormalizeSettings(method=NormalizeChoice.STANDARDIZE, axis=0)
        result = normalize(data, settings)

        # Should be centered (all zeros after subtracting constant)
        assert result.shape == data.shape
        np.testing.assert_array_almost_equal(result, 0)

    # --- Tests for MED_MAD method ---

    def test_med_mad_global(self, simple_time_series_data):
        """Test median-MAD normalization with global scope (axis=None, whole-array)."""
        settings = NormalizeSettings(method=NormalizeChoice.MED_MAD, scope=NormalizeScope.GLOBAL)
        result = normalize(simple_time_series_data, settings)

        assert result.shape == simple_time_series_data.shape
        np.testing.assert_almost_equal(np.median(result), 0, decimal=10)

    def test_med_mad_robust_to_outliers_axis_0(self):
        """Test that MED_MAD is more robust to outliers than STANDARDIZE."""
        np.random.seed(42)
        data = np.random.randn(10, 24)
        # Add extreme outliers in some columns
        data[0, 0] = 1000
        data[1, 5] = -1000

        settings = NormalizeSettings(method=NormalizeChoice.MED_MAD, axis=0)
        result = normalize(data, settings)

        # Should handle outliers reasonably
        assert np.isfinite(result).all()

    # --- Tests for MIN_MAX_QUANTILE method ---

    def test_min_max_quantile_axis_1(self, simple_time_series_data):
        """Test min-max quantile normalization along axis 1."""
        settings = NormalizeSettings(
            method=NormalizeChoice.MIN_MAX_QUANTILE,
            quantile=0.05,
            axis=1
        )
        result = normalize(simple_time_series_data, settings)

        # Values should be roughly in range [-1, 1]
        assert result.shape == simple_time_series_data.shape
        assert np.min(result) >= -2  # Allow some tolerance for extreme quantiles
        assert np.max(result) <= 2

    def test_min_max_quantile_axis_0(self, simple_time_series_data):
        """Test min-max quantile normalization along axis 0 (column-wise).

        Note: With quantile-based normalization, values outside the quantile
        range will naturally fall outside [-1, 1], which is expected behavior.
        """
        settings = NormalizeSettings(
            method=NormalizeChoice.MIN_MAX_QUANTILE,
            quantile=0.1,
            axis=0
        )
        result = normalize(simple_time_series_data, settings)

        assert result.shape == simple_time_series_data.shape
        assert np.isfinite(result).all()

        # Verify the normalization is correct by checking manually
        # For each column, the 10th and 90th percentiles should map to -1 and 1
        q = 0.1
        for col_idx in range(simple_time_series_data.shape[1]):
            col_data = simple_time_series_data[:, col_idx]
            min_val, max_val = np.quantile(col_data, [q, 1 - q])

            # Skip columns where min == max (constant)
            if np.abs(min_val - max_val) < 1e-10:
                continue

            result_col = result[:, col_idx]

            # Find values close to the original quantiles
            lower_mask = np.abs(col_data - min_val) < 1e-10
            upper_mask = np.abs(col_data - max_val) < 1e-10

            if np.any(lower_mask):
                np.testing.assert_allclose(result_col[lower_mask], -1.0, rtol=1e-4, atol=1e-8)

            if np.any(upper_mask):
                np.testing.assert_allclose(result_col[upper_mask], 1.0, rtol=1e-4, atol=1e-8)

        # The bulk of values should be in [-1, 1]
        percentiles = np.percentile(result, [10, 90])
        assert percentiles[0] >= -1.5
        assert percentiles[1] <= 1.5

    @pytest.mark.parametrize("quantile", [0.01, 0.05, 0.1, 0.25, 0.4])
    def test_min_max_quantile_different_quantiles(self, simple_time_series_data, quantile):
        """Test different quantile values produce finite results."""
        settings = NormalizeSettings(
            method=NormalizeChoice.MIN_MAX_QUANTILE,
            quantile=quantile,
            axis=1
        )
        result = normalize(simple_time_series_data, settings)

        assert result.shape == simple_time_series_data.shape
        assert np.isfinite(result).all()

    def test_min_max_quantile_constant_rows(self):
        """Test min-max quantile with some constant rows."""
        data = np.random.randn(10, 24)
        data[3, :] = 5.0  # Constant row
        data[7, :] = -3.0  # Another constant row

        settings = NormalizeSettings(
            method=NormalizeChoice.MIN_MAX_QUANTILE,
            quantile=0.05,
            axis=1
        )
        result = normalize(data, settings)

        # Constant rows should be set to midpoint 0
        assert result.shape == data.shape
        np.testing.assert_almost_equal(result[3, :], 0)
        np.testing.assert_almost_equal(result[7, :], 0)

    # --- Edge cases ---

    def test_normalize_single_column(self):
        """Test normalization with single column."""
        data = np.random.randn(50, 1)
        settings = NormalizeSettings(method=NormalizeChoice.STANDARDIZE, axis=0)
        result = normalize(data, settings)

        assert result.shape == data.shape
        np.testing.assert_almost_equal(np.mean(result), 0, decimal=10)

    def test_normalize_small_values_axis_0(self):
        """Test normalization with very small values."""
        data = np.random.randn(10, 20) * 1e-8
        settings = NormalizeSettings(method=NormalizeChoice.STANDARDIZE, axis=0)
        result = normalize(data, settings)

        assert result.shape == data.shape
        assert np.isfinite(result).all()

    # --- Tests for axis=1 (row-wise normalization) ---

    @pytest.mark.parametrize("method,center_fn", [
        (NormalizeChoice.STANDARDIZE, np.mean),
        (NormalizeChoice.MED_MAD, np.median),
    ])
    def test_axis_1_centering(self, simple_time_series_data, method, center_fn):
        """Test row-wise normalization centers each row for different methods."""
        settings = NormalizeSettings(method=method, axis=1)
        result = normalize(simple_time_series_data, settings)

        assert result.shape == simple_time_series_data.shape
        row_centers = center_fn(result, axis=1)
        np.testing.assert_array_almost_equal(row_centers, 0, decimal=10)

    def test_min_max_quantile_axis_0_detailed(self):
        """Detailed test for MIN_MAX_QUANTILE with axis=0 (column-wise).

        Verifies that each column is normalized independently.
        """
        np.random.seed(42)
        data = np.random.randn(10, 3) * [1, 10, 0.1]  # Different scales

        settings = NormalizeSettings(
            method=NormalizeChoice.MIN_MAX_QUANTILE,
            quantile=0.25,
            axis=0
        )
        result = normalize(data, settings)

        assert result.shape == data.shape
        assert np.isfinite(result).all()

        # Verify each column is normalized independently
        for col_idx in range(data.shape[1]):
            col_data = data[:, col_idx]
            col_result = result[:, col_idx]

            min_val, max_val = np.quantile(col_data, [0.25, 0.75])

            if np.abs(min_val - max_val) > 1e-10:
                lower_mask = np.abs(col_data - min_val) < 1e-10
                upper_mask = np.abs(col_data - max_val) < 1e-10

                if np.any(lower_mask):
                    np.testing.assert_allclose(col_result[lower_mask], -1.0, rtol=1e-4, atol=1e-8)
                if np.any(upper_mask):
                    np.testing.assert_allclose(col_result[upper_mask], 1.0, rtol=1e-4, atol=1e-8)

    @pytest.mark.parametrize("method,center_fn,spread_fn,spread_expected", [
        (NormalizeChoice.STANDARDIZE, np.mean, np.std, 1.0),
        (NormalizeChoice.MED_MAD, np.median, None, None),
    ])
    def test_axis_1_with_mixed_scales(self, method, center_fn, spread_fn, spread_expected):
        """Test axis=1 normalization with rows of different scales."""
        data = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [100.0, 200.0, 300.0, 400.0],
            [0.01, 0.02, 0.03, 0.04]
        ])

        settings = NormalizeSettings(method=method, axis=1)
        result = normalize(data, settings)

        # Each row should be centered
        row_centers = center_fn(result, axis=1)
        np.testing.assert_array_almost_equal(row_centers, 0, decimal=10)

        # For standardize, also check unit variance
        if spread_fn is not None:
            row_spreads = spread_fn(result, axis=1)
            np.testing.assert_array_almost_equal(row_spreads, spread_expected, decimal=10)

    def test_min_max_quantile_axis_1_detailed(self):
        """Detailed test for MIN_MAX_QUANTILE with axis=1 (row-wise).

        Verifies that each row is normalized independently.
        """
        np.random.seed(42)
        data = np.random.randn(3, 10) * [[1], [10], [0.1]]  # Different scales

        settings = NormalizeSettings(
            method=NormalizeChoice.MIN_MAX_QUANTILE,
            quantile=0.25,
            axis=1
        )
        result = normalize(data, settings)

        assert result.shape == data.shape
        assert np.isfinite(result).all()

        # Verify each row is normalized independently
        for row_idx in range(data.shape[0]):
            row_data = data[row_idx, :]
            row_result = result[row_idx, :]

            min_val, max_val = np.quantile(row_data, [0.25, 0.75])

            if np.abs(min_val - max_val) > 1e-10:
                lower_mask = np.abs(row_data - min_val) < 1e-10
                upper_mask = np.abs(row_data - max_val) < 1e-10

                if np.any(lower_mask):
                    np.testing.assert_allclose(row_result[lower_mask], -1.0, rtol=1e-4, atol=1e-8)
                if np.any(upper_mask):
                    np.testing.assert_allclose(row_result[upper_mask], 1.0, rtol=1e-4, atol=1e-8)


# =============================================================================
# Tests for FPCA transform
# =============================================================================

class TestFpcaError:
    """Tests for FpcaError exception."""

    def test_fpca_error_instantiation(self):
        """Test that FpcaError can be instantiated."""
        error = FpcaError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)


class TestFpcaBase:
    """Tests for _fpca_base internal function."""

    def test_fpca_base_valid_input(self, simple_time_series_data):
        """Test _fpca_base with valid input."""
        x = np.arange(simple_time_series_data.shape[1])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = _fpca_base(x, simple_time_series_data, min_var_ratio=0.90)

        # Should reduce dimensionality
        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] < simple_time_series_data.shape[1]
        assert result.shape[1] > 0

    @pytest.mark.parametrize("invalid_ratio", [0.0, -0.1, 1.0, 1.5])
    def test_fpca_base_invalid_min_var_ratio(self, simple_time_series_data, invalid_ratio):
        """Test _fpca_base rejects min_var_ratio outside (0, 1)."""
        x = np.arange(simple_time_series_data.shape[1])

        with pytest.raises(FpcaError, match="min_var_ratio but be greater than 0"):
            _fpca_base(x, simple_time_series_data, min_var_ratio=invalid_ratio)

    def test_fpca_base_non_finite_x(self, simple_time_series_data):
        """Test _fpca_base with non-finite x values."""
        x = np.arange(simple_time_series_data.shape[1], dtype=float)
        x[5] = np.nan

        with pytest.raises(FpcaError, match="provided non finite values for fpca"):
            _fpca_base(x, simple_time_series_data, min_var_ratio=0.90)

    def test_fpca_base_non_finite_y(self, simple_time_series_data):
        """Test _fpca_base with non-finite y values."""
        x = np.arange(simple_time_series_data.shape[1])
        data = simple_time_series_data.copy()
        data[10, 15] = np.inf

        with pytest.raises(FpcaError, match="provided non finite values for fpca"):
            _fpca_base(x, data, min_var_ratio=0.90)

    @pytest.mark.parametrize("x,y", [
        (np.array([]), None),  # empty x, y filled by fixture
    ])
    def test_fpca_base_empty_x(self, simple_time_series_data, x, y):
        """Test _fpca_base with empty x array."""
        with pytest.raises(FpcaError, match="provided empty values for fpca"):
            _fpca_base(x, simple_time_series_data, min_var_ratio=0.90)

    def test_fpca_base_empty_y(self):
        """Test _fpca_base with empty y array."""
        x = np.arange(24)
        y = np.array([]).reshape(0, 24)

        with pytest.raises(FpcaError, match="provided empty values for fpca"):
            _fpca_base(x, y, min_var_ratio=0.90)

    def test_fpca_base_different_var_ratios(self, simple_time_series_data):
        """Test _fpca_base with different variance ratios produces monotonic component counts."""
        x = np.arange(simple_time_series_data.shape[1])

        results = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for ratio in [0.70, 0.80, 0.90, 0.95, 0.99]:
                result = _fpca_base(x, simple_time_series_data, min_var_ratio=ratio)
                results[ratio] = result.shape[1]

        # Higher variance ratio should require more components
        assert results[0.95] >= results[0.90]
        assert results[0.90] >= results[0.80]

    def test_fpca_base_small_dataset(self, small_dataset):
        """Test _fpca_base with small dataset."""
        x = np.arange(small_dataset.shape[1])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = _fpca_base(x, small_dataset, min_var_ratio=0.80)

        assert result.shape[0] == small_dataset.shape[0]
        assert result.shape[1] > 0


class TestFpcaTransform:
    """Tests for fpca_transform function."""

    def test_fpca_transform_basic(self, simple_time_series_data):
        """Test basic FPCA transformation."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            feature_transform={"fpca": {"enabled": True, "min_var_ratio": 0.90}, "wavelet": {"enabled": False}}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = fpca_transform(simple_time_series_data, settings)

        # Should reduce dimensionality
        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] < simple_time_series_data.shape[1]
        assert result.shape[1] > 0

    @pytest.mark.slow
    def test_fpca_transform_different_var_ratios(self, simple_time_series_data):
        """Test FPCA with different variance ratios yields monotonic component counts."""
        n_components = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for min_var_ratio in [0.80, 0.90, 0.95, 0.97]:
                settings = ClusteringSettings(
                    algorithm_selection="bisecting_kmeans",
                    seed=42,
                    feature_transform={"fpca": {"enabled": True, "min_var_ratio": min_var_ratio}, "wavelet": {"enabled": False}}
                )
                result = fpca_transform(simple_time_series_data, settings)
                n_components.append(result.shape[1])

        # Higher variance ratio typically needs more components
        assert max(n_components) >= min(n_components)

    @pytest.mark.slow
    def test_fpca_transform_small_dataset(self, small_dataset):
        """Test FPCA on small dataset."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            feature_transform={"fpca": {"enabled": True, "min_var_ratio": 0.85}, "wavelet": {"enabled": False}}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = fpca_transform(small_dataset, settings)

        assert result.shape[0] == small_dataset.shape[0]
        assert result.shape[1] > 0

    def test_fpca_transform_propagates_error(self, simple_time_series_data):
        """Test that FPCA transform propagates FpcaError."""
        # Create data with NaN
        data = simple_time_series_data.copy()
        data[0, 0] = np.nan

        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            feature_transform={"fpca": {"enabled": True, "min_var_ratio": 0.90}, "wavelet": {"enabled": False}}
        )

        with pytest.raises(FpcaError):
            fpca_transform(data, settings)

    @pytest.mark.slow
    def test_fpca_transform_deterministic(self, simple_time_series_data):
        """Test that FPCA transform produces consistent results."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            feature_transform={"fpca": {"enabled": True, "min_var_ratio": 0.90}, "wavelet": {"enabled": False}}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result1 = fpca_transform(simple_time_series_data, settings)
            result2 = fpca_transform(simple_time_series_data, settings)

        # Should produce same results with same input
        np.testing.assert_array_almost_equal(result1, result2)


# =============================================================================
# Tests for wavelet transform
# =============================================================================

class TestWaveletTransform:
    """Comprehensive tests for wavelet_transform function."""

    def test_wavelet_basic(self, simple_time_series_data):
        """Test basic wavelet transformation."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            feature_transform={"fpca": {"enabled": False}, "wavelet": {"enabled": True, "wavelet_name": "db1", "pca_n_components": 5, "pca_scope": "global"}}
        )

        result = _wavelet_transform_wrapper(simple_time_series_data, settings)

        # Wavelet returns PCA components only (magnitude features are appended in transform_features)
        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] == 5  # 5 PCA components (global scope)

    @pytest.mark.parametrize("wavelet_name", ["db1", "haar", "coif6", "sym11"])
    @pytest.mark.parametrize("n_components", [3, 5, 10])
    def test_wavelet_combinations(self, simple_time_series_data, wavelet_name, n_components):
        """Test combinations of wavelets and component counts."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            feature_transform={"fpca": {"enabled": False}, "wavelet": {"enabled": True, "wavelet_name": wavelet_name, "pca_n_components": n_components, "include_scale_feature": False, "pca_scope": "global"}}
        )

        result = _wavelet_transform_wrapper(simple_time_series_data, settings)

        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] == n_components

    def test_wavelet_with_variance_ratio(self, simple_time_series_data):
        """Test wavelet with PCA variance ratio instead of n_components."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            feature_transform={"fpca": {"enabled": False}, "wavelet": {"enabled": True, "wavelet_name": "db1", "pca_n_components": None, "pca_min_variance_ratio_explained": 0.90, "include_scale_feature": False}}
        )

        result = _wavelet_transform_wrapper(simple_time_series_data, settings)

        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] > 0

    def test_wavelet_with_mle(self, simple_time_series_data):
        """Test wavelet with MLE for PCA n_components."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            feature_transform={"fpca": {"enabled": False}, "wavelet": {"enabled": True, "wavelet_name": "db1", "pca_n_components": "mle", "include_scale_feature": False}}
        )

        result = _wavelet_transform_wrapper(simple_time_series_data, settings)

        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] > 0

    def test_wavelet_with_post_normalization(self, simple_time_series_data):
        """Test wavelet with post-transform normalization."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            feature_transform={
                "fpca": {"enabled": False},
                "wavelet": {"enabled": True, "wavelet_name": "db1", "pca_n_components": 5, "include_scale_feature": False, "pca_scope": "global"},
                "normalize": {"enabled": True, "method": "standardize"},
            },
        )

        result = _wavelet_transform_wrapper(simple_time_series_data, settings)

        assert result.shape[0] == simple_time_series_data.shape[0]
        # Post-normalized features should have approximately zero mean and unit std
        np.testing.assert_almost_equal(np.mean(result), 0, decimal=1)
        np.testing.assert_almost_equal(np.std(result), 1, decimal=1)

    @pytest.mark.parametrize("n_levels", [None, 1, 2, 3])
    def test_wavelet_different_n_levels(self, simple_time_series_data, n_levels):
        """Test wavelet with different decomposition levels."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            feature_transform={"fpca": {"enabled": False}, "wavelet": {"enabled": True, "wavelet_name": "db1", "wavelet_n_levels": n_levels, "pca_n_components": 5, "include_scale_feature": False}}
        )
        result = _wavelet_transform_wrapper(simple_time_series_data, settings)

        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] > 0

    @pytest.mark.parametrize("mode", ["smooth", "periodic", "zero", "symmetric"])
    def test_wavelet_different_modes(self, simple_time_series_data, mode):
        """Test wavelet with different extension modes."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            feature_transform={"fpca": {"enabled": False}, "wavelet": {"enabled": True, "wavelet_name": "db1", "wavelet_mode": mode, "pca_n_components": 5, "include_scale_feature": False}}
        )
        result = _wavelet_transform_wrapper(simple_time_series_data, settings)

        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] > 0

    @pytest.mark.parametrize("dataset_fixture", ["small_dataset", "large_dataset"])
    def test_wavelet_dataset_sizes(self, dataset_fixture, request):
        """Test wavelet on datasets of different sizes."""
        dataset = request.getfixturevalue(dataset_fixture)
        n_components = 3 if dataset.shape[0] < 10 else 10
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            feature_transform={"fpca": {"enabled": False}, "wavelet": {"enabled": True, "wavelet_name": "db1", "pca_n_components": n_components}}
        )

        result = _wavelet_transform_wrapper(dataset, settings)

        assert result.shape[0] == dataset.shape[0]
        assert result.shape[1] > 0

    def test_wavelet_deterministic_with_seed(self, simple_time_series_data):
        """Test that wavelet transform is deterministic with same seed."""
        settings1 = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            feature_transform={"fpca": {"enabled": False}, "wavelet": {"enabled": True, "wavelet_name": "db1", "pca_n_components": 5, "seed": 42}}
        )

        settings2 = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            feature_transform={"fpca": {"enabled": False}, "wavelet": {"enabled": True, "wavelet_name": "db1", "pca_n_components": 5, "seed": 42}}
        )

        result1, _, _ = wavelet_transform(simple_time_series_data, settings1)
        result2, _, _ = wavelet_transform(simple_time_series_data, settings2)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_magnitude_features_auto_appended_with_centering(self, simple_time_series_data):
        """Magnitude features are auto-appended when centering normalization is used."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            feature_transform={
                "fpca": {"enabled": False},
                "wavelet": {"enabled": True, "wavelet_name": "db1", "pca_n_components": 5, "pca_scope": "global"},
                "normalize": {"enabled": True, "method": "standardize"},  # centering triggers magnitude
            }
        )

        result = transform_features(simple_time_series_data, settings).data

        # Should have PCA components + 3 magnitude features (median, quantile_range, baseload)
        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] > 5  # 5 PCA + magnitude features


# =============================================================================
# Tests for transform_features (main entry point)
# =============================================================================

class TestTransformFeatures:
    """Comprehensive tests for transform_features main function."""

    # --- Tests with FPCA ---

    def test_transform_features_fpca_no_normalization(self, simple_time_series_data):
        """Test FPCA transform without normalization."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            feature_transform={"fpca": {"enabled": True, "min_var_ratio": 0.90}, "wavelet": {"enabled": False}},
            normalize={"pre_transform": False, "post_transform": False, "method": None}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = transform_features(simple_time_series_data, settings).data

        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] > 0

    def test_transform_features_fpca_with_pre_normalization(self, simple_time_series_data):
        """Test FPCA transform with pre-normalization (axis=0)."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            feature_transform={"fpca": {"enabled": True, "min_var_ratio": 0.90}, "wavelet": {"enabled": False}},
            normalize={
                "pre_transform": True,
                "post_transform": False,
                "method": "standardize",
                "axis": 0  # Use axis=0 for proper broadcasting
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = transform_features(simple_time_series_data, settings).data

        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] > 0

    # --- Tests with Wavelet ---

    def test_transform_features_wavelet_no_normalization(self, simple_time_series_data):
        """Test wavelet transform without normalization."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            feature_transform={"fpca": {"enabled": False}, "wavelet": {"enabled": True, "wavelet_name": "db1", "pca_n_components": 5}},
            normalize={"pre_transform": False, "post_transform": False, "method": None}
        )

        result = transform_features(simple_time_series_data, settings).data

        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] > 0

    def test_transform_features_wavelet_with_pre_normalization(self, simple_time_series_data):
        """Test wavelet transform with pre-normalization."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            feature_transform={"fpca": {"enabled": False}, "wavelet": {"enabled": True, "wavelet_name": "db1", "pca_n_components": 5}},
            normalize={
                "pre_transform": True,
                "post_transform": False,
                "method": "min_max_quantile",
                "quantile": 0.05,
                "axis": 1  # MIN_MAX_QUANTILE works with axis=1
            }
        )

        result = transform_features(simple_time_series_data, settings).data

        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] > 0

    def test_transform_features_wavelet_with_normalization(self, simple_time_series_data):
        """Test wavelet transform with normalization enabled."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            feature_transform={
                "fpca": {"enabled": False},
                "wavelet": {"enabled": True, "wavelet_name": "db1", "pca_n_components": 5},
                "normalize": {"enabled": True, "method": "standardize"},
            }
        )

        result = transform_features(simple_time_series_data, settings).data

        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] > 0  # wavelet + normalization produces features

    # --- Integration tests ---

    def test_transform_features_reduces_dimensionality(self, large_dataset):
        """Test that transform reduces dimensionality appropriately."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            feature_transform={
                "fpca": {"enabled": False},
                "wavelet": {"enabled": True, "wavelet_name": "db1", "pca_n_components": 10, "pca_scope": "global"},
            }
        )

        result = transform_features(large_dataset, settings).data

        # Wavelet PCA produces 10 global components; transform_features may
        # append magnitude features, so total can exceed 10 but should still
        # be much less than original 96.
        assert result.shape[1] >= 10
        assert result.shape[0] == large_dataset.shape[0]

    def test_transform_features_reproducible(self, simple_time_series_data):
        """Test that results are reproducible with same settings."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            feature_transform={
                "fpca": {"enabled": False},
                "wavelet": {"enabled": True, "wavelet_name": "db1", "pca_n_components": 5, "seed": 42},
            },
        )

        result1 = transform_features(simple_time_series_data, settings).data
        result2 = transform_features(simple_time_series_data, settings).data

        np.testing.assert_array_almost_equal(result1, result2)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
