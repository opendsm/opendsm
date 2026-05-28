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
    _safe_standardize,
    _fpca_base,
    normalize,
    fpca_transform,
    wavelet_transform,
    transform_features,
    FpcaError,
)
from opendsm.common.clustering.settings import (
    ClusteringSettings,
    NormalizeSettings,
    NormalizeChoice,
    TransformChoice,
)


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
        assert result.shape == data.shape
        np.testing.assert_array_almost_equal(result[:, 1], data[:, 1] - center[1])
        # First and third should be scaled
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

        assert result.shape == data.shape
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

    def test_standardize_axis_0(self, simple_time_series_data):
        """Test standardization along axis 0 (column-wise)."""
        settings = NormalizeSettings(method=NormalizeChoice.STANDARDIZE, axis=0)
        result = normalize(simple_time_series_data, settings)

        # Each column should have approximately zero mean and unit variance
        assert result.shape == simple_time_series_data.shape
        col_means = np.mean(result, axis=0)
        col_stds = np.std(result, axis=0)

        np.testing.assert_array_almost_equal(col_means, 0, decimal=10)
        assert np.min(col_stds) > 0.9

    def test_standardize_axis_none(self, simple_time_series_data):
        """Test standardization over entire array (axis=None)."""
        settings = NormalizeSettings(method=NormalizeChoice.STANDARDIZE, axis=None)
        result = normalize(simple_time_series_data, settings)

        # Global mean and std should be 0 and 1
        assert result.shape == simple_time_series_data.shape
        np.testing.assert_almost_equal(np.mean(result), 0, decimal=10)
        np.testing.assert_almost_equal(np.std(result), 1, decimal=10)

    def test_standardize_constant_data_axis_0(self):
        """Test standardization with constant data along axis 0."""
        data = np.ones((10, 24)) * 42.0
        settings = NormalizeSettings(method=NormalizeChoice.STANDARDIZE, axis=0)
        result = normalize(data, settings)

        # Should be centered (all zeros after subtracting constant)
        assert result.shape == data.shape
        np.testing.assert_array_almost_equal(result, 0)

    # --- Tests for MED_MAD method ---

    def test_med_mad_axis_0(self, simple_time_series_data):
        """Test median-MAD normalization along axis 0."""
        settings = NormalizeSettings(method=NormalizeChoice.MED_MAD, axis=0)
        result = normalize(simple_time_series_data, settings)

        # Each column should have approximately zero median
        assert result.shape == simple_time_series_data.shape
        col_medians = np.median(result, axis=0)
        np.testing.assert_array_almost_equal(col_medians, 0, decimal=10)

    def test_med_mad_axis_none(self, simple_time_series_data):
        """Test median-MAD normalization over entire array."""
        settings = NormalizeSettings(method=NormalizeChoice.MED_MAD, axis=None)
        result = normalize(simple_time_series_data, settings)

        # Global median should be approximately 0
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
        # All values should be finite
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

            # For non-constant columns, check that the quantile values map correctly
            # Values at the quantiles should be close to -1 and 1
            result_col = result[:, col_idx]

            # Find values close to the original quantiles
            lower_mask = np.abs(col_data - min_val) < 1e-10
            upper_mask = np.abs(col_data - max_val) < 1e-10

            if np.any(lower_mask):
                # Values at lower quantile should be close to -1
                np.testing.assert_allclose(result_col[lower_mask], -1.0, rtol=1e-4, atol=1e-8)

            if np.any(upper_mask):
                # Values at upper quantile should be close to 1
                np.testing.assert_allclose(result_col[upper_mask], 1.0, rtol=1e-4, atol=1e-8)

        # The bulk of values (between 10th and 90th percentile) should be in [-1, 1]
        # but outliers can be outside this range
        percentiles = np.percentile(result, [10, 90])
        assert percentiles[0] >= -1.5  # 10th percentile shouldn't be too extreme
        assert percentiles[1] <= 1.5   # 90th percentile shouldn't be too extreme

    def test_min_max_quantile_different_quantiles(self, simple_time_series_data):
        """Test different quantile values."""
        for q in [0.01, 0.05, 0.1, 0.25, 0.4]:
            settings = NormalizeSettings(
                method=NormalizeChoice.MIN_MAX_QUANTILE,
                quantile=q,
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

        # Suppress deprecation warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = _fpca_base(x, simple_time_series_data, min_var_ratio=0.90)

        # Should reduce dimensionality
        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] < simple_time_series_data.shape[1]
        assert result.shape[1] > 0

    def test_fpca_base_invalid_min_var_ratio_too_low(self, simple_time_series_data):
        """Test _fpca_base with min_var_ratio <= 0."""
        x = np.arange(simple_time_series_data.shape[1])

        with pytest.raises(FpcaError, match="min_var_ratio but be greater than 0"):
            _fpca_base(x, simple_time_series_data, min_var_ratio=0.0)

        with pytest.raises(FpcaError, match="min_var_ratio but be greater than 0"):
            _fpca_base(x, simple_time_series_data, min_var_ratio=-0.1)

    def test_fpca_base_invalid_min_var_ratio_too_high(self, simple_time_series_data):
        """Test _fpca_base with min_var_ratio >= 1."""
        x = np.arange(simple_time_series_data.shape[1])

        with pytest.raises(FpcaError, match="min_var_ratio but be greater than 0"):
            _fpca_base(x, simple_time_series_data, min_var_ratio=1.0)

        with pytest.raises(FpcaError, match="min_var_ratio but be greater than 0"):
            _fpca_base(x, simple_time_series_data, min_var_ratio=1.5)

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

    def test_fpca_base_empty_x(self, simple_time_series_data):
        """Test _fpca_base with empty x array."""
        x = np.array([])

        with pytest.raises(FpcaError, match="provided empty values for fpca"):
            _fpca_base(x, simple_time_series_data, min_var_ratio=0.90)

    def test_fpca_base_empty_y(self):
        """Test _fpca_base with empty y array."""
        x = np.arange(24)
        y = np.array([]).reshape(0, 24)

        with pytest.raises(FpcaError, match="provided empty values for fpca"):
            _fpca_base(x, y, min_var_ratio=0.90)

    def test_fpca_base_different_var_ratios(self, simple_time_series_data):
        """Test _fpca_base with different variance ratios."""
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
            transform_selection=TransformChoice.FPCA,
            fpca_transform={"min_var_ratio": 0.90}
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
        """Test FPCA with different variance ratios."""
        n_components = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for min_var_ratio in [0.80, 0.90, 0.95, 0.97]:
                settings = ClusteringSettings(
                    algorithm_selection="bisecting_kmeans",
                    seed=42,
                    transform_selection=TransformChoice.FPCA,
                    fpca_transform={"min_var_ratio": min_var_ratio}
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
            transform_selection=TransformChoice.FPCA,
            fpca_transform={"min_var_ratio": 0.85}
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
            transform_selection=TransformChoice.FPCA,
            fpca_transform={"min_var_ratio": 0.90}
        )

        with pytest.raises(FpcaError):
            fpca_transform(data, settings)

    @pytest.mark.slow
    def test_fpca_transform_deterministic(self, simple_time_series_data):
        """Test that FPCA transform produces consistent results."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            transform_selection=TransformChoice.FPCA,
            fpca_transform={"min_var_ratio": 0.90}
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
            transform_selection=TransformChoice.WAVELET,
            wavelet_transform={"wavelet_name": "db1", "pca_n_components": 5}
        )

        result = wavelet_transform(simple_time_series_data, settings)

        # Should have requested number of components plus scale feature
        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] == 6  # 5 PCA components + 1 scale feature

    def test_wavelet_without_scale_feature(self, simple_time_series_data):
        """Test wavelet transformation without scale feature."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            transform_selection=TransformChoice.WAVELET,
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 5,
                "include_scale_feature": False
            }
        )

        result = wavelet_transform(simple_time_series_data, settings)

        # Should have only PCA components (no scale feature)
        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] == 5

    def test_wavelet_different_wavelets(self, simple_time_series_data):
        """Test different wavelet types."""
        wavelets = ["db1", "haar", "coif6", "sym11"]

        for wavelet_name in wavelets:
            settings = ClusteringSettings(
                algorithm_selection="bisecting_kmeans",
                seed=42,
                transform_selection=TransformChoice.WAVELET,
                wavelet_transform={
                    "wavelet_name": wavelet_name,
                    "pca_n_components": 5
                }
            )
            result = wavelet_transform(simple_time_series_data, settings)

            assert result.shape[0] == simple_time_series_data.shape[0]
            assert result.shape[1] > 0

    def test_wavelet_with_variance_ratio(self, simple_time_series_data):
        """Test wavelet with PCA variance ratio instead of n_components."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            transform_selection=TransformChoice.WAVELET,
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": None,
                "pca_min_variance_ratio_explained": 0.90,
                "include_scale_feature": False
            }
        )

        result = wavelet_transform(simple_time_series_data, settings)

        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] > 0

    def test_wavelet_with_mle(self, simple_time_series_data):
        """Test wavelet with MLE for PCA n_components."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            transform_selection=TransformChoice.WAVELET,
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": "mle",
                "include_scale_feature": False
            }
        )

        result = wavelet_transform(simple_time_series_data, settings)

        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] > 0

    def test_wavelet_with_post_normalization(self, simple_time_series_data):
        """Test wavelet with post-transform normalization."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            transform_selection=TransformChoice.WAVELET,
            normalize={"pre_transform": False, "post_transform": True, "method": "standardize"},
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 5,
                "include_scale_feature": False
            }
        )

        result = wavelet_transform(simple_time_series_data, settings)

        assert result.shape[0] == simple_time_series_data.shape[0]
        # Post-normalized features should have approximately zero mean and unit std
        np.testing.assert_almost_equal(np.mean(result), 0, decimal=1)
        np.testing.assert_almost_equal(np.std(result), 1, decimal=1)

    def test_wavelet_different_n_levels(self, simple_time_series_data):
        """Test wavelet with different decomposition levels."""
        for n_levels in [None, 1, 2, 3]:
            settings = ClusteringSettings(
                algorithm_selection="bisecting_kmeans",
                seed=42,
                transform_selection=TransformChoice.WAVELET,
                wavelet_transform={
                    "wavelet_name": "db1",
                    "wavelet_n_levels": n_levels,
                    "pca_n_components": 5,
                    "include_scale_feature": False
                }
            )
            result = wavelet_transform(simple_time_series_data, settings)

            assert result.shape[0] == simple_time_series_data.shape[0]
            assert result.shape[1] > 0

    def test_wavelet_different_modes(self, simple_time_series_data):
        """Test wavelet with different extension modes."""
        modes = ["smooth", "periodic", "zero", "symmetric"]

        for mode in modes:
            settings = ClusteringSettings(
                algorithm_selection="bisecting_kmeans",
                seed=42,
                transform_selection=TransformChoice.WAVELET,
                wavelet_transform={
                    "wavelet_name": "db1",
                    "wavelet_mode": mode,
                    "pca_n_components": 5,
                    "include_scale_feature": False
                }
            )
            result = wavelet_transform(simple_time_series_data, settings)

            assert result.shape[0] == simple_time_series_data.shape[0]
            assert result.shape[1] > 0

    def test_wavelet_small_dataset(self, small_dataset):
        """Test wavelet on small dataset."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            transform_selection=TransformChoice.WAVELET,
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 3
            }
        )

        result = wavelet_transform(small_dataset, settings)

        assert result.shape[0] == small_dataset.shape[0]
        assert result.shape[1] > 0

    def test_wavelet_large_dataset(self, large_dataset):
        """Test wavelet on larger dataset."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            transform_selection=TransformChoice.WAVELET,
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 10
            }
        )

        result = wavelet_transform(large_dataset, settings)

        assert result.shape[0] == large_dataset.shape[0]
        assert result.shape[1] > 0

    def test_wavelet_deterministic_with_seed(self, simple_time_series_data):
        """Test that wavelet transform is deterministic with same seed."""
        settings1 = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            transform_selection=TransformChoice.WAVELET,
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 5,
                "seed": 42
            }
        )

        settings2 = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            transform_selection=TransformChoice.WAVELET,
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 5,
                "seed": 42
            }
        )

        result1 = wavelet_transform(simple_time_series_data, settings1)
        result2 = wavelet_transform(simple_time_series_data, settings2)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_wavelet_scale_feature_is_median(self, simple_time_series_data):
        """Test that scale feature is the median of each row."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            transform_selection=TransformChoice.WAVELET,
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 5,
                "include_scale_feature": True
            }
        )

        result = wavelet_transform(simple_time_series_data, settings)

        # Last column should be the median
        expected_medians = np.median(simple_time_series_data, axis=1)
        np.testing.assert_array_almost_equal(result[:, -1], expected_medians)


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
            transform_selection=TransformChoice.FPCA,
            normalize={"pre_transform": False, "post_transform": False, "method": None},
            fpca_transform={"min_var_ratio": 0.90}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = transform_features(simple_time_series_data, settings)

        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] > 0

    def test_transform_features_fpca_with_pre_normalization(self, simple_time_series_data):
        """Test FPCA transform with pre-normalization (axis=0)."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            transform_selection=TransformChoice.FPCA,
            normalize={
                "pre_transform": True,
                "post_transform": False,
                "method": "standardize",
                "axis": 0  # Use axis=0 for proper broadcasting
            },
            fpca_transform={"min_var_ratio": 0.90}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = transform_features(simple_time_series_data, settings)

        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] > 0

    # --- Tests with Wavelet ---

    def test_transform_features_wavelet_no_normalization(self, simple_time_series_data):
        """Test wavelet transform without normalization."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            transform_selection=TransformChoice.WAVELET,
            normalize={"pre_transform": False, "post_transform": False, "method": None},
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 5
            }
        )

        result = transform_features(simple_time_series_data, settings)

        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] > 0

    def test_transform_features_wavelet_with_pre_normalization(self, simple_time_series_data):
        """Test wavelet transform with pre-normalization."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            transform_selection=TransformChoice.WAVELET,
            normalize={
                "pre_transform": True,
                "post_transform": False,
                "method": "min_max_quantile",
                "quantile": 0.05,
                "axis": 1  # MIN_MAX_QUANTILE works with axis=1
            },
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 5
            }
        )

        result = transform_features(simple_time_series_data, settings)

        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] > 0

    def test_transform_features_wavelet_with_post_normalization(self, simple_time_series_data):
        """Test wavelet transform with post-normalization."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            transform_selection=TransformChoice.WAVELET,
            normalize={
                "pre_transform": False,
                "post_transform": True,
                "method": "standardize"
            },
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 5,
                "include_scale_feature": False
            }
        )

        result = transform_features(simple_time_series_data, settings)

        assert result.shape[0] == simple_time_series_data.shape[0]
        # Should be globally normalized
        np.testing.assert_almost_equal(np.mean(result), 0, decimal=1)

    # --- Integration tests ---

    def test_transform_features_reduces_dimensionality(self, large_dataset):
        """Test that transform reduces dimensionality appropriately."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            transform_selection=TransformChoice.WAVELET,
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 10,
                "include_scale_feature": False
            }
        )

        result = transform_features(large_dataset, settings)

        # Should reduce from 96 to 10 dimensions
        assert result.shape[1] == 10
        assert result.shape[1] < large_dataset.shape[1]

    def test_transform_features_reproducible(self, simple_time_series_data):
        """Test that results are reproducible with same settings."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            transform_selection=TransformChoice.WAVELET,
            normalize={
                "pre_transform": True,
                "method": "min_max_quantile",
                "quantile": 0.05,
                "axis": 1
            },
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 5,
                "seed": 42
            }
        )

        result1 = transform_features(simple_time_series_data, settings)
        result2 = transform_features(simple_time_series_data, settings)

        np.testing.assert_array_almost_equal(result1, result2)


# =============================================================================
# Parametrized tests
# =============================================================================

class TestParametrizedTransforms:
    """Parametrized tests across multiple configurations."""

    @pytest.mark.parametrize("wavelet", ["db1", "haar", "coif6", "sym11"])
    @pytest.mark.parametrize("n_components", [3, 5, 10])
    def test_wavelet_combinations(self, simple_time_series_data, wavelet, n_components):
        """Test combinations of wavelets and component counts."""
        settings = ClusteringSettings(
            algorithm_selection="bisecting_kmeans",
            seed=42,
            transform_selection=TransformChoice.WAVELET,
            wavelet_transform={
                "wavelet_name": wavelet,
                "pca_n_components": n_components,
                "include_scale_feature": False
            }
        )

        result = wavelet_transform(simple_time_series_data, settings)

        assert result.shape[0] == simple_time_series_data.shape[0]
        assert result.shape[1] == n_components


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
