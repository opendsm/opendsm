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

"""Comprehensive test suite for hourly_interpolation module."""

import numpy as np
import pandas as pd
import pytest

from opendsm.common.hourly_interpolation import (
    interpolate,
    _autocorr_fcn,
    _AUTOCORR_FFT_LAG_THRESHOLD,
)
from opendsm.common.test_data import load_test_data



# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def real_hourly_data():
    """Load real hourly energy data for testing."""
    df_baseline, df_reporting = load_test_data("hourly_treatment_data")

    # Get first meter's baseline data
    meter_id = df_baseline.index.get_level_values(0)[0]
    df = df_baseline.loc[meter_id].copy()

    return df


@pytest.fixture
def real_hourly_data_with_gaps():
    """Load real hourly data and introduce missing values."""
    df_baseline, df_reporting = load_test_data("hourly_treatment_data")

    # Get first meter's baseline data
    meter_id = df_baseline.index.get_level_values(0)[0]
    df = df_baseline.loc[meter_id].copy()

    # Introduce missing values at various locations
    df.loc[df.index[10:15], "temperature"] = np.nan
    df.loc[df.index[100:105], "ghi"] = np.nan
    df.loc[df.index[[50, 150, 250]], "observed"] = np.nan

    return df


@pytest.fixture
def simple_series():
    """Create simple time series data for unit testing."""
    np.random.seed(42)
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])


@pytest.fixture
def periodic_series():
    """Create periodic time series data for autocorrelation testing."""
    np.random.seed(42)
    t = np.arange(0, 100)

    return np.sin(2 * np.pi * t / 24) + 0.1 * np.random.randn(100)


@pytest.fixture
def series_with_nans():
    """Create time series with missing values."""
    np.random.seed(42)
    data = np.arange(20, dtype=float)
    # Introduce some NaN values
    data[[2, 5, 8, 15]] = np.nan

    return data


# =============================================================================
# Tests for interpolate
# =============================================================================

class TestInterpolate:
    """Tests for interpolate function."""

    def test_interpolate_no_missing_values_adds_flags(self, real_hourly_data):
        """Test that interpolation on complete data adds flags but doesn't change values."""
        df_original = real_hourly_data.copy()
        # Ensure no NaNs
        df_original = df_original.dropna()

        result = interpolate(df_original, columns=["temperature"])

        # Original values should be unchanged
        np.testing.assert_array_equal(
            result["temperature"].values,
            df_original["temperature"].values
        )
        # Interpolation flag should exist and be all False
        assert "interpolated_temperature" in result.columns
        assert result["interpolated_temperature"].sum() == 0

    def test_interpolate_fills_missing_values(self, real_hourly_data_with_gaps):
        """Test that interpolation fills missing values."""
        df_copy = real_hourly_data_with_gaps.copy()
        initial_nan_count = df_copy["temperature"].isna().sum()

        assert initial_nan_count > 0  # Verify we have NaNs to start

        result = interpolate(df_copy, columns=["temperature"])

        # All NaN values should be filled
        assert result["temperature"].isna().sum() == 0

    def test_interpolate_sets_correct_flags(self, real_hourly_data_with_gaps):
        """Test that interpolation sets correct flags for filled values."""
        df_copy = real_hourly_data_with_gaps.copy()
        initial_nan_indices = df_copy[df_copy["temperature"].isna()].index
        initial_nan_count = len(initial_nan_indices)

        result = interpolate(df_copy, columns=["temperature"])

        # Interpolation flag should be True for originally missing values
        assert "interpolated_temperature" in result.columns
        assert result.loc[initial_nan_indices, "interpolated_temperature"].all()

        # Total number of True flags should equal number of originally missing values
        assert result["interpolated_temperature"].sum() == initial_nan_count

    @pytest.mark.parametrize("column", ["temperature", "ghi", "observed"])
    def test_interpolate_different_columns(self, real_hourly_data_with_gaps, column):
        """Test interpolation works for different columns."""
        df_copy = real_hourly_data_with_gaps.copy()

        result = interpolate(df_copy, columns=[column])

        # Should have no NaN values after interpolation
        assert result[column].isna().sum() == 0
        # Should have interpolation flag
        assert f"interpolated_{column}" in result.columns

    def test_interpolate_default_columns(self, real_hourly_data_with_gaps):
        """Test that interpolate uses default columns when not specified."""
        df_copy = real_hourly_data_with_gaps.copy()

        result = interpolate(df_copy)

        # Should interpolate default columns: temperature, ghi, observed
        for col in ["temperature", "ghi", "observed"]:
            if col in df_copy.columns:
                assert result[col].isna().sum() == 0
                assert f"interpolated_{col}" in result.columns

    def test_interpolate_skips_nonexistent_columns(self, real_hourly_data):
        """Test that interpolate skips columns not in DataFrame."""
        df_copy = real_hourly_data.copy()

        # Request interpolation of non-existent column
        result = interpolate(df_copy, columns=["nonexistent_column"])

        # Should not raise error, just skip the column
        assert "interpolated_nonexistent_column" not in result.columns

    def test_interpolate_preserves_existing_flags(self, real_hourly_data):
        """Test that interpolate doesn't overwrite existing interpolation flags."""
        df = real_hourly_data.copy().head(100)
        df.loc[df.index[10], "temperature"] = np.nan
        df["interpolated_temperature"] = False

        result = interpolate(df, columns=["temperature"])

        # Should skip creating new flag since it already exists
        # Original flag should remain unchanged
        assert result["interpolated_temperature"].equals(df["interpolated_temperature"])

    def test_interpolate_handles_all_nan_column(self):
        """Test that interpolate handles column with all NaN values."""
        dates = pd.date_range("2023-01-01", periods=100, freq="h")
        df = pd.DataFrame({
            "temperature": np.full(100, np.nan)
        }, index=dates)

        result = interpolate(df, columns=["temperature"])

        # Should still create interpolation flag
        assert "interpolated_temperature" in result.columns
        # All values remain NaN (nothing to interpolate from)
        assert result["temperature"].isna().all()


# =============================================================================
# Integration tests
# =============================================================================

class TestInterpolateIntegration:
    """Integration tests for complete interpolation pipeline."""

    def test_realistic_interpolation_on_real_data(self, real_hourly_data_with_gaps):
        """Test complete interpolation on real hourly energy data with gaps."""
        df_copy = real_hourly_data_with_gaps.copy()

        # Count missing values before interpolation
        missing_temp = df_copy["temperature"].isna().sum()
        missing_ghi = df_copy["ghi"].isna().sum()
        missing_obs = df_copy["observed"].isna().sum()

        result = interpolate(df_copy)

        # All values should be interpolated
        assert result["temperature"].isna().sum() == 0
        assert result["ghi"].isna().sum() == 0
        assert result["observed"].isna().sum() == 0

        # Interpolated values should be reasonable (not extreme outliers)
        # Temperature in Fahrenheit
        assert result["temperature"].min() > -30
        assert result["temperature"].max() < 120
        # GHI (Global Horizontal Irradiance) in W/m²
        assert result["ghi"].min() >= 0
        assert result["ghi"].max() < 2000
        # Observed energy values should be non-negative and within reasonable range
        assert result["observed"].min() >= 0
        assert result["observed"].max() < result["observed"].quantile(0.99) * 2.0

        # Flags should be set correctly
        assert result["interpolated_temperature"].sum() == missing_temp
        assert result["interpolated_ghi"].sum() == missing_ghi
        assert result["interpolated_observed"].sum() == missing_obs

    def test_multiple_columns_interpolated_independently(self):
        """Test that multiple columns are interpolated independently."""
        df_baseline, _ = load_test_data("hourly_treatment_data")
        meter_id = df_baseline.index.get_level_values(0)[0]
        df = df_baseline.loc[meter_id].copy()

        # Different missing patterns for each column
        df.loc[df.index[10:15], "temperature"] = np.nan
        df.loc[df.index[50:55], "ghi"] = np.nan
        df.loc[df.index[100:105], "observed"] = np.nan

        result = interpolate(df)

        # Each column should be fully interpolated
        assert result["temperature"].isna().sum() == 0
        assert result["ghi"].isna().sum() == 0
        assert result["observed"].isna().sum() == 0

        # Each should have correct flags
        assert result["interpolated_temperature"].sum() == 5
        assert result["interpolated_ghi"].sum() == 5
        assert result["interpolated_observed"].sum() == 5

    def test_interpolation_maintains_approximate_statistics(self):
        """Test that interpolation maintains approximate statistical properties."""
        df_baseline, _ = load_test_data("hourly_treatment_data")
        meter_id = df_baseline.index.get_level_values(0)[0]
        df = df_baseline.loc[meter_id].copy()

        # Store original statistics
        original_mean = df["observed"].mean()
        original_std = df["observed"].std()

        # Remove 5% of values randomly
        np.random.seed(42)
        n_to_remove = int(0.05 * len(df))
        remove_indices = np.random.choice(len(df), size=n_to_remove, replace=False)
        df.loc[df.index[remove_indices], "observed"] = np.nan

        result = interpolate(df)

        # Interpolated data should have similar statistics
        result_mean = result["observed"].mean()
        result_std = result["observed"].std()

        # Within 10% of original statistics
        assert abs(result_mean - original_mean) < 0.1 * original_mean
        assert abs(result_std - original_std) < 0.2 * original_std

    def test_large_gap_interpolation(self):
        """Test interpolation with large consecutive gaps."""
        df_baseline, _ = load_test_data("hourly_treatment_data")
        meter_id = df_baseline.index.get_level_values(0)[0]
        df = df_baseline.loc[meter_id].copy().head(1000)

        # Create large gap (48 hours)
        df.loc[df.index[400:448], "temperature"] = np.nan

        result = interpolate(df, columns=["temperature"])

        # Should fill the gap
        assert result["temperature"].isna().sum() == 0
        # Interpolated values should be reasonable
        gap_values = result.loc[df.index[400:448], "temperature"]
        assert gap_values.min() > df["temperature"].min() - 10
        assert gap_values.max() < df["temperature"].max() + 10

    def test_interpolation_respects_temporal_patterns(self, real_hourly_data_with_gaps):
        """Test that interpolation respects temporal patterns in data."""
        df_copy = real_hourly_data_with_gaps.copy()

        # Interpolate temperature (which has strong daily patterns)
        result = interpolate(df_copy, columns=["temperature"])

        # Check that interpolated values follow reasonable patterns
        # Temperature should still show daily variation
        result_hourly_mean = result.groupby(result.index.hour)["temperature"].mean()

        # Should have variation across hours (not flat)
        hourly_std = result_hourly_mean.std()
        assert hourly_std > 1.0  # At least some daily variation


# =============================================================================
# Comparison tests: autocorr interpolation vs linear interpolation
# =============================================================================

class TestInterpolateVsLinear:
    """Tests demonstrating autocorr interpolation outperforms linear on real data."""

    def test_autocorr_more_accurate_than_linear_on_large_gap(self):
        """Test autocorr interpolation outperforms linear on large gaps with daily cycles."""
        # Load real data with strong daily patterns
        df_baseline, _ = load_test_data("hourly_treatment_data")
        meter_id = df_baseline.index.get_level_values(0)[0]
        df = df_baseline.loc[meter_id].copy().head(24 * 7)  # 1 week

        # Save original temperature values
        original_temp = df["temperature"].copy()

        # Create a 24-hour gap (full day) to test daily pattern recognition
        # Autocorr-based methods shine with larger gaps where patterns matter
        gap_start = 24 * 3  # Day 3
        gap_end = gap_start + 24
        df.loc[df.index[gap_start:gap_end], "temperature"] = np.nan

        # Method 1: Autocorr-based interpolation
        df_autocorr = df.copy()
        result_autocorr = interpolate(df_autocorr, columns=["temperature"])

        # Method 2: Linear interpolation
        df_linear = df.copy()
        df_linear["temperature"] = df_linear["temperature"].interpolate(method="linear")

        # Compare against original values in the gap
        gap_indices = df.index[gap_start:gap_end]
        original_values = original_temp.loc[gap_indices]
        autocorr_values = result_autocorr.loc[gap_indices, "temperature"]
        linear_values = df_linear.loc[gap_indices, "temperature"]

        # Calculate Mean Absolute Error (MAE)
        mae_autocorr = np.mean(np.abs(autocorr_values - original_values))
        mae_linear = np.mean(np.abs(linear_values - original_values))

        # For large gaps (24+ hours), autocorr should outperform linear
        assert mae_autocorr < mae_linear, \
            f"Autocorr MAE ({mae_autocorr:.2f}°F) should be < Linear MAE ({mae_linear:.2f}°F) for large gaps"

        # Calculate improvement percentage
        improvement = (mae_linear - mae_autocorr) / mae_linear * 100
        assert improvement > 5, \
            f"Autocorr should improve accuracy by >5% on large gaps, got {improvement:.1f}%"

    def test_autocorr_handles_energy_consumption_patterns_better(self):
        """Test autocorr captures energy consumption patterns that linear misses."""
        # Load real energy consumption data
        df_baseline, _ = load_test_data("hourly_treatment_data")
        meter_id = df_baseline.index.get_level_values(0)[0]
        df = df_baseline.loc[meter_id].copy().head(24 * 21)  # 3 weeks

        original_observed = df["observed"].copy()

        # Create a 48-hour gap (2 days) to test weekly pattern recognition
        gap_start = 24 * 10  # Middle of week 2
        gap_end = gap_start + 48
        df.loc[df.index[gap_start:gap_end], "observed"] = np.nan

        # Autocorr interpolation
        df_autocorr = df.copy()
        result_autocorr = interpolate(df_autocorr, columns=["observed"])

        # Linear interpolation
        df_linear = df.copy()
        df_linear["observed"] = df_linear["observed"].interpolate(method="linear")

        # Compare against original
        gap_indices = df.index[gap_start:gap_end]
        original_values = original_observed.loc[gap_indices]
        autocorr_values = result_autocorr.loc[gap_indices, "observed"]
        linear_values = df_linear.loc[gap_indices, "observed"]

        # Calculate errors
        mae_autocorr = np.mean(np.abs(autocorr_values - original_values))
        mae_linear = np.mean(np.abs(linear_values - original_values))

        rmse_autocorr = np.sqrt(np.mean((autocorr_values - original_values) ** 2))
        rmse_linear = np.sqrt(np.mean((linear_values - original_values) ** 2))

        # Autocorr should outperform linear on both metrics
        assert mae_autocorr < mae_linear, \
            f"Autocorr MAE ({mae_autocorr:.2f}) should be < Linear MAE ({mae_linear:.2f})"

        assert rmse_autocorr < rmse_linear, \
            f"Autocorr RMSE ({rmse_autocorr:.2f}) should be < Linear RMSE ({rmse_linear:.2f})"

    def test_ghi_solar_irradiance_patterns_autocorr_vs_linear(self):
        """Test autocorr handles solar irradiance patterns better than linear."""
        # Load real GHI (Global Horizontal Irradiance) data
        df_baseline, _ = load_test_data("hourly_treatment_data")
        meter_id = df_baseline.index.get_level_values(0)[0]
        df = df_baseline.loc[meter_id].copy().head(24 * 7)  # 1 week

        original_ghi = df["ghi"].copy()

        # Create gap during daylight hours (when solar patterns are important)
        gap_start = 24 * 3 + 8   # Day 3, 8am
        gap_end = gap_start + 10  # 10-hour gap during daylight
        df.loc[df.index[gap_start:gap_end], "ghi"] = np.nan

        # Autocorr interpolation
        df_autocorr = df.copy()
        result_autocorr = interpolate(df_autocorr, columns=["ghi"])

        # Linear interpolation
        df_linear = df.copy()
        df_linear["ghi"] = df_linear["ghi"].interpolate(method="linear")

        # Compare against original
        gap_indices = df.index[gap_start:gap_end]
        original_values = original_ghi.loc[gap_indices]
        autocorr_values = result_autocorr.loc[gap_indices, "ghi"]
        linear_values = df_linear.loc[gap_indices, "ghi"]

        # Calculate errors
        mae_autocorr = np.mean(np.abs(autocorr_values - original_values))
        mae_linear = np.mean(np.abs(linear_values - original_values))

        # Autocorr should handle solar patterns better
        assert mae_autocorr < mae_linear, \
            f"Autocorr MAE ({mae_autocorr:.2f} W/m²) should be < Linear MAE ({mae_linear:.2f} W/m²)"


# =============================================================================
# _autocorr_fcn — characterization & correctness (masked loop vs FFT branch)
# =============================================================================

def _deterministic_acf_signal(n=2000):
    """Reproducible gappy signal: daily + weekly cycles + slow ramp, fixed gaps."""
    t = np.arange(n, dtype=float)
    x = np.sin(2 * np.pi * t / 24) + 0.5 * np.sin(2 * np.pi * t / 168) + 0.1 * t / n
    x[::37] = np.nan
    x[100:115] = np.nan

    return x


def _naive_acf(x, lags):
    """Reference ACF: masked mean/var, sum of valid lag products / n / var."""
    xm = np.ma.masked_invalid(x)
    mean = xm.mean()
    var = xm.var()

    if not np.isfinite(var) or var == 0:
        var = 1.0

    xc = xm - mean
    n = len(x)
    out = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        if lag == 0:
            out[i] = 1.0
        elif lag < n:
            out[i] = float(np.ma.filled(np.ma.sum(xc[lag:] * xc[:-lag]) / n / var, 0.0))

    return out


def _pos_corr(result):
    """{lag: corr} for lags >= 0 from the mirrored _autocorr_fcn output."""
    return {int(lag): float(corr) for lag, corr in result if lag >= 0}


class TestAutocorrFcn:
    """The masked-loop and FFT branches must agree and stay pinned to the original."""

    # Pinned from the original masked-loop implementation on _deterministic_acf_signal().
    REFERENCE = {
        0: 1.0,
        1: 0.9122603913,
        2: 0.8362625721,
        3: 0.7155700045,
        4: 0.5588584332,
        5: 0.37680267,
        6: 0.1817630883,
        7: -0.0130593771,
        8: -0.1945236306,
    }

    def test_loop_branch_matches_pinned_reference(self):
        """Few lags (<= threshold) take the masked loop and reproduce the pinned values."""
        assert len(self.REFERENCE) <= _AUTOCORR_FFT_LAG_THRESHOLD
        x = _deterministic_acf_signal()
        corr = _pos_corr(_autocorr_fcn(x, np.arange(0, len(self.REFERENCE)), exclude_0=False))

        for lag, expected in self.REFERENCE.items():
            assert corr[lag] == pytest.approx(expected, abs=1e-8), f"lag {lag}"

    def test_fft_branch_matches_pinned_reference(self):
        """Many lags (> threshold) take the FFT path and reproduce the same pinned values."""
        x = _deterministic_acf_signal()
        corr = _pos_corr(
            _autocorr_fcn(x, np.arange(0, _AUTOCORR_FFT_LAG_THRESHOLD + 50), exclude_0=False)
        )

        for lag, expected in self.REFERENCE.items():
            assert corr[lag] == pytest.approx(expected, abs=1e-8), f"lag {lag}"

    def test_loop_and_fft_branches_agree(self):
        """The two branches return identical correlations on their shared lags."""
        x = _deterministic_acf_signal()
        few = _pos_corr(
            _autocorr_fcn(x, np.arange(0, _AUTOCORR_FFT_LAG_THRESHOLD), exclude_0=False)
        )
        many = _pos_corr(
            _autocorr_fcn(x, np.arange(0, _AUTOCORR_FFT_LAG_THRESHOLD + 100), exclude_0=False)
        )

        for lag in few:
            assert few[lag] == pytest.approx(many[lag], abs=1e-10), f"lag {lag}"

    @pytest.mark.parametrize("n_lags", [6, _AUTOCORR_FFT_LAG_THRESHOLD + 30])
    def test_matches_naive_reference(self, n_lags):
        """Both branches match an independent naive masked-loop reference."""
        x = _deterministic_acf_signal()
        lags = np.arange(0, n_lags)
        got = _pos_corr(_autocorr_fcn(x, lags, exclude_0=False))
        expected = _naive_acf(x, lags)

        for lag in lags:
            assert got[int(lag)] == pytest.approx(expected[lag], abs=1e-10), f"lag {lag}"

    def test_recovers_ar1_theoretical_acf(self):
        """AR(1) with phi=0.7 has ACF(k)=0.7**k; the FFT branch recovers it from data."""
        rng = np.random.default_rng(0)
        phi = 0.7
        e = rng.standard_normal(20_000)
        x = np.empty_like(e)
        x[0] = e[0]

        for i in range(1, len(e)):
            x[i] = phi * x[i - 1] + e[i]

        corr = _pos_corr(_autocorr_fcn(x, np.arange(0, 30), exclude_0=False))

        for k in range(1, 6):
            assert corr[k] == pytest.approx(phi ** k, abs=0.03), f"lag {k}"

    def test_constant_input_is_zero_beyond_lag0(self):
        """Constant input (var=0) is guarded: lag 0 = 1, all other lags = 0."""
        x = np.full(500, 7.0)
        corr = _pos_corr(_autocorr_fcn(x, np.arange(0, 6), exclude_0=False))

        assert corr[0] == 1.0
        assert all(corr[lag] == 0.0 for lag in range(1, 6))

    def test_all_nan_input_is_finite(self):
        """All-NaN input does not crash and yields finite correlations."""
        x = np.full(200, np.nan)
        result = _autocorr_fcn(x, np.arange(0, 6))

        assert np.isfinite(result[:, 1]).all()

    def test_lags_at_least_n_are_zero(self):
        """Lags >= series length return 0 with no out-of-range access."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        corr = _pos_corr(_autocorr_fcn(x, np.arange(0, 6), exclude_0=False))

        assert corr[4] == 0.0
        assert corr[5] == 0.0

    def test_exclude_0_drops_zero_lag_and_mirrors(self):
        """exclude_0=True removes lag 0; correlations are symmetric in lag sign."""
        x = _deterministic_acf_signal()
        lags = np.arange(0, 5)
        with_0 = _autocorr_fcn(x, lags, exclude_0=False)
        without_0 = _autocorr_fcn(x, lags, exclude_0=True)

        assert 0 in with_0[:, 0]
        assert 0 not in without_0[:, 0]

        signed = {int(lag): float(corr) for lag, corr in with_0}

        for lag in range(1, 5):
            assert signed[lag] == pytest.approx(signed[-lag], abs=1e-12), f"lag {lag}"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
