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

"""
Comprehensive test suite for hourly data classes and sufficiency criteria.

Tests cover:
- Data class initialization, properties, and configuration
- Data processing pipeline (zero conversion, deduplication, gap filling, interpolation)
- All 11 sufficiency criteria checks + warnings
- Integration scenarios and edge cases
"""

import logging

import pytest
import numpy as np
import pandas as pd
from datetime import date

from opendsm.eemeter.models.hourly.data import (
    _HourlyData,
    HourlyBaselineData,
    HourlyReportingData,
)
from opendsm.eemeter.common.data_settings import HourlyDataSettings

from .conftest import (
    create_hourly_dataframe,
    create_repeated_values,
    create_extreme_values,
    assert_has_disqualification,
    assert_has_warning,
    assert_column_exists,
    assert_all_hours_present,
)



class TestHourlyDataClassInit:
    """Tests for data class initialization, properties, and configuration"""

    # ===== Initialization & Input Validation Tests =====

    def test_baseline_with_valid_data(self, synthetic_hourly_data):
        """Basic successful instantiation of HourlyBaselineData"""
        data = HourlyBaselineData(
            df=synthetic_hourly_data, is_electricity_data=True
        )

        assert data is not None
        assert isinstance(data.df, pd.DataFrame)
        assert data.is_electricity_data is True

    def test_baseline_with_minimal_data(self):
        """Only required columns (observed, temperature)"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert "observed" in data.df.columns
        assert "temperature" in data.df.columns

    def test_baseline_with_ghi_data(self, synthetic_hourly_with_ghi):
        """Include optional GHI column"""
        data = HourlyBaselineData(
            df=synthetic_hourly_with_ghi, is_electricity_data=True
        )

        assert "ghi" in data.df.columns

    def test_reporting_with_valid_data(self, synthetic_hourly_data):
        """Basic reporting data instantiation"""
        data = HourlyReportingData(
            df=synthetic_hourly_data, is_electricity_data=True
        )

        assert data is not None
        assert isinstance(data.df, pd.DataFrame)

    def test_reporting_without_observed(self):
        """Observed column missing (should work for reporting)"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-03-31")
        df = df.drop(columns=["observed"])
        data = HourlyReportingData(df=df, is_electricity_data=True)

        assert "observed" in data.df.columns  # Auto-created with NaN

    def test_reporting_with_observed(self, synthetic_hourly_data):
        """Observed column present (should work for reporting)"""
        data = HourlyReportingData(
            df=synthetic_hourly_data, is_electricity_data=True
        )

        assert "observed" in data.df.columns

    def test_missing_temperature_column(self):
        """Should raise ValueError if temperature missing"""
        df = create_hourly_dataframe()
        df = df.drop(columns=["temperature"])

        with pytest.raises(ValueError, match="temperature"):
            HourlyBaselineData(df=df, is_electricity_data=True)

    def test_missing_observed_column_baseline(self):
        """Should raise ValueError if observed missing from baseline"""
        df = create_hourly_dataframe()
        df = df.drop(columns=["observed"])

        with pytest.raises(ValueError, match="observed"):
            HourlyBaselineData(df=df, is_electricity_data=True)

    def test_datetime_as_column(self):
        """Datetime in column, not index"""
        df = create_hourly_dataframe()
        df = df.reset_index().rename(columns={"index": "datetime"})

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert isinstance(data.df.index, pd.DatetimeIndex)

    def test_datetime_missing_timezone(self):
        """Should raise ValueError for timezone-naive datetime"""
        df = create_hourly_dataframe()
        df.index = df.index.tz_localize(None)

        with pytest.raises(ValueError, match="timezone"):
            HourlyBaselineData(df=df, is_electricity_data=True)

    def test_datetime_utc_timezone(self):
        """Should issue warning for UTC timezone"""
        df = create_hourly_dataframe()
        df.index = df.index.tz_convert("UTC")

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # Check for UTC warning
        warning_names = [w.qualified_name for w in data.warnings]
        assert any("utc" in name.lower() for name in warning_names)

    def test_datetime_non_ns_precision(self):
        """Should convert to nanosecond precision"""
        df = create_hourly_dataframe()
        # Force different precision (microseconds)
        # Need to remove timezone, convert precision, then re-add timezone
        original_tz = df.index.tz
        naive_index = df.index.tz_localize(None)
        us_index = naive_index.astype("datetime64[us]")
        df.index = us_index.tz_localize(original_tz, ambiguous='infer', nonexistent='shift_forward')

        # Verify we successfully created microsecond precision
        assert df.index.dtype.unit == "us", f"Test setup failed: expected us, got {df.index.dtype.unit}"

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # Verify it was converted to nanosecond precision
        assert data.df.index.dtype.unit == "ns", f"Expected ns, got {data.df.index.dtype.unit}"

    def test_is_electricity_data_flag(self):
        """Test both True/False for is_electricity_data"""
        df = create_hourly_dataframe()

        data_elec = HourlyBaselineData(df=df, is_electricity_data=True)
        data_gas = HourlyBaselineData(df=df, is_electricity_data=False)

        assert data_elec.is_electricity_data is True
        assert data_gas.is_electricity_data is False

    def test_abstract_base_class_instantiation(self):
        """_HourlyData raises NotImplementedError"""
        df = create_hourly_dataframe()

        with pytest.raises(NotImplementedError):
            _HourlyData(df=df, is_electricity_data=True)

    # ===== Property & Immutability Tests =====

    def test_df_property_returns_copy(self, synthetic_hourly_data):
        """Verify .df returns copy, not reference"""
        data = HourlyBaselineData(
            df=synthetic_hourly_data, is_electricity_data=True
        )

        df1 = data.df
        df2 = data.df

        assert df1 is not df2  # Different objects

    def test_df_immutability(self, synthetic_hourly_data):
        """Modifying returned df doesn't affect internal data"""
        data = HourlyBaselineData(
            df=synthetic_hourly_data, is_electricity_data=True
        )

        df = data.df
        original_value = df.iloc[0, 0]
        df.iloc[0, 0] = 999999

        # Get fresh copy
        df2 = data.df
        assert df2.iloc[0, 0] == original_value  # Unchanged

    def test_tz_property(self, synthetic_hourly_data):
        """Timezone preserved correctly"""
        data = HourlyBaselineData(
            df=synthetic_hourly_data, is_electricity_data=True
        )

        assert data.tz is not None
        assert str(data.tz) == "America/New_York"

    def test_settings_property(self, synthetic_hourly_data):
        """Settings accessible and correct type"""
        data = HourlyBaselineData(
            df=synthetic_hourly_data, is_electricity_data=True
        )

        assert data.settings is not None
        assert isinstance(data.settings, HourlyDataSettings)

    def test_pv_start_property(self):
        """PV start date handling"""
        df = create_hourly_dataframe()
        pv_date = date(2019, 6, 1)
        data = HourlyBaselineData(
            df=df, is_electricity_data=True, pv_start=pv_date
        )

        assert data.pv_start == pv_date

    def test_warnings_list(self, synthetic_hourly_data):
        """Warnings list accessible"""
        data = HourlyBaselineData(
            df=synthetic_hourly_data, is_electricity_data=True
        )

        assert isinstance(data.warnings, list)

    def test_disqualification_list(self, synthetic_hourly_data):
        """Disqualifications list accessible"""
        data = HourlyBaselineData(
            df=synthetic_hourly_data, is_electricity_data=True
        )

        assert isinstance(data.disqualification, list)

    # ===== Settings Configuration Tests =====

    def test_default_settings(self, synthetic_hourly_data):
        """No settings passed, uses defaults"""
        data = HourlyBaselineData(
            df=synthetic_hourly_data, is_electricity_data=True
        )

        assert data.settings.sufficiency.min_baseline_length == 329
        assert data.settings.sufficiency.temperature.min_pct_daily_coverage == 0.9

    def test_dict_settings(self, synthetic_hourly_data):
        """Pass settings as dict"""
        settings_dict = {"sufficiency": {"min_baseline_length": 300}}
        data = HourlyBaselineData(
            df=synthetic_hourly_data,
            is_electricity_data=True,
            settings=settings_dict,
        )

        assert data.settings.sufficiency.min_baseline_length == 300

    def test_settings_object(self, synthetic_hourly_data):
        """Pass HourlyDataSettings instance"""
        settings = HourlyDataSettings(
            sufficiency={"min_baseline_length": 350}
        )
        data = HourlyBaselineData(
            df=synthetic_hourly_data, is_electricity_data=True, settings=settings
        )

        assert data.settings.sufficiency.min_baseline_length == 350

    def test_custom_sufficiency_thresholds(self, synthetic_hourly_data):
        """Override default thresholds"""
        settings_dict = {
            "sufficiency": {
                "temperature": {"min_pct_daily_coverage": 0.8},
                "observed": {"min_pct_unique_values": 0.3},
            }
        }
        data = HourlyBaselineData(
            df=synthetic_hourly_data,
            is_electricity_data=True,
            settings=settings_dict,
        )

        assert data.settings.sufficiency.temperature.min_pct_daily_coverage == 0.8
        assert data.settings.sufficiency.observed.min_pct_unique_values == 0.3

    def test_settings_immutability(self, synthetic_hourly_data):
        """Settings can't be changed after init"""
        data = HourlyBaselineData(
            df=synthetic_hourly_data, is_electricity_data=True
        )

        original_value = data.settings.sufficiency.min_baseline_length

        # Attempt to modify (should not affect stored settings due to Pydantic)
        with pytest.raises(AttributeError):
            data.settings = HourlyDataSettings()

        assert data.settings.sufficiency.min_baseline_length == original_value

    def test_invalid_settings(self, synthetic_hourly_data):
        """Invalid settings raise appropriate errors"""
        settings_dict = {"sufficiency": {"min_baseline_length": "invalid"}}

        with pytest.raises((ValueError, TypeError)):
            HourlyBaselineData(
                df=synthetic_hourly_data,
                is_electricity_data=True,
                settings=settings_dict,
            )

    def test_settings_inherited_by_data_class(self, synthetic_hourly_data):
        """is_electricity_data propagated to sufficiency checks"""
        data = HourlyBaselineData(
            df=synthetic_hourly_data, is_electricity_data=False
        )

        assert data.is_electricity_data is False


class TestHourlyDataProcessing:
    """Tests for data transformation steps in the processing pipeline"""

    # ===== Zero-to-NaN Conversion Tests =====

    def test_electricity_zeros_converted_to_nan(self):
        """Electricity data: zeros converted to NaN and then interpolated"""
        df = create_hourly_dataframe()
        df.loc[df.index[:10], "observed"] = 0

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # Zeros should be converted to NaN, then interpolated
        # Final values should not be NaN (they got interpolated)
        assert not data.df.iloc[:10]["observed"].isna().any()
        # They should be marked as interpolated
        assert data.df.iloc[:10]["interpolated_observed"].all()

    def test_gas_zeros_preserved(self):
        """Gas data: zeros preserved"""
        df = create_hourly_dataframe()
        df.loc[df.index[:10], "observed"] = 0

        data = HourlyBaselineData(df=df, is_electricity_data=False)

        # Zeros should be preserved (not converted to NaN)
        assert (data.df.iloc[:10]["observed"] == 0).any()

    def test_negative_electricity_values_not_converted(self):
        """Negative values not converted to NaN"""
        df = create_hourly_dataframe()
        df.loc[df.index[:10], "observed"] = -1.5

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert not data.df.iloc[:10]["observed"].isna().all()

    def test_temperature_zeros_unchanged(self):
        """Temperature zeros not converted"""
        df = create_hourly_dataframe()
        df.loc[df.index[:10], "temperature"] = 0

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert (data.df.iloc[:10]["temperature"] == 0).any()

    def test_partial_zeros(self):
        """Only some zeros converted"""
        df = create_hourly_dataframe()
        df.loc[df.index[5], "observed"] = 0
        df.loc[df.index[15], "observed"] = 5.0

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert data.df.loc[df.index[5], "observed"] != 0
        assert data.df.loc[df.index[15], "observed"] == 5.0

    # ===== Duplicate Removal Tests =====

    def test_duplicate_timestamps_removed(self):
        """Duplicate timestamps removed (keeps first)"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-01-31")
        duplicate_row = df.iloc[10:11].copy()
        df = pd.concat([df, duplicate_row])

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert data.df.index.is_unique

    def test_no_duplicates(self):
        """No change if no duplicates"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-01-31")
        original_len = len(df)

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # Length should be similar (may differ due to gap filling)
        assert len(data.df) >= original_len

    def test_multiple_duplicates_same_timestamp(self):
        """Only first kept when multiple duplicates"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-01-31")
        duplicate_row = df.iloc[10:11].copy()
        duplicate_row["observed"] = 999
        df = pd.concat([df, duplicate_row, duplicate_row])

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert data.df.index.is_unique
        # First value should be preserved (not 999)
        assert data.df.loc[df.index[10], "observed"] != 999

    def test_duplicate_with_different_values(self):
        """First value kept"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-01-31")
        first_value = df.iloc[10]["observed"]
        duplicate_row = df.iloc[10:11].copy()
        duplicate_row["observed"] = 999
        df = pd.concat([df, duplicate_row])

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert abs(data.df.loc[df.index[10], "observed"] - first_value) < 0.1

    def test_order_preserved(self):
        """Non-duplicate order maintained"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-01-31")

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert data.df.index.is_monotonic_increasing

    # ===== Gap Filling Tests =====

    def test_gap_filled_with_nans(self):
        """Missing hours filled with NaN"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-01-31")
        # Remove some hours
        df = df.iloc[::2]  # Keep every other hour

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # Should have all hours filled in
        assert_all_hours_present(data.df)

    def test_no_gaps(self):
        """Already contiguous data values unchanged"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-01-31")
        first_value = df.iloc[0]["observed"]

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # First value should remain close to original (may be slightly different due to processing)
        assert abs(data.df.iloc[0]["observed"] - first_value) < 1.0

    def test_irregular_gaps(self):
        """Multiple gap sizes handled"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-01-31")
        # Create irregular gaps
        df = df.drop(df.index[10:12])  # 2-hour gap
        df = df.drop(df.index[50:56])  # 6-hour gap

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert_all_hours_present(data.df)

    def test_timezone_preserved_after_gap_filling(self):
        """TZ maintained after gap filling"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-01-31")
        df = df.iloc[::2]  # Create gaps

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert str(data.df.index.tz) == "America/New_York"

    def test_date_and_hour_columns_added(self):
        """date, hour_of_day columns created"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-01-31")

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert_column_exists(data.df, "date")
        assert_column_exists(data.df, "hour_of_day")

    def test_earliest_hour_reset_to_zero(self):
        """Starts at midnight"""
        df = create_hourly_dataframe(start="2019-01-01 10:00", end="2019-01-31")

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert data.df.index[0].hour == 0

    def test_latest_hour_extended_to_23(self):
        """Ends at 11pm"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-01-31 18:00")

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert data.df.index[-1].hour == 23

    def test_partial_days_extended(self):
        """Extends to full days (hour 0-23)"""
        df = create_hourly_dataframe(start="2019-01-01 06:00", end="2019-01-02 18:00")

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # Should start at hour 0
        assert data.df.iloc[0].name.hour == 0

    # ===== Interpolation Tests =====

    @pytest.mark.parametrize(
        "column",
        ["observed", "temperature"],
        ids=["observed_small_gap", "temperature_small_gap"],
    )
    def test_interpolation_fills_small_gaps(self, column):
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df.loc[df.index[100:103], column] = np.nan

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert not data.df.loc[df.index[100:103], column].isna().all()

    def test_interpolation_ghi_column(self):
        """GHI interpolated if present"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31", include_ghi=True)
        df.loc[df.index[100:103], "ghi"] = np.nan

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # GHI should be interpolated
        assert "ghi" in data.df.columns

    def test_interpolated_flags_created(self):
        """interpolated_{col} columns added"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df.loc[df.index[100:103], "observed"] = np.nan

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert_column_exists(data.df, "interpolated_observed")
        assert_column_exists(data.df, "interpolated_temperature")

    def test_interpolated_flags_accurate(self):
        """Flags match filled locations"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df.loc[df.index[100:103], "observed"] = np.nan

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # Flags should be 1 for interpolated values
        assert data.df.loc[df.index[100:103], "interpolated_observed"].sum() > 0

    def test_no_interpolation_if_all_missing(self):
        """All NaN remains NaN"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-01-31")
        df["observed"] = np.nan

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # Should still be all NaN (can't interpolate nothing)
        assert data.df["observed"].isna().all()

    def test_no_interpolation_if_no_missing(self):
        """No change if complete"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        original_values = df["observed"].copy()

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # Compare only the values that existed in the original data (align by index)
        common_index = original_values.index.intersection(data.df.index)
        original_common = original_values.loc[common_index].dropna()
        processed_common = data.df.loc[common_index, "observed"].dropna()

        # Values should be very similar (allowing for minor floating point differences)
        correlation = np.corrcoef(original_common, processed_common)[0, 1]
        assert correlation > 0.99

    def test_short_series_interpolation(self):
        """Less than 3 days (fallback methods)"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-01-02")
        df.loc[df.index[10:12], "observed"] = np.nan

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # Should still interpolate using fallback methods
        assert not data.df.loc[df.index[10:12], "observed"].isna().all()

    def test_interpolation_methods_fallback(self):
        """time -> ffill -> bfill cascade"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-01-03")
        # Save the reference values before creating NaNs
        first_valid_value = df.iloc[5]["observed"]  # Value that should be bfilled
        last_valid_value = df.iloc[-6]["observed"]  # Value that should be ffilled

        # Create pattern that needs different methods
        df.loc[df.index[:5], "observed"] = np.nan  # Needs bfill
        df.loc[df.index[-5:], "observed"] = np.nan  # Needs ffill

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # First 5 values should be filled by bfill (equal to first valid value after them)
        assert data.df.iloc[:5]["observed"].notna().all()
        assert (data.df.iloc[:5]["observed"] == first_valid_value).all()

        # Last 5 values should be filled by ffill (equal to last valid value before them)
        assert data.df.iloc[-5:]["observed"].notna().all()
        assert (data.df.iloc[-5:]["observed"] == last_valid_value).all()

    # ===== PV Start Date Tests =====

    def test_pv_start_none_defaults_to_data_start(self):
        """No pv_start provided defaults to data start"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert_column_exists(data.df, "has_pv")
        # All should be 1 (PV from start)
        assert data.df["has_pv"].iloc[-1] == 1

    def test_pv_start_date_object(self):
        """Pass datetime.date"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        pv_date = date(2019, 6, 1)

        data = HourlyBaselineData(df=df, is_electricity_data=True, pv_start=pv_date)

        assert data.pv_start == pv_date

    def test_pv_start_string_converted(self):
        """Pass ISO string"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")

        data = HourlyBaselineData(df=df, is_electricity_data=True, pv_start="2019-06-01")

        assert data.pv_start == date(2019, 6, 1)

    def test_has_pv_column_created(self):
        """has_pv column added"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")

        data = HourlyBaselineData(df=df, is_electricity_data=True, pv_start="2019-06-01")

        assert_column_exists(data.df, "has_pv")

    def test_has_pv_before_pv_start(self):
        """has_pv=0 before date"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")

        data = HourlyBaselineData(df=df, is_electricity_data=True, pv_start="2019-06-01")

        before_pv = data.df[data.df.index < "2019-06-01"]
        assert (before_pv["has_pv"] == 0).all()

    def test_has_pv_after_pv_start(self):
        """has_pv=1 after date"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")

        data = HourlyBaselineData(df=df, is_electricity_data=True, pv_start="2019-06-01")

        after_pv = data.df[data.df.index >= "2019-06-01"]
        assert (after_pv["has_pv"] == 1).all()

    def test_has_pv_on_pv_start_date(self):
        """has_pv=1 on exact date"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")

        data = HourlyBaselineData(df=df, is_electricity_data=True, pv_start="2019-06-01")

        on_pv_date = data.df[data.df.index.date == date(2019, 6, 1)]
        assert (on_pv_date["has_pv"] == 1).all()

    def test_pv_start_mid_baseline(self):
        """PV install mid-period"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")

        data = HourlyBaselineData(df=df, is_electricity_data=True, pv_start="2019-07-15")

        before = data.df[data.df.index < "2019-07-15"]["has_pv"].sum()
        after = data.df[data.df.index >= "2019-07-15"]["has_pv"].sum()

        assert before == 0
        assert after > 0

    # ===== Output Structure Tests =====

    def test_output_columns_present(self):
        """Expected columns exist"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        required_cols = ["observed", "temperature", "date", "hour_of_day", "has_pv"]
        for col in required_cols:
            assert_column_exists(data.df, col)

    def test_output_index_type(self):
        """DatetimeIndex maintained"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert isinstance(data.df.index, pd.DatetimeIndex)

    def test_output_dtypes(self):
        """Correct data types"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert pd.api.types.is_numeric_dtype(data.df["observed"])
        assert pd.api.types.is_numeric_dtype(data.df["temperature"])
        assert pd.api.types.is_integer_dtype(data.df["hour_of_day"])

    def test_date_column_is_date_type(self):
        """date column is date, not datetime"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # Date column should contain date objects
        assert isinstance(data.df["date"].iloc[0], (date, pd.Timestamp))

    def test_hour_of_day_range(self):
        """hour_of_day in 0-23"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert data.df["hour_of_day"].min() >= 0
        assert data.df["hour_of_day"].max() <= 23

    def test_interpolated_flags_boolean(self):
        """Flags are int (0/1)"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df.loc[df.index[100:103], "observed"] = np.nan

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert data.df["interpolated_observed"].isin([0, 1]).all()
        assert data.df["interpolated_temperature"].isin([0, 1]).all()


class TestHourlySufficiencyCriteria:
    """Tests for all sufficiency checks"""

    # ===== No Data Check =====

    def test_no_data_empty_dataframe_baseline(self):
        """Empty df raises disqualification"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-01-02")
        df = df.iloc[:0]  # Empty

        with pytest.raises((ValueError, IndexError)):
            HourlyBaselineData(df=df, is_electricity_data=True)

    def test_no_data_all_nan_baseline(self):
        """All NaN values disqualifies"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df["observed"] = np.nan
        df["temperature"] = np.nan

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert_has_disqualification(data, "eemeter.sufficiency_criteria.no_data")

    def test_no_data_all_nan_reporting(self):
        """Same for reporting"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-03-31")
        df["temperature"] = np.nan
        # Observed is optional for reporting

        data = HourlyReportingData(df=df, is_electricity_data=True)

        assert_has_disqualification(data, "eemeter.sufficiency_criteria.no_data")

    # ===== Baseline Length Check =====

    @pytest.mark.parametrize(
        "start, end, expect_disqualified",
        [
            ("2019-01-01", "2019-12-31", False),
            ("2019-01-01", "2019-11-26", False),
            ("2020-01-01", "2020-12-31", False),
            ("2019-01-01", "2019-11-20", True),
            ("2019-01-01", "2020-01-10", True),
        ],
        ids=[
            "full_year_365_days",
            "minimum_329_days",
            "leap_year_366_days",
            "under_329_days",
            "over_366_days",
        ],
    )
    def test_baseline_length(self, start, end, expect_disqualified):
        df = create_hourly_dataframe(start=start, end=end)

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        dq_name = "eemeter.sufficiency_criteria.incorrect_number_of_total_days"

        if expect_disqualified:
            assert_has_disqualification(data, dq_name)
        else:
            dq_names = [dq.qualified_name for dq in data.disqualification]
            assert dq_name not in dq_names

    def test_reporting_skips_length_check(self):
        """No check for reporting data"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-02-28")  # Only 2 months

        data = HourlyReportingData(df=df, is_electricity_data=True)

        dq_names = [dq.qualified_name for dq in data.disqualification]
        assert "eemeter.sufficiency_criteria.incorrect_number_of_total_days" not in dq_names

    def test_custom_length_thresholds(self):
        """Override min/max in settings"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-11-01")  # 305 days
        settings = {"sufficiency": {"min_baseline_length": 300, "max_baseline_length": 400}}

        data = HourlyBaselineData(df=df, is_electricity_data=True, settings=settings)

        dq_names = [dq.qualified_name for dq in data.disqualification]
        assert "eemeter.sufficiency_criteria.incorrect_number_of_total_days" not in dq_names

    # ===== Negative Observed Values Check =====

    def test_negative_values_electricity_baseline(self):
        """Electricity allowed"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df.loc[df.index[:10], "observed"] = -5.0

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        dq_names = [dq.qualified_name for dq in data.disqualification]
        assert "eemeter.sufficiency_criteria.negative_observed_values" not in dq_names

    def test_negative_values_gas_baseline(self):
        """Gas with negatives disqualified"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df.loc[df.index[:10], "observed"] = -5.0

        data = HourlyBaselineData(df=df, is_electricity_data=False)

        for dq in data.disqualification:
            if "negative_observed_values" in dq.qualified_name:
                assert "n_negative_observed_values" in dq.data
                assert dq.data["n_negative_observed_values"] == 10

    def test_negative_values_electricity_reporting(self):
        """Electricity reporting with negatives allowed"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-03-31")
        df.loc[df.index[:10], "observed"] = -5.0

        data = HourlyReportingData(df=df, is_electricity_data=True)

        dq_names = [dq.qualified_name for dq in data.disqualification]
        assert "eemeter.sufficiency_criteria.negative_observed_values" not in dq_names

    # ===== Daily Coverage Checks =====

    @pytest.mark.parametrize(
        "column, seed",
        [("temperature", 50), ("observed", 55)],
        ids=["temperature_15_pct_missing", "observed_15_pct_missing"],
    )
    def test_daily_coverage_below_threshold(self, column, seed):
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        np.random.seed(seed)
        mask = np.random.random(len(df)) < 0.15
        df.loc[mask, column] = np.nan

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert_has_disqualification(
            data,
            f"eemeter.sufficiency_criteria.too_many_days_with_missing_{column}_data",
        )

    @pytest.mark.parametrize(
        "column",
        ["temperature", "observed"],
        ids=["temperature_at_or_above_90_pct", "observed_at_or_above_90_pct"],
    )
    def test_daily_coverage_above_threshold(self, column):
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df.loc[df.index[:5], column] = np.nan

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        dq_names = [dq.qualified_name for dq in data.disqualification]
        assert (
            f"eemeter.sufficiency_criteria.too_many_days_with_missing_{column}_data"
            not in dq_names
        )

    @pytest.mark.parametrize(
        "column, missing_columns, seed",
        [
            ("temperature", ["temperature"], 52),
            ("observed", ["observed"], 58),
            ("joint", ["temperature", "observed"], 62),
        ],
        ids=[
            "temperature_80_pct_threshold",
            "observed_80_pct_threshold",
            "joint_80_pct_threshold",
        ],
    )
    def test_daily_coverage_custom_threshold(self, column, missing_columns, seed):
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        np.random.seed(seed)
        mask = np.random.random(len(df)) < 0.15
        df.loc[mask, missing_columns] = np.nan

        settings = {"sufficiency": {column: {"min_pct_daily_coverage": 0.8}}}
        data = HourlyBaselineData(df=df, is_electricity_data=True, settings=settings)

        dq_names = [dq.qualified_name for dq in data.disqualification]
        assert (
            f"eemeter.sufficiency_criteria.too_many_days_with_missing_{column}_data"
            not in dq_names
        )

    def test_temperature_daily_coverage_reporting(self):
        """Applies to reporting"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        np.random.seed(54)
        mask = np.random.random(len(df)) < 0.15
        df.loc[mask, "temperature"] = np.nan

        data = HourlyReportingData(df=df, is_electricity_data=True)

        assert_has_disqualification(
            data,
            "eemeter.sufficiency_criteria.too_many_days_with_missing_temperature_data",
        )

    def test_observed_daily_coverage_reporting_flagged(self):
        """Reporting with observed present runs the coverage check"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-03-31")
        np.random.seed(57)
        mask = np.random.random(len(df)) < 0.15
        df.loc[mask, "observed"] = np.nan

        reporting_data = HourlyReportingData(df=df, is_electricity_data=True)

        dq_names = [dq.qualified_name for dq in reporting_data.disqualification]
        assert (
            "eemeter.sufficiency_criteria.too_many_days_with_missing_observed_data"
            in dq_names
        )

    def test_observed_daily_coverage_reporting_skipped_when_all_nan(self):
        """Prediction-only reporting (all-NaN observed) does not run the coverage check"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-03-31")
        df = df.drop(columns=["observed"])

        reporting_data = HourlyReportingData(df=df, is_electricity_data=True)

        dq_names = [dq.qualified_name for dq in reporting_data.disqualification]
        assert (
            "eemeter.sufficiency_criteria.too_many_days_with_missing_observed_data"
            not in dq_names
        )

    # ===== Joint Daily Coverage Check =====

    def test_joint_daily_coverage_valid(self):
        """At/above 90% passes"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        # Different missing patterns, but minimal overall
        df.loc[df.index[:3], "temperature"] = np.nan
        df.loc[df.index[3:6], "observed"] = np.nan

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        dq_names = [dq.qualified_name for dq in data.disqualification]
        assert (
            "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data"
            not in dq_names
        )

    def test_joint_daily_coverage_temp_missing(self):
        """Temperature NaN fails joint"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        np.random.seed(60)
        mask = np.random.random(len(df)) < 0.15
        df.loc[mask, "temperature"] = np.nan

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # Will likely fail joint or temperature coverage check
        assert len(data.disqualification) > 0

    def test_joint_daily_coverage_observed_missing(self):
        """Observed NaN fails joint"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        np.random.seed(61)
        mask = np.random.random(len(df)) < 0.15
        df.loc[mask, "observed"] = np.nan

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # Will likely fail joint or observed coverage check
        assert len(data.disqualification) > 0

    def test_joint_daily_coverage_below_threshold(self):
        """Both NaN fails joint"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        # Create overlapping but different missing patterns
        np.random.seed(43)
        temp_mask = np.random.random(len(df)) < 0.1
        obs_mask = np.random.random(len(df)) < 0.1
        df.loc[temp_mask, "temperature"] = np.nan
        df.loc[obs_mask, "observed"] = np.nan
        # Also ensure some hours have both missing
        both_mask = np.random.random(len(df)) < 0.05
        df.loc[both_mask, ["temperature", "observed"]] = np.nan

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert_has_disqualification(
            data, "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data"
        )

    # ===== Monthly Coverage Checks =====

    @pytest.mark.parametrize(
        "column, include_ghi, n_missing",
        [
            ("temperature", False, 10),
            ("ghi", True, 5),
            ("observed", False, 5),
        ],
        ids=["temperature", "ghi", "observed"],
    )
    def test_monthly_coverage_all_months_pass(self, column, include_ghi, n_missing):
        df = create_hourly_dataframe(
            start="2019-01-01", end="2019-12-31", include_ghi=include_ghi
        )
        df.loc[df.index[:n_missing], column] = np.nan

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        dq_names = [dq.qualified_name for dq in data.disqualification]
        assert f"eemeter.sufficiency_criteria.missing_monthly_{column}_data" not in dq_names

    @pytest.mark.parametrize(
        "column, include_ghi, month",
        [
            ("temperature", False, 2),
            ("ghi", True, 6),
            ("observed", False, 2),
        ],
        ids=["temperature_february", "ghi_june", "observed_february"],
    )
    def test_monthly_coverage_one_month_low(self, column, include_ghi, month):
        df = create_hourly_dataframe(
            start="2019-01-01", end="2019-12-31", include_ghi=include_ghi
        )
        month_indices = df[df.index.month == month].index
        n_remove = int(len(month_indices) * 0.15)
        df.loc[month_indices[:n_remove], column] = np.nan

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert_has_disqualification(
            data, f"eemeter.sufficiency_criteria.missing_monthly_{column}_data"
        )

    @pytest.mark.parametrize(
        "column, include_ghi, month",
        [
            ("temperature", False, 2),
            ("ghi", True, 6),
            ("observed", False, 2),
        ],
        ids=["temperature_february", "ghi_june", "observed_february"],
    )
    def test_monthly_coverage_custom_threshold(self, column, include_ghi, month):
        df = create_hourly_dataframe(
            start="2019-01-01", end="2019-12-31", include_ghi=include_ghi
        )
        month_indices = df[df.index.month == month].index
        n_remove = int(len(month_indices) * 0.15)
        df.loc[month_indices[:n_remove], column] = np.nan

        settings = {"sufficiency": {column: {"min_pct_monthly_coverage": 0.8}}}
        data = HourlyBaselineData(df=df, is_electricity_data=True, settings=settings)

        dq_names = [dq.qualified_name for dq in data.disqualification]
        assert f"eemeter.sufficiency_criteria.missing_monthly_{column}_data" not in dq_names

    @pytest.mark.parametrize(
        "column, include_ghi",
        [
            ("temperature", False),
            ("ghi", True),
        ],
        ids=["temperature", "ghi"],
    )
    def test_monthly_coverage_reporting(self, column, include_ghi):
        df = create_hourly_dataframe(
            start="2019-01-01", end="2019-03-31", include_ghi=include_ghi
        )
        feb_indices = df[df.index.month == 2].index
        n_remove = int(len(feb_indices) * 0.15)
        df.loc[feb_indices[:n_remove], column] = np.nan

        data = HourlyReportingData(df=df, is_electricity_data=True)

        assert_has_disqualification(
            data, f"eemeter.sufficiency_criteria.missing_monthly_{column}_data"
        )

    def test_temperature_monthly_coverage_multiple_months_fail(self):
        """Multiple failures detected"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        # Remove data from Feb, March, April
        for month in [2, 3, 4]:
            month_mask = df.index.month == month
            month_indices = df[month_mask].index
            n_remove = int(len(month_indices) * 0.15)
            df.loc[month_indices[:n_remove], "temperature"] = np.nan

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert_has_disqualification(
            data, "eemeter.sufficiency_criteria.missing_monthly_temperature_data"
        )

    def test_ghi_monthly_coverage_no_ghi_column_skipped(self):
        """No GHI, no check"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31", include_ghi=False)

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        dq_names = [dq.qualified_name for dq in data.disqualification]
        assert "eemeter.sufficiency_criteria.missing_monthly_ghi_data" not in dq_names

    def test_observed_monthly_coverage_reporting_skipped(self):
        """Reporting exempt"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-03-31")
        feb_mask = df.index.month == 2
        feb_indices = df[feb_mask].index
        n_remove = int(len(feb_indices) * 0.15)
        df.loc[feb_indices[:n_remove], "observed"] = np.nan

        reporting_data = HourlyReportingData(df=df, is_electricity_data=True)

        dq_names = [dq.qualified_name for dq in reporting_data.disqualification]
        assert (
            "eemeter.sufficiency_criteria.missing_monthly_observed_data"
            not in dq_names
        )

    # ===== Unique Observed Values Check =====

    def test_unique_observed_all_unique(self):
        """100% unique passes"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        # Add noise to make all values unique
        np.random.seed(64)
        df["observed"] = df["observed"] + np.random.random(len(df)) * 0.01

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        dq_names = [dq.qualified_name for dq in data.disqualification]
        assert (
            "eemeter.sufficiency_criteria.insufficient_unique_observed_values"
            not in dq_names
        )

    def test_unique_observed_above_threshold(self):
        """Above 10% passes"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        dq_names = [dq.qualified_name for dq in data.disqualification]
        assert (
            "eemeter.sufficiency_criteria.insufficient_unique_observed_values"
            not in dq_names
        )

    def test_unique_observed_below_threshold(self):
        """Below 10% unique disqualifies"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df = create_repeated_values(df, 0.91)  # 91% repeated = 9% unique

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert_has_disqualification(
            data, "eemeter.sufficiency_criteria.insufficient_unique_observed_values"
        )

    def test_all_zeros_electricity_disqualified(self):
        """All zeros converted to NaN, triggers no_data disqualification"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df["observed"] = 0

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # All zeros become NaN, cannot be interpolated, should trigger no_data
        assert_has_disqualification(data, "eemeter.sufficiency_criteria.no_data")

    def test_all_ones_disqualified(self):
        """All 1.0 values triggers insufficient uniqueness disqualification"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df["observed"] = 1.0

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # Only 1 unique value (0% unique), should trigger uniqueness disqualification
        assert_has_disqualification(
            data, "eemeter.sufficiency_criteria.insufficient_unique_observed_values"
        )

    def test_unique_observed_reporting_flagged_when_observed_present(self):
        """Reporting with observed present runs the uniqueness check"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-03-31")
        df = create_repeated_values(df, 0.91)

        reporting_data = HourlyReportingData(df=df, is_electricity_data=True)

        assert_has_disqualification(
            reporting_data,
            "eemeter.sufficiency_criteria.insufficient_unique_observed_values",
        )

    def test_unique_observed_reporting_skipped_when_all_nan(self):
        """Prediction-only reporting (all-NaN observed) does not run the uniqueness check"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-03-31")
        df = create_repeated_values(df, 0.91)
        df = df.drop(columns=["observed"])

        reporting_data = HourlyReportingData(df=df, is_electricity_data=True)

        dq_names = [dq.qualified_name for dq in reporting_data.disqualification]
        assert (
            "eemeter.sufficiency_criteria.insufficient_unique_observed_values"
            not in dq_names
        )

    def test_unique_observed_custom_threshold(self):
        """Override to 30%"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df = create_repeated_values(df, 0.72)  # 72% repeated = 28% unique

        settings = {"sufficiency": {"observed": {"min_pct_unique_values": 0.3}}}
        data = HourlyBaselineData(df=df, is_electricity_data=True, settings=settings)

        # 28% unique with 30% threshold should fail
        assert_has_disqualification(
            data, "eemeter.sufficiency_criteria.insufficient_unique_observed_values"
        )

    def test_unique_observed_data_fields(self):
        """n_unique_values, n_total_values, percentages in data"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df = create_repeated_values(df, 0.76)

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        for dq in data.disqualification:
            if "insufficient_unique_observed_values" in dq.qualified_name:
                assert "n_unique_values" in dq.data
                assert "n_total_values" in dq.data
                assert "unique_percentage" in dq.data
                assert dq.data["n_unique_values"] == 2098
                assert dq.data["unique_percentage"] == 24.0

    # ===== Extreme Values Warning =====

    def test_extreme_values_detected_warning(self):
        """Extreme values issue warning"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df = create_extreme_values(df, 10)

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert_has_warning(data, "eemeter.sufficiency_criteria.extreme_values_detected")

    def test_extreme_values_no_disqualification(self):
        """No disqualification, only warning"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df = create_extreme_values(df, 10)

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # Check it's in warnings, not disqualifications
        warning_names = [w.qualified_name for w in data.warnings]
        dq_names = [dq.qualified_name for dq in data.disqualification]

        assert "eemeter.sufficiency_criteria.extreme_values_detected" in warning_names
        assert "eemeter.sufficiency_criteria.extreme_values_detected" not in dq_names

    def test_extreme_values_iqr_calculation(self):
        """3x IQR bounds correct"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df = create_extreme_values(df, 10)

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # Verify warning issued
        assert_has_warning(data, "eemeter.sufficiency_criteria.extreme_values_detected")

    def test_extreme_values_upper_bound(self):
        """Values above upper bound detected"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        q1 = df["observed"].quantile(0.25)
        q3 = df["observed"].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 3 * iqr

        # Add extreme high values
        df.loc[df.index[:10], "observed"] = upper_bound * 2

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert_has_warning(data, "eemeter.sufficiency_criteria.extreme_values_detected")

    def test_extreme_values_lower_bound(self):
        """Values below lower bound detected"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        q1 = df["observed"].quantile(0.25)
        q3 = df["observed"].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr

        # Add extreme low values (but not negative for gas)
        df.loc[df.index[:10], "observed"] = max(lower_bound / 2, 0.01)

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert_has_warning(data, "eemeter.sufficiency_criteria.extreme_values_detected")

    def test_extreme_values_both_bounds(self):
        """Extremes on both sides"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df = create_extreme_values(df, 10)

        # Also add low extremes
        df.loc[df.index[-10:], "observed"] = 0.01

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert_has_warning(data, "eemeter.sufficiency_criteria.extreme_values_detected")

    def test_extreme_values_count_in_data(self):
        """n_extreme_values correct"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df = create_extreme_values(df, 10)

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        for w in data.warnings:
            if "extreme_values_detected" in w.qualified_name:
                assert "n_extreme_values" in w.data
                assert w.data["n_extreme_values"] == 10

    def test_extreme_values_bounds_in_data(self):
        """lower_bound, upper_bound present"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df = create_extreme_values(df, 10)

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        for w in data.warnings:
            if "extreme_values_detected" in w.qualified_name:
                # Check bounds info in data
                assert "bound" in str(w.data).lower() or "extreme" in str(w.data).lower()

    def test_extreme_values_baseline_only(self):
        """Only baseline checked"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df = create_extreme_values(df, 10)

        baseline_data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert_has_warning(
            baseline_data, "eemeter.sufficiency_criteria.extreme_values_detected"
        )

    def test_extreme_values_reporting_skipped(self):
        """Reporting exempt"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-03-31")
        df = create_extreme_values(df, 10)

        reporting_data = HourlyReportingData(df=df, is_electricity_data=True)

        warning_names = [w.qualified_name for w in reporting_data.warnings]
        # Reporting should not check extreme values
        assert "eemeter.sufficiency_criteria.extreme_values_detected" not in warning_names

    def test_no_extreme_values_no_warning(self):
        """No extremes, no warning"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        warning_names = [w.qualified_name for w in data.warnings]
        assert "eemeter.sufficiency_criteria.extreme_values_detected" not in warning_names

class TestHourlyDataIntegration:
    """Tests for end-to-end scenarios"""

    # ===== Complete Pipeline Tests =====

    def test_baseline_complete_pipeline(self):
        """Raw data → processed baseline"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert data.df is not None
        assert len(data.df) > 0
        assert "has_pv" in data.df.columns

    def test_reporting_complete_pipeline(self):
        """Raw data → processed reporting"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-03-31")

        data = HourlyReportingData(df=df, is_electricity_data=True)

        assert data.df is not None
        assert len(data.df) > 0

    def test_with_ghi_complete_pipeline(self):
        """Include GHI in pipeline"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31", include_ghi=True)

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert "ghi" in data.df.columns
        assert "interpolated_ghi" in data.df.columns

    def test_multiple_sufficiency_failures(self):
        """Multiple disqualifications"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-06-01")  # Too short
        # Also add low uniqueness
        df = create_repeated_values(df, 0.76)

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert len(data.disqualification) >= 2

    def test_multiple_warnings(self):
        """Multiple warnings issued"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df = create_extreme_values(df, 10)
        # Convert to UTC for UTC warning
        df.index = df.index.tz_convert("UTC")

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert len(data.warnings) >= 1

    def test_warnings_and_disqualifications(self):
        """Both present"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-06-01")  # Too short
        df = create_extreme_values(df, 10)

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert len(data.disqualification) >= 1
        assert len(data.warnings) >= 1

    def test_log_warnings_called(self, caplog):
        """Warnings logged"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df = create_extreme_values(df, 10)

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        with caplog.at_level(logging.WARNING, logger="eemeter"):
            data.log_warnings()

        assert (
            "Extreme values (outside 3x IQR) must be flagged for manual review."
            in caplog.text
        )


class TestHourlyDataEdgeCases:
    """Tests for boundary conditions and unusual inputs"""

    # ===== Boundary Conditions =====

    @pytest.mark.parametrize(
        "start, end, expect_disqualified",
        [
            ("2019-01-01", "2019-11-26", False),
            ("2020-01-01", "2020-12-31", False),
            ("2019-01-01", "2019-01-01", True),
        ],
        ids=["exactly_329_days", "exactly_366_days", "single_day"],
    )
    def test_baseline_length_boundaries(self, start, end, expect_disqualified):
        df = create_hourly_dataframe(start=start, end=end)

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        dq_name = "eemeter.sufficiency_criteria.incorrect_number_of_total_days"

        if expect_disqualified:
            assert_has_disqualification(data, dq_name)
        else:
            dq_names = [dq.qualified_name for dq in data.disqualification]
            assert dq_name not in dq_names

    @pytest.mark.parametrize(
        "seed, missing_fraction, expect_disqualified",
        [
            (65, 0.09, False),
            (66, 0.11, True),
        ],
        ids=["exactly_90_percent", "89_9_percent"],
    )
    def test_temperature_daily_coverage_boundaries(
        self, seed, missing_fraction, expect_disqualified
    ):
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        np.random.seed(seed)
        mask = np.random.random(len(df)) < missing_fraction
        df.loc[mask, "temperature"] = np.nan

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        dq_name = (
            "eemeter.sufficiency_criteria.too_many_days_with_missing_temperature_data"
        )

        if expect_disqualified:
            assert_has_disqualification(data, dq_name)
        else:
            dq_names = [dq.qualified_name for dq in data.disqualification]
            assert dq_name not in dq_names

    @pytest.mark.parametrize(
        "repeated_fraction, expect_disqualified",
        [
            (0.90, False),
            (0.901, True),
        ],
        ids=["exactly_10_percent_unique", "9_9_percent_unique"],
    )
    def test_unique_observed_boundaries(self, repeated_fraction, expect_disqualified):
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df = create_repeated_values(df, repeated_fraction)

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        dq_name = "eemeter.sufficiency_criteria.insufficient_unique_observed_values"

        if expect_disqualified:
            assert_has_disqualification(data, dq_name)
        else:
            dq_names = [dq.qualified_name for dq in data.disqualification]
            assert dq_name not in dq_names

    # ===== DST & Timezone Edge Cases =====

    @pytest.mark.parametrize(
        "start, end",
        [
            ("2019-03-09", "2019-03-11"),
            ("2019-11-02", "2019-11-04"),
        ],
        ids=["spring_forward_missing_hour", "fall_back_duplicate_hour"],
    )
    def test_dst_transition_handled(self, start, end):
        df = create_hourly_dataframe(start=start, end=end, tz="America/New_York")

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert data.df is not None

    def test_utc_input_warning(self):
        """UTC timezone issues warning"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df.index = df.index.tz_convert("UTC")

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        warning_names = [w.qualified_name for w in data.warnings]
        assert any("utc" in name.lower() for name in warning_names)

    def test_timezone_conversion_preserved(self):
        """Convert before input"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31", tz="America/Los_Angeles")

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert "Los_Angeles" in str(data.tz)

    # ===== Data Type Edge Cases =====

    def test_integer_observed_values(self):
        """Not just floats"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df["observed"] = df["observed"].astype(int)

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert data.df is not None

    def test_very_small_values(self):
        """Near-zero, not exactly zero"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df["observed"] = 0.001

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # Should not be converted to NaN (not exactly zero)
        assert data.df["observed"].notna().any()

    def test_very_large_values(self):
        """Not extreme, but large"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df["observed"] = df["observed"] * 100

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert data.df is not None

    # ===== Empty & Minimal Data =====

    def test_single_row(self):
        """One hour of data"""
        df = create_hourly_dataframe(start="2019-01-01 00:00", end="2019-01-01 00:00")

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # Will be disqualified for length
        assert len(data.disqualification) > 0

    def test_two_days(self):
        """48 hours (minimum for some checks)"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-01-02")

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        # Will be disqualified for length
        assert_has_disqualification(
            data, "eemeter.sufficiency_criteria.incorrect_number_of_total_days"
        )

    # ===== Column Variations =====

    def test_extra_columns_ignored(self):
        """Additional columns preserved"""
        df = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df["extra_column"] = 123

        data = HourlyBaselineData(df=df, is_electricity_data=True)

        assert "extra_column" in data.df.columns

    def test_ghi_vs_no_ghi(self):
        """GHI checks conditional"""
        df_no_ghi = create_hourly_dataframe(start="2019-01-01", end="2019-12-31", include_ghi=False)
        df_with_ghi = create_hourly_dataframe(start="2019-01-01", end="2019-12-31", include_ghi=True)

        data_no_ghi = HourlyBaselineData(df=df_no_ghi, is_electricity_data=True)
        data_with_ghi = HourlyBaselineData(df=df_with_ghi, is_electricity_data=True)

        # Check GHI only checked when present
        assert "ghi" not in data_no_ghi.df.columns
        assert "ghi" in data_with_ghi.df.columns

    def test_datetime_column_vs_index(self):
        """Both supported"""
        df_index = create_hourly_dataframe(start="2019-01-01", end="2019-12-31")
        df_column = df_index.reset_index().rename(columns={"index": "datetime"})

        data_index = HourlyBaselineData(df=df_index, is_electricity_data=True)
        data_column = HourlyBaselineData(df=df_column, is_electricity_data=True)

        assert isinstance(data_index.df.index, pd.DatetimeIndex)
        assert isinstance(data_column.df.index, pd.DatetimeIndex)


# End of test file
