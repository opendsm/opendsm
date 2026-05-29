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

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from opendsm.common.test_data import load_test_data

_TEST_METER = 110596


@pytest.fixture
def hourly_data():
    baseline, reporting = load_test_data("hourly_treatment_data")
    return baseline.loc[_TEST_METER], reporting.loc[_TEST_METER]


@pytest.fixture
def baseline(hourly_data):
    baseline, _ = hourly_data
    baseline.loc[baseline["observed"] > 513, "observed"] = (
        0  # quick extreme value removal
    )
    return baseline


@pytest.fixture
def reporting(hourly_data):
    _, reporting = hourly_data
    return reporting


# ===== Synthetic Data Generators =====


def create_hourly_dataframe(
    start="2019-01-01",
    end="2019-12-31",
    tz="US/Eastern",
    include_ghi=False,
    observed_mean=5.0,
    temperature_mean=60.0,
    add_noise=True,
):
    """Create synthetic hourly data for testing"""
    start_dt = pd.Timestamp(start, tz=tz)
    end_dt = pd.Timestamp(end, tz=tz)
    index = pd.date_range(start_dt, end_dt, freq="h")

    n = len(index)

    # Temperature: seasonal + daily cycle + noise
    day_of_year = index.dayofyear
    hour_of_day = index.hour
    seasonal = 20 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    daily = 10 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    noise = np.random.normal(0, 3, n) if add_noise else 0
    temperature = temperature_mean + seasonal + daily + noise

    # Observed: correlated with temperature + noise
    base_load = observed_mean
    temp_correlation = 0.02 * (temperature - temperature_mean)
    obs_noise = np.random.normal(0, 0.5, n) if add_noise else 0
    observed = np.maximum(0, base_load + temp_correlation + obs_noise)

    df = pd.DataFrame(
        {"observed": observed, "temperature": temperature}, index=index
    )

    if include_ghi:
        # GHI: daytime only, seasonal variation
        is_daytime = (hour_of_day >= 6) & (hour_of_day <= 18)
        sun_angle = np.sin(np.pi * (hour_of_day - 6) / 12)
        seasonal_ghi = 300 + 200 * np.sin(2 * np.pi * (day_of_year - 172) / 365)
        ghi = np.where(is_daytime, seasonal_ghi * sun_angle, 0)
        df["ghi"] = ghi

    return df


def create_gaps(df, gap_pattern):
    """Add strategic gaps to dataframe

    Args:
        df: DataFrame with hourly data
        gap_pattern: str - 'single_hour', 'multi_hour', 'full_day', 'month'
    """
    df = df.copy()

    if gap_pattern == "single_hour":
        # Random single hour gaps
        mask = np.random.random(len(df)) > 0.95
        df.loc[mask, ["observed", "temperature"]] = np.nan

    elif gap_pattern == "multi_hour":
        # 6-hour consecutive gaps
        start_idx = len(df) // 4
        df.iloc[start_idx : start_idx + 6, df.columns.get_indexer(["observed", "temperature"])] = np.nan

    elif gap_pattern == "full_day":
        # Full 24-hour gap
        start_idx = len(df) // 3
        df.iloc[start_idx : start_idx + 24, df.columns.get_indexer(["observed", "temperature"])] = np.nan

    elif gap_pattern == "month":
        # February missing
        feb_mask = df.index.month == 2
        df.loc[feb_mask, ["observed", "temperature"]] = np.nan

    return df


def create_repeated_values(df, repeat_pct):
    """Reduce uniqueness by repeating values

    Args:
        df: DataFrame with hourly data
        repeat_pct: float - percentage of values to repeat (e.g., 0.75 for 75%)
    """
    df = df.copy()
    n_repeat = int(len(df) * repeat_pct)
    repeated_value = df["observed"].iloc[0]
    df.loc[df.index[:n_repeat], "observed"] = repeated_value

    return df


def create_extreme_values(df, n_extreme, column="observed"):
    """Add outliers beyond 3x IQR

    Args:
        df: DataFrame with hourly data
        n_extreme: int - number of extreme values to add
        column: str - column to add extremes to
    """
    df = df.copy()
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr

    # Add values beyond upper bound
    extreme_indices = np.random.choice(len(df), n_extreme, replace=False)
    df.loc[df.index[extreme_indices], column] = upper_bound * 2

    return df


@pytest.fixture
def synthetic_hourly_data():
    """Clean 365 days of hourly data, US/Eastern timezone"""
    return create_hourly_dataframe()


@pytest.fixture
def synthetic_hourly_with_gaps():
    """Data with single-hour gaps for testing interpolation"""
    df = create_hourly_dataframe()
    return create_gaps(df, "single_hour")


@pytest.fixture
def synthetic_hourly_with_ghi():
    """Data including GHI column"""
    return create_hourly_dataframe(include_ghi=True)


@pytest.fixture
def synthetic_hourly_low_coverage():
    """Data with month missing (below 90% coverage)"""
    df = create_hourly_dataframe()
    return create_gaps(df, "month")


@pytest.fixture
def synthetic_hourly_low_uniqueness():
    """Data with <25% unique values (75% repeated)"""
    df = create_hourly_dataframe()
    return create_repeated_values(df, 0.76)


@pytest.fixture
def synthetic_hourly_with_extremes():
    """Data with outliers beyond 3x IQR"""
    df = create_hourly_dataframe()
    return create_extreme_values(df, 10)


@pytest.fixture
def dst_transition_data():
    """Data spanning DST transitions (spring forward, fall back)"""
    # March 10, 2019: spring forward (2am doesn't exist)
    # November 3, 2019: fall back (1am repeats)
    return create_hourly_dataframe(start="2019-03-01", end="2019-11-30")


# ===== Helper Assertion Functions =====


def assert_disqualification(data, expected_qualified_names):
    """Assert data has exactly these disqualifications"""
    if isinstance(expected_qualified_names, str):
        expected_qualified_names = [expected_qualified_names]

    actual = {dq.qualified_name for dq in data.disqualification}
    expected = set(expected_qualified_names)

    assert actual == expected, f"Expected {expected}, got {actual}"


def assert_warning(data, expected_qualified_names):
    """Assert data has exactly these warnings"""
    if isinstance(expected_qualified_names, str):
        expected_qualified_names = [expected_qualified_names]

    actual = {w.qualified_name for w in data.warnings}
    expected = set(expected_qualified_names)

    assert actual == expected, f"Expected {expected}, got {actual}"


def assert_has_disqualification(data, qualified_name):
    """Assert data has at least this disqualification"""
    names = {dq.qualified_name for dq in data.disqualification}
    assert qualified_name in names, f"{qualified_name} not in {names}"


def assert_has_warning(data, qualified_name):
    """Assert data has at least this warning"""
    names = {w.qualified_name for w in data.warnings}
    assert qualified_name in names, f"{qualified_name} not in {names}"


def assert_no_disqualifications(data):
    """Assert data passes all checks"""
    actual = [dq.qualified_name for dq in data.disqualification]
    assert len(actual) == 0, f"Expected no disqualifications, got {actual}"


def assert_column_exists(df, column):
    """Assert column present in dataframe"""
    assert column in df.columns, f"Column {column} not found in {list(df.columns)}"


def assert_all_hours_present(df):
    """Assert dataframe has contiguous hourly index"""
    expected_hours = (df.index.max() - df.index.min()).total_seconds() / 3600 + 1
    actual_hours = len(df)
    assert actual_hours == expected_hours, f"Expected {expected_hours} hours, got {actual_hours}"


# ===== Fixtures for Comprehensive HourlyModel Testing =====


@pytest.fixture
def baseline_with_supplemental_features(baseline):
    """Baseline data with custom supplemental features for testing"""
    baseline = baseline.copy()
    np.random.seed(42)
    baseline['custom_ts_feature'] = np.random.randn(len(baseline))
    baseline['custom_cat_feature'] = np.random.choice(['A', 'B', 'C'], len(baseline))

    return baseline


@pytest.fixture
def baseline_extreme_temperatures(baseline):
    """Baseline data with extreme temperature values for edge bin testing"""
    baseline_wide = baseline.copy()
    baseline_wide.loc[baseline_wide.index[:100], 'temperature'] = -20
    baseline_wide.loc[baseline_wide.index[-100:], 'temperature'] = 110

    return baseline_wide


@pytest.fixture
def dst_transition_baseline():
    """Baseline data spanning DST transitions for testing DST handling"""
    return create_hourly_dataframe(start="2019-01-01", end="2019-12-31", tz="US/Eastern")


@pytest.fixture
def dst_transition_reporting():
    """Reporting data spanning DST transitions for testing DST handling"""
    return create_hourly_dataframe(start="2020-01-01", end="2020-12-31", tz="US/Eastern")