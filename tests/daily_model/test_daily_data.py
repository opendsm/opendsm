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
from datetime import datetime

from opendsm.eemeter.common.transform import get_baseline_data
from opendsm.eemeter.models.daily.data import DailyBaselineData, DailyReportingData
from opendsm.eemeter.samples import load_sample
import numpy as np
import pandas as pd
from pandas import Timestamp, DatetimeIndex, DataFrame
import pytest

TEMPERATURE_SEED = 29
METER_SEED = 41
NUM_DAYS_IN_YEAR = 365


@pytest.fixture
def get_datetime_index(request):
    # Request = [frequency , is_timezone_aware]

    # Create a DateTimeIndex at 30-minute intervals
    datetime_index = pd.date_range(
        start="2023-01-01",
        end="2024-01-01",
        inclusive="left",
        freq=request.param[0],
        tz="US/Eastern" if request.param[1] else None,
    )

    return datetime_index


@pytest.fixture
def get_datetime_index_half_hourly_with_timezone():
    # Create a DateTimeIndex at 30-minute intervals
    datetime_index = pd.date_range(
        start="2023-01-01",
        end="2024-01-01",
        inclusive="left",
        freq="30min",
        tz="US/Eastern",
    )

    return datetime_index


@pytest.fixture
def get_datetime_index_hourly_with_timezone():
    # Create a DateTimeIndex at 30-minute intervals
    datetime_index = pd.date_range(
        start="2023-01-01",
        end="2024-01-01",
        inclusive="left",
        freq="h",
        tz="US/Eastern",
    )

    return datetime_index


@pytest.fixture
def get_datetime_index_daily_with_timezone():
    # Create a DateTimeIndex at daily intervals
    datetime_index = pd.date_range(
        start="2023-01-01",
        end="2024-01-01",
        inclusive="left",
        freq="D",
        tz="US/Eastern",
    )

    return datetime_index


@pytest.fixture
def get_datetime_index_daily_without_timezone():
    # Create a DateTimeIndex at daily intervals
    datetime_index = pd.date_range(
        start="2023-01-01", end="2024-01-01", inclusive="left", freq="D"
    )

    return datetime_index


@pytest.fixture
def get_temperature_data_half_hourly(get_datetime_index_half_hourly_with_timezone):
    datetime_index = get_datetime_index_half_hourly_with_timezone

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' column with random data
    temperature_mean = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={"temperature": temperature_mean}, index=datetime_index)

    return df


@pytest.fixture
def get_temperature_data_hourly(get_datetime_index_hourly_with_timezone):
    datetime_index = get_datetime_index_hourly_with_timezone

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' column with random data
    temperature_mean = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={"temperature": temperature_mean}, index=datetime_index)

    return df


@pytest.fixture
def get_meter_data_daily(get_datetime_index_daily_with_timezone):
    datetime_index = get_datetime_index_daily_with_timezone

    np.random.seed(METER_SEED)
    # Create a 'meter_value' column with random data
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={"observed": meter_value}, index=datetime_index)

    return df


@pytest.fixture
def get_meter_data_daily_with_extreme_values_and_negative_values(
    get_datetime_index_daily_with_timezone,
):
    datetime_index = get_datetime_index_daily_with_timezone

    np.random.seed(METER_SEED)
    # Create a 'meter_value' column with random data
    # Last 60 will be for extreme values
    meter_value = np.random.normal(loc=0.0, scale=100.0, size=len(datetime_index) - 60)
    median = np.median(meter_value)
    q75, q25 = np.percentile(meter_value, [75, 25])
    iqr = q75 - q25

    # Generate some extreme values more than thrice the interquartile range from the median
    extreme_values_right = (
        median + (3 * iqr) + np.random.normal(loc=0.0, scale=100.0, size=30)
    )
    extreme_values_left = median - (
        (3 * iqr) + np.random.normal(loc=0.0, scale=100.0, size=30)
    )

    meter_value = np.concatenate(
        (extreme_values_right, meter_value, extreme_values_left)
    )

    # Create the DataFrame
    df = pd.DataFrame(data={"observed": meter_value}, index=datetime_index)

    return df


@pytest.fixture
def get_temperature_data_daily(get_datetime_index_daily_with_timezone):
    datetime_index = get_datetime_index_daily_with_timezone

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' column with random data
    temperature_mean = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={"temperature": temperature_mean}, index=datetime_index)

    return df


# Check that a missing timezone raises a Value Error
@pytest.mark.parametrize("get_datetime_index", [["D", False]], indirect=True)
def test_daily_baseline_data_with_missing_timezone(get_datetime_index):
    datetime_index = get_datetime_index

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))

    np.random.seed(METER_SEED)
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(
        data={"meter": meter_value, "temperature": temperature_mean},
        index=datetime_index,
    )

    with pytest.raises(ValueError):
        cls = DailyBaselineData(df, is_electricity_data=True)


# Check that a missing datetime index and column raises a Value Error
def test_daily_baseline_data_with_missing_datetime_index_and_column():
    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(NUM_DAYS_IN_YEAR)

    np.random.seed(METER_SEED)
    meter_value = np.random.rand(NUM_DAYS_IN_YEAR)

    # Create the DataFrame
    df = pd.DataFrame(data={"meter": meter_value, "temperature": temperature_mean})

    with pytest.raises(ValueError):
        cls = DailyBaselineData(df, is_electricity_data=True)


@pytest.mark.parametrize("get_datetime_index", [["D", True]], indirect=True)
def test_daily_baseline_data_with_datetime_column(get_datetime_index):
    df = pd.DataFrame()
    df["datetime"] = get_datetime_index
    np.random.seed(TEMPERATURE_SEED)
    df["temperature"] = np.random.rand(len(get_datetime_index))
    np.random.seed(METER_SEED)
    df["observed"] = np.random.rand(len(get_datetime_index))

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert round(cls.df.observed.sum(), 2) == round(df.observed.sum(), 2)
    assert len(cls.warnings) == 1
    assert (
        cls.warnings[0].qualified_name
        == "eemeter.sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency"
    )
    assert len(cls.disqualification) == 0


@pytest.mark.parametrize("get_datetime_index", [["D", True]], indirect=True)
def test_daily_baseline_data_with_same_daily_frequencies(get_datetime_index):
    datetime_index = get_datetime_index

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))

    np.random.seed(METER_SEED)
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(
        data={"observed": meter_value, "temperature": temperature_mean},
        index=datetime_index,
    )

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert round(cls.df.observed.sum(), 2) == round(df.observed.sum(), 2)
    assert len(cls.warnings) == 1
    assert (
        cls.warnings[0].qualified_name
        == "eemeter.sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency"
    )
    assert len(cls.disqualification) == 0


@pytest.mark.parametrize(
    "get_datetime_index", [["30min", True], ["h", True]], indirect=True
)
def test_daily_baseline_data_with_same_hourly_frequencies(get_datetime_index):
    datetime_index = get_datetime_index

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))

    np.random.seed(METER_SEED)
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(
        data={"observed": meter_value, "temperature": temperature_mean},
        index=datetime_index,
    )

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    # TODO: Because of the weird behaviour of as_freq() on the last hour for downsampling, so we can't add it
    assert round(cls.df.observed.sum(), 2) == round(df.observed[:-1].sum(), 2)
    assert len(cls.warnings) == 0
    assert len(cls.disqualification) == 0


def test_daily_baseline_data_with_daily_and_half_hourly_frequencies(
    get_temperature_data_half_hourly, get_meter_data_daily
):
    # Create a DataFrame with uneven frequency
    df = get_temperature_data_half_hourly

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_daily

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how="outer")

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert round(cls.df.observed.sum(), 2) == round(df.observed.sum(), 2)
    assert len(cls.warnings) == 0
    assert len(cls.disqualification) == 0


def test_daily_baseline_data_with_daily_and_hourly_frequencies(
    get_meter_data_daily, get_temperature_data_hourly
):
    df = get_temperature_data_hourly

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_daily

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how="outer")

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert round(cls.df.observed.sum(), 2) == round(df.observed.sum(), 2)
    assert len(cls.warnings) == 0
    assert len(cls.disqualification) == 0


def test_daily_baseline_data_with_extreme_values_in_daily_and_hourly_frequencies(
    get_meter_data_daily_with_extreme_values_and_negative_values,
    get_temperature_data_hourly,
):
    df = get_temperature_data_hourly

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_daily_with_extreme_values_and_negative_values

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how="outer")

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert round(cls.df.observed.sum(), 2) == round(df.observed.sum(), 2)
    assert len(cls.warnings) == 1
    assert (
        cls.warnings[0].qualified_name
        == "eemeter.sufficiency_criteria.extreme_values_detected"
    )
    assert len(cls.disqualification) == 0


def test_daily_baseline_data_with_extreme_and_negative_values_in_daily_and_hourly_frequencies(
    get_meter_data_daily_with_extreme_values_and_negative_values,
    get_temperature_data_hourly,
):
    df = get_temperature_data_hourly

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_daily_with_extreme_values_and_negative_values

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how="outer")

    cls = DailyBaselineData(df, is_electricity_data=False)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert round(cls.df.observed.sum(), 2) == round(df.observed.sum(), 2)
    assert len(cls.warnings) == 1
    assert (
        cls.warnings[0].qualified_name
        == "eemeter.sufficiency_criteria.extreme_values_detected"
    )
    assert len(cls.disqualification) == 1
    assert (
        cls.disqualification[0].qualified_name
        == "eemeter.sufficiency_criteria.negative_observed_values"
    )


def test_daily_baseline_data_with_specific_hourly_input():
    meter, temperature, _ = load_sample("il-electricity-cdd-hdd-hourly")
    meter = meter[meter.index.year == 2017]
    temperature = temperature[temperature.index.year == 2017]
    cls = DailyBaselineData.from_series(meter, temperature, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    # TODO: Because of the weird behaviour of as_freq() on the last hour for downsampling, so we can't add it
    assert round(cls.df.observed.sum(), 2) == round(meter.value[:-1].sum(), 2)
    assert len(cls.warnings) == 2
    assert [warning.qualified_name for warning in cls.warnings] == [
        "eemeter.data_quality.utc_index",
        "eemeter.sufficiency_criteria.extreme_values_detected",
    ]
    assert len(cls.disqualification) == 0


def test_daily_baseline_data_with_specific_daily_input():
    meter, temperature, _ = load_sample("il-electricity-cdd-hdd-daily")
    meter = meter[meter.index.year == 2017]
    temperature = temperature[temperature.index.year == 2017]
    cls = DailyBaselineData.from_series(meter, temperature, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert round(cls.df.observed.sum(), 2) == round(meter.value.sum(), 2)
    assert len(cls.warnings) == 2
    assert [warning.qualified_name for warning in cls.warnings] == [
        "eemeter.data_quality.utc_index",
        "eemeter.sufficiency_criteria.extreme_values_detected",
    ]
    assert len(cls.disqualification) == 0


def test_daily_baseline_data_with_missing_specific_daily_input():
    meter, temperature, _ = load_sample("il-electricity-cdd-hdd-daily")
    meter = meter[meter.index.year == 2017]
    # Set 1 month meter data to NaN
    meter.loc[meter.index.month == 4] = np.nan
    temperature = temperature[temperature.index.year == 2017]
    cls = DailyBaselineData.from_series(meter, temperature, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert round(cls.df.observed.sum(), 2) == round(meter.value.sum(), 2)
    assert len(cls.warnings) == 2
    assert [warning.qualified_name for warning in cls.warnings] == [
        "eemeter.data_quality.utc_index",
        "eemeter.sufficiency_criteria.extreme_values_detected",
    ]
    assert len(cls.disqualification) == 0


def test_daily_baseline_data_with_missing_hourly_temperature_data(
    get_meter_data_daily, get_temperature_data_hourly
):
    df = get_temperature_data_hourly

    # Create a mask for Tuesdays and Thursdays
    mask = df.index.dayofweek.isin([1, 3])

    # Set 60% of the temperature data as missing on Tuesdays and Thursdays
    df.loc[df[mask].sample(frac=0.6).index, "temperature"] = np.nan

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_daily

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how="outer")

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert round(cls.df.observed.sum(), 2) == round(df.observed.sum(), 2)
    assert len(cls.warnings) == 1
    assert (
        cls.warnings[0].qualified_name
        == "eemeter.sufficiency_criteria.missing_high_frequency_temperature_data"
    )
    assert len(cls.disqualification) == 3
    expected_disqualifications = [
        "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
        "eemeter.sufficiency_criteria.too_many_days_with_missing_temperature_data",
        "eemeter.sufficiency_criteria.missing_monthly_temperature_data",
    ]
    assert all(
        disqualification.qualified_name in expected_disqualifications
        for disqualification in cls.disqualification
    )


def test_daily_baseline_data_with_missing_half_hourly_temperature_data(
    get_meter_data_daily, get_temperature_data_half_hourly
):
    df = get_temperature_data_half_hourly

    # Create a mask for Tuesdays and Thursdays
    mask = df.index.dayofweek.isin([1, 3])

    # Set 60% of the temperature data as missing on Tuesdays and Thursdays
    # This should cause the high frequency temperature check to fail on these days
    df.loc[df[mask].sample(frac=0.6).index, "temperature"] = np.nan

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_daily

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how="outer")

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert round(cls.df.observed.sum(), 2) == round(df.observed.sum(), 2)
    assert len(cls.warnings) == 1
    assert (
        cls.warnings[0].qualified_name
        == "eemeter.sufficiency_criteria.missing_high_frequency_temperature_data"
    )
    assert len(cls.disqualification) == 3
    expected_disqualifications = [
        "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
        "eemeter.sufficiency_criteria.too_many_days_with_missing_temperature_data",
        "eemeter.sufficiency_criteria.missing_monthly_temperature_data",
    ]
    assert all(
        disqualification.qualified_name in expected_disqualifications
        for disqualification in cls.disqualification
    )


def test_daily_baseline_data_with_missing_daily_temperature_data(
    get_meter_data_daily, get_temperature_data_daily
):
    df = get_temperature_data_daily

    # Set 60% of the temperature data as missing on Tuesdays and Thursdays
    # This should cause the high frequency temperature check to fail on these days
    df.loc[df.index.dayofweek.isin([1, 3]), "temperature"] = np.nan

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_daily

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how="outer")

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert round(cls.df.observed.sum(), 2) == round(df.observed.sum(), 2)
    assert len(cls.warnings) == 1
    assert (
        cls.warnings[0].qualified_name
        == "eemeter.sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency"
    )
    assert len(cls.disqualification) == 3
    expected_disqualifications = [
        "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
        "eemeter.sufficiency_criteria.too_many_days_with_missing_temperature_data",
        "eemeter.sufficiency_criteria.missing_monthly_temperature_data",
    ]
    assert all(
        disqualification.qualified_name in expected_disqualifications
        for disqualification in cls.disqualification
    )


def test_daily_baseline_data_with_missing_meter_data(
    get_meter_data_daily, get_temperature_data_hourly
):
    df = get_temperature_data_hourly

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_daily

    # Set Tuesdays & Thursdays data as missing
    df_meter.loc[df_meter.index.dayofweek.isin([1, 3]), "observed"] = np.nan

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how="outer")

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert round(cls.df.observed.sum(), 2) == round(df.observed.sum(), 2)
    assert len(cls.warnings) == 0
    # assert all(warning.qualified_name in expected_warnings for warning in cls.warnings)
    assert len(cls.disqualification) == 2
    expected_disqualifications = [
        "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
        "eemeter.sufficiency_criteria.too_many_days_with_missing_observed_data",
    ]
    assert all(
        disqualification.qualified_name in expected_disqualifications
        for disqualification in cls.disqualification
    )


def test_daily_baseline_data_with_missing_meter_data_37_days(
    get_meter_data_daily, get_temperature_data_hourly
):
    df = get_temperature_data_hourly

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_daily

    # Set Tuesdays & Thursdays data as missing
    df_meter.loc[df_meter.index[1:38], "observed"] = np.nan

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how="outer")

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert round(cls.df.observed.sum(), 2) == round(df.observed.sum(), 2)
    assert len(cls.warnings) == 0
    # assert all(warning.qualified_name in expected_warnings for warning in cls.warnings)
    assert len(cls.disqualification) == 2
    expected_disqualifications = [
        "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
        "eemeter.sufficiency_criteria.too_many_days_with_missing_observed_data",
    ]
    assert all(
        disqualification.qualified_name in expected_disqualifications
        for disqualification in cls.disqualification
    )


def test_duplicate_datetime_index_values():
    # Create a Timestamp with a specific date
    timestamp = pd.Timestamp("2023-01-01")

    # Create an Index with 365 identical timestamps
    datetime_index = pd.DatetimeIndex([timestamp] * 365, tz="US/Eastern")

    # Create random values for 'observed' and 'temperature'
    observed = np.random.rand(len(datetime_index))
    temperature = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(
        data={"observed": observed, "temperature": temperature}, index=datetime_index
    )

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == 1


@pytest.mark.parametrize(
    "get_datetime_index", [["30min", True], ["h", True]], indirect=True
)
def test_daily_reporting_data_with_half_hourly_and_hourly_frequencies(
    get_datetime_index,
):
    datetime_index = get_datetime_index

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={"temperature": temperature_mean}, index=datetime_index)

    cls = DailyReportingData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert len(cls.warnings) == 0
    assert len(cls.disqualification) == 0


@pytest.mark.parametrize(
    "get_datetime_index", [["30min", True], ["h", True]], indirect=True
)
def test_daily_reporting_data_with_missing_half_hourly_and_hourly_frequencies(
    get_datetime_index,
):
    datetime_index = get_datetime_index

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={"temperature": temperature_mean}, index=datetime_index)

    # Create a mask for Tuesdays and Thursdays
    mask = df.index.dayofweek.isin([1, 3])

    # Set 60% of the temperature data as missing on Tuesdays and Thursdays
    # This should cause the high frequency temperature check to fail on these days
    df.loc[df[mask].sample(frac=0.6, random_state=42).index, "temperature"] = np.nan

    cls = DailyReportingData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR

    if datetime_index.freq == "30min":
        assert len(cls.df.temperature.dropna()) == 268
    elif datetime_index.freq == "h":
        assert len(cls.df.temperature.dropna()) == 270

    assert len(cls.warnings) == 1
    assert (
        cls.warnings[0].qualified_name
        == "eemeter.sufficiency_criteria.missing_high_frequency_temperature_data"
    )
    expected_disqualifications = [
        "eemeter.sufficiency_criteria.missing_monthly_temperature_data",
        "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
        "eemeter.sufficiency_criteria.too_many_days_with_missing_temperature_data",
    ]
    assert all(
        disqualification.qualified_name in expected_disqualifications
        for disqualification in cls.disqualification
    )


def test_daily_reporting_data_high_frequency_temperature_warning_gives_proper_results():

    datetime_index = pd.date_range(
        "2023-01-01", "2023-01-08", freq="h", tz="US/Eastern"
    )
    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={"temperature": temperature_mean}, index=datetime_index)

    # Nan all of 2023-01-01
    df.loc["2023-01-01 06:00":"2023-01-01 18:00", "temperature"] = np.nan

    cls = DailyReportingData(df, is_electricity_data=True)

    assert len(cls.warnings) == 1
    assert (
        cls.warnings[0].qualified_name
        == "eemeter.sufficiency_criteria.missing_high_frequency_temperature_data"
    )
    # Should return just the day that has too many nulls.
    assert len(cls.warnings[0].data) == 1


@pytest.mark.parametrize("get_datetime_index", [["D", True]], indirect=True)
def test_daily_reporting_data_with_missing_daily_frequencies(get_datetime_index):
    datetime_index = get_datetime_index

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={"temperature": temperature_mean}, index=datetime_index)

    # Create a mask for Tuesdays and Thursdays
    mask = df.index.dayofweek.isin([1, 3])

    # Set 60% of the temperature data as missing on Tuesdays and Thursdays
    # This should cause the high frequency temperature check to fail on these days
    df.loc[df[mask].sample(frac=0.6, random_state=42).index, "temperature"] = np.nan

    cls = DailyReportingData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert len(cls.df.temperature.dropna()) == len(df.temperature.dropna())
    assert len(cls.warnings) == 1
    assert (
        cls.warnings[0].qualified_name
        == "eemeter.sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency"
    )
    expected_disqualifications = [
        "eemeter.sufficiency_criteria.missing_monthly_temperature_data",
        "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
        "eemeter.sufficiency_criteria.too_many_days_with_missing_temperature_data",
    ]
    assert all(
        disqualification.qualified_name in expected_disqualifications
        for disqualification in cls.disqualification
    )


@pytest.fixture
def baseline_data_daily_params(il_electricity_cdd_hdd_daily):
    def _baseline(tz="UTC", hour=0):
        meter_data = il_electricity_cdd_hdd_daily["meter_data"]
        temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
        blackout_start_date = il_electricity_cdd_hdd_daily["blackout_start_date"]
        baseline_meter_data, warnings = get_baseline_data(
            meter_data, end=blackout_start_date
        )
        baseline_meter_data.index = map(
            lambda x: x.replace(hour=x.hour + hour), baseline_meter_data.index
        )
        baseline_meter_data.index = baseline_meter_data.index.tz_convert(tz)
        return baseline_meter_data, temperature_data

    yield _baseline


@pytest.mark.parametrize(
    ["tz", "hour"], [["US/Pacific", 3], ["US/Eastern", 8], ["Europe/London", 13]]
)
def test_offset_temperature_aggregations(baseline_data_daily_params, tz, hour):
    baseline_meter_data, temp_series = baseline_data_daily_params(tz=tz, hour=hour)
    baseline = DailyBaselineData.from_series(
        baseline_meter_data, temp_series, is_electricity_data=True
    )
    tz = baseline_meter_data.index.tz

    abs_diff = 0
    for day in baseline.df.index:
        abs_diff += abs(
            temp_series[day : day + pd.Timedelta(hours=23)].mean()
            - baseline.df.temperature.loc[day].squeeze()
        )
    assert abs_diff < 0.000001


def test_non_ns_datetime_index():
    meter, temperature, _ = load_sample("il-electricity-cdd-hdd-hourly")
    meter = meter[meter.index.year == 2017]
    temperature = temperature[temperature.index.year == 2017]

    # convert to microseconds
    meter.index = meter.index.astype("datetime64[us, UTC]")
    temperature.index = temperature.index.astype("datetime64[us, UTC]")
    cls = DailyBaselineData.from_series(meter, temperature, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR


def test_offset_aggregations_hourly(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"]
    blackout_start_date = il_electricity_cdd_hdd_hourly["blackout_start_date"]
    baseline_meter_data, warnings = get_baseline_data(
        meter_data, end=blackout_start_date
    )
    baseline_meter_data = baseline_meter_data.iloc[3:]  # begin from 3AM UTC
    baseline = DailyBaselineData.from_series(
        baseline_meter_data, temperature_data, is_electricity_data=True
    )
    assert baseline is not None
    assert len(baseline.df) == NUM_DAYS_IN_YEAR


def test_dst_handling():
    # 2020-03-08 02:00 is nonexistent, should push to 03:00
    tz = "America/New_York"
    idx = DatetimeIndex(
        [Timestamp("2020-03-07 02", tz=tz), Timestamp("2021-03-06 02", tz=tz)]
    )
    df = DataFrame({"observed": [1] * 2, "temperature": [50] * 2}, index=idx)
    baseline = DailyBaselineData(df, is_electricity_data=True)
    assert len(baseline.df) == 365
    hours, counts = np.unique(baseline.df.index.hour, return_counts=True)
    assert (hours == [2, 3]).all()
    assert (counts == [364, 1]).all()

    # 2020-11-01 01:00 is ambiguous, single index should be chosen
    tz = "America/New_York"
    idx = DatetimeIndex(
        [Timestamp("2020-03-07 01", tz=tz), Timestamp("2021-03-06 01", tz=tz)]
    )
    df = DataFrame({"observed": [1] * 2, "temperature": [50] * 2}, index=idx)
    baseline = DailyBaselineData(df, is_electricity_data=True)
    assert len(baseline.df) == 365
    assert (baseline.df.index.hour == 1).all()
