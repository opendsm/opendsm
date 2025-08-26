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
from opendsm.eemeter.models.billing.data import (
    BillingBaselineData,
    BillingReportingData,
)
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

    # Create a DateTimeIndex at given frequency and timezone if requested
    inclusive = "both" if request.param[0] in ["MS", "2MS"] else "left"
    datetime_index = pd.date_range(
        start="2023-01-01",
        end="2024-01-01",
        inclusive=inclusive,
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
def get_datetime_index_monthly_with_timezone():
    # Create a DateTimeIndex at daily intervals
    datetime_index = pd.date_range(
        start="2023-01-01",
        end="2024-01-01",
        inclusive="both",
        freq="MS",
        tz="US/Eastern",
    )

    return datetime_index


@pytest.fixture
def get_datetime_index_bimonthly_with_timezone():
    # Create a DateTimeIndex at daily intervals
    datetime_index = pd.date_range(
        start="2023-01-01",
        end="2024-01-01",
        inclusive="both",
        freq="2MS",
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
def get_temperature_data_daily(get_datetime_index_daily_with_timezone):
    datetime_index = get_datetime_index_daily_with_timezone

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
def get_meter_data_monthly(get_datetime_index_monthly_with_timezone):
    datetime_index = get_datetime_index_monthly_with_timezone

    np.random.seed(METER_SEED)
    # Create a 'meter_value' column with random data
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={"observed": meter_value}, index=datetime_index)
    df.iloc[-1, df.columns.get_loc("observed")] = np.nan

    return df


@pytest.fixture
def get_meter_data_bimonthly(get_datetime_index_bimonthly_with_timezone):
    datetime_index = get_datetime_index_bimonthly_with_timezone

    np.random.seed(METER_SEED)
    # Create a 'meter_value' column with random data
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={"observed": meter_value}, index=datetime_index)
    df.iloc[-1, df.columns.get_loc("observed")] = np.nan

    return df


# Check that a missing timezone raises a Value Error
@pytest.mark.parametrize("get_datetime_index", [["D", False]], indirect=True)
def test_billing_baseline_data_with_missing_timezone(get_datetime_index):
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
        cls = BillingBaselineData(df, is_electricity_data=True)


# Check that a missing datetime index and column raises a Value Error
def test_billing_baseline_data_with_missing_datetime_index_and_column():
    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(NUM_DAYS_IN_YEAR)

    np.random.seed(METER_SEED)
    meter_value = np.random.rand(NUM_DAYS_IN_YEAR)

    # Create the DataFrame
    df = pd.DataFrame(data={"meter": meter_value, "temperature": temperature_mean})

    with pytest.raises(ValueError):
        cls = BillingBaselineData(df, is_electricity_data=True)


@pytest.mark.parametrize("get_datetime_index", [["MS", True]], indirect=True)
def test_billing_baseline_data_with_monthly_frequencies(get_datetime_index):
    datetime_index = get_datetime_index

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))

    np.random.seed(METER_SEED)
    meter_value = np.random.rand(len(datetime_index))
    meter_value[-1] = np.nan

    # Create the DataFrame
    df = pd.DataFrame(
        data={"observed": meter_value, "temperature": temperature_mean},
        index=datetime_index,
    )
    df.index = df.index[:-1].union([df.index[-1] - pd.Timedelta(days=1)])

    cls = BillingBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert round(cls.df.observed.sum(), 2) == round(df.observed.sum(), 2)
    assert len(cls.warnings) == 1
    assert (
        cls.warnings[0].qualified_name
        == "eemeter.sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency"
    )
    # DQ because only 12 days worth of temperature data is available
    assert len(cls.disqualification) == 2
    assert set([dq.qualified_name for dq in cls.disqualification]) == set([
        "eemeter.sufficiency_criteria.too_many_days_with_missing_temperature_data",
        "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
    ])


@pytest.mark.parametrize("get_datetime_index", [["2MS", True]], indirect=True)
def test_billing_baseline_data_with_bimonthly_frequencies(get_datetime_index):
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
    df.index = df.index[:-1].union([df.index[-1] - pd.Timedelta(days=1)])
    df.iloc[-1, df.columns.get_loc("observed")] = np.nan

    cls = BillingBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    # Because two months are missing
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert round(cls.df.observed.sum(), 2) == round(df.observed.sum(), 2)
    assert len(cls.warnings) == 1
    assert (
        cls.warnings[0].qualified_name
        == "eemeter.sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency"
    )
    # DQ because only 6 days worth of temperature data is available
    assert len(cls.disqualification) == 2
    assert set([dq.qualified_name for dq in cls.disqualification]) == set(
        [
            "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
            "eemeter.sufficiency_criteria.too_many_days_with_missing_temperature_data",
        ]
    )


def test_billing_baseline_data_with_monthly_hourly_frequencies(
    get_meter_data_monthly, get_temperature_data_hourly
):
    # Create a DataFrame with uneven frequency
    df = get_temperature_data_hourly

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_monthly

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how="outer")
    df = df[:-1]  # when using dataframe input, rows are exact length

    cls = BillingBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert round(cls.df.observed.sum(), 2) == round(df.observed.sum(), 2)
    assert len(cls.warnings) == 0
    assert len(cls.disqualification) == 0


def test_billing_baseline_data_with_bimonthly_hourly_frequencies(
    get_meter_data_bimonthly, get_temperature_data_hourly
):
    # Create a DataFrame with uneven frequency
    df = get_temperature_data_hourly

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_bimonthly

    # Merge 'df' and 'df_meter' in a left join, as df input should not have trailing nan
    df = df.merge(df_meter, left_index=True, right_index=True, how="left")

    cls = BillingBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert round(cls.df.observed.sum(), 2) == round(df.observed.sum(), 2)
    assert len(cls.warnings) == 0
    assert len(cls.disqualification) == 0


def test_billing_baseline_data_with_monthly_daily_frequencies(
    get_meter_data_monthly, get_temperature_data_daily
):
    # Create a DataFrame with uneven frequency
    df = get_temperature_data_daily

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_monthly

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how="outer")
    df = df[:-1]  # when using dataframe input, rows are exact length

    cls = BillingBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert round(cls.df.observed.sum(), 2) == round(df.observed.sum(), 2)
    assert len(cls.warnings) == 1
    assert (
        cls.warnings[0].qualified_name
        == "eemeter.sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency"
    )
    assert len(cls.disqualification) == 0


def test_billing_baseline_data_with_bimonthly_daily_frequencies(
    get_meter_data_bimonthly, get_temperature_data_daily
):
    # Create a DataFrame with uneven frequency
    df = get_temperature_data_daily

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_bimonthly

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how="outer")
    df = df[:-1]  # when using dataframe input, rows are exact length

    cls = BillingBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    # assert round(cls.df.observed.sum(), 2) == round(df.observed[:-1].sum(), 2)
    assert round(cls.df.observed.sum(), 2) == round(df.observed.sum(), 2)
    assert len(cls.warnings) == 1
    assert (
        cls.warnings[0].qualified_name
        == "eemeter.sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency"
    )
    assert len(cls.disqualification) == 0


def test_billing_baseline_data_with_specific_hourly_input():
    meter, temperature, _ = load_sample("il-electricity-cdd-hdd-hourly")
    # Take the extra month for billing data
    meter = meter[
        (meter.index.year == 2017)
        | ((meter.index.year == 2018) & (meter.index.month == 1))
    ]
    temperature = temperature[
        (temperature.index.year == 2017)
        | ((temperature.index.year == 2018) & (temperature.index.month == 1))
    ]

    cls = BillingBaselineData.from_series(meter, temperature, is_electricity_data=True)

    assert cls.df is not None
    assert (
        len(cls.df) == (meter.index[-1] - meter.index[0]).days + 1
    )  # hourly series does not have trailing nan
    assert round(cls.df.observed.sum(), 2) == round(meter.value.sum(), 2)
    assert len(cls.warnings) == 2
    assert [warning.qualified_name for warning in cls.warnings] == [
        "eemeter.data_quality.utc_index",
        "eemeter.sufficiency_criteria.inferior_model_usage",
    ]
    assert len(cls.disqualification) == 1
    assert (
        cls.disqualification[0].qualified_name
        == "eemeter.sufficiency_criteria.incorrect_number_of_total_days"
    )


def test_billing_baseline_data_with_specific_daily_input():
    meter, temperature, _ = load_sample("il-electricity-cdd-hdd-daily")
    # Take the extra month for billing data
    meter = meter[
        (meter.index.year == 2017)
        | ((meter.index.year == 2018) & (meter.index.month == 1))
    ]
    temperature = temperature[
        (temperature.index.year == 2017)
        | ((temperature.index.year == 2018) & (temperature.index.month == 1))
    ]
    cls = BillingBaselineData.from_series(meter, temperature, is_electricity_data=True)

    assert cls.df is not None
    assert (
        len(cls.df) == (meter.index[-1] - meter.index[0]).days + 1
    )  # daily series does not have trailing nan
    assert round(cls.df.observed.sum(), 2) == round(meter.value.sum(), 2)
    assert len(cls.warnings) == 2
    assert [warning.qualified_name for warning in cls.warnings] == [
        "eemeter.data_quality.utc_index",
        "eemeter.sufficiency_criteria.inferior_model_usage",
    ]
    assert len(cls.disqualification) == 1
    assert (
        cls.disqualification[0].qualified_name
        == "eemeter.sufficiency_criteria.incorrect_number_of_total_days"
    )


def test_billing_baseline_data_with_specific_missing_daily_input():
    meter, temperature, _ = load_sample("il-electricity-cdd-hdd-daily")
    # Take the extra month for billing data
    meter = meter[
        (meter.index.year == 2017)
        | ((meter.index.year == 2018) & (meter.index.month == 1))
    ]
    temperature = temperature[
        (temperature.index.year == 2017)
        | ((temperature.index.year == 2018) & (temperature.index.month == 1))
    ]
    # Set 1 month meter data to NaN
    meter.loc[meter.index.month == 4] = np.nan

    cls = BillingBaselineData.from_series(meter, temperature, is_electricity_data=True)

    assert cls.df is not None
    assert (
        len(cls.df) == (meter.index[-1] - meter.index[0]).days + 1
    )  # daily series does not have trailing nan
    assert round(cls.df.observed.sum(), 2) == round(meter.value.sum(), 2)
    assert len(cls.warnings) == 2
    assert [warning.qualified_name for warning in cls.warnings] == [
        "eemeter.data_quality.utc_index",
        "eemeter.sufficiency_criteria.inferior_model_usage",
    ]
    assert len(cls.disqualification) == 1
    assert (
        cls.disqualification[0].qualified_name
        == "eemeter.sufficiency_criteria.incorrect_number_of_total_days"
    )


def test_billing_baseline_data_with_specific_monthly_input():
    meter, temperature, _ = load_sample("il-electricity-cdd-hdd-billing_monthly")
    # Take the extra month for billing data
    meter = meter[
        (meter.index.year == 2017)
        | ((meter.index.year == 2018) & (meter.index.month == 1))
    ]
    temperature = temperature[
        (temperature.index.year == 2017)
        | ((temperature.index.year == 2018) & (temperature.index.month == 1))
    ]
    cls = BillingBaselineData.from_series(meter, temperature, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == (meter.index[-1] - meter.index[0]).days
    assert round(cls.df.observed.sum(), 2) == round(meter.value.sum(), 2)
    assert len(cls.warnings) == 1
    assert set([warning.qualified_name for warning in cls.warnings]) == set(
        ["eemeter.data_quality.utc_index"]
    )
    assert len(cls.disqualification) == 0


@pytest.mark.parametrize(
    "get_datetime_index", [["30min", True], ["h", True]], indirect=True
)
def test_billing_reporting_data_with_missing_half_hourly_frequencies(
    get_datetime_index,
):
    datetime_index = get_datetime_index
    datetime_index = datetime_index[datetime_index.year == 2023]

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

    cls = BillingReportingData(df, is_electricity_data=True)

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
    assert len(cls.disqualification) == 3
    expected_disqualifications = [
        "eemeter.sufficiency_criteria.missing_monthly_temperature_data",
        "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
        "eemeter.sufficiency_criteria.too_many_days_with_missing_temperature_data",
    ]
    assert all(
        disqualification.qualified_name in expected_disqualifications
        for disqualification in cls.disqualification
    )


@pytest.mark.parametrize("get_datetime_index", [["D", True]], indirect=True)
def test_billing_reporting_data_with_missing_daily_frequencies(get_datetime_index):
    datetime_index = get_datetime_index
    datetime_index = datetime_index[datetime_index.year == 2023]

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

    cls = BillingReportingData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert len(cls.df.temperature.dropna()) == len(df.temperature.dropna())
    assert len(cls.warnings) == 1
    assert (
        cls.warnings[0].qualified_name
        == "eemeter.sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency"
    )
    assert len(cls.disqualification) == 3
    expected_disqualifications = [
        "eemeter.sufficiency_criteria.missing_monthly_temperature_data",
        "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
        "eemeter.sufficiency_criteria.too_many_days_with_missing_temperature_data",
    ]
    assert all(
        disqualification.qualified_name in expected_disqualifications
        for disqualification in cls.disqualification
    )


def test_dst_handling():
    # 2020-03-08 02:00 is nonexistent, should push to 03:00
    tz = "America/New_York"
    idx = DatetimeIndex(
        [
            Timestamp("2020-03-07 02", tz=tz),
            Timestamp("2020-04-06 02", tz=tz),
            Timestamp("2020-05-06 02", tz=tz),
        ]
    )
    df = DataFrame({"observed": [1] * 3, "temperature": [50] * 3}, index=idx)
    baseline = BillingBaselineData(df, is_electricity_data=True)
    assert len(baseline.df) == 61
    hours = np.unique(baseline.df.index.hour)
    assert (hours == [2, 3]).all()

    # 2020-11-01 01:00 is ambiguous, single index should be chosen
    tz = "America/New_York"
    idx = DatetimeIndex(
        [
            Timestamp("2020-10-31 01", tz=tz),
            Timestamp("2020-11-28 01", tz=tz),
            Timestamp("2020-12-28 01", tz=tz),
        ]
    )
    df = DataFrame({"observed": [1] * 3, "temperature": [50] * 3}, index=idx)
    baseline = BillingBaselineData(df, is_electricity_data=True)
    assert (baseline.df.index.hour == 1).all()
