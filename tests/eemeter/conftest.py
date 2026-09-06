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
import pandas as pd
import pytest



TEMPERATURE_SEED = 29
METER_SEED = 41


@pytest.fixture
def get_datetime_index(request):
    # Request = [frequency , is_timezone_aware]

    # Create a DateTimeIndex at given frequency and timezone if requested
    if request.param[0] in ["MS", "2MS"]:
        inclusive = "both"
    else:
        inclusive = "left"

    if request.param[1]:
        tz = "America/New_York"
    else:
        tz = None

    datetime_index = pd.date_range(
        start="2023-01-01",
        end="2024-01-01",
        inclusive=inclusive,
        freq=request.param[0],
        tz=tz,
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
        tz="America/New_York",
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
        tz="America/New_York",
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
        tz="America/New_York",
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
