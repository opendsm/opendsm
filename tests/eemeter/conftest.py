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

from opendsm.eemeter.models.billing.model import BillingModel
from opendsm.eemeter.models.daily.data import DailyReportingData
from opendsm.eemeter.models.daily.model import DailyModel

from .constants import METER_SEED, TEMPERATURE_SEED



@pytest.fixture
def get_datetime_index(request):
    # request.param: (frequency, is_timezone_aware)
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
    datetime_index = pd.date_range(
        start="2023-01-01", end="2024-01-01", inclusive="left", freq="D"
    )

    return datetime_index


@pytest.fixture
def get_temperature_data_half_hourly(get_datetime_index_half_hourly_with_timezone):
    datetime_index = get_datetime_index_half_hourly_with_timezone

    np.random.seed(TEMPERATURE_SEED)
    temperature_mean = np.random.rand(len(datetime_index))

    df = pd.DataFrame(data={"temperature": temperature_mean}, index=datetime_index)

    return df


@pytest.fixture
def get_temperature_data_hourly(get_datetime_index_hourly_with_timezone):
    datetime_index = get_datetime_index_hourly_with_timezone

    np.random.seed(TEMPERATURE_SEED)
    temperature_mean = np.random.rand(len(datetime_index))

    df = pd.DataFrame(data={"temperature": temperature_mean}, index=datetime_index)

    return df


@pytest.fixture
def get_temperature_data_daily(get_datetime_index_daily_with_timezone):
    datetime_index = get_datetime_index_daily_with_timezone

    np.random.seed(TEMPERATURE_SEED)
    temperature_mean = np.random.rand(len(datetime_index))

    df = pd.DataFrame(data={"temperature": temperature_mean}, index=datetime_index)

    return df


@pytest.fixture
def get_meter_data_daily(get_datetime_index_daily_with_timezone):
    datetime_index = get_datetime_index_daily_with_timezone

    np.random.seed(METER_SEED)
    meter_value = np.random.rand(len(datetime_index))

    df = pd.DataFrame(data={"observed": meter_value}, index=datetime_index)

    return df


@pytest.fixture(scope="session")
def baseline_data_daily(comstock_daily):
    df_b, _ = comstock_daily

    return df_b.reset_index().copy()


@pytest.fixture(scope="session")
def baseline_model_daily(baseline_data_daily):
    """The daily model fit once on the ComStock baseline meter; tests read it and never
    mutate it."""
    model = DailyModel().fit(
        baseline_data_daily, is_electricity_data=True, enforce_compliance=False
    )

    return model


@pytest.fixture(scope="session")
def reporting_data_daily(comstock_daily):
    _, df_r = comstock_daily
    data = DailyReportingData(df=df_r.reset_index().copy(), is_electricity_data=True)

    return data


@pytest.fixture(scope="session")
def baseline_data_billing(comstock_monthly):
    df_b, _ = comstock_monthly

    return df_b.reset_index().copy()


@pytest.fixture(scope="session")
def baseline_model_billing(baseline_data_billing):
    """The billing model fit once on the ComStock baseline meter; tests read it and never
    mutate it."""
    model = BillingModel().fit(
        baseline_data_billing, is_electricity_data=True, enforce_compliance=False
    )

    return model
