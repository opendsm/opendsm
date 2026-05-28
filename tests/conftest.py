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

import json
import importlib.resources

import pytest

from opendsm.common.test_data import load_test_data
from opendsm.eemeter.samples import load_sample


# ComStock meter ids selected for diverse load behavior.
# Greedy-diverse pick across 6 standardized dimensions
# (log_mean, peak/baseload, temp_corr, day/night ratio, wd/we ratio, CV):
#   116756 - central / representative commercial building
#   115041 - smallest mean, strongest workweek pattern (wd/we ratio 3.06)
#   115092 - medium scale, moderate workweek pattern
#   116975 - flatter load shape (low peak/baseload, low CV)
#   117154 - largest scale (~2000 kWh mean), near-symmetric wd/we
COMSTOCK_METER_IDS = [116756, 115041, 115092, 116975, 117154]
COMSTOCK_DEFAULT_METER_ID = COMSTOCK_METER_IDS[0]


def _meter_subset(df, meter_id):
    return df.xs(meter_id, level="id")


@pytest.fixture(scope="session")
def _comstock_hourly_all():
    """All 100 meters of ComStock hourly treatment data, (baseline, reporting)."""
    return load_test_data("hourly_treatment_data")


@pytest.fixture(scope="session")
def _comstock_daily_all():
    """All 100 meters aggregated daily (mean temperature, sum observed)."""
    return load_test_data("daily_treatment_data")


@pytest.fixture(scope="session")
def _comstock_monthly_all():
    """All 100 meters aggregated monthly."""
    return load_test_data("monthly_treatment_data")


@pytest.fixture
def comstock_hourly(_comstock_hourly_all):
    """Single-meter hourly baseline+reporting DataFrames."""
    df_b, df_r = _comstock_hourly_all
    return _meter_subset(df_b, COMSTOCK_DEFAULT_METER_ID), _meter_subset(df_r, COMSTOCK_DEFAULT_METER_ID)


@pytest.fixture
def comstock_daily(_comstock_daily_all):
    """Single-meter daily baseline+reporting DataFrames."""
    df_b, df_r = _comstock_daily_all
    return _meter_subset(df_b, COMSTOCK_DEFAULT_METER_ID), _meter_subset(df_r, COMSTOCK_DEFAULT_METER_ID)


@pytest.fixture
def comstock_monthly(_comstock_monthly_all):
    """Single-meter monthly baseline+reporting DataFrames."""
    df_b, df_r = _comstock_monthly_all
    return _meter_subset(df_b, COMSTOCK_DEFAULT_METER_ID), _meter_subset(df_r, COMSTOCK_DEFAULT_METER_ID)


@pytest.fixture(params=COMSTOCK_METER_IDS, ids=[str(i) for i in COMSTOCK_METER_IDS])
def comstock_hourly_diverse(request, _comstock_hourly_all):
    """Parametrized: yields each of the 5 diverse meters' hourly baseline+reporting."""
    df_b, df_r = _comstock_hourly_all
    return _meter_subset(df_b, request.param), _meter_subset(df_r, request.param)


@pytest.fixture(params=COMSTOCK_METER_IDS, ids=[str(i) for i in COMSTOCK_METER_IDS])
def comstock_daily_diverse(request, _comstock_daily_all):
    """Parametrized: yields each of the 5 diverse meters' daily baseline+reporting."""
    df_b, df_r = _comstock_daily_all
    return _meter_subset(df_b, request.param), _meter_subset(df_r, request.param)


@pytest.fixture(params=COMSTOCK_METER_IDS, ids=[str(i) for i in COMSTOCK_METER_IDS])
def comstock_monthly_diverse(request, _comstock_monthly_all):
    """Parametrized: yields each of the 5 diverse meters' monthly baseline+reporting."""
    df_b, df_r = _comstock_monthly_all
    return _meter_subset(df_b, request.param), _meter_subset(df_r, request.param)


@pytest.fixture
def sample_metadata():
    with importlib.resources.files("opendsm.eemeter.samples").joinpath(
        "metadata.json"
    ).open("rb") as f:
        metadata = json.loads(f.read().decode("utf-8"))
    return metadata


def _from_sample(sample, tempF=True):
    meter_data, temperature_data, metadata = load_sample(sample, tempF=tempF)
    return {
        "meter_data": meter_data,
        "temperature_data": temperature_data,
        "blackout_start_date": metadata["blackout_start_date"],
        "blackout_end_date": metadata["blackout_end_date"],
    }


@pytest.fixture
def il_electricity_cdd_hdd_hourly():
    return _from_sample("il-electricity-cdd-hdd-hourly")


@pytest.fixture
def il_electricity_cdd_hdd_daily():
    return _from_sample("il-electricity-cdd-hdd-daily")


@pytest.fixture
def il_electricity_cdd_hdd_billing_monthly():
    return _from_sample("il-electricity-cdd-hdd-billing_monthly")


@pytest.fixture
def il_electricity_cdd_hdd_billing_bimonthly():
    return _from_sample("il-electricity-cdd-hdd-billing_bimonthly")


@pytest.fixture
def il_gas_hdd_only_hourly():
    return _from_sample("il-gas-hdd-only-hourly")


@pytest.fixture
def uk_electricity_hdd_only_hourly_sample_1():
    return _from_sample("uk-electricity-hdd-only-hourly-sample-1", tempF=False)


@pytest.fixture
def uk_electricity_hdd_only_hourly_sample_2():
    return _from_sample("uk-electricity-hdd-only-hourly-sample-2", tempF=False)
