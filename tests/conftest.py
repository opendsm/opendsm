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

from opendsm.common.test_data import load_test_data

from syrupy_extensions import TolerantJSONSnapshotExtension


@pytest.fixture
def snapshot(snapshot):
    """Default the project's `snapshot` fixture to the tolerant JSON extension."""
    return snapshot.use_extension(TolerantJSONSnapshotExtension)


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


def _meter_subset(df, meter_id, freq=None):
    sub = df.xs(meter_id, level="id")
    if freq is not None:
        sub = sub.asfreq(freq)

    return sub


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


@pytest.fixture(scope="session")
def comstock_hourly(_comstock_hourly_all):
    """Single-meter hourly baseline+reporting DataFrames with freq='h'."""
    df_b, df_r = _comstock_hourly_all
    return (
        _meter_subset(df_b, COMSTOCK_DEFAULT_METER_ID, freq="h"),
        _meter_subset(df_r, COMSTOCK_DEFAULT_METER_ID, freq="h"),
    )


@pytest.fixture(scope="session")
def comstock_daily(_comstock_daily_all):
    """Single-meter daily baseline+reporting DataFrames."""
    df_b, df_r = _comstock_daily_all
    return _meter_subset(df_b, COMSTOCK_DEFAULT_METER_ID), _meter_subset(df_r, COMSTOCK_DEFAULT_METER_ID)


@pytest.fixture(scope="session")
def comstock_monthly(_comstock_monthly_all):
    """Single-meter monthly baseline+reporting DataFrames."""
    df_b, df_r = _comstock_monthly_all
    return _meter_subset(df_b, COMSTOCK_DEFAULT_METER_ID), _meter_subset(df_r, COMSTOCK_DEFAULT_METER_ID)


@pytest.fixture(params=COMSTOCK_METER_IDS, ids=[str(i) for i in COMSTOCK_METER_IDS])
def comstock_hourly_diverse(request, _comstock_hourly_all):
    """Parametrized: yields each of the 5 diverse meters' hourly baseline+reporting with freq='h'."""
    df_b, df_r = _comstock_hourly_all
    return (
        _meter_subset(df_b, request.param, freq="h"),
        _meter_subset(df_r, request.param, freq="h"),
    )


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


