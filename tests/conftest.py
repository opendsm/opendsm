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

import sys

import pytest

from opendsm.common.test_data import load_test_data

from syrupy_extensions import TolerantJSONSnapshotExtension


@pytest.fixture
def snapshot(snapshot):
    """Default the project's `snapshot` fixture to the tolerant JSON extension."""
    return snapshot.use_extension(TolerantJSONSnapshotExtension)


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.regression tests off linux-x86_64; nlopt/BLAS optimizer convergence drifts beyond tolerance on macOS/Windows."""
    if sys.platform == "linux":
        return

    skip = pytest.mark.skip(
        reason="Snapshot pinned to linux-x86_64; cross-platform optimizer/BLAS drift exceeds tolerance"
    )
    for item in items:
        if "regression" in item.keywords:
            item.add_marker(skip)


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


def _to_fahrenheit(df):
    """ComStock parquet ships temperature in Celsius; opendsm data classes expect Fahrenheit."""
    out = df.copy()
    out["temperature"] = out["temperature"] * 9.0 / 5.0 + 32.0

    return out


@pytest.fixture(scope="session")
def _comstock_hourly_all():
    """All 100 meters of ComStock hourly treatment data, (baseline, reporting)."""
    df_b, df_r = load_test_data("hourly_treatment_data")

    return _to_fahrenheit(df_b), _to_fahrenheit(df_r)


@pytest.fixture(scope="session")
def _comstock_daily_all():
    """All 100 meters aggregated daily (mean temperature, sum observed)."""
    df_b, df_r = load_test_data("daily_treatment_data")

    return _to_fahrenheit(df_b), _to_fahrenheit(df_r)


@pytest.fixture(scope="session")
def _comstock_monthly_all():
    """All 100 meters aggregated monthly."""
    df_b, df_r = load_test_data("monthly_treatment_data")

    return _to_fahrenheit(df_b), _to_fahrenheit(df_r)


@pytest.fixture
def comstock_hourly(_comstock_hourly_all):
    """Single-meter hourly baseline+reporting DataFrames with freq='h'."""
    df_b, df_r = _comstock_hourly_all
    return (
        _meter_subset(df_b, COMSTOCK_DEFAULT_METER_ID, freq="h"),
        _meter_subset(df_r, COMSTOCK_DEFAULT_METER_ID, freq="h"),
    )


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


