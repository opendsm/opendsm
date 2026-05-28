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

from opendsm.eemeter.models.hourly_caltrack.design_matrices import (
    create_caltrack_hourly_preliminary_design_matrix,
    create_caltrack_hourly_segmented_design_matrices,
    create_caltrack_daily_design_matrix,
    create_caltrack_billing_design_matrix,
)
from opendsm.eemeter.common.features import (
    estimate_hour_of_week_occupancy,
    fit_temperature_bins,
)
from opendsm.eemeter.models.hourly_caltrack.segmentation import segment_time_series


def _meter_temp(df):
    """Return (meter_df with 'value' column, temperature series) from a comstock baseline DataFrame."""
    meter = df[["observed"]].rename(columns={"observed": "value"}).copy()
    temperature = df["temperature"].copy()

    return meter, temperature


def test_create_caltrack_hourly_preliminary_design_matrix(comstock_hourly, snapshot):
    df_b, _ = comstock_hourly
    meter, temperature = _meter_temp(df_b)
    design_matrix = create_caltrack_hourly_preliminary_design_matrix(
        meter[:1000], temperature
    )
    assert design_matrix.shape == (1000, 7)
    assert sorted(design_matrix.columns) == [
        "cdd_65",
        "hdd_50",
        "hour_of_week",
        "meter_value",
        "n_hours_dropped",
        "n_hours_kept",
        "temperature_mean",
    ]
    # In newer pandas, categorical columns (like hour_of_week) arent included in sum
    design_matrix.hour_of_week = design_matrix.hour_of_week.astype(float)
    snapshot.assert_match(round(float(design_matrix.sum().sum()), 2), "design_matrix_sum")


def test_create_caltrack_daily_design_matrix(comstock_daily, comstock_hourly, snapshot):
    df_daily, _ = comstock_daily
    df_hourly, _ = comstock_hourly
    meter, _ = _meter_temp(df_daily)
    temperature = df_hourly["temperature"]
    design_matrix = create_caltrack_daily_design_matrix(meter[:100], temperature)
    assert design_matrix.shape == (100, 6)
    assert sorted(design_matrix.columns) == [
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
        "temperature_mean",
        "temperature_not_null",
        "temperature_null",
    ]
    snapshot.assert_match(round(float(design_matrix.sum().sum()), 2), "design_matrix_sum")


def test_create_caltrack_billing_design_matrix(comstock_monthly, comstock_hourly, snapshot):
    df_monthly, _ = comstock_monthly
    df_hourly, _ = comstock_hourly
    meter, _ = _meter_temp(df_monthly)
    temperature = df_hourly["temperature"]
    meter = meter.dropna()
    design_matrix = create_caltrack_billing_design_matrix(meter[:10], temperature)
    assert sorted(design_matrix.columns) == [
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
        "temperature_mean",
        "temperature_not_null",
        "temperature_null",
    ]
    snapshot.assert_match(round(float(design_matrix.sum().sum()), 2), "design_matrix_sum")
    snapshot.assert_match(list(design_matrix.shape), "design_matrix_shape")


@pytest.fixture
def preliminary_hourly_design_matrix(comstock_hourly):
    df_b, _ = comstock_hourly
    meter, temperature = _meter_temp(df_b)

    return create_caltrack_hourly_preliminary_design_matrix(meter[:1000], temperature)


@pytest.fixture
def segmentation(preliminary_hourly_design_matrix):
    return segment_time_series(
        preliminary_hourly_design_matrix.index, "three_month_weighted"
    )


@pytest.fixture
def occupancy_lookup(preliminary_hourly_design_matrix, segmentation):
    return estimate_hour_of_week_occupancy(
        preliminary_hourly_design_matrix, segmentation=segmentation
    )


@pytest.fixture
def temperature_bins(preliminary_hourly_design_matrix, segmentation, occupancy_lookup):
    return fit_temperature_bins(
        preliminary_hourly_design_matrix,
        segmentation=segmentation,
        occupancy_lookup=occupancy_lookup,
    )


def test_create_caltrack_hourly_segmented_design_matrices(
    preliminary_hourly_design_matrix, segmentation, occupancy_lookup, temperature_bins, snapshot
):
    occupied_temperature_bins, unoccupied_temperature_bins = temperature_bins
    design_matrices = create_caltrack_hourly_segmented_design_matrices(
        preliminary_hourly_design_matrix,
        segmentation,
        occupancy_lookup,
        occupied_temperature_bins,
        unoccupied_temperature_bins,
    )

    for segment_name in ("dec-jan-feb-weighted", "mar-apr-may-weighted"):
        design_matrix = design_matrices[segment_name]
        design_matrix.hour_of_week = design_matrix.hour_of_week.astype(float)
        snapshot.assert_match(list(design_matrix.shape), f"{segment_name}_shape")
        snapshot.assert_match(sorted(design_matrix.columns), f"{segment_name}_columns")
        snapshot.assert_match(round(float(design_matrix.sum().sum()), 2), f"{segment_name}_sum")


def test_create_caltrack_billing_design_matrix_empty_temp(comstock_monthly, comstock_hourly):
    df_monthly, _ = comstock_monthly
    df_hourly, _ = comstock_hourly
    meter, _ = _meter_temp(df_monthly)
    meter = meter.dropna()
    temperature = df_hourly["temperature"]
    with pytest.raises(ValueError):
        create_caltrack_billing_design_matrix(meter[:10], temperature[:0])


def test_create_caltrack_billing_design_matrix_partial_empty_temp(comstock_monthly, comstock_hourly):
    df_monthly, _ = comstock_monthly
    df_hourly, _ = comstock_hourly
    meter, _ = _meter_temp(df_monthly)
    meter = meter.dropna()
    temperature = df_hourly["temperature"]
    design_matrix = create_caltrack_billing_design_matrix(meter[:10], temperature[:200])
    assert design_matrix is not None
