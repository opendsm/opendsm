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


def test_create_caltrack_hourly_preliminary_design_matrix(
    il_electricity_cdd_hdd_hourly,
):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"]
    design_matrix = create_caltrack_hourly_preliminary_design_matrix(
        meter_data[:1000], temperature_data
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
    assert round(design_matrix.sum().sum(), 2) == 136352.61


def test_create_caltrack_daily_design_matrix(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    design_matrix = create_caltrack_daily_design_matrix(
        meter_data[:100], temperature_data
    )
    assert design_matrix.shape == (100, 6)
    assert sorted(design_matrix.columns) == [
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
        "temperature_mean",
        "temperature_not_null",
        "temperature_null",
    ]
    assert round(design_matrix.sum().sum(), 2) == 9267.06


def test_create_caltrack_billing_design_matrix(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    design_matrix = create_caltrack_billing_design_matrix(
        meter_data[:10], temperature_data
    )
    assert design_matrix.shape == (275, 6)
    assert sorted(design_matrix.columns) == [
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
        "temperature_mean",
        "temperature_not_null",
        "temperature_null",
    ]
    assert round(design_matrix.sum().sum(), 2) == 29925.27


@pytest.fixture
def preliminary_hourly_design_matrix(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"]
    return create_caltrack_hourly_preliminary_design_matrix(
        meter_data[:1000], temperature_data
    )


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
    preliminary_hourly_design_matrix, segmentation, occupancy_lookup, temperature_bins
):
    occupied_temperature_bins, unoccupied_temperature_bins = temperature_bins
    design_matrices = create_caltrack_hourly_segmented_design_matrices(
        preliminary_hourly_design_matrix,
        segmentation,
        occupancy_lookup,
        occupied_temperature_bins,
        unoccupied_temperature_bins,
    )

    design_matrix = design_matrices["dec-jan-feb-weighted"]
    assert design_matrix.shape == (1000, 8)
    assert sorted(design_matrix.columns) == [
        "bin_0_occupied",
        "bin_0_unoccupied",
        "bin_1_unoccupied",
        "bin_2_unoccupied",
        "bin_3_unoccupied",
        "hour_of_week",
        "meter_value",
        "weight",
    ]
    design_matrix.hour_of_week = design_matrix.hour_of_week.astype(float)
    assert round(design_matrix.sum().sum(), 2) == 126210.07

    design_matrix = design_matrices["mar-apr-may-weighted"]
    assert design_matrix.shape == (1000, 5)
    assert sorted(design_matrix.columns) == [
        "bin_0_occupied",
        "bin_0_unoccupied",
        "hour_of_week",
        "meter_value",
        "weight",
    ]
    design_matrix.hour_of_week = design_matrix.hour_of_week.astype(float)
    assert round(design_matrix.sum().sum(), 2) == 167659.28


def test_create_caltrack_billing_design_matrix_empty_temp(
    il_electricity_cdd_hdd_billing_monthly,
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"][:0]
    with pytest.raises(ValueError):
        design_matrix = create_caltrack_billing_design_matrix(
            meter_data[:10], temperature_data
        )


def test_create_caltrack_billing_design_matrix_partial_empty_temp(
    il_electricity_cdd_hdd_billing_monthly,
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"][:200]
    design_matrix = create_caltrack_billing_design_matrix(
        meter_data[:10], temperature_data
    )
    assert "n_days_kept" in design_matrix.columns
