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
"""Regression tests pinning the caltrack hourly fit & predict outputs."""

import pytest

from opendsm.eemeter.common.features import (
    estimate_hour_of_week_occupancy,
    fit_temperature_bins,
)
from opendsm.eemeter.models.hourly_caltrack.design_matrices import (
    create_caltrack_hourly_preliminary_design_matrix,
    create_caltrack_hourly_segmented_design_matrices,
)
from opendsm.eemeter.models.hourly_caltrack.model import fit_caltrack_hourly_model
from opendsm.eemeter.models.hourly_caltrack.segmentation import segment_time_series


@pytest.fixture(scope="session")
def caltrack_baseline_preliminary(comstock_hourly):
    df_b, _ = comstock_hourly
    meter_data = df_b["observed"].rename("value").to_frame()

    return create_caltrack_hourly_preliminary_design_matrix(
        meter_data, df_b["temperature"]
    )


@pytest.fixture(scope="session")
def caltrack_hourly_baseline_model(caltrack_baseline_preliminary):
    preliminary = caltrack_baseline_preliminary
    segmentation = segment_time_series(preliminary.index, "three_month_weighted")
    occupancy = estimate_hour_of_week_occupancy(preliminary, segmentation=segmentation)
    occ_bins, unocc_bins = fit_temperature_bins(
        preliminary, segmentation=segmentation, occupancy_lookup=occupancy
    )
    design = create_caltrack_hourly_segmented_design_matrices(
        preliminary, segmentation, occupancy, occ_bins, unocc_bins
    )

    return fit_caltrack_hourly_model(
        design, occupancy, occ_bins, unocc_bins, segment_type="three_month_weighted"
    )


def _summary(series):
    return {
        "sum": float(series.sum()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "n": int(series.shape[0]),
    }


@pytest.mark.slow
@pytest.mark.regression
def test_caltrack_hourly_baseline_predict_regression(
    caltrack_hourly_baseline_model, caltrack_baseline_preliminary, snapshot
):
    preliminary = caltrack_baseline_preliminary
    predicted = caltrack_hourly_baseline_model.predict(
        preliminary.index, preliminary["temperature_mean"]
    ).result["predicted_usage"]

    assert _summary(predicted) == snapshot(name="predicted_summary")
    assert predicted.values.tolist() == snapshot(name="predicted_values")
