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
import numpy as np
import pandas as pd
import pytest

from opendsm.eemeter.models.hourly_caltrack.model import (
    caltrack_hourly_fit_feature_processor,
    caltrack_hourly_prediction_feature_processor,
    fit_caltrack_hourly_model_segment,
    fit_caltrack_hourly_model,
)
from opendsm.eemeter.common.features import (
    compute_time_features,
)


@pytest.fixture
def segmented_data():
    index = pd.date_range(start="2017-01-01", periods=24, freq="h", tz="UTC")
    time_features = compute_time_features(index)
    segmented_data = pd.DataFrame(
        {
            "hour_of_week": time_features.hour_of_week,
            "temperature_mean": np.linspace(0, 100, 24),
            "meter_value": np.linspace(10, 70, 24),
            "weight": np.ones((24,)),
        },
        index=index,
    )
    return segmented_data


@pytest.fixture
def occupancy_lookup():
    index = pd.Categorical(range(168))
    occupancy = pd.Series([i % 2 == 0 for i in range(168)], index=index)
    return pd.DataFrame(
        {"dec-jan-feb-weighted": occupancy, "jan-feb-mar-weighted": occupancy}
    )


@pytest.fixture
def occupied_temperature_bins():
    bins = pd.Series([True, True, True], index=[30, 60, 90])
    return pd.DataFrame({"dec-jan-feb-weighted": bins, "jan-feb-mar-weighted": bins})


@pytest.fixture
def unoccupied_temperature_bins():
    bins = pd.Series([False, True, True], index=[30, 60, 90])
    return pd.DataFrame({"dec-jan-feb-weighted": bins, "jan-feb-mar-weighted": bins})


def test_caltrack_hourly_fit_feature_processor(
    segmented_data,
    occupancy_lookup,
    occupied_temperature_bins,
    unoccupied_temperature_bins,
):
    result = caltrack_hourly_fit_feature_processor(
        "dec-jan-feb-weighted",
        segmented_data,
        occupancy_lookup,
        occupied_temperature_bins,
        unoccupied_temperature_bins,
    )
    assert list(result.columns) == [
        "meter_value",
        "hour_of_week",
        "bin_0_occupied",
        "bin_1_occupied",
        "bin_2_occupied",
        "bin_3_occupied",
        "bin_0_unoccupied",
        "bin_1_unoccupied",
        "bin_2_unoccupied",
        "weight",
    ]
    assert result.shape == (24, 10)
    result.hour_of_week = result.hour_of_week.astype(float)
    assert round(result.sum().sum(), 2) == 5916.0


def test_caltrack_hourly_prediction_feature_processor(
    segmented_data,
    occupancy_lookup,
    occupied_temperature_bins,
    unoccupied_temperature_bins,
):
    result = caltrack_hourly_prediction_feature_processor(
        "dec-jan-feb-weighted",
        segmented_data,
        occupancy_lookup,
        occupied_temperature_bins,
        unoccupied_temperature_bins,
    )
    assert list(result.columns) == [
        "hour_of_week",
        "bin_0_occupied",
        "bin_1_occupied",
        "bin_2_occupied",
        "bin_3_occupied",
        "bin_0_unoccupied",
        "bin_1_unoccupied",
        "bin_2_unoccupied",
        "weight",
    ]
    assert result.shape == (24, 9)
    result.hour_of_week = result.hour_of_week.astype(float)
    assert round(result.sum().sum(), 2) == 4956.0


@pytest.fixture
def segmented_design_matrices(
    segmented_data,
    occupancy_lookup,
    occupied_temperature_bins,
    unoccupied_temperature_bins,
):
    return {
        "dec-jan-feb-weighted": caltrack_hourly_fit_feature_processor(
            "dec-jan-feb-weighted",
            segmented_data,
            occupancy_lookup,
            occupied_temperature_bins,
            unoccupied_temperature_bins,
        )
    }


def test_fit_caltrack_hourly_model_segment(segmented_design_matrices):
    segment_name = "dec-jan-feb-weighted"
    segment_data = segmented_design_matrices[segment_name]
    segment_model = fit_caltrack_hourly_model_segment(segment_name, segment_data)
    assert segment_model.formula == (
        "meter_value ~ C(hour_of_week) - 1 + bin_0_occupied"
        " + bin_1_occupied + bin_2_occupied + bin_3_occupied"
        " + bin_0_unoccupied + bin_1_unoccupied + bin_2_unoccupied"
    )
    assert segment_model.segment_name == "dec-jan-feb-weighted"
    assert len(segment_model.model_params.keys()) == 31
    assert segment_model.model is not None
    assert segment_model.warnings is not None
    prediction = segment_model.predict(segment_data)
    assert round(prediction.sum(), 2) == 960.0


@pytest.fixture
def temps():
    index = pd.date_range(start="2017-01-01", periods=24, freq="h", tz="UTC")
    temps = pd.Series(np.linspace(0, 100, 24), index=index)
    return temps


@pytest.mark.parametrize("segment_type", ["three_month_weighted"])
def test_fit_caltrack_hourly_model(
    segmented_design_matrices,
    occupancy_lookup,
    occupied_temperature_bins,
    unoccupied_temperature_bins,
    temps,
    segment_type,
):
    segmented_model_results = fit_caltrack_hourly_model(
        segmented_design_matrices,
        occupancy_lookup,
        occupied_temperature_bins,
        unoccupied_temperature_bins,
        segment_type=segment_type,
    )

    assert segmented_model_results.model.segment_models is not None
    assert str(segmented_model_results).startswith("CalTRACKHourlyModelResults")
    prediction = segmented_model_results.predict(temps.index, temps).result


@pytest.mark.parametrize("segment_type", ["single", "three_month_weighted"])
def test_serialize_caltrack_hourly_model(
    segmented_design_matrices,
    occupancy_lookup,
    occupied_temperature_bins,
    unoccupied_temperature_bins,
    temps,
    segment_type,
):
    segmented_model = fit_caltrack_hourly_model(
        segmented_design_matrices,
        occupancy_lookup,
        occupied_temperature_bins,
        unoccupied_temperature_bins,
        segment_type=segment_type,
    )
    assert json.dumps(segmented_model.json())


@pytest.fixture
def segmented_data_nans():
    num_periods = 200
    index = pd.date_range(start="2017-01-01", periods=num_periods, freq="h", tz="UTC")
    time_features = compute_time_features(index)
    segmented_data = pd.DataFrame(
        {
            "hour_of_week": time_features.hour_of_week,
            "temperature_mean": np.linspace(0, 100, num_periods),
            "meter_value": np.linspace(10, 70, num_periods),
            "weight": np.ones((num_periods,)),
        },
        index=index,
    )
    return segmented_data


@pytest.fixture
def occupancy_lookup_nans():
    index = pd.Categorical(range(168))
    occupancy = pd.Series([i % 2 == 0 for i in range(168)], index=index)
    occupancy_nans = pd.Series([np.nan for i in range(168)], index=index)
    return pd.DataFrame(
        {
            "dec-jan-feb-weighted": occupancy,
            "jan-feb-mar-weighted": occupancy,
            "apr-may-jun-weighted": occupancy_nans,
        }
    )


@pytest.fixture
def temperature_bins_nans():
    bins = pd.Series([True, True, True], index=[30, 60, 90])
    bins_nans = pd.Series([False, False, False], index=[30, 60, 90])
    return pd.DataFrame(
        {
            "dec-jan-feb-weighted": bins,
            "jan-feb-mar-weighted": bins,
            "apr-may-jun-weighted": bins_nans,
        }
    )


@pytest.fixture
def segmented_design_matrices_nans(
    segmented_data_nans, occupancy_lookup_nans, temperature_bins_nans
):
    return {
        "dec-jan-feb-weighted": caltrack_hourly_fit_feature_processor(
            "dec-jan-feb-weighted",
            segmented_data_nans,
            occupancy_lookup_nans,
            temperature_bins_nans,
            temperature_bins_nans,
        ),
        "apr-may-jun-weighted": caltrack_hourly_fit_feature_processor(
            "apr-may-jun-weighted",
            segmented_data_nans,
            occupancy_lookup_nans,
            temperature_bins_nans,
            temperature_bins_nans,
        ),
    }


@pytest.mark.parametrize("segment_type", ["three_month_weighted"])
def test_fit_caltrack_hourly_model_nans_less_than_week_predict(
    segmented_design_matrices_nans,
    occupancy_lookup_nans,
    temperature_bins_nans,
    temps_extended,
    temps,
    segment_type,
):
    segmented_model_results = fit_caltrack_hourly_model(
        segmented_design_matrices_nans,
        occupancy_lookup_nans,
        temperature_bins_nans,
        temperature_bins_nans,
        segment_type=segment_type,
    )

    assert segmented_model_results.model.segment_models is not None
    assert segmented_model_results.model.model_lookup["jan"].model is not None
    assert segmented_model_results.model.model_lookup["may"].model is not None
    assert segmented_model_results.model.model_lookup["may"].warnings == []
    prediction = segmented_model_results.predict(temps.index, temps).result
    assert prediction.shape[0] == 24
    assert prediction["predicted_usage"].sum().round() == 955.0


@pytest.fixture
def segmented_data_nans_less_than_week():
    num_periods = 4
    index = pd.date_range(start="2017-01-01", periods=num_periods, freq="h", tz="UTC")
    time_features = compute_time_features(index)
    segmented_data = pd.DataFrame(
        {
            "hour_of_week": time_features.hour_of_week,
            "temperature_mean": np.linspace(0, 100, num_periods),
            "meter_value": np.linspace(10, 70, num_periods),
            "weight": np.ones((num_periods,)),
        },
        index=index,
    )
    return segmented_data


@pytest.fixture
def occupancy_lookup_nans_less_than_week():
    index = pd.Categorical(range(168))
    occupancy = pd.Series([i % 2 == 0 for i in range(168)], index=index)
    occupancy_nans_less_than_week = pd.Series([np.nan for i in range(168)], index=index)
    return pd.DataFrame(
        {
            "dec-jan-feb-weighted": occupancy,
            "jan-feb-mar-weighted": occupancy,
            "apr-may-jun-weighted": occupancy_nans_less_than_week,
        }
    )


@pytest.fixture
def temperature_bins_nans_less_than_week():
    bins = pd.Series([True, True, True], index=[30, 60, 90])
    bins_nans_less_than_week = pd.Series([False, False, False], index=[30, 60, 90])
    return pd.DataFrame(
        {
            "dec-jan-feb-weighted": bins,
            "jan-feb-mar-weighted": bins,
            "apr-may-jun-weighted": bins_nans_less_than_week,
        }
    )


@pytest.fixture
def segmented_design_matrices_nans_less_than_week(
    segmented_data_nans_less_than_week,
    occupancy_lookup_nans_less_than_week,
    temperature_bins_nans_less_than_week,
):
    return {
        "dec-jan-feb-weighted": caltrack_hourly_fit_feature_processor(
            "dec-jan-feb-weighted",
            segmented_data_nans_less_than_week,
            occupancy_lookup_nans_less_than_week,
            temperature_bins_nans_less_than_week,
            temperature_bins_nans_less_than_week,
        ),
        "apr-may-jun-weighted": caltrack_hourly_fit_feature_processor(
            "apr-may-jun-weighted",
            segmented_data_nans_less_than_week,
            occupancy_lookup_nans_less_than_week,
            temperature_bins_nans_less_than_week,
            temperature_bins_nans_less_than_week,
        ),
    }


@pytest.fixture
def temps_extended():
    index = pd.date_range(start="2017-01-01", periods=168, freq="h", tz="UTC")
    temps = pd.Series(1, index=index)
    return temps


@pytest.mark.parametrize("segment_type", ["three_month_weighted"])
def test_fit_caltrack_hourly_model_nans_less_than_week_fit(
    segmented_design_matrices_nans_less_than_week,
    occupancy_lookup_nans_less_than_week,
    temperature_bins_nans_less_than_week,
    temps_extended,
    segment_type,
):
    segmented_model_results = fit_caltrack_hourly_model(
        segmented_design_matrices_nans_less_than_week,
        occupancy_lookup_nans_less_than_week,
        temperature_bins_nans_less_than_week,
        temperature_bins_nans_less_than_week,
        segment_type=segment_type,
    )

    assert segmented_model_results.model.segment_models is not None
    prediction = segmented_model_results.predict(
        temps_extended.index, temps_extended
    ).result
    assert prediction.shape[0] == 168
    assert prediction.dropna().shape[0] == 4


@pytest.fixture
def segmented_design_matrices_empty_models(
    segmented_data,
    occupancy_lookup,
    occupied_temperature_bins,
    unoccupied_temperature_bins,
):
    return {
        "dec-jan-feb-weighted": caltrack_hourly_fit_feature_processor(
            "dec-jan-feb-weighted",
            segmented_data[:0],
            occupancy_lookup,
            occupied_temperature_bins,
            unoccupied_temperature_bins,
        )
    }


@pytest.mark.parametrize("segment_type", ["three_month_weighted"])
def test_predict_caltrack_hourly_model_empty_models(
    temps,
    segmented_design_matrices_empty_models,
    occupancy_lookup,
    occupied_temperature_bins,
    unoccupied_temperature_bins,
    segment_type,
):
    segmented_model_results = fit_caltrack_hourly_model(
        segmented_design_matrices_empty_models,
        occupancy_lookup,
        occupied_temperature_bins,
        unoccupied_temperature_bins,
        segment_type=segment_type,
    )

    assert segmented_model_results.model.segment_models is not None
    prediction = segmented_model_results.predict(temps.index, temps).result
    assert prediction.shape[0] == 24
    assert prediction.dropna().shape[0] == 0


@pytest.fixture
def occupancy_lookup_zeroes():
    index = pd.Categorical(range(168))
    occupancy = pd.Series([False] * 168, index=index)
    return pd.DataFrame(
        {"dec-jan-feb-weighted": occupancy, "jan-feb-mar-weighted": occupancy}
    )


@pytest.fixture
def segmented_design_matrices_single_mode(
    segmented_data,
    occupancy_lookup_zeroes,
    occupied_temperature_bins,
    unoccupied_temperature_bins,
):
    return {
        "dec-jan-feb-weighted": caltrack_hourly_fit_feature_processor(
            "dec-jan-feb-weighted",
            segmented_data,
            occupancy_lookup_zeroes,
            occupied_temperature_bins,
            unoccupied_temperature_bins,
        )
    }


def test_fit_caltrack_hourly_model_segment_single_mode(
    segmented_design_matrices_single_mode,
):
    segment_name = "dec-jan-feb-weighted"
    segment_data = segmented_design_matrices_single_mode[segment_name]
    segment_model = fit_caltrack_hourly_model_segment(segment_name, segment_data)
    assert segment_model.formula == (
        "meter_value ~ C(hour_of_week) - 1 + bin_0_occupied + bin_1_occupied"
        " + bin_2_occupied + bin_3_occupied + bin_0_unoccupied + bin_1_unoccupied"
        " + bin_2_unoccupied"
    )
    assert segment_model.segment_name == "dec-jan-feb-weighted"
    assert len(segment_model.model_params.keys()) == 31
    assert segment_model.model is not None
    assert segment_model.warnings is not None
    prediction = segment_model.predict(segment_data)
    assert round(prediction.sum(), 2) == 960.0
