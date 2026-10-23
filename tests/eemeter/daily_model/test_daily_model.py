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
import logging.handlers

import pytest

import numpy as np

from opendsm.eemeter import DailyModel, DailyBaselineData, DailyReportingData
from opendsm.eemeter.common.exceptions import (
    DataSufficiencyError,
    DisqualifiedModelError,
)

from ..helpers import nonstandard_settings_warnings



@pytest.fixture
def daily_series(comstock_daily):
    """(meter_df, temperature_series) extracted from ComStock daily baseline."""
    df_b, _ = comstock_daily
    meter = df_b[["observed"]].rename(columns={"observed": "value"}).copy()
    meter.index = meter.index.tz_convert("America/Los_Angeles")
    temp = df_b["temperature"].copy()
    temp.index = temp.index.tz_convert("UTC")

    return meter, temp


@pytest.fixture
def bad_daily_series(daily_series):
    meter, temp = daily_series
    meter.iloc[:50] += meter["value"].median() * 50

    return meter, temp


@pytest.fixture
def missing_daily_data(bad_daily_series) -> DailyBaselineData:
    meter, temp = bad_daily_series
    meter = meter[:-90]
    baseline_data = DailyBaselineData.from_series(meter, temp, is_electricity_data=True)

    return baseline_data


@pytest.fixture
def bad_daily_data(bad_daily_series) -> DailyBaselineData:
    meter, temp = bad_daily_series
    baseline_data = DailyBaselineData.from_series(meter, temp, is_electricity_data=True)

    return baseline_data


@pytest.mark.slow
def test_disqualified_data_error(missing_daily_data):
    with pytest.raises(DataSufficiencyError):
        model = DailyModel().fit(missing_daily_data)
    model = DailyModel().fit(missing_daily_data, ignore_disqualification=True)
    with pytest.raises(DisqualifiedModelError):
        model.predict(bad_daily_data)
    model.predict(missing_daily_data, ignore_disqualification=True)


def test_model_cvrmse_error(bad_daily_data):
    model = DailyModel().fit(bad_daily_data)
    with pytest.raises(DisqualifiedModelError):
        model.predict(bad_daily_data)
    model.predict(bad_daily_data, ignore_disqualification=True)


def test_timezone_behavior(daily_series):
    # TODO probably move some of this to dataclass tests
    meter, temp = daily_series
    # ensure that meter is using local tz
    assert str(meter.index.tz) == "America/Los_Angeles"
    assert str(temp.index.tz) == "UTC"

    baseline_data = DailyBaselineData.from_series(meter, temp, is_electricity_data=True)

    # require is_electricity_data flag when passing meter data
    with pytest.raises(ValueError):
        DailyReportingData.from_series(meter, temp)

    # fail when passing timezone both through index as well as param
    with pytest.raises(ValueError):
        DailyReportingData.from_series(meter, temp, tzinfo=meter.index.tz)

    model = DailyModel().fit(baseline_data)

    # fail when attempting to predict on data with different timezone from baseline
    reporting_data_no_meter_utc = DailyReportingData.from_series(None, temp)
    assert model.baseline_timezone != reporting_data_no_meter_utc.tz
    with pytest.raises(ValueError):
        model.predict(reporting_data_no_meter_utc)

    reporting_data = DailyReportingData.from_series(
        meter, temp, is_electricity_data=True
    )
    res1 = model.predict(reporting_data)
    reporting_data_no_meter = DailyReportingData.from_series(
        None, temp, tzinfo=meter.index.tz
    )
    res2 = model.predict(reporting_data_no_meter)
    assert round((res1["temperature"] - res2["temperature"]).sum(), 2) == 0
    assert round((res1["predicted"] - res2["predicted"]).sum(), 2) == 0


def test_predict_df_matches_input_index(daily_series):
    meter, temp = daily_series
    baseline_data = DailyBaselineData.from_series(meter, temp, is_electricity_data=True)
    baseline_model = DailyModel().fit(baseline_data)

    temp[temp.index.day > 20] = np.nan
    reporting_data_missing_temp = DailyBaselineData.from_series(
        meter, temp, is_electricity_data=True
    )
    res = baseline_model.predict(reporting_data_missing_temp)
    assert len(res) == len(reporting_data_missing_temp.df)


def test_daily_predict_before_fit_raises(daily_series):
    """Predicting on an unfitted DailyModel raises RuntimeError, not AttributeError."""
    meter, temp = daily_series
    data = DailyBaselineData.from_series(meter, temp, is_electricity_data=True)

    with pytest.raises(RuntimeError, match="must be fit"):
        DailyModel().predict(data)


def test_json_daily(baseline_model_daily, reporting_data_daily):
    metered_savings_dataframe = baseline_model_daily.predict(reporting_data_daily)
    total_metered_savings = (
        metered_savings_dataframe["observed"] - metered_savings_dataframe["predicted"]
    ).sum()

    json_str = baseline_model_daily.to_json()
    loaded_model = DailyModel.from_json(json_str)
    prediction_json = loaded_model.predict(reporting_data_daily)
    total_metered_savings_loaded = (
        prediction_json["observed"] - prediction_json["predicted"]
    ).sum()

    assert total_metered_savings == total_metered_savings_loaded


def test_daily_model_rejects_settings_of_another_type():
    with pytest.raises(TypeError, match="must be None, a dict, or a DailySettings"):
        DailyModel(settings="legacy")


@pytest.fixture(scope="module")
def default_fitted_daily_model(comstock_daily):
    df_b, _ = comstock_daily
    baseline_data = DailyBaselineData(df=df_b.reset_index(), is_electricity_data=True)

    return DailyModel().fit(baseline_data, ignore_disqualification=True)


@pytest.fixture(scope="module")
def nonstandard_daily_fit(comstock_daily):
    """(model, log_text): a DailyModel fit with segment_minimum_count=8, and the text of the
    records the 'eemeter' logger emitted during that fit."""
    df_b, _ = comstock_daily
    baseline_data = DailyBaselineData(df=df_b.reset_index(), is_electricity_data=True)
    handler = logging.handlers.BufferingHandler(capacity=1000)
    logger = logging.getLogger("eemeter")
    logger.addHandler(handler)
    try:
        model = DailyModel(settings={"segment_minimum_count": 8}).fit(
            baseline_data, ignore_disqualification=True
        )
    finally:
        logger.removeHandler(handler)

    log_text = "\n".join(record.getMessage() for record in handler.buffer)

    return model, log_text


@pytest.mark.slow
def test_default_settings_fit_has_no_nonstandard_settings_warning(default_fitted_daily_model):
    assert nonstandard_settings_warnings(default_fitted_daily_model.warnings) == []


@pytest.mark.slow
def test_nonstandard_setting_fit_logs_the_deviation(nonstandard_daily_fit):
    _, log_text = nonstandard_daily_fit

    assert "segment_minimum_count" in log_text


@pytest.mark.slow
def test_nonstandard_setting_fit_carries_deviation_warning(nonstandard_daily_fit):
    baseline_model, _ = nonstandard_daily_fit

    matching = nonstandard_settings_warnings(baseline_model.warnings)
    assert len(matching) == 1

    deviation = matching[0].data["deviations"]["segment_minimum_count"]
    assert deviation["value"] == 8
    assert deviation["default"] == 6


@pytest.mark.slow
def test_legacy_preset_fit_has_no_nonstandard_settings_warning(comstock_daily):
    df_b, _ = comstock_daily
    baseline_data = DailyBaselineData(df=df_b.reset_index(), is_electricity_data=True)
    baseline_model = DailyModel(settings={"preset": "legacy"}).fit(
        baseline_data, ignore_disqualification=True
    )

    assert nonstandard_settings_warnings(baseline_model.warnings) == []


@pytest.mark.slow
def test_nonstandard_settings_warning_survives_to_dict_round_trip(nonstandard_daily_fit):
    baseline_model, _ = nonstandard_daily_fit

    rebuilt = DailyModel.from_dict(baseline_model.to_dict())

    matching = nonstandard_settings_warnings(rebuilt.warnings)
    assert len(matching) == 1
    deviation = matching[0].data["deviations"]["segment_minimum_count"]
    assert deviation == {"value": 8, "default": 6}


@pytest.mark.slow
def test_from_dict_loads_payload_without_preset_as_current_with_deviations(
    default_fitted_daily_model,
):
    """A payload whose settings carry legacy values and no `preset` key loads under the
    current preset, keeps every stored value, and reports each legacy value against the
    current default."""
    legacy_settings = {
        "developer_mode": False,
        "silent_developer_mode": False,
        "algorithm_choice": "nlopt_sbplx",
        "initial_guess_algorithm_choice": "nlopt_direct",
        "full_model": "hdd_tidd_cdd",
        "allow_smooth_model": False,
        "alpha_minimum": -100.0,
        "alpha_selection": 2.0,
        "alpha_final_type": "last",
        "alpha_final": 2.0,
        "final_bounds_scalar": 1.0,
        "regularization_alpha": 0.001,
        "regularization_percent_lasso": 1.0,
        "segment_minimum_count": 10,
        "maximum_slope_oom_scalar": 2.0,
        "initial_step_percentage": 0.1,
        "split_selection": {
            "criteria": "bic",
            "penalty_multiplier": 0.24,
            "penalty_power": 2.061,
            "allow_separate_summer": False,
            "allow_separate_shoulder": False,
            "allow_separate_winter": False,
            "allow_separate_weekday_weekend": False,
            "reduce_splits_by_gaussian": False,
            "reduce_splits_num_std": None,
        },
        "season": {
            "january": "winter",
            "february": "winter",
            "march": "shoulder",
            "april": "shoulder",
            "may": "shoulder",
            "june": "summer",
            "july": "summer",
            "august": "summer",
            "september": "summer",
            "october": "shoulder",
            "november": "winter",
            "december": "winter",
            "options": ["summer", "shoulder", "winter"],
        },
        "weekday_weekend": {
            "monday": "weekday",
            "tuesday": "weekday",
            "wednesday": "weekday",
            "thursday": "weekday",
            "friday": "weekday",
            "saturday": "weekend",
            "sunday": "weekend",
            "options": ["weekday", "weekend"],
        },
        "uncertainty_alpha": 0.1,
        "cvrmse_threshold": 1.0,
        "pnrmse_threshold": 1.6,
    }
    model_dict = default_fitted_daily_model.to_dict()
    model_dict["settings"] = legacy_settings

    model = DailyModel.from_dict(model_dict)

    settings = model.settings
    assert settings.preset == "current"
    assert settings.allow_smooth_model is False
    assert settings.alpha_final == 2.0
    assert settings.segment_minimum_count == 10
    assert settings.split_selection.allow_separate_summer is False
    assert settings.split_selection.allow_separate_shoulder is False
    assert settings.split_selection.allow_separate_winter is False
    assert settings.split_selection.allow_separate_weekday_weekend is False
    assert settings.split_selection.reduce_splits_by_gaussian is False
    assert settings.split_selection.reduce_splits_num_std is None
    expected_deviations = {
        "allow_smooth_model": {"value": False, "default": True},
        "alpha_final": {"value": 2.0, "default": "adaptive"},
        "segment_minimum_count": {"value": 10, "default": 6},
        "split_selection.allow_separate_summer": {"value": False, "default": True},
        "split_selection.allow_separate_shoulder": {"value": False, "default": True},
        "split_selection.allow_separate_winter": {"value": False, "default": True},
        "split_selection.allow_separate_weekday_weekend": {"value": False, "default": True},
        "split_selection.reduce_splits_by_gaussian": {"value": False, "default": True},
        "split_selection.reduce_splits_num_std": {"value": None, "default": [1.4, 0.89]},
    }
    assert model.settings_deviations == expected_deviations


def test_legacy_deserialization_daily(comstock_daily, snapshot):
    legacy_model_dict = {
        "model_type": "hdd_only",
        "formula": "meter_value ~ hdd_46",
        "status": "QUALIFIED",
        "model_params": {"intercept": 12, "beta_hdd": 2, "heating_balance_point": 50},
        "r_squared_adj": 0.3,
        "warnings": [],
    }
    serialized_str = json.dumps(legacy_model_dict)
    baseline_model = DailyModel.from_2_0_json(serialized_str)

    _, df_r = comstock_daily
    df_r = df_r.copy()
    df_r.index = df_r.index.tz_convert("UTC")
    reporting_data = DailyReportingData(df=df_r.reset_index(), is_electricity_data=True)
    metered_savings_dataframe = baseline_model.predict(reporting_data)
    total_metered_savings = (
        metered_savings_dataframe["observed"] - metered_savings_dataframe["predicted"]
    ).sum()

    assert round(float(total_metered_savings), 2) == snapshot(name="total_metered_savings")


@pytest.mark.slow
def test_settings_deviations_property_matches_the_warning_payload(nonstandard_daily_fit):
    """The `settings_deviations` property equals the nonstandard-settings warning's
    `deviations` payload."""
    model, _ = nonstandard_daily_fit

    (warning,) = nonstandard_settings_warnings(model.warnings)

    assert model.settings_deviations == warning.data["deviations"]


@pytest.mark.slow
def test_settings_deviations_property_is_empty_for_default_settings(default_fitted_daily_model):
    assert default_fitted_daily_model.settings_deviations == {}


@pytest.mark.slow
def test_settings_deviations_property_recomputes_after_a_round_trip(nonstandard_daily_fit):
    """A model rebuilt through `to_dict`/`from_dict` reports the same settings deviations
    as the original."""
    model, _ = nonstandard_daily_fit

    rebuilt = DailyModel.from_dict(model.to_dict())

    assert rebuilt.settings_deviations == model.settings_deviations
    assert rebuilt.settings_deviations["segment_minimum_count"]["value"] == 8


def test_settings_deviations_property_is_available_before_fit():
    model = DailyModel(settings={"segment_minimum_count": 8})

    assert model.settings_deviations["segment_minimum_count"]["default"] == 6
