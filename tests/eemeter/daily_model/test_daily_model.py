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
import pandas as pd

from opendsm.eemeter import DailyModel
from opendsm.eemeter.models.daily.data import DailyBaselineData, DailyReportingData
from opendsm.eemeter.common.data_settings import DailyDataSettings
from opendsm.eemeter.models.daily.utilities.settings import DailySettings
from opendsm.eemeter.common.exceptions import (
    DataSufficiencyError,
    DisqualifiedModelError,
)

from ..helpers import nonstandard_settings_warnings



def _daily_frame(meter, temperature):
    """Combine a meter frame (column 'value') and a temperature series into a model input frame."""
    meter = meter.rename(columns={"value": "observed"})
    temperature = temperature.tz_convert(meter.index.tz).rename("temperature")

    return pd.concat([meter, temperature], axis=1)


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
def missing_daily_data(bad_daily_series) -> pd.DataFrame:
    meter, temp = bad_daily_series
    meter = meter[:-90]

    return _daily_frame(meter, temp)


@pytest.fixture
def bad_daily_data(bad_daily_series) -> pd.DataFrame:
    meter, temp = bad_daily_series

    return _daily_frame(meter, temp)


@pytest.mark.slow
def test_disqualified_data_error(missing_daily_data, bad_daily_data):
    with pytest.raises(DataSufficiencyError):
        DailyModel().fit(missing_daily_data, is_electricity_data=True)
    model = DailyModel().fit(
        missing_daily_data, is_electricity_data=True, ignore_disqualification=True
    )
    with pytest.raises(DisqualifiedModelError):
        model.predict(bad_daily_data)
    model.predict(missing_daily_data, ignore_disqualification=True)


def test_model_cvrmse_error(bad_daily_data):
    model = DailyModel().fit(bad_daily_data, is_electricity_data=True)
    with pytest.raises(DisqualifiedModelError):
        model.predict(bad_daily_data)
    model.predict(bad_daily_data, ignore_disqualification=True)


def test_timezone_behavior(daily_series):
    # TODO probably move some of this to dataclass tests
    meter, temp = daily_series
    # ensure that meter is using local tz
    assert str(meter.index.tz) == "America/Los_Angeles"
    assert str(temp.index.tz) == "UTC"

    baseline_df = _daily_frame(meter, temp)
    model = DailyModel().fit(baseline_df, is_electricity_data=True)

    # fail when attempting to predict on data with different timezone from baseline
    reporting_no_meter_utc = pd.DataFrame({"temperature": temp})
    assert str(model.baseline_timezone) != str(reporting_no_meter_utc.index.tz)
    with pytest.raises(ValueError):
        model.predict(reporting_no_meter_utc)

    reporting_df = _daily_frame(meter, temp)
    res1 = model.predict(reporting_df)
    reporting_no_meter = pd.DataFrame({"temperature": temp.tz_convert(meter.index.tz)})
    res2 = model.predict(reporting_no_meter)
    assert round((res1["temperature"] - res2["temperature"]).sum(), 2) == 0
    assert round((res1["predicted"] - res2["predicted"]).sum(), 2) == 0


def test_predict_df_matches_input_index(daily_series):
    meter, temp = daily_series
    baseline_df = _daily_frame(meter, temp)
    model = DailyModel().fit(baseline_df, is_electricity_data=True)

    temp = temp.copy()
    temp[temp.index.day > 20] = np.nan
    reporting_df = _daily_frame(meter, temp)
    res = model.predict(reporting_df)
    assert len(res) == len(reporting_df)


def test_daily_predict_before_fit_raises(daily_series):
    """Predicting on an unfitted DailyModel raises RuntimeError, not AttributeError."""
    meter, temp = daily_series
    df = _daily_frame(meter, temp)

    with pytest.raises(RuntimeError, match="must be fit"):
        DailyModel().predict(df)


def test_fit_with_dataframe_and_is_electricity_data_fits(daily_series):
    meter, temp = daily_series
    df = _daily_frame(meter, temp)

    model = DailyModel().fit(df, is_electricity_data=True)

    assert model.is_fitted


def test_fit_rejects_data_class_instance(daily_series):
    meter, temp = daily_series
    df = _daily_frame(meter, temp)
    data = DailyBaselineData(df, is_electricity_data=True)

    with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
        DailyModel().fit(data, is_electricity_data=True)


def test_fit_rejects_positional_is_electricity_data(daily_series):
    meter, temp = daily_series
    df = _daily_frame(meter, temp)

    with pytest.raises(TypeError):
        DailyModel().fit(df, True)


def test_predict_frame_matches_predict_on_built_reporting_data(daily_series):
    """predict(df) reproduces predicting on a separately constructed reporting data object."""
    meter, temp = daily_series
    baseline_df = _daily_frame(meter, temp)
    model = DailyModel().fit(baseline_df, is_electricity_data=True)

    reporting_df = _daily_frame(meter, temp)
    reporting_data = DailyReportingData(
        reporting_df, True, settings=DailyDataSettings()
    )
    expected = model._predict_data(reporting_data)

    result = model.predict(reporting_df)

    np.testing.assert_allclose(
        result["predicted"].values, expected["predicted"].values, atol=1e-9
    )


def test_predict_without_observed_column_works(daily_series):
    meter, temp = daily_series
    baseline_df = _daily_frame(meter, temp)
    model = DailyModel().fit(baseline_df, is_electricity_data=True)

    reporting_df = pd.DataFrame({"temperature": temp.tz_convert(meter.index.tz)})
    result = model.predict(reporting_df)

    assert "predicted" in result.columns
    assert "observed" not in result.columns or result["observed"].isna().all()


def test_predict_does_not_trim_edge_nan_rows(daily_series):
    meter, temp = daily_series
    df = _daily_frame(meter, temp)
    model = DailyModel().fit(df, is_electricity_data=True)

    padded_temp = temp.tz_convert(meter.index.tz)
    padded_temp.iloc[:3] = np.nan
    padded_temp.iloc[-3:] = np.nan
    reporting_df = pd.DataFrame({"temperature": padded_temp})

    result = model.predict(reporting_df)

    assert len(result) == len(reporting_df)


def test_edge_trimmed_frame_fits_same_coefficients_with_warning(daily_series):
    """A baseline frame padded with all-NaN edge rows fits identically and logs one trim warning."""
    meter, temp = daily_series
    df = _daily_frame(meter, temp)
    unpadded_model = DailyModel().fit(df, is_electricity_data=True)

    n_leading, n_trailing = 4, 6
    leading = pd.DataFrame(
        {"observed": np.nan, "temperature": np.nan},
        index=df.index[0] - pd.to_timedelta(np.arange(n_leading, 0, -1), unit="D"),
    )
    trailing = pd.DataFrame(
        {"observed": np.nan, "temperature": np.nan},
        index=df.index[-1] + pd.to_timedelta(np.arange(1, n_trailing + 1), unit="D"),
    )
    padded_df = pd.concat([leading, df, trailing])

    padded_model = DailyModel().fit(padded_df, is_electricity_data=True)

    trim_warnings = [
        w
        for w in padded_model.warnings
        if w.qualified_name == "eemeter.data_quality.edge_rows_trimmed"
    ]
    assert len(trim_warnings) == 1
    assert trim_warnings[0].data == {"leading": n_leading, "trailing": n_trailing}

    for key, submodel in unpadded_model.params.submodels.items():
        padded_submodel = padded_model.params.submodels[key]
        assert padded_submodel.coefficients.intercept == pytest.approx(
            submodel.coefficients.intercept
        )


def test_fit_disqualified_frame_raises_with_matching_disqualification_list():
    """A disqualified baseline frame raises DataSufficiencyError carrying the disqualifications."""
    idx = pd.date_range("2020-01-01", periods=30, freq="D", tz="UTC")
    df = pd.DataFrame(
        {"observed": np.linspace(10, 20, 30), "temperature": np.linspace(30, 90, 30)},
        index=idx,
    )

    model = DailyModel()
    with pytest.raises(DataSufficiencyError) as exc_info:
        model.fit(df, is_electricity_data=True)

    assert len(exc_info.value.disqualification) > 0
    assert exc_info.value.disqualification == model.disqualification


def test_json_daily(comstock_daily, baseline_model_daily):
    _, df_r = comstock_daily
    metered_savings_dataframe = baseline_model_daily.predict(df_r.reset_index())
    total_metered_savings = (
        metered_savings_dataframe["observed"] - metered_savings_dataframe["predicted"]
    ).sum()

    json_str = baseline_model_daily.to_json()
    loaded_model = DailyModel.from_json(json_str)
    prediction_json = loaded_model.predict(df_r.reset_index())
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
    model = DailyModel().fit(
        df_b.reset_index(), is_electricity_data=True, ignore_disqualification=True
    )

    return model


@pytest.fixture(scope="module")
def nonstandard_daily_fit(comstock_daily):
    """(model, log_text): a DailyModel fit with segment_minimum_count=8, and the text of the
    records the 'eemeter' logger emitted during that fit."""
    df_b, _ = comstock_daily
    handler = logging.handlers.BufferingHandler(capacity=1000)
    logger = logging.getLogger("eemeter")
    logger.addHandler(handler)
    try:
        model = DailyModel(settings={"segment_minimum_count": 8}).fit(
            df_b.reset_index(), is_electricity_data=True, ignore_disqualification=True
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
    baseline_model = DailyModel(settings={"preset": "legacy"}).fit(
        df_b.reset_index(), is_electricity_data=True, ignore_disqualification=True
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
    baseline_model = DailyModel.from_2_0_json(serialized_str, is_electricity_data=True)

    _, df_r = comstock_daily
    df_r = df_r.copy()
    df_r.index = df_r.index.tz_convert("UTC")
    metered_savings_dataframe = baseline_model.predict(df_r.reset_index())
    total_metered_savings = (
        metered_savings_dataframe["observed"] - metered_savings_dataframe["predicted"]
    ).sum()

    assert round(float(total_metered_savings), 2) == snapshot(name="total_metered_savings")


def test_meter_fact_round_trips_through_to_dict_from_dict(default_fitted_daily_model):
    model_dict = default_fitted_daily_model.to_dict()
    rebuilt = DailyModel.from_dict(model_dict)

    assert rebuilt._is_electricity_data == default_fitted_daily_model._is_electricity_data


def test_from_dict_requires_keyword_when_payload_lacks_meter_fact(
    default_fitted_daily_model, comstock_daily
):
    model_dict = default_fitted_daily_model.to_dict()
    del model_dict["info"]["is_electricity_data"]

    with pytest.raises(ValueError):
        DailyModel.from_dict(model_dict)

    rebuilt = DailyModel.from_dict(model_dict, is_electricity_data=True)
    prior_format = [
        w for w in rebuilt.warnings if w.qualified_name == "eemeter.serialization.prior_format"
    ]
    assert len(prior_format) == 1

    _, df_r = comstock_daily
    result = rebuilt.predict(df_r.reset_index())
    assert "predicted" in result.columns


def test_from_dict_rejects_keyword_when_payload_carries_meter_fact(default_fitted_daily_model):
    model_dict = default_fitted_daily_model.to_dict()

    with pytest.raises(ValueError):
        DailyModel.from_dict(model_dict, is_electricity_data=True)


def test_daily_settings_data_block_has_sufficiency_defaults():
    settings = DailySettings()

    assert settings.data.sufficiency.min_baseline_length == np.ceil(0.9 * 365)


def test_data_mapping_on_settings_instance_builds_daily_data_settings():
    settings = DailySettings(data={"sufficiency": {"min_baseline_length": 30}})

    assert isinstance(settings.data, DailyDataSettings)
    assert settings.data.sufficiency.min_baseline_length == 30


def test_sufficiency_settings_survive_to_dict_round_trip(comstock_daily):
    df_b, _ = comstock_daily
    model = DailyModel(settings={"data": {"sufficiency": {"min_baseline_length": 100}}}).fit(
        df_b.reset_index(), is_electricity_data=True, ignore_disqualification=True
    )

    rebuilt = DailyModel.from_dict(model.to_dict())

    assert rebuilt.settings.data.sufficiency.min_baseline_length == 100


def test_changed_min_baseline_length_fires_nonstandard_settings_warning(comstock_daily):
    df_b, _ = comstock_daily
    model = DailyModel(settings={"data": {"sufficiency": {"min_baseline_length": 100}}}).fit(
        df_b.reset_index(), is_electricity_data=True, ignore_disqualification=True
    )

    assert nonstandard_settings_warnings(model.warnings) != []


def test_changed_requested_start_does_not_fire_nonstandard_settings_warning(comstock_daily):
    df_b, _ = comstock_daily
    model = DailyModel(
        settings={"data": {"sufficiency": {"requested_start": df_b.index.min()}}}
    ).fit(df_b.reset_index(), is_electricity_data=True, ignore_disqualification=True)

    assert nonstandard_settings_warnings(model.warnings) == []


def test_baseline_df_is_none_before_fit(daily_series):
    assert DailyModel().baseline_df is None


def test_baseline_df_is_dataframe_after_fit(daily_series):
    meter, temp = daily_series
    df = _daily_frame(meter, temp)

    model = DailyModel().fit(df, is_electricity_data=True)

    assert isinstance(model.baseline_df, pd.DataFrame)


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
