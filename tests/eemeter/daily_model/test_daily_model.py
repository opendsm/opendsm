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


def test_json_daily(comstock_daily):
    df_b, df_r = comstock_daily
    baseline_model = DailyModel().fit(
        df_b.reset_index(), is_electricity_data=True, ignore_disqualification=True
    )

    metered_savings_dataframe = baseline_model.predict(df_r.reset_index())
    total_metered_savings = (
        metered_savings_dataframe["observed"] - metered_savings_dataframe["predicted"]
    ).sum()

    json_str = baseline_model.to_json()
    loaded_model = DailyModel.from_json(json_str)
    prediction_json = loaded_model.predict(df_r.reset_index())
    total_metered_savings_loaded = (
        prediction_json["observed"] - prediction_json["predicted"]
    ).sum()

    assert total_metered_savings == total_metered_savings_loaded


def test_to_dict_tags_model_type_daily(comstock_daily):
    """to_dict tags a daily payload with model_type='daily', and from_dict
    tolerates the tag on a round trip."""
    df_b, _ = comstock_daily
    baseline_model = DailyModel().fit(
        df_b.reset_index(), is_electricity_data=True, ignore_disqualification=True
    )

    model_dict = baseline_model.to_dict()

    assert model_dict["model_type"] == "daily"

    rebuilt = DailyModel.from_dict(model_dict)

    assert rebuilt.to_dict()["model_type"] == "daily"


def test_daily_model_rejects_legacy_model_kwarg():
    with pytest.raises(TypeError):
        DailyModel(model="legacy")


@pytest.fixture(scope="module")
def default_fitted_daily_model(comstock_daily):
    df_b, _ = comstock_daily
    model = DailyModel().fit(
        df_b.reset_index(), is_electricity_data=True, ignore_disqualification=True
    )

    return model


def _has_nonstandard_settings_warning(warnings):
    return any(w.qualified_name == "eemeter.settings.nonstandard" for w in warnings)


def test_default_settings_fit_has_no_nonstandard_settings_warning(default_fitted_daily_model):
    assert not _has_nonstandard_settings_warning(default_fitted_daily_model.warnings)


def test_nonstandard_setting_fit_carries_deviation_warning(comstock_daily, caplog):
    df_b, _ = comstock_daily
    with caplog.at_level("WARNING"):
        baseline_model = DailyModel(settings={"segment_minimum_count": 8}).fit(
            df_b.reset_index(), is_electricity_data=True, ignore_disqualification=True
        )
    assert "segment_minimum_count" in caplog.text

    matching = [
        w
        for w in baseline_model.warnings
        if w.qualified_name == "eemeter.settings.nonstandard"
    ]
    assert len(matching) == 1

    deviation = matching[0].data["deviations"]["segment_minimum_count"]
    assert deviation["value"] == 8
    assert deviation["default"] == 6


def test_legacy_preset_fit_has_no_nonstandard_settings_warning(comstock_daily):
    df_b, _ = comstock_daily
    baseline_model = DailyModel(settings={"preset": "legacy"}).fit(
        df_b.reset_index(), is_electricity_data=True, ignore_disqualification=True
    )

    assert not _has_nonstandard_settings_warning(baseline_model.warnings)


def test_nonstandard_settings_warning_survives_to_dict_round_trip(comstock_daily):
    df_b, _ = comstock_daily
    baseline_model = DailyModel(settings={"segment_minimum_count": 8}).fit(
        df_b.reset_index(), is_electricity_data=True, ignore_disqualification=True
    )

    rebuilt = DailyModel.from_dict(baseline_model.to_dict())

    assert _has_nonstandard_settings_warning(rebuilt.warnings)

    matching = [
        w
        for w in rebuilt.warnings
        if w.qualified_name == "eemeter.settings.nonstandard"
    ]
    deviation = matching[0].data["deviations"]["segment_minimum_count"]
    assert deviation == {"value": 8, "default": 6}


def test_from_dict_accepts_developer_mode_settings_without_preset(default_fitted_daily_model):
    model_dict = default_fitted_daily_model.to_dict()
    model_dict["settings"]["developer_mode"] = True
    del model_dict["settings"]["preset"]

    DailyModel.from_dict(model_dict)


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

    assert _has_nonstandard_settings_warning(model.warnings)


def test_changed_requested_start_does_not_fire_nonstandard_settings_warning(comstock_daily):
    df_b, _ = comstock_daily
    model = DailyModel(
        settings={"data": {"sufficiency": {"requested_start": df_b.index.min()}}}
    ).fit(df_b.reset_index(), is_electricity_data=True, ignore_disqualification=True)

    assert not _has_nonstandard_settings_warning(model.warnings)


def test_baseline_df_is_none_before_fit(daily_series):
    assert DailyModel().baseline_df is None


def test_baseline_df_is_dataframe_after_fit(daily_series):
    meter, temp = daily_series
    df = _daily_frame(meter, temp)

    model = DailyModel().fit(df, is_electricity_data=True)

    assert isinstance(model.baseline_df, pd.DataFrame)


def test_settings_deviations_property_matches_the_warning_payload(comstock_daily):
    """The property and the warning report the same deviation, from one formatter."""
    df_b, _ = comstock_daily
    model = DailyModel(settings={"segment_minimum_count": 8}).fit(
        df_b.reset_index(), is_electricity_data=True, ignore_disqualification=True
    )

    warning = next(
        w for w in model.warnings if w.qualified_name == "eemeter.settings.nonstandard"
    )

    assert model.settings_deviations == warning.data["deviations"]


def test_settings_deviations_property_is_empty_for_default_settings(default_fitted_daily_model):
    assert default_fitted_daily_model.settings_deviations == {}


def test_settings_deviations_property_recomputes_after_a_round_trip(comstock_daily):
    """The property is derived from the model's own settings, so deserialization
    restores it without storing it."""
    df_b, _ = comstock_daily
    model = DailyModel(settings={"segment_minimum_count": 8}).fit(
        df_b.reset_index(), is_electricity_data=True, ignore_disqualification=True
    )

    rebuilt = DailyModel.from_dict(model.to_dict())

    assert rebuilt.settings_deviations == model.settings_deviations
    assert rebuilt.settings_deviations["segment_minimum_count"]["value"] == 8


def test_settings_deviations_property_is_available_before_fit():
    model = DailyModel(settings={"segment_minimum_count": 8})

    assert model.settings_deviations["segment_minimum_count"]["default"] == 6
