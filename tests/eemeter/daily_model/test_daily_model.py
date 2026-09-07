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

from opendsm.eemeter import DailyModel, DailyBaselineData, DailyReportingData
from opendsm.eemeter.common.exceptions import (
    DataSufficiencyError,
    DisqualifiedModelError,
)



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


def test_json_daily(comstock_daily):
    df_b, df_r = comstock_daily
    baseline_data = DailyBaselineData(df=df_b.reset_index(), is_electricity_data=True)
    baseline_model = DailyModel().fit(baseline_data, ignore_disqualification=True)

    reporting_data = DailyReportingData(df=df_r.reset_index(), is_electricity_data=True)
    metered_savings_dataframe = baseline_model.predict(reporting_data)
    total_metered_savings = (
        metered_savings_dataframe["observed"] - metered_savings_dataframe["predicted"]
    ).sum()

    json_str = baseline_model.to_json()
    loaded_model = DailyModel.from_json(json_str)
    prediction_json = loaded_model.predict(reporting_data)
    total_metered_savings_loaded = (
        prediction_json["observed"] - prediction_json["predicted"]
    ).sum()

    assert total_metered_savings == total_metered_savings_loaded


def test_to_dict_tags_model_type_daily(comstock_daily):
    """to_dict tags a daily payload with model_type='daily', and from_dict
    tolerates the tag on a round trip."""
    df_b, _ = comstock_daily
    baseline_data = DailyBaselineData(df=df_b.reset_index(), is_electricity_data=True)
    baseline_model = DailyModel().fit(baseline_data, ignore_disqualification=True)

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
    baseline_data = DailyBaselineData(df=df_b.reset_index(), is_electricity_data=True)

    return DailyModel().fit(baseline_data, ignore_disqualification=True)


def _has_nonstandard_settings_warning(warnings):
    return any(w.qualified_name == "eemeter.settings.nonstandard" for w in warnings)


def test_default_settings_fit_has_no_nonstandard_settings_warning(default_fitted_daily_model):
    assert not _has_nonstandard_settings_warning(default_fitted_daily_model.warnings)


def test_nonstandard_setting_fit_carries_deviation_warning(comstock_daily, caplog):
    df_b, _ = comstock_daily
    baseline_data = DailyBaselineData(df=df_b.reset_index(), is_electricity_data=True)
    with caplog.at_level("WARNING"):
        baseline_model = DailyModel(settings={"segment_minimum_count": 8}).fit(
            baseline_data, ignore_disqualification=True
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
    baseline_data = DailyBaselineData(df=df_b.reset_index(), is_electricity_data=True)
    baseline_model = DailyModel(settings={"preset": "legacy"}).fit(
        baseline_data, ignore_disqualification=True
    )

    assert not _has_nonstandard_settings_warning(baseline_model.warnings)


def test_nonstandard_settings_warning_survives_to_dict_round_trip(comstock_daily):
    df_b, _ = comstock_daily
    baseline_data = DailyBaselineData(df=df_b.reset_index(), is_electricity_data=True)
    baseline_model = DailyModel(settings={"segment_minimum_count": 8}).fit(
        baseline_data, ignore_disqualification=True
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