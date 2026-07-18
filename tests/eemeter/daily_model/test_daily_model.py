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
    meter.index = meter.index.tz_convert("US/Pacific")
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
    assert str(meter.index.tz) == "US/Pacific"
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