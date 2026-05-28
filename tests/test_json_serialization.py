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

from opendsm.eemeter import (
    DailyBaselineData,
    DailyReportingData,
    DailyModel,
    BillingBaselineData,
    BillingReportingData,
    BillingModel,
    HourlyCaltrackModel,
    HourlyCaltrackBaselineData,
    HourlyCaltrackReportingData,
)


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


def test_json_billing(comstock_monthly):
    df_b, df_r = comstock_monthly
    baseline_data = BillingBaselineData(df=df_b.reset_index(), is_electricity_data=True)
    baseline_model = BillingModel().fit(baseline_data, ignore_disqualification=True)

    reporting_data = BillingReportingData(df=df_r.reset_index(), is_electricity_data=True)
    metered_savings_dataframe = baseline_model.predict(reporting_data)
    total_metered_savings = (
        metered_savings_dataframe["observed"] - metered_savings_dataframe["predicted"]
    ).sum()

    json_str = baseline_model.to_json()
    loaded_model = BillingModel.from_json(json_str)
    prediction_json = loaded_model.predict(reporting_data)
    total_metered_savings_loaded = (
        prediction_json["observed"] - prediction_json["predicted"]
    ).sum()

    assert total_metered_savings == total_metered_savings_loaded


def test_json_hourly_with_zeros(comstock_hourly):
    df_b, _ = comstock_hourly
    meter = df_b[["observed"]].rename(columns={"observed": "value"}).copy()
    meter["value"] = 0
    temperature = df_b["temperature"]

    baseline = HourlyCaltrackBaselineData.from_series(meter, temperature, is_electricity_data=True)
    assert baseline.df["observed"].isnull().all()
    reporting = HourlyCaltrackReportingData.from_series(meter, temperature, is_electricity_data=True)
    assert reporting.df["observed"].isnull().all()


def test_json_caltrack_hourly(comstock_hourly):
    df_b, df_r = comstock_hourly
    meter_b = df_b[["observed"]].rename(columns={"observed": "value"}).copy()
    meter_r = df_r[["observed"]].rename(columns={"observed": "value"}).copy()
    temperature = df_b["temperature"]

    baseline = HourlyCaltrackBaselineData.from_series(meter_b, temperature, is_electricity_data=True)
    baseline_model = HourlyCaltrackModel().fit(baseline)

    reporting = HourlyCaltrackReportingData.from_series(meter_r, df_r["temperature"], is_electricity_data=True)
    result1 = baseline_model.predict(reporting)

    json_str = baseline_model.to_json()
    m = HourlyCaltrackModel.from_json(json_str)
    result2 = m.predict(reporting)

    assert result1["predicted"].sum() == result2["predicted"].sum()
    assert (
        baseline_model.model.totals_metrics["dec-jan-feb-weighted"].observed_length
        == m.model.totals_metrics["dec-jan-feb-weighted"].observed_length
    )


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


def test_legacy_deserialization_hourly(request, comstock_hourly, snapshot):
    with open(request.fspath.dirname + "/legacy_hourly.json", "r") as f:
        legacy_str = f.read()
    baseline_model = HourlyCaltrackModel.from_2_0_json(legacy_str)

    _, df_r = comstock_hourly
    meter = df_r[["observed"]].rename(columns={"observed": "value"}).copy()
    temperature = df_r["temperature"]

    reporting = HourlyCaltrackReportingData.from_series(meter, temperature, is_electricity_data=True)
    metered_savings_dataframe = baseline_model.predict(reporting)
    total_metered_savings = (
        metered_savings_dataframe["observed"] - metered_savings_dataframe["predicted"]
    ).sum()

    assert round(float(total_metered_savings), 2) == snapshot(name="total_metered_savings")
