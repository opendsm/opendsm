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

from opendsm.eemeter.samples import load_sample
from opendsm.eemeter.common.transform import get_baseline_data, get_reporting_data
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


def test_json_daily():
    meter_data, temperature_data, sample_metadata = load_sample(
        "il-electricity-cdd-hdd-daily"
    )

    blackout_start_date = sample_metadata["blackout_start_date"]
    blackout_end_date = sample_metadata["blackout_end_date"]

    # fit baseline model
    baseline_meter_data, warnings = get_baseline_data(
        meter_data, end=blackout_start_date, max_days=365
    )
    baseline_data = DailyBaselineData.from_series(
        baseline_meter_data, temperature_data, is_electricity_data=True
    )
    baseline_model = DailyModel().fit(baseline_data, ignore_disqualification=True)

    # predict on reporting year and calculate savings
    reporting_meter_data, warnings = get_reporting_data(
        meter_data, start=blackout_end_date, max_days=365
    )
    reporting_data = DailyReportingData.from_series(
        reporting_meter_data, temperature_data, is_electricity_data=True
    )
    metered_savings_dataframe = baseline_model.predict(reporting_data)
    total_metered_savings = (
        metered_savings_dataframe["observed"] - metered_savings_dataframe["predicted"]
    ).sum()

    # serialize, deserialize model
    json_str = baseline_model.to_json()
    loaded_model = DailyModel.from_json(json_str)

    # compute metered savings from the loaded model
    prediction_json = loaded_model.predict(reporting_data)
    total_metered_savings_loaded = (
        prediction_json["observed"] - prediction_json["predicted"]
    ).sum()

    # compare results
    assert total_metered_savings == total_metered_savings_loaded


def test_json_billing():
    meter_data, temperature_data, sample_metadata = load_sample(
        "il-electricity-cdd-hdd-billing_monthly"
    )

    blackout_start_date = sample_metadata["blackout_start_date"]
    blackout_end_date = sample_metadata["blackout_end_date"]

    # fit baseline model
    baseline_meter_data, warnings = get_baseline_data(
        meter_data, end=blackout_start_date, max_days=365
    )
    baseline_data = BillingBaselineData.from_series(
        baseline_meter_data, temperature_data, is_electricity_data=True
    )
    baseline_model = BillingModel().fit(baseline_data, ignore_disqualification=True)

    # predict on reporting year and calculate savings
    reporting_meter_data, warnings = get_reporting_data(
        meter_data, start=blackout_end_date, max_days=365
    )
    reporting_data = BillingReportingData.from_series(
        reporting_meter_data, temperature_data, is_electricity_data=True
    )
    metered_savings_dataframe = baseline_model.predict(reporting_data)
    total_metered_savings = (
        metered_savings_dataframe["observed"] - metered_savings_dataframe["predicted"]
    ).sum()

    # serialize, deserialize model
    json_str = baseline_model.to_json()
    loaded_model = BillingModel.from_json(json_str)

    # compute metered savings from the loaded model
    prediction_json = loaded_model.predict(reporting_data)
    total_metered_savings_loaded = (
        prediction_json["observed"] - prediction_json["predicted"]
    ).sum()

    # compare results
    assert total_metered_savings == total_metered_savings_loaded


def test_json_hourly_with_zeros():
    meter_data, temperature_data, sample_metadata = load_sample(
        "il-electricity-cdd-hdd-hourly"
    )
    meter_data["value"] = 0
    baseline = HourlyCaltrackBaselineData.from_series(
        meter_data, temperature_data, is_electricity_data=True
    )
    assert baseline.df["observed"].isnull().all()
    reporting = HourlyCaltrackReportingData.from_series(
        meter_data, temperature_data, is_electricity_data=True
    )
    assert reporting.df["observed"].isnull().all()


def test_json_caltrack_hourly():
    meter_data, temperature_data, sample_metadata = load_sample(
        "il-electricity-cdd-hdd-hourly"
    )

    blackout_start_date = sample_metadata["blackout_start_date"]
    blackout_end_date = sample_metadata["blackout_end_date"]

    # get meter data suitable for fitting a baseline model
    baseline_meter_data, warnings = get_baseline_data(
        meter_data, end=blackout_start_date, max_days=365
    )
    baseline = HourlyCaltrackBaselineData.from_series(
        baseline_meter_data, temperature_data, is_electricity_data=True
    )

    # build a CalTRACK hourly model
    baseline_model = HourlyCaltrackModel().fit(baseline)

    # get a year of reporting period data
    reporting_meter_data, warnings = get_reporting_data(
        meter_data, start=blackout_end_date, max_days=365
    )
    reporting = HourlyCaltrackReportingData.from_series(
        reporting_meter_data, temperature_data, is_electricity_data=True
    )

    result1 = baseline_model.predict(reporting)

    # serialize, deserialize
    json_str = baseline_model.to_json()
    m = HourlyCaltrackModel.from_json(json_str)

    result2 = m.predict(reporting)

    assert result1["predicted"].sum() == result2["predicted"].sum()

    # Check that model metrics are properly serialized/serialized
    assert (
        baseline_model.model.totals_metrics["dec-jan-feb-weighted"].observed_length
        == m.model.totals_metrics["dec-jan-feb-weighted"].observed_length
    )


def test_legacy_deserialization_daily():
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

    meter_data, temperature_data, sample_metadata = load_sample(
        "il-electricity-cdd-hdd-daily"
    )
    blackout_end_date = sample_metadata["blackout_end_date"]

    # predict on reporting year and calculate savings
    reporting_meter_data, warnings = get_reporting_data(
        meter_data, start=blackout_end_date, max_days=365
    )
    reporting_data = DailyReportingData.from_series(
        reporting_meter_data, temperature_data, is_electricity_data=True
    )
    metered_savings_dataframe = baseline_model.predict(reporting_data)
    total_metered_savings = (
        metered_savings_dataframe["observed"] - metered_savings_dataframe["predicted"]
    ).sum()

    assert round(total_metered_savings, 2) == 891.2


def test_legacy_deserialization_hourly(request):
    with open(request.fspath.dirname + "/legacy_hourly.json", "r") as f:
        legacy_str = f.read()
    baseline_model = HourlyCaltrackModel.from_2_0_json(legacy_str)

    meter_data, temperature_data, sample_metadata = load_sample(
        "il-electricity-cdd-hdd-hourly"
    )
    blackout_end_date = sample_metadata["blackout_end_date"]

    reporting_meter_data, warnings = get_reporting_data(
        meter_data, start=blackout_end_date, max_days=365
    )
    reporting = HourlyCaltrackReportingData.from_series(
        reporting_meter_data, temperature_data, is_electricity_data=True
    )

    metered_savings_dataframe = baseline_model.predict(reporting)
    total_metered_savings = (
        metered_savings_dataframe["observed"] - metered_savings_dataframe["predicted"]
    ).sum()

    assert round(total_metered_savings, 2) == -52454.02
