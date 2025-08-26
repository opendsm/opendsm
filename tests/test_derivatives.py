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
import numpy as np
import pandas as pd
import pytest

from opendsm.eemeter.models.hourly_caltrack.design_matrices import (
    create_caltrack_billing_design_matrix,
    create_caltrack_hourly_preliminary_design_matrix,
    create_caltrack_hourly_segmented_design_matrices,
)
from opendsm.eemeter.models.hourly_caltrack.model import fit_caltrack_hourly_model
from opendsm.eemeter.models.hourly_caltrack.derivatives import (
    metered_savings,
    modeled_savings,
)
from opendsm.eemeter.common.features import (
    estimate_hour_of_week_occupancy,
    fit_temperature_bins,
)
from opendsm.eemeter.models.hourly_caltrack.segmentation import segment_time_series
from opendsm.eemeter.common.transform import get_baseline_data, get_reporting_data
from opendsm.eemeter.models.daily.model import DailyModel
from opendsm.eemeter.models.daily.data import DailyBaselineData, DailyReportingData
from opendsm.eemeter.models.billing.model import BillingModel
from opendsm.eemeter.models.billing.data import (
    BillingBaselineData,
    BillingReportingData,
)


@pytest.fixture
def baseline_data_daily(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    blackout_start_date = il_electricity_cdd_hdd_daily["blackout_start_date"]
    baseline_meter_data, warnings = get_baseline_data(
        meter_data, end=blackout_start_date
    )
    baseline_data = DailyBaselineData.from_series(
        baseline_meter_data, temperature_data, is_electricity_data=True
    )

    return baseline_data


@pytest.fixture
def baseline_model_daily(baseline_data_daily):
    model_results = DailyModel().fit(baseline_data_daily, ignore_disqualification=True)
    return model_results


@pytest.fixture
def reporting_data_daily(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    blackout_end_date = il_electricity_cdd_hdd_daily["blackout_end_date"]
    reporting_meter_data, warnings = get_reporting_data(
        meter_data, start=blackout_end_date
    )
    reporting_data = DailyBaselineData.from_series(
        reporting_meter_data, temperature_data, is_electricity_data=True
    )
    return reporting_data


@pytest.fixture
def reporting_model_daily(reporting_data_daily):
    model_results = DailyModel().fit(reporting_data_daily, ignore_disqualification=True)
    return model_results


@pytest.fixture
def reporting_meter_data_daily():
    index = pd.date_range("2011-01-01", freq="D", periods=60, tz="UTC")
    return pd.DataFrame({"value": 1}, index=index)


@pytest.fixture
def reporting_temperature_data():
    index = pd.date_range("2011-01-01", freq="D", periods=60, tz="UTC")
    return pd.Series(np.arange(30.0, 90.0), index=index).asfreq("h").ffill()


def test_metered_savings_cdd_hdd_daily(
    baseline_model_daily, reporting_meter_data_daily, reporting_temperature_data
):
    reporting_data = DailyReportingData.from_series(
        reporting_meter_data_daily, reporting_temperature_data, is_electricity_data=True
    )
    results = baseline_model_daily.predict(reporting_data)
    metered_savings = results["predicted"] - results["observed"]

    # platform difference on Windows requires bigger tolerance here
    assert np.isclose(metered_savings.sum(), 1630, rtol=1e-2)


@pytest.fixture
def baseline_model_billing(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    blackout_start_date = il_electricity_cdd_hdd_billing_monthly["blackout_start_date"]
    baseline_meter_data, warnings = get_baseline_data(
        meter_data, end=blackout_start_date
    )
    baseline_data = BillingBaselineData.from_series(
        baseline_meter_data, temperature_data, is_electricity_data=True
    )
    model_results = BillingModel().fit(baseline_data, ignore_disqualification=True)
    return model_results


@pytest.fixture
def reporting_model_billing(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    meter_data.value = meter_data.value - 50
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    blackout_start_date = il_electricity_cdd_hdd_billing_monthly["blackout_start_date"]
    baseline_meter_data, warnings = get_baseline_data(
        meter_data, end=blackout_start_date
    )
    baseline_data = BillingBaselineData.from_series(
        baseline_meter_data, temperature_data, is_electricity_data=True
    )
    model_results = BillingModel().fit(baseline_data, ignore_disqualification=True)
    return model_results


@pytest.fixture
def reporting_meter_data_billing():
    index = pd.date_range("2011-01-01", freq="MS", periods=13, tz="UTC")
    return pd.DataFrame({"value": 1}, index=index)


def test_metered_savings_cdd_hdd_billing(
    baseline_model_billing, reporting_meter_data_billing, reporting_temperature_data
):
    reporting_data = BillingReportingData.from_series(
        reporting_meter_data_billing,
        reporting_temperature_data,
        is_electricity_data=True,
    )
    results = baseline_model_billing.predict(reporting_data)
    metered_savings = (results["predicted"] - results["observed"]).sum()
    assert np.isclose(metered_savings, 1605.14, rtol=1e-3)


def test_metered_savings_cdd_hdd_billing_no_reporting_data(
    baseline_model_billing, reporting_meter_data_billing, reporting_temperature_data
):
    # TODO test makes less sense without the use of derivatives functions. can just be merged with other predict() tests
    results = baseline_model_billing.predict(
        BillingReportingData.from_series(
            None, reporting_temperature_data, is_electricity_data=True
        )
    )
    assert list(results.columns) == [
        "season",
        "day_of_week",
        "weekday_weekend",
        "temperature",
        "predicted",
        "predicted_unc",
        "heating_load",
        "cooling_load",
        "model_split",
        "model_type",
    ]
    predicted_sum = results.predicted.sum()
    assert np.isclose(predicted_sum, 1607.1, rtol=1e-3)


def test_metered_savings_cdd_hdd_billing_single_record_reporting_data(
    baseline_model_billing, reporting_meter_data_billing, reporting_temperature_data
):
    # results, error_bands = metered_savings(
    #     baseline_model_billing,
    #     reporting_meter_data_billing[:1],
    #     reporting_temperature_data,
    #     billing_data=True,
    # )
    results = baseline_model_billing.predict(
        BillingReportingData.from_series(
            reporting_meter_data_billing[:1],
            reporting_temperature_data,
            is_electricity_data=True,
        )
    )
    assert list(results.columns) == [
        "season",
        "day_of_week",
        "weekday_weekend",
        "temperature",
        "predicted",
        "predicted_unc",
        "heating_load",
        "cooling_load",
        "model_split",
        "model_type",
    ]
    assert round(results.predicted.sum(), 2) == 0.0


@pytest.fixture
def baseline_model_billing_single_record_baseline_data(
    il_electricity_cdd_hdd_billing_monthly,
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    blackout_start_date = il_electricity_cdd_hdd_billing_monthly["blackout_start_date"]
    baseline_meter_data, warnings = get_baseline_data(
        meter_data, end=blackout_start_date
    )
    baseline_data = create_caltrack_billing_design_matrix(
        baseline_meter_data, temperature_data
    ).rename(columns={"meter_value": "observed", "temperature_mean": "temperature"})
    baseline_data = baseline_data[:60]

    model_results = BillingModel().fit(
        BillingBaselineData(baseline_data, is_electricity_data=True),
        ignore_disqualification=True,
    )
    return model_results


def test_metered_savings_cdd_hdd_billing_single_record_baseline_data(
    baseline_model_billing_single_record_baseline_data,
    reporting_meter_data_billing,
    reporting_temperature_data,
):
    # results, error_bands = metered_savings(
    #     baseline_model_billing_single_record_baseline_data,
    #     reporting_meter_data_billing,
    #     reporting_temperature_data,
    #     billing_data=True,
    # )
    results = baseline_model_billing_single_record_baseline_data.predict(
        BillingReportingData.from_series(
            reporting_meter_data_billing,
            reporting_temperature_data,
            is_electricity_data=True,
        ),
        ignore_disqualification=True,
    )
    assert list(results.columns) == [
        "season",
        "day_of_week",
        "weekday_weekend",
        "temperature",
        "observed",
        "predicted",
        "predicted_unc",
        "heating_load",
        "cooling_load",
        "model_split",
        "model_type",
    ]
    metered_savings = (results.predicted - results.observed).sum()
    assert np.isclose(metered_savings, 1785.8, rtol=1e-2)


@pytest.fixture
def reporting_meter_data_billing_wrong_timestamp():
    index = pd.date_range("2003-01-01", freq="MS", periods=13, tz="UTC")
    return pd.DataFrame({"value": 1}, index=index)


def test_metered_savings_cdd_hdd_billing_reporting_data_wrong_timestamp(
    reporting_meter_data_billing_wrong_timestamp,
    reporting_temperature_data,
):
    with pytest.raises(ValueError):
        BillingReportingData.from_series(
            reporting_meter_data_billing_wrong_timestamp,
            reporting_temperature_data,
            is_electricity_data=True,
        )


def test_modeled_savings_cdd_hdd_daily(
    baseline_model_daily,
    reporting_model_daily,
    reporting_meter_data_daily,
    reporting_temperature_data,
):
    reporting_data = DailyReportingData.from_series(
        reporting_meter_data_daily, reporting_temperature_data, is_electricity_data=True
    )
    baseline_model_result = baseline_model_daily.predict(reporting_data)
    reporting_model_result = reporting_model_daily.predict(reporting_data)
    modeled_savings = (
        baseline_model_result["predicted"] - reporting_model_result["predicted"]
    )
    assert np.isclose(modeled_savings.sum(), 177.02, rtol=0.1)


# TODO move to dataclass testing
def test_modeled_savings_daily_empty_temperature_data(
    baseline_model_daily, reporting_model_daily
):
    index = pd.DatetimeIndex([], tz="UTC", name="dt", freq="h")
    temperature_data = pd.Series([], index=index).to_frame()

    with pytest.raises(ValueError):
        reporting = DailyReportingData(temperature_data, True)


@pytest.fixture
def baseline_model_hourly(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"]
    blackout_start_date = il_electricity_cdd_hdd_hourly["blackout_start_date"]
    baseline_meter_data, warnings = get_baseline_data(
        meter_data, end=blackout_start_date
    )
    preliminary_hourly_design_matrix = create_caltrack_hourly_preliminary_design_matrix(
        baseline_meter_data, temperature_data
    )
    segmentation = segment_time_series(
        preliminary_hourly_design_matrix.index, "three_month_weighted"
    )
    occupancy_lookup = estimate_hour_of_week_occupancy(
        preliminary_hourly_design_matrix, segmentation=segmentation
    )
    occupied_temperature_bins, unoccupied_temperature_bins = fit_temperature_bins(
        preliminary_hourly_design_matrix,
        segmentation=segmentation,
        occupancy_lookup=occupancy_lookup,
    )
    design_matrices = create_caltrack_hourly_segmented_design_matrices(
        preliminary_hourly_design_matrix,
        segmentation,
        occupancy_lookup,
        occupied_temperature_bins,
        unoccupied_temperature_bins,
    )
    segmented_model = fit_caltrack_hourly_model(
        design_matrices,
        occupancy_lookup,
        occupied_temperature_bins,
        unoccupied_temperature_bins,
        segment_type="three_month_weighted",
    )
    return segmented_model


@pytest.fixture
def reporting_model_hourly(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"]
    blackout_end_date = il_electricity_cdd_hdd_hourly["blackout_end_date"]
    reporting_meter_data, warnings = get_reporting_data(
        meter_data, start=blackout_end_date
    )
    preliminary_hourly_design_matrix = create_caltrack_hourly_preliminary_design_matrix(
        reporting_meter_data, temperature_data
    )
    segmentation = segment_time_series(
        preliminary_hourly_design_matrix.index, "three_month_weighted"
    )
    occupancy_lookup = estimate_hour_of_week_occupancy(
        preliminary_hourly_design_matrix, segmentation=segmentation
    )
    occupied_temperature_bins, unoccupied_temperature_bins = fit_temperature_bins(
        preliminary_hourly_design_matrix,
        segmentation=segmentation,
        occupancy_lookup=occupancy_lookup,
    )
    design_matrices = create_caltrack_hourly_segmented_design_matrices(
        preliminary_hourly_design_matrix,
        segmentation,
        occupancy_lookup,
        occupied_temperature_bins,
        unoccupied_temperature_bins,
    )
    segmented_model = fit_caltrack_hourly_model(
        design_matrices,
        occupancy_lookup,
        occupied_temperature_bins,
        unoccupied_temperature_bins,
        segment_type="three_month_weighted",
    )
    return segmented_model


@pytest.fixture
def reporting_meter_data_hourly():
    index = pd.date_range("2011-01-01", freq="D", periods=60, tz="UTC")
    return pd.DataFrame({"value": 1}, index=index).asfreq("h").ffill()


def test_metered_savings_cdd_hdd_hourly(
    baseline_model_hourly, reporting_meter_data_hourly, reporting_temperature_data
):
    results, error_bands = metered_savings(
        baseline_model_hourly, reporting_meter_data_hourly, reporting_temperature_data
    )
    assert list(results.columns) == [
        "reporting_observed",
        "counterfactual_usage",
        "metered_savings",
    ]
    assert round(results.metered_savings.sum(), 2) == -403.7
    assert error_bands is None


def test_modeled_savings_cdd_hdd_hourly(
    baseline_model_hourly,
    reporting_model_hourly,
    reporting_meter_data_hourly,
    reporting_temperature_data,
):
    # using reporting data for convenience, but intention is to use normal data
    results, error_bands = modeled_savings(
        baseline_model_hourly,
        reporting_model_hourly,
        reporting_meter_data_hourly.index,
        reporting_temperature_data,
    )
    assert list(results.columns) == [
        "modeled_baseline_usage",
        "modeled_reporting_usage",
        "modeled_savings",
    ]
    assert round(results.modeled_savings.sum(), 1) == 55.3
    assert error_bands is None


@pytest.fixture
def normal_year_temperature_data():
    index = pd.date_range("2015-01-01", freq="D", periods=365, tz="UTC")
    np.random.seed(0)
    return pd.Series(np.random.rand(365) * 30 + 45, index=index).asfreq("h").ffill()


def test_modeled_savings_cdd_hdd_billing(
    baseline_model_billing, reporting_model_billing, normal_year_temperature_data
):
    # results, error_bands = modeled_savings(
    #     baseline_model_billing,
    #     reporting_model_billing,
    #     pd.date_range("2015-01-01", freq="D", periods=365, tz="UTC"),
    #     normal_year_temperature_data,
    # )
    meter_data = meter_data = pd.DataFrame(
        {"observed": np.nan}, index=normal_year_temperature_data.index
    )
    results = baseline_model_billing.predict(
        BillingReportingData.from_series(
            meter_data, normal_year_temperature_data, is_electricity_data=True
        )
    )

    assert list(results.columns) == [
        "season",
        "day_of_week",
        "weekday_weekend",
        "temperature",
        "predicted",
        "predicted_unc",
        "heating_load",
        "cooling_load",
        "model_split",
        "model_type",
    ]
    predicted_sum = results.predicted.sum()
    assert np.isclose(predicted_sum, 8245.37, rtol=1e-2)


@pytest.fixture
def reporting_meter_data_billing_not_aligned():
    index = pd.date_range("2001-01-01", freq="MS", periods=13, tz="UTC")
    return pd.DataFrame({"value": None}, index=index)


def test_metered_savings_not_aligned_reporting_data(
    reporting_meter_data_billing_not_aligned,
    reporting_temperature_data,
):
    with pytest.raises(ValueError):
        BillingReportingData.from_series(
            reporting_meter_data_billing_not_aligned,
            reporting_temperature_data,
            is_electricity_data=True,
        )


@pytest.fixture
def baseline_model_billing_single_record(il_electricity_cdd_hdd_billing_monthly):
    # using two records until bounds failure is fixed
    baseline_meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"][-3:]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    blackout_start_date = il_electricity_cdd_hdd_billing_monthly["blackout_start_date"]
    baseline_data = create_caltrack_billing_design_matrix(
        baseline_meter_data, temperature_data
    ).rename(columns={"meter_value": "observed", "temperature_mean": "temperature"})
    model_results = BillingModel().fit(
        BillingBaselineData(baseline_data, is_electricity_data=True),
        ignore_disqualification=True,
    )
    return model_results


def test_metered_savings_model_single_record(
    baseline_model_billing_single_record,
    reporting_meter_data_billing,
    reporting_temperature_data,
):
    # results, error_bands = metered_savings(
    #     baseline_model_billing_single_record,
    #     reporting_meter_data_billing,
    #     reporting_temperature_data,
    #     billing_data=True,
    # )

    results = baseline_model_billing_single_record.predict(
        BillingReportingData.from_series(
            reporting_meter_data_billing,
            reporting_temperature_data,
            is_electricity_data=True,
        ),
        ignore_disqualification=True,
    )
    assert list(results.columns) == [
        "season",
        "day_of_week",
        "weekday_weekend",
        "temperature",
        "observed",
        "predicted",
        "predicted_unc",
        "heating_load",
        "cooling_load",
        "model_split",
        "model_type",
    ]
    metered_savings = (results.predicted - results.observed).sum()
    assert np.isclose(metered_savings, 1436.72, rtol=1e-3)


@pytest.fixture
def baseline_model_hourly_single_segment(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"]
    blackout_start_date = il_electricity_cdd_hdd_hourly["blackout_start_date"]
    baseline_meter_data, warnings = get_baseline_data(
        meter_data, end=blackout_start_date
    )
    preliminary_hourly_design_matrix = create_caltrack_hourly_preliminary_design_matrix(
        baseline_meter_data, temperature_data
    )
    segmentation = segment_time_series(
        preliminary_hourly_design_matrix.index, "three_month_weighted"
    )
    occupancy_lookup = estimate_hour_of_week_occupancy(
        preliminary_hourly_design_matrix, segmentation=segmentation
    )
    occupied_temperature_bins, unoccupied_temperature_bins = fit_temperature_bins(
        preliminary_hourly_design_matrix,
        segmentation=segmentation,
        occupancy_lookup=occupancy_lookup,
    )
    design_matrices = create_caltrack_hourly_segmented_design_matrices(
        preliminary_hourly_design_matrix,
        segmentation,
        occupancy_lookup,
        occupied_temperature_bins,
        unoccupied_temperature_bins,
    )
    segmented_model = fit_caltrack_hourly_model(
        design_matrices,
        occupancy_lookup,
        occupied_temperature_bins,
        unoccupied_temperature_bins,
        segment_type="three_month_weighted",
    )
    return segmented_model
