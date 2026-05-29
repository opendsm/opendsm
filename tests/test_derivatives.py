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
from opendsm.eemeter.models.daily.model import DailyModel
from opendsm.eemeter.models.daily.data import DailyBaselineData, DailyReportingData
from opendsm.eemeter.models.billing.model import BillingModel
from opendsm.eemeter.models.billing.data import (
    BillingBaselineData,
    BillingReportingData,
)

from regression_metrics import regression_block


@pytest.fixture(scope="session")
def baseline_data_daily(comstock_daily):
    df_b, _ = comstock_daily

    return DailyBaselineData(df=df_b.reset_index(), is_electricity_data=True)


@pytest.fixture(scope="session")
def baseline_model_daily(baseline_data_daily):
    return DailyModel().fit(baseline_data_daily, ignore_disqualification=True)


@pytest.fixture(scope="session")
def reporting_data_daily(comstock_daily):
    _, df_r = comstock_daily

    return DailyReportingData(df=df_r.reset_index(), is_electricity_data=True)


@pytest.fixture(scope="session")
def reporting_model_daily(comstock_daily):
    # Reporting-period DailyModel is trained as a baseline fit on the reporting period;
    # use DailyBaselineData here because DailyReportingData has no observed values to fit on.
    _, df_r = comstock_daily

    return DailyModel().fit(
        DailyBaselineData(df=df_r.reset_index(), is_electricity_data=True),
        ignore_disqualification=True,
    )


@pytest.fixture
def reporting_meter_data_daily():
    index = pd.date_range("2019-01-01", freq="D", periods=60, tz="America/Chicago")
    return pd.DataFrame({"value": 1}, index=index)


@pytest.fixture
def reporting_temperature_data():
    index = pd.date_range("2019-01-01", freq="D", periods=60, tz="America/Chicago")
    return pd.Series(np.arange(30.0, 90.0), index=index).asfreq("h").ffill()


@pytest.mark.regression
def test_metered_savings_cdd_hdd_daily(
    baseline_model_daily,
    reporting_meter_data_daily,
    reporting_temperature_data,
    snapshot,
):
    reporting_data = DailyReportingData.from_series(
        reporting_meter_data_daily, reporting_temperature_data, is_electricity_data=True
    )
    results = baseline_model_daily.predict(reporting_data)

    assert regression_block(results, freq="daily") == snapshot(name="regression")


@pytest.fixture(scope="session")
def baseline_model_billing(comstock_monthly):
    df_b, _ = comstock_monthly
    baseline_data = BillingBaselineData(df=df_b.reset_index(), is_electricity_data=True)

    return BillingModel().fit(baseline_data, ignore_disqualification=True)


@pytest.fixture(scope="session")
def reporting_model_billing(comstock_monthly):
    df_b, _ = comstock_monthly
    df_shifted = df_b.copy()
    df_shifted["observed"] = df_shifted["observed"] - 50
    baseline_data = BillingBaselineData(
        df=df_shifted.reset_index(), is_electricity_data=True
    )

    return BillingModel().fit(baseline_data, ignore_disqualification=True)


@pytest.fixture
def reporting_meter_data_billing():
    index = pd.date_range("2019-01-01", freq="MS", periods=13, tz="America/Chicago")
    return pd.DataFrame({"value": 1}, index=index)


@pytest.mark.regression
def test_metered_savings_cdd_hdd_billing(
    baseline_model_billing,
    reporting_meter_data_billing,
    reporting_temperature_data,
    snapshot,
):
    reporting_data = BillingReportingData.from_series(
        reporting_meter_data_billing,
        reporting_temperature_data,
        is_electricity_data=True,
    )
    results = baseline_model_billing.predict(reporting_data)

    assert regression_block(results, freq="daily") == snapshot(name="regression")


@pytest.mark.regression
def test_metered_savings_cdd_hdd_billing_no_reporting_data(
    baseline_model_billing,
    reporting_meter_data_billing,
    reporting_temperature_data,
    snapshot,
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

    assert regression_block(results, freq="daily") == snapshot(name="regression")


@pytest.mark.regression
def test_metered_savings_cdd_hdd_billing_single_record_reporting_data(
    baseline_model_billing,
    reporting_meter_data_billing,
    reporting_temperature_data,
    snapshot,
):
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

    assert regression_block(results, freq="daily") == snapshot(name="regression")


@pytest.fixture(scope="session")
def baseline_model_billing_single_record_baseline_data(comstock_monthly, comstock_hourly):
    df_monthly, _ = comstock_monthly
    df_hourly, _ = comstock_hourly
    meter_data = df_monthly[["observed"]].rename(columns={"observed": "value"}).dropna()
    meter_data.index = meter_data.index.tz_convert("UTC")
    temperature_data = df_hourly["temperature"]
    temperature_data.index = temperature_data.index.tz_convert("UTC")

    baseline_data = create_caltrack_billing_design_matrix(
        meter_data, temperature_data
    ).rename(columns={"meter_value": "observed", "temperature_mean": "temperature"})
    baseline_data = baseline_data[:60]

    return BillingModel().fit(
        BillingBaselineData(baseline_data, is_electricity_data=True),
        ignore_disqualification=True,
    )


@pytest.mark.regression
def test_metered_savings_cdd_hdd_billing_single_record_baseline_data(
    baseline_model_billing_single_record_baseline_data,
    reporting_meter_data_billing,
    reporting_temperature_data,
    snapshot,
):
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
    assert regression_block(results, freq="daily") == snapshot(name="regression")


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


@pytest.mark.regression
def test_modeled_savings_cdd_hdd_daily(
    baseline_model_daily,
    reporting_model_daily,
    reporting_meter_data_daily,
    reporting_temperature_data,
    snapshot,
):
    reporting_data = DailyReportingData.from_series(
        reporting_meter_data_daily, reporting_temperature_data, is_electricity_data=True
    )
    baseline_model_result = baseline_model_daily.predict(reporting_data)
    reporting_model_result = reporting_model_daily.predict(reporting_data)
    modeled_savings_df = pd.DataFrame(
        {
            "predicted": baseline_model_result["predicted"] - reporting_model_result["predicted"],
            "temperature": baseline_model_result["temperature"],
        },
        index=baseline_model_result.index,
    )

    assert regression_block(modeled_savings_df, freq="daily") == snapshot(name="regression")


# TODO move to dataclass testing
def test_modeled_savings_daily_empty_temperature_data(
    baseline_model_daily, reporting_model_daily
):
    index = pd.DatetimeIndex([], tz="UTC", name="dt", freq="h")
    temperature_data = pd.Series([], index=index).to_frame()

    with pytest.raises(ValueError):
        reporting = DailyReportingData(temperature_data, True)


def _fit_caltrack_hourly(meter_data, temperature_data):
    preliminary = create_caltrack_hourly_preliminary_design_matrix(
        meter_data, temperature_data
    )
    segmentation = segment_time_series(preliminary.index, "three_month_weighted")
    occupancy_lookup = estimate_hour_of_week_occupancy(
        preliminary, segmentation=segmentation
    )
    occ_bins, unocc_bins = fit_temperature_bins(
        preliminary, segmentation=segmentation, occupancy_lookup=occupancy_lookup
    )
    design = create_caltrack_hourly_segmented_design_matrices(
        preliminary, segmentation, occupancy_lookup, occ_bins, unocc_bins
    )

    return fit_caltrack_hourly_model(
        design,
        occupancy_lookup,
        occ_bins,
        unocc_bins,
        segment_type="three_month_weighted",
    )


@pytest.fixture(scope="session")
def baseline_model_hourly(comstock_hourly):
    df_b, _ = comstock_hourly
    meter_data = df_b[["observed"]].rename(columns={"observed": "value"}).copy()
    meter_data.index = meter_data.index.tz_convert("UTC")
    temperature_data = df_b["temperature"].copy()
    temperature_data.index = temperature_data.index.tz_convert("UTC")

    return _fit_caltrack_hourly(meter_data, temperature_data)


@pytest.fixture(scope="session")
def reporting_model_hourly(comstock_hourly):
    _, df_r = comstock_hourly
    meter_data = df_r[["observed"]].rename(columns={"observed": "value"}).copy()
    meter_data.index = meter_data.index.tz_convert("UTC")
    temperature_data = df_r["temperature"].copy()
    temperature_data.index = temperature_data.index.tz_convert("UTC")

    return _fit_caltrack_hourly(meter_data, temperature_data)


@pytest.fixture
def reporting_meter_data_hourly():
    index = pd.date_range("2019-01-01", freq="D", periods=60, tz="America/Chicago")
    return pd.DataFrame({"value": 1}, index=index).asfreq("h").ffill()


@pytest.mark.regression
def test_metered_savings_cdd_hdd_hourly(
    baseline_model_hourly,
    reporting_meter_data_hourly,
    reporting_temperature_data,
    snapshot,
):
    results, error_bands = metered_savings(
        baseline_model_hourly, reporting_meter_data_hourly, reporting_temperature_data
    )
    assert list(results.columns) == [
        "reporting_observed",
        "counterfactual_usage",
        "metered_savings",
    ]
    df = pd.DataFrame(
        {
            "observed": results["reporting_observed"],
            "predicted": results["counterfactual_usage"],
            "temperature": reporting_temperature_data.reindex(results.index),
        }
    )

    assert regression_block(df, freq="hourly") == snapshot(name="regression")
    assert error_bands is None


@pytest.mark.regression
def test_modeled_savings_cdd_hdd_hourly(
    baseline_model_hourly,
    reporting_model_hourly,
    reporting_meter_data_hourly,
    reporting_temperature_data,
    snapshot,
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
    df = pd.DataFrame(
        {
            "predicted": results["modeled_savings"],
            "temperature": reporting_temperature_data.reindex(results.index),
        }
    )

    assert regression_block(df, freq="hourly") == snapshot(name="regression")
    assert error_bands is None


@pytest.fixture
def normal_year_temperature_data():
    index = pd.date_range("2019-01-01", freq="D", periods=365, tz="America/Chicago")
    np.random.seed(0)
    return pd.Series(np.random.rand(365) * 30 + 45, index=index).asfreq("h").ffill()


@pytest.mark.regression
def test_modeled_savings_cdd_hdd_billing(
    baseline_model_billing,
    reporting_model_billing,
    normal_year_temperature_data,
    snapshot,
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

    assert regression_block(results, freq="daily") == snapshot(name="regression")


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


@pytest.fixture(scope="session")
def baseline_model_billing_single_record(comstock_monthly, comstock_hourly):
    df_monthly, _ = comstock_monthly
    df_hourly, _ = comstock_hourly
    meter_data = df_monthly[["observed"]].rename(columns={"observed": "value"}).dropna()
    meter_data.index = meter_data.index.tz_convert("UTC")
    temperature_data = df_hourly["temperature"]
    temperature_data.index = temperature_data.index.tz_convert("UTC")
    # 4 monthly records → 3 billing periods (minimum that exercises the optimizer
    # without collapsing observed stdev to zero on the design matrix)
    baseline_meter_data = meter_data[-4:]
    baseline_data = create_caltrack_billing_design_matrix(
        baseline_meter_data, temperature_data
    ).rename(columns={"meter_value": "observed", "temperature_mean": "temperature"})

    return BillingModel().fit(
        BillingBaselineData(baseline_data, is_electricity_data=True),
        ignore_disqualification=True,
    )


@pytest.mark.regression
def test_metered_savings_model_single_record(
    baseline_model_billing_single_record,
    reporting_meter_data_billing,
    reporting_temperature_data,
    snapshot,
):
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

    assert regression_block(results, freq="daily") == snapshot(name="regression")


@pytest.fixture(scope="session")
def baseline_model_hourly_single_segment(comstock_hourly):
    df_b, _ = comstock_hourly
    meter_data = df_b[["observed"]].rename(columns={"observed": "value"}).copy()
    meter_data.index = meter_data.index.tz_convert("UTC")
    temperature_data = df_b["temperature"].copy()
    temperature_data.index = temperature_data.index.tz_convert("UTC")

    return _fit_caltrack_hourly(meter_data, temperature_data)
