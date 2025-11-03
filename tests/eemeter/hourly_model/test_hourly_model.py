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

from opendsm.eemeter import (
    HourlyBaselineData,
    HourlyReportingData,
    HourlyModel,
    HourlySolarSettings,
    HourlyNonSolarSettings,
)
from opendsm.eemeter.models.hourly.settings import BaseHourlySettings
from opendsm.eemeter.common.exceptions import (
    DataSufficiencyError,
    DisqualifiedModelError,
)
import numpy as np
import pandas as pd
import pytest
from math import ceil



def test_good_data(baseline, reporting):
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    reporting_data = HourlyReportingData(reporting, is_electricity_data=True)
    hm = HourlyModel().fit(baseline_data)
    p1 = hm.predict(reporting_data)
    assert np.isclose(
        p1["predicted"].sum(), 1135000, rtol=1e-2
    )  # quick check that model fit isn't changing drastically
    serialized = hm.to_json()
    hm2 = HourlyModel.from_json(serialized)
    p2 = hm2.predict(reporting_data)
    assert p1.equals(p2)


def test_misaligned_data(baseline, reporting):
    reporting.index = reporting.index.shift(8, freq="h")
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    reporting_data = HourlyReportingData(reporting, is_electricity_data=True)
    hm = HourlyModel().fit(baseline_data)
    hm.predict(reporting_data)


def test_tz_naive(baseline):
    baseline.index = baseline.index.tz_localize(None)
    with pytest.raises(ValueError):
        HourlyBaselineData(baseline, is_electricity_data=True)


def test_tz_mismatch(baseline):
    # might allow automatic adjustment from the model in the future, but hard requirement for now
    baseline.index = baseline.index.tz_convert("US/Pacific")
    reporting = baseline.copy()
    reporting.index = reporting.index.tz_convert("US/Eastern")
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    reporting_data = HourlyReportingData(reporting, is_electricity_data=True)
    hm = HourlyModel().fit(baseline_data)
    with pytest.raises(ValueError):
        hm.predict(reporting_data)


def test_predict_missing_fit_features(baseline, reporting):
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    hm = HourlyModel(settings=HourlySolarSettings()).fit(baseline_data)
    reporting.drop("ghi", axis=1, inplace=True)
    reporting_data = HourlyReportingData(reporting, is_electricity_data=True)
    with pytest.raises(ValueError):
        hm.predict(reporting_data)


def test_nonsolar_predict_with_ghi(baseline, reporting, caplog):
    baseline.drop("ghi", axis=1, inplace=True)
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    hm = HourlyModel().fit(baseline_data)
    reporting_data = HourlyReportingData(reporting, is_electricity_data=True)
    with caplog.at_level("WARNING"):
        hm.predict(reporting_data)
        assert "GHI" in caplog.text


def test_forced_solar_model_fit_no_ghi(baseline):
    baseline = baseline.drop("ghi", axis=1)
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    with pytest.raises(ValueError):
        HourlyModel(settings=HourlySolarSettings()).fit(baseline_data)


def test_forced_nonsolar_model_fit_with_ghi(baseline):
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    hm = HourlyModel(settings=HourlyNonSolarSettings()).fit(baseline_data)
    assert [
        w for w in hm.warnings if w.qualified_name == "eemeter.potential_model_mismatch"
    ]


def test_no_data(baseline):
    baseline["observed"] = 0
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(baseline_data)


def test_negative_meter_values(baseline):
    baseline.loc["2018-01-08", "observed"] = -1

    # gas data can't be negative
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=False)
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(baseline_data)

    # elec can
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    HourlyModel().fit(baseline_data)


def test_invalid_baseline_lengths(baseline):
    # TODO import min/max length from constants
    MAX_BASELINE_HOURS = 8760
    MIN_BASELINE_HOURS = ceil(MAX_BASELINE_HOURS * 0.9) - 24
    short_df = baseline.iloc[:MIN_BASELINE_HOURS]

    extra_days = baseline.iloc[-24*2:]
    extra_days.index += pd.Timedelta(days=2)
    long_df = pd.concat([baseline, extra_days])

    short_baseline = HourlyBaselineData(short_df, is_electricity_data=True)
    long_baseline = HourlyBaselineData(long_df, is_electricity_data=True)
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(short_baseline)
    hm_short = HourlyModel().fit(short_baseline, ignore_disqualification=True)
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(long_baseline)
    hm_long = HourlyModel().fit(long_baseline, ignore_disqualification=True)


def test_low_freq_temp(baseline):
    baseline["temperature"] = baseline["temperature"].resample("D").mean()
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    assert_dq(
        baseline_data,
        ["eemeter.sufficiency_criteria.too_many_days_with_missing_temperature_data"],
    )
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(baseline_data)


def test_low_freq_meter(baseline):
    baseline["observed"] = baseline["observed"].resample("D").mean()
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    assert_dq(
        baseline_data,
        ["eemeter.sufficiency_criteria.too_many_days_with_missing_observed_data"],
    )
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(baseline_data)


def test_monthly_percentage(baseline):
    missing_idx = pd.date_range(
        start=baseline.index.min(), end=baseline.index.max(), freq="h"
    )
    # create datetimeindex where a little over 10% of days are missing in feb, but still 90% overall
    missing_idx = missing_idx[missing_idx.day < 4]
    invalid_baseline = baseline[~baseline.index.isin(missing_idx)]
    # create datetimeindex where a little under 10% of days are missing in feb
    missing_idx = missing_idx[missing_idx.day < 3]
    valid_baseline = baseline[~baseline.index.isin(missing_idx)]

    invalid_temp = baseline.copy()
    invalid_temp.loc[invalid_temp.index.day < 5, "temperature"] = np.nan

    invalid_meter = baseline.copy()
    invalid_meter.loc[invalid_meter.index.day < 5, "observed"] = np.nan

    baseline_data = HourlyBaselineData(invalid_baseline, is_electricity_data=True)
    assert_dq(
        baseline_data, ["eemeter.sufficiency_criteria.missing_monthly_temperature_data"]
    )
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(baseline_data)
    baseline_data = HourlyBaselineData(valid_baseline, is_electricity_data=True)
    HourlyModel().fit(baseline_data)

    baseline_data = HourlyBaselineData(invalid_temp, is_electricity_data=True)
    assert_dq(
        baseline_data,
        [
            "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
            "eemeter.sufficiency_criteria.missing_monthly_temperature_data",
            "eemeter.sufficiency_criteria.too_many_days_with_missing_temperature_data",
        ],
    )
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(baseline_data)

    baseline_data = HourlyBaselineData(invalid_meter, is_electricity_data=True)
    assert_dq(
        baseline_data,
        [
            "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
            "eemeter.sufficiency_criteria.missing_monthly_observed_data",
            "eemeter.sufficiency_criteria.too_many_days_with_missing_observed_data",
        ],
    )
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(baseline_data)


def test_monthly_ghi_percentage(baseline):
    # create datetimeindex where a little over 10% of days are missing in feb, but still 90% overall
    missing_idx = pd.date_range(
        start=baseline.index.min(), end=baseline.index.max(), freq="h"
    )
    missing_idx = missing_idx[missing_idx.day < 4]

    invalid_ghi = baseline.copy()
    invalid_ghi.loc[invalid_ghi.index.day < 5, "ghi"] = np.nan

    baseline_data = HourlyBaselineData(invalid_ghi, is_electricity_data=True)
    assert_dq(
        baseline_data,
        [
            "eemeter.sufficiency_criteria.missing_monthly_ghi_data",
        ],
    )
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(baseline_data)


def test_hourly_fit_daily_threshold(baseline):
    """confirm that days with >50% interpolated data are excluded from fit step"""

    # bit fragile testing private methods this way, but fine for now
    m = HourlyModel()
    b1 = baseline.copy()
    b1.loc["2018-01-08":"2018-01-08 11", "temperature"] = np.nan
    b1 = m._add_categorical_features(b1)
    b1 = m._daily_fitting_sufficiency(b1)
    assert b1.loc["2018-01-08", "include_date"].sum() == 24

    b2 = baseline.copy()
    b2.loc["2018-01-08":"2018-01-08 12", "temperature"] = np.nan
    b2 = m._add_categorical_features(b2)
    b2 = m._daily_fitting_sufficiency(b2)
    assert b2.loc["2018-01-08", "include_date"].sum() == 0
    assert b2.loc["2018-01-09", "include_date"].sum() == 24


@pytest.mark.filterwarnings("ignore:Objective did not converge.")
def test_hourly_error_metric_dq(baseline):
    baseline["observed"] = np.random.normal(-1, 10, len(baseline)) ** 3
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    model = HourlyModel().fit(baseline_data)
    assert_dq(baseline_data, ["eemeter.model_fit_metrics"])
    with pytest.raises(DisqualifiedModelError):
        model.predict(baseline_data)


def assert_dq(data, expected_disqualifications):
    remaining_dq = set(expected_disqualifications)
    for dq in data.disqualification:
        if dq.qualified_name in remaining_dq:
            remaining_dq.remove(dq.qualified_name)
    assert not remaining_dq


def test_hourly_dict_settings():
    m = HourlyModel(settings={"train_features": ["feature_col"]})
    assert isinstance(m.settings, HourlyNonSolarSettings)
    assert set(m.settings.train_features) == {"temperature", "feature_col"}
    m = HourlyModel(settings={"train_features": ["feature_col", "ghi"]})
    assert isinstance(m.settings, HourlySolarSettings)
    assert set(m.settings.train_features) == {"temperature", "ghi", "feature_col"}
    m = HourlyModel(settings={"cvrmse_threshold": 1.0})
    assert isinstance(m.settings, BaseHourlySettings)
    assert m.settings.train_features == None
