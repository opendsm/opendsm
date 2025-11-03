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
import pytest

import numpy as np

from opendsm.eemeter import DailyModel, DailyBaselineData, DailyReportingData
from opendsm.eemeter.samples import load_sample
from opendsm.eemeter.common.transform import get_baseline_data
from opendsm.eemeter.common.exceptions import (
    DataSufficiencyError,
    DisqualifiedModelError,
)


@pytest.fixture
def daily_series():
    meter_data, temperature_data, sample_metadata = load_sample(
        "il-electricity-cdd-hdd-daily"
    )
    blackout_start_date = sample_metadata["blackout_start_date"]
    meter_data.index = meter_data.index.tz_convert("US/Pacific")

    baseline_meter_data, warnings = get_baseline_data(
        meter_data, end=blackout_start_date, max_days=365
    )
    baseline_meter_data = baseline_meter_data[:-1]  # drop nan
    return baseline_meter_data, temperature_data


@pytest.fixture
def bad_daily_series(daily_series):
    meter, temp = daily_series
    meter[:50] += 3000
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