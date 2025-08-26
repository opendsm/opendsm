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
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import pytz

from opendsm.eemeter.common.transform import (
    as_freq,
    clean_caltrack_billing_data,
    downsample_and_clean_caltrack_daily_data,
    clean_caltrack_billing_daily_data,
    day_counts,
    get_baseline_data,
    get_reporting_data,
    get_terms,
    remove_duplicates,
    NoBaselineDataError,
    NoReportingDataError,
    overwrite_partial_rows_with_nan,
    add_freq,
    trim,
    format_energy_data_for_caltrack,
    format_temperature_data_for_caltrack,
)


def test_as_freq_not_series(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    assert meter_data.shape == (27, 1)
    with pytest.raises(ValueError):
        as_freq(meter_data, freq="h")


def test_as_freq_hourly(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    assert meter_data.shape == (27, 1)
    as_hourly = as_freq(meter_data.value, freq="h")
    assert as_hourly.shape == (18961,)
    assert round(meter_data.value.sum(), 1) == round(as_hourly.sum(), 1) == 21290.2


def test_as_freq_daily(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    assert meter_data.shape == (27, 1)
    as_daily = as_freq(meter_data.value, freq="D")
    assert as_daily.shape == (792,)
    assert round(meter_data.value.sum(), 1) == round(as_daily.sum(), 1) == 21290.2


def test_as_freq_daily_all_nones_instantaneous(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    meter_data["value"] = np.nan
    assert meter_data.shape == (27, 1)
    as_daily = as_freq(meter_data.value, freq="D", series_type="instantaneous")
    assert as_daily.shape == (792,)
    assert round(meter_data.value.sum(), 1) == round(as_daily.sum(), 1) == 0


def test_as_freq_daily_all_nones(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    meter_data["value"] = np.nan
    assert meter_data.shape == (27, 1)
    as_daily = as_freq(meter_data.value, freq="D")
    assert as_daily.shape == (792,)
    assert round(meter_data.value.sum(), 1) == round(as_daily.sum(), 1) == 0


def test_as_freq_month_start(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    assert meter_data.shape == (27, 1)
    as_month_start = as_freq(meter_data.value, freq="MS")
    assert as_month_start.shape == (28,)
    assert round(meter_data.value.sum(), 1) == round(as_month_start.sum(), 1) == 21290.2


def test_as_freq_hourly_temperature(il_electricity_cdd_hdd_billing_monthly):
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    assert temperature_data.shape == (19417,)
    as_hourly = as_freq(temperature_data, freq="h", series_type="instantaneous")
    assert as_hourly.shape == (19417,)
    assert round(temperature_data.mean(), 1) == round(as_hourly.mean(), 1) == 54.6


def test_as_freq_daily_temperature(il_electricity_cdd_hdd_billing_monthly):
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    assert temperature_data.shape == (19417,)
    as_daily = as_freq(temperature_data, freq="D", series_type="instantaneous")
    assert as_daily.shape == (811,)
    assert abs(temperature_data.mean() - as_daily.mean()) <= 0.1


def test_as_freq_month_start_temperature(il_electricity_cdd_hdd_billing_monthly):
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    assert temperature_data.shape == (19417,)
    as_month_start = as_freq(temperature_data, freq="MS", series_type="instantaneous")
    assert as_month_start.shape == (29,)
    assert round(as_month_start.mean(), 1) == 53.4


def test_as_freq_daily_temperature_monthly(il_electricity_cdd_hdd_billing_monthly):
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    temperature_data = temperature_data.groupby(pd.Grouper(freq="MS")).mean()
    assert temperature_data.shape == (28,)
    as_daily = as_freq(temperature_data, freq="D", series_type="instantaneous")
    assert as_daily.shape == (824,)
    assert round(as_daily.mean(), 1) == 54.5


def test_as_freq_empty():
    meter_data = pd.DataFrame({"value": []})
    empty_meter_data = as_freq(meter_data.value, freq="h")
    assert empty_meter_data.empty


def test_as_freq_perserves_nulls(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    monthly_with_nulls = meter_data[meter_data.index.year != 2016].reindex(
        meter_data.index
    )
    daily_with_nulls = as_freq(monthly_with_nulls.value, freq="D")
    assert (
        round(monthly_with_nulls.value.sum(), 2)
        == round(daily_with_nulls.sum(), 2)
        == 11094.05
    )
    assert monthly_with_nulls.value.isnull().sum() == 13
    assert daily_with_nulls.isnull().sum() == 365


def test_day_counts(il_electricity_cdd_hdd_billing_monthly):
    data = il_electricity_cdd_hdd_billing_monthly["meter_data"].value
    counts = day_counts(data.index)
    assert counts.shape == (27,)
    assert counts.iloc[0] == 29.0
    assert pd.isnull(counts.iloc[-1])
    assert counts.sum() == 790.0


def test_day_counts_empty_series():
    index = pd.DatetimeIndex([])
    index.freq = None
    data = pd.Series([], index=index)
    counts = day_counts(data.index)
    assert counts.shape == (0,)


def test_get_baseline_data(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    baseline_data, warnings = get_baseline_data(meter_data)
    assert meter_data.shape == baseline_data.shape == (19417, 1)
    assert len(warnings) == 0


def test_get_baseline_data_with_timezones(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    baseline_data, warnings = get_baseline_data(
        meter_data.tz_convert("America/New_York")
    )
    assert len(warnings) == 0
    baseline_data, warnings = get_baseline_data(
        meter_data.tz_convert("Australia/Sydney")
    )
    assert len(warnings) == 0


def test_get_baseline_data_with_end(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    blackout_start_date = il_electricity_cdd_hdd_hourly["blackout_start_date"]
    baseline_data, warnings = get_baseline_data(meter_data, end=blackout_start_date)
    assert meter_data.shape != baseline_data.shape == (8761, 1)
    assert len(warnings) == 0


def test_get_baseline_data_with_end_no_max_days(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    blackout_start_date = il_electricity_cdd_hdd_hourly["blackout_start_date"]
    baseline_data, warnings = get_baseline_data(
        meter_data, end=blackout_start_date, max_days=None
    )
    assert meter_data.shape != baseline_data.shape == (9595, 1)
    assert len(warnings) == 0


def test_get_baseline_data_empty(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    blackout_start_date = il_electricity_cdd_hdd_hourly["blackout_start_date"]
    with pytest.raises(NoBaselineDataError):
        get_baseline_data(meter_data, end=pd.Timestamp("2000").tz_localize("UTC"))


def test_get_baseline_data_start_gap(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    start = meter_data.index.min() - timedelta(days=1)
    baseline_data, warnings = get_baseline_data(meter_data, start=start, max_days=None)
    assert meter_data.shape == baseline_data.shape == (19417, 1)
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.qualified_name == "eemeter.get_baseline_data.gap_at_baseline_start"
    assert (
        warning.description
        == "Data does not have coverage at requested baseline start date."
    )
    assert warning.data == {
        "data_start": "2015-11-22T06:00:00+00:00",
        "requested_start": "2015-11-21T06:00:00+00:00",
    }


def test_get_baseline_data_end_gap(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    end = meter_data.index.max() + timedelta(days=1)
    baseline_data, warnings = get_baseline_data(meter_data, end=end, max_days=None)
    assert meter_data.shape == baseline_data.shape == (19417, 1)
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.qualified_name == "eemeter.get_baseline_data.gap_at_baseline_end"
    assert (
        warning.description
        == "Data does not have coverage at requested baseline end date."
    )
    assert warning.data == {
        "data_end": "2018-02-08T06:00:00+00:00",
        "requested_end": "2018-02-09T06:00:00+00:00",
    }


def test_get_baseline_data_with_overshoot(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    baseline_data, warnings = get_baseline_data(
        meter_data,
        end=datetime(2016, 11, 9, tzinfo=pytz.UTC),
        max_days=32,
        allow_billing_period_overshoot=True,
    )
    assert baseline_data.shape == (2, 1)
    assert round(baseline_data.value.sum(), 2) == 632.31
    assert len(warnings) == 0

    baseline_data, warnings = get_baseline_data(
        meter_data,
        end=datetime(2016, 11, 9, tzinfo=pytz.UTC),
        max_days=32,
        allow_billing_period_overshoot=False,
    )
    assert baseline_data.shape == (1, 1)
    assert round(baseline_data.value.sum(), 2) == 0
    assert len(warnings) == 0

    baseline_data, warnings = get_baseline_data(
        meter_data,
        end=datetime(2016, 11, 9, tzinfo=pytz.UTC),
        max_days=25,
        allow_billing_period_overshoot=True,
    )
    assert baseline_data.shape == (1, 1)
    assert round(baseline_data.value.sum(), 2) == 0
    assert len(warnings) == 0


def test_get_baseline_data_with_ignored_gap(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    baseline_data, warnings = get_baseline_data(
        meter_data,
        end=datetime(2016, 11, 9, tzinfo=pytz.UTC),
        max_days=45,
        ignore_billing_period_gap_for_day_count=True,
    )
    assert baseline_data.shape == (2, 1)
    assert round(baseline_data.value.sum(), 2) == 632.31
    assert len(warnings) == 0

    baseline_data, warnings = get_baseline_data(
        meter_data,
        end=datetime(2016, 11, 9, tzinfo=pytz.UTC),
        max_days=45,
        ignore_billing_period_gap_for_day_count=False,
    )
    assert baseline_data.shape == (1, 1)
    assert round(baseline_data.value.sum(), 2) == 0
    assert len(warnings) == 0

    baseline_data, warnings = get_baseline_data(
        meter_data,
        end=datetime(2016, 11, 9, tzinfo=pytz.UTC),
        max_days=25,
        ignore_billing_period_gap_for_day_count=True,
    )
    assert baseline_data.shape == (1, 1)
    assert round(baseline_data.value.sum(), 2) == 0
    assert len(warnings) == 0


def test_get_baseline_data_with_overshoot_and_ignored_gap(
    il_electricity_cdd_hdd_billing_monthly,
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    baseline_data, warnings = get_baseline_data(
        meter_data,
        end=datetime(2016, 11, 9, tzinfo=pytz.UTC),
        max_days=25,
        allow_billing_period_overshoot=True,
        ignore_billing_period_gap_for_day_count=True,
    )
    assert baseline_data.shape == (2, 1)
    assert round(baseline_data.value.sum(), 2) == 632.31
    assert len(warnings) == 0

    baseline_data, warnings = get_baseline_data(
        meter_data,
        end=datetime(2016, 11, 9, tzinfo=pytz.UTC),
        max_days=25,
        allow_billing_period_overshoot=False,
        ignore_billing_period_gap_for_day_count=False,
    )
    assert baseline_data.shape == (1, 1)
    assert round(baseline_data.value.sum(), 2) == 0
    assert len(warnings) == 0


def test_get_baseline_data_n_days_billing_period_overshoot(
    il_electricity_cdd_hdd_billing_monthly,
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    baseline_data, warnings = get_baseline_data(
        meter_data,
        end=datetime(2017, 11, 9, tzinfo=pytz.UTC),
        max_days=45,
        allow_billing_period_overshoot=True,
        n_days_billing_period_overshoot=45,
        ignore_billing_period_gap_for_day_count=True,
    )
    assert baseline_data.shape == (2, 1)
    assert round(baseline_data.value.sum(), 2) == 526.25
    assert len(warnings) == 0


def test_get_baseline_data_too_far_from_date(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    end_date = datetime(2020, 11, 9, tzinfo=pytz.UTC)
    max_days = 45
    baseline_data, warnings = get_baseline_data(
        meter_data,
        end=end_date,
        max_days=max_days,
        ignore_billing_period_gap_for_day_count=True,
    )
    assert baseline_data.shape == (2, 1)
    assert round(baseline_data.value.sum(), 2) == 1393.4
    assert len(warnings) == 0
    with pytest.raises(NoBaselineDataError):
        get_baseline_data(
            meter_data,
            end=end_date,
            max_days=max_days,
            n_days_billing_period_overshoot=45,
            ignore_billing_period_gap_for_day_count=True,
        )
    baseline_data, warnings = get_baseline_data(
        meter_data,
        end=end_date,
        max_days=max_days,
        allow_billing_period_overshoot=True,
        ignore_billing_period_gap_for_day_count=True,
    )
    assert baseline_data.shape == (3, 1)
    assert round(baseline_data.value.sum(), 2) == 2043.92
    assert len(warnings) == 0
    # Includes 3 data points because data at index -3 is closer to start target
    # then data at index -2
    start_target = baseline_data.index[-1] - timedelta(days=max_days)
    assert abs((baseline_data.index[0] - start_target).days) < abs(
        (baseline_data.index[1] - start_target).days
    )
    with pytest.raises(NoBaselineDataError):
        get_baseline_data(
            meter_data,
            end=end_date,
            max_days=max_days,
            allow_billing_period_overshoot=True,
            n_days_billing_period_overshoot=45,
            ignore_billing_period_gap_for_day_count=True,
        )


def test_get_reporting_data(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    reporting_data, warnings = get_reporting_data(meter_data)
    assert meter_data.shape == reporting_data.shape == (19417, 1)
    assert len(warnings) == 0


def test_get_reporting_data_with_timezones(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    reporting_data, warnings = get_reporting_data(
        meter_data.tz_convert("America/New_York")
    )
    assert len(warnings) == 0
    reporting_data, warnings = get_reporting_data(
        meter_data.tz_convert("Australia/Sydney")
    )
    assert len(warnings) == 0


def test_get_reporting_data_with_start(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    blackout_end_date = il_electricity_cdd_hdd_hourly["blackout_end_date"]
    reporting_data, warnings = get_reporting_data(meter_data, start=blackout_end_date)
    assert meter_data.shape != reporting_data.shape == (8761, 1)
    assert len(warnings) == 0


def test_get_reporting_data_with_start_no_max_days(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    blackout_end_date = il_electricity_cdd_hdd_hourly["blackout_end_date"]
    reporting_data, warnings = get_reporting_data(
        meter_data, start=blackout_end_date, max_days=None
    )
    assert meter_data.shape != reporting_data.shape == (9607, 1)
    assert len(warnings) == 0


def test_get_reporting_data_empty(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    blackout_end_date = il_electricity_cdd_hdd_hourly["blackout_end_date"]
    with pytest.raises(NoReportingDataError):
        get_reporting_data(meter_data, start=pd.Timestamp("2030").tz_localize("UTC"))


def test_get_reporting_data_start_gap(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    start = meter_data.index.min() - timedelta(days=1)
    reporting_data, warnings = get_reporting_data(
        meter_data, start=start, max_days=None
    )
    assert meter_data.shape == reporting_data.shape == (19417, 1)
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.qualified_name == "eemeter.get_reporting_data.gap_at_reporting_start"
    assert (
        warning.description
        == "Data does not have coverage at requested reporting start date."
    )
    assert warning.data == {
        "data_start": "2015-11-22T06:00:00+00:00",
        "requested_start": "2015-11-21T06:00:00+00:00",
    }


def test_get_reporting_data_end_gap(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    end = meter_data.index.max() + timedelta(days=1)
    reporting_data, warnings = get_reporting_data(meter_data, end=end, max_days=None)
    assert meter_data.shape == reporting_data.shape == (19417, 1)
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.qualified_name == "eemeter.get_reporting_data.gap_at_reporting_end"
    assert (
        warning.description
        == "Data does not have coverage at requested reporting end date."
    )
    assert warning.data == {
        "data_end": "2018-02-08T06:00:00+00:00",
        "requested_end": "2018-02-09T06:00:00+00:00",
    }


def test_get_reporting_data_with_overshoot(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    reporting_data, warnings = get_reporting_data(
        meter_data,
        start=datetime(2016, 9, 9, tzinfo=pytz.UTC),
        max_days=30,
        allow_billing_period_overshoot=True,
    )
    assert reporting_data.shape == (2, 1)
    assert round(reporting_data.value.sum(), 2) == 632.31
    assert len(warnings) == 0

    reporting_data, warnings = get_reporting_data(
        meter_data,
        start=datetime(2016, 9, 9, tzinfo=pytz.UTC),
        max_days=30,
        allow_billing_period_overshoot=False,
    )
    assert reporting_data.shape == (1, 1)
    assert round(reporting_data.value.sum(), 2) == 0
    assert len(warnings) == 0

    reporting_data, warnings = get_reporting_data(
        meter_data,
        start=datetime(2016, 9, 9, tzinfo=pytz.UTC),
        max_days=25,
        allow_billing_period_overshoot=True,
    )
    assert reporting_data.shape == (1, 1)
    assert round(reporting_data.value.sum(), 2) == 0
    assert len(warnings) == 0


def test_get_reporting_data_with_ignored_gap(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    reporting_data, warnings = get_reporting_data(
        meter_data,
        start=datetime(2016, 9, 9, tzinfo=pytz.UTC),
        max_days=45,
        ignore_billing_period_gap_for_day_count=True,
    )
    assert reporting_data.shape == (2, 1)
    assert round(reporting_data.value.sum(), 2) == 632.31
    assert len(warnings) == 0

    reporting_data, warnings = get_reporting_data(
        meter_data,
        start=datetime(2016, 9, 9, tzinfo=pytz.UTC),
        max_days=45,
        ignore_billing_period_gap_for_day_count=False,
    )
    assert reporting_data.shape == (1, 1)
    assert round(reporting_data.value.sum(), 2) == 0
    assert len(warnings) == 0

    reporting_data, warnings = get_reporting_data(
        meter_data,
        start=datetime(2016, 9, 9, tzinfo=pytz.UTC),
        max_days=25,
        ignore_billing_period_gap_for_day_count=True,
    )
    assert reporting_data.shape == (1, 1)
    assert round(reporting_data.value.sum(), 2) == 0
    assert len(warnings) == 0


def test_get_reporting_data_with_overshoot_and_ignored_gap(
    il_electricity_cdd_hdd_billing_monthly,
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    reporting_data, warnings = get_reporting_data(
        meter_data,
        start=datetime(2016, 9, 9, tzinfo=pytz.UTC),
        max_days=25,
        allow_billing_period_overshoot=True,
        ignore_billing_period_gap_for_day_count=True,
    )
    assert reporting_data.shape == (2, 1)
    assert round(reporting_data.value.sum(), 2) == 632.31
    assert len(warnings) == 0

    reporting_data, warnings = get_reporting_data(
        meter_data,
        start=datetime(2016, 9, 9, tzinfo=pytz.UTC),
        max_days=25,
        allow_billing_period_overshoot=False,
        ignore_billing_period_gap_for_day_count=False,
    )
    assert reporting_data.shape == (1, 1)
    assert round(reporting_data.value.sum(), 2) == 0
    assert len(warnings) == 0


def test_get_terms_unrecognized_method(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]

    with pytest.raises(ValueError):
        get_terms(meter_data.index, term_lengths=[365], method="unrecognized")


def test_get_terms_unsorted_index(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]

    with pytest.raises(ValueError):
        get_terms(meter_data.index[::-1], term_lengths=[365])


def test_get_terms_bad_term_labels(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]

    with pytest.raises(ValueError):
        terms = get_terms(
            meter_data.index,
            term_lengths=[60, 60, 60],
            term_labels=["abc", "def"],  # too short
        )


def test_get_terms_default_term_labels(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]

    terms = get_terms(meter_data.index, term_lengths=[60, 60, 60])
    assert [t.label for t in terms] == ["term_001", "term_002", "term_003"]


def test_get_terms_custom_term_labels(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]

    terms = get_terms(
        meter_data.index, term_lengths=[60, 60, 60], term_labels=["abc", "def", "ghi"]
    )
    assert [t.label for t in terms] == ["abc", "def", "ghi"]


def test_get_terms_empty_index_input(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]

    terms = get_terms(meter_data.index[:0], term_lengths=[60, 60, 60])
    assert len(terms) == 0


def test_get_terms_strict(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]

    strict_terms = get_terms(
        meter_data.index,
        term_lengths=[365, 365],
        term_labels=["year1", "year2"],
        start=datetime(2016, 1, 15, tzinfo=pytz.UTC),
        method="strict",
    )

    assert len(strict_terms) == 2

    year1 = strict_terms[0]
    assert year1.label == "year1"
    assert year1.index.shape == (12,)
    assert (
        year1.target_start_date
        == pd.Timestamp("2016-01-15 00:00:00+0000", tz="UTC").to_pydatetime()
    )
    assert (
        year1.target_end_date
        == pd.Timestamp("2017-01-14 00:00:00+0000", tz="UTC").to_pydatetime()
    )
    assert year1.target_term_length_days == 365
    assert (
        year1.actual_start_date
        == year1.index[0]
        == pd.Timestamp("2016-01-22 06:00:00+0000", tz="UTC")
    )
    assert (
        year1.actual_end_date
        == year1.index[-1]
        == pd.Timestamp("2016-12-19 06:00:00+0000", tz="UTC")
    )
    assert year1.actual_term_length_days == 332
    assert year1.complete

    year2 = strict_terms[1]
    assert year2.index.shape == (13,)
    assert year2.label == "year2"
    assert year2.target_start_date == pd.Timestamp("2016-12-19 06:00:00+0000", tz="UTC")
    assert (
        year2.target_end_date
        == pd.Timestamp("2018-01-14 00:00:00+0000", tz="UTC").to_pydatetime()
    )
    assert year2.target_term_length_days == 365
    assert (
        year2.actual_start_date
        == year2.index[0]
        == pd.Timestamp("2016-12-19 06:00:00+00:00", tz="UTC")
    )
    assert (
        year2.actual_end_date
        == year2.index[-1]
        == pd.Timestamp("2017-12-22 06:00:00+0000", tz="UTC")
    )
    assert year2.actual_term_length_days == 368
    assert year2.complete


def test_get_terms_nearest(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    nearest_terms = get_terms(
        meter_data.index,
        term_lengths=[365, 365],
        term_labels=["year1", "year2"],
        start=datetime(2016, 1, 15, tzinfo=pytz.UTC),
        method="nearest",
    )

    assert len(nearest_terms) == 2

    year1 = nearest_terms[0]
    assert year1.label == "year1"
    assert year1.index.shape == (13,)
    assert year1.index[0] == pd.Timestamp("2016-01-22 06:00:00+0000", tz="UTC")
    assert year1.index[-1] == pd.Timestamp("2017-01-21 06:00:00+0000", tz="UTC")
    assert (
        year1.target_start_date
        == pd.Timestamp("2016-01-15 00:00:00+0000", tz="UTC").to_pydatetime()
    )
    assert year1.target_term_length_days == 365
    assert year1.actual_term_length_days == 365
    assert year1.complete

    year2 = nearest_terms[1]
    assert year2.label == "year2"
    assert year2.index.shape == (13,)
    assert year2.index[0] == pd.Timestamp("2017-01-21 06:00:00+0000", tz="UTC")
    assert year2.index[-1] == pd.Timestamp("2018-01-20 06:00:00+0000", tz="UTC")
    assert year2.target_start_date == pd.Timestamp("2017-01-21 06:00:00+0000", tz="UTC")
    assert year1.target_term_length_days == 365
    assert year2.actual_term_length_days == 364
    assert not year2.complete  # no remaining index

    # check completeness case with a shorter final term
    nearest_terms = get_terms(
        meter_data.index,
        term_lengths=[365, 340],
        term_labels=["year1", "year2"],
        start=datetime(2016, 1, 15, tzinfo=pytz.UTC),
        method="nearest",
    )
    year2 = nearest_terms[1]
    assert year2.label == "year2"
    assert year2.index.shape == (12,)
    assert year2.index[0] == pd.Timestamp("2017-01-21 06:00:00+0000", tz="UTC")
    assert year2.index[-1] == pd.Timestamp("2017-12-22 06:00:00+00:00", tz="UTC")
    assert year2.target_start_date == pd.Timestamp("2017-01-21 06:00:00+0000", tz="UTC")
    assert year2.target_term_length_days == 340
    assert year2.actual_term_length_days == 335
    assert year2.complete  # has remaining index


def test_term_repr(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]

    terms = get_terms(meter_data.index, term_lengths=[60, 60, 60])
    assert repr(terms[0]) == (
        "Term(label=term_001, target_term_length_days=60, actual_term_length_days=29,"
        " complete=True)"
    )


def test_remove_duplicates_df():
    index = pd.DatetimeIndex(["2017-01-01", "2017-01-02", "2017-01-02"])
    df = pd.DataFrame({"value": [1, 2, 3]}, index=index)
    assert df.shape == (3, 1)
    df_dedupe = remove_duplicates(df)
    assert df_dedupe.shape == (2, 1)
    assert list(df_dedupe.value) == [1, 2]


def test_remove_duplicates_series():
    index = pd.DatetimeIndex(["2017-01-01", "2017-01-02", "2017-01-02"])
    series = pd.Series([1, 2, 3], index=index)
    assert series.shape == (3,)
    series_dedupe = remove_duplicates(series)
    assert series_dedupe.shape == (2,)
    assert list(series_dedupe) == [1, 2]


def test_as_freq_hourly_to_daily(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]

    meter_data.iloc[-1, meter_data.columns.get_loc("value")] = np.nan
    assert meter_data.shape == (19417, 1)
    as_daily = as_freq(meter_data.value, freq="D")
    assert as_daily.shape == (811,)
    assert round(meter_data.value.sum(), 1) == round(as_daily.sum(), 1) == 21926.0


def test_as_freq_daily_to_daily(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    assert meter_data.shape == (810, 1)
    as_daily = as_freq(meter_data.value, freq="D")
    assert as_daily.shape == (810,)
    assert round(meter_data.value.sum(), 1) == round(as_daily.sum(), 1) == 21925.8


def test_as_freq_hourly_to_daily_include_coverage(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    meter_data.iloc[-1, meter_data.columns.get_loc("value")] = np.nan
    assert meter_data.shape == (19417, 1)
    as_daily = as_freq(meter_data.value, freq="D", include_coverage=True)
    assert as_daily.shape == (811, 2)
    assert round(meter_data.value.sum(), 1) == round(as_daily.value.sum(), 1) == 21926.0


def test_clean_caltrack_billing_daily_data_billing(
    il_electricity_cdd_hdd_billing_monthly,
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    cleaned_data = clean_caltrack_billing_daily_data(meter_data, "billing_monthly")
    assert cleaned_data.shape == (27, 1)
    pd.testing.assert_frame_equal(meter_data, cleaned_data)


def test_clean_caltrack_billing_daily_data_daily(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    cleaned_data = clean_caltrack_billing_daily_data(meter_data, "daily")
    assert cleaned_data.shape == (810, 1)
    pd.testing.assert_frame_equal(meter_data, cleaned_data)


def test_clean_caltrack_billing_daily_data_daily_local_tz(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    meter_data.index += timedelta(hours=6)
    meter_data = meter_data.tz_convert("America/Chicago")
    cleaned_data = clean_caltrack_billing_daily_data(meter_data, "daily")
    assert cleaned_data.shape == (810, 1)
    pd.testing.assert_frame_equal(meter_data, cleaned_data)


def test_clean_caltrack_billing_daily_data_hourly(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    cleaned_data = clean_caltrack_billing_daily_data(meter_data, "hourly")
    assert cleaned_data.shape == (811, 1)


def test_clean_caltrack_daily_data_hourly(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    cleaned_data = downsample_and_clean_caltrack_daily_data(meter_data)
    assert cleaned_data.shape == (811, 1)


def test_clean_caltrack_daily_data_hourly_local_tz(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    meter_data = meter_data.tz_convert("America/Chicago")
    cleaned_data = downsample_and_clean_caltrack_daily_data(meter_data)
    assert cleaned_data.shape == (810, 1)


def test_clean_caltrack_billing_data_estimated(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    meter_data["estimated"] = False
    estimated_col_index = meter_data.columns.get_loc("estimated")
    meter_data.iloc[:, estimated_col_index] = False
    meter_data.iloc[2, estimated_col_index] = True
    meter_data.iloc[5, estimated_col_index] = True
    meter_data.iloc[6, estimated_col_index] = True
    meter_data.iloc[10, estimated_col_index] = True

    cleaned_data = clean_caltrack_billing_data(meter_data, "billing_monthly")
    assert cleaned_data.dropna().shape[0] == cleaned_data.shape[0] - 2


def test_clean_caltrack_billing_data_uneven_datetimes(
    il_electricity_cdd_hdd_billing_monthly,
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    too_short_meter_data = pd.concat(
        [
            meter_data,
            pd.DataFrame(
                data=[{"value": 100}],
                index=[datetime(2017, 1, 1, 6).replace(tzinfo=pytz.UTC)],
            ),
        ]
    ).sort_index()
    cleaned_data = clean_caltrack_billing_data(too_short_meter_data, "billing_monthly")
    assert cleaned_data.dropna().shape[0] == cleaned_data.shape[0] - 3

    too_long_meter_data = meter_data.drop(
        [datetime(2016, 12, 19, 6).replace(tzinfo=pytz.UTC)]
    )
    cleaned_data = clean_caltrack_billing_data(too_long_meter_data, "billing_monthly")

    too_long_meter_data = meter_data.drop(
        [
            datetime(2016, 12, 19, 6).replace(tzinfo=pytz.UTC),
            datetime(2017, 1, 21, 6).replace(tzinfo=pytz.UTC),
        ]
    )
    cleaned_data = clean_caltrack_billing_data(too_long_meter_data, "billing_bimonthly")
    assert cleaned_data.dropna().shape[0] == cleaned_data.shape[0] - 2
    assert cleaned_data.dropna().shape[0] == cleaned_data.shape[0] - 2

    pre_empty_meter_data = meter_data[:0]
    cleaned_data = clean_caltrack_billing_data(pre_empty_meter_data, "billing_monthly")
    assert cleaned_data.empty

    post_empty_meter_data = meter_data[:4].drop(
        [
            datetime(2015, 12, 21, 6).replace(tzinfo=pytz.UTC),
            datetime(2016, 1, 22, 6).replace(tzinfo=pytz.UTC),
        ]
    )
    assert not post_empty_meter_data["value"].dropna().empty
    cleaned_data = clean_caltrack_billing_data(post_empty_meter_data, "billing_monthly")
    assert cleaned_data.empty


def test_overwrite_partial_rows_with_nan(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    meter_data["other_column"] = meter_data["value"]
    meter_data.iloc[:3, meter_data.columns.get_loc("other_column")] = np.nan
    meter_data_nanned = overwrite_partial_rows_with_nan(meter_data)
    assert pd.isnull(meter_data_nanned["value"][:3]).all()


import pandas as pd


def test_add_freq(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]

    # make DateTimeIndex timezone-naive
    meter_data.index = meter_data.index.tz_localize(None)

    # infer frequency
    meter_data.index = add_freq(meter_data.index)
    assert meter_data.index.freq == "h"


def test_trim_two_dataframes(
    uk_electricity_hdd_only_hourly_sample_1, uk_electricity_hdd_only_hourly_sample_2
):
    df1 = uk_electricity_hdd_only_hourly_sample_1["meter_data"]
    df2 = uk_electricity_hdd_only_hourly_sample_2["meter_data"]

    df1_trimmed, df2_trimmed = trim(df1, df2)

    assert (
        df1.index[0] == df1.index.min()
        and df2.index[0] == df2.index.min()
        and df1.index[0] != df2.index[0]
    )

    assert (
        df1.index[-1] == df1.index.max()
        and df2.index[-1] == df2.index.max()
        and df1.index[-1] != df2.index[-1]
    )

    assert df1_trimmed.index[0] == df2_trimmed.index[0]
    assert df1_trimmed.index.min() == df2_trimmed.index.min()
    assert df1_trimmed.index[-1] == df2_trimmed.index[-1]
    assert df1_trimmed.index.max() == df2_trimmed.index.max()


def test_format_temperature_data_for_caltrack(il_electricity_cdd_hdd_hourly):
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"]

    # temperature_data to pd.DateFrame
    temperature_data = pd.DataFrame(temperature_data)

    # flipping df
    temperature_data = temperature_data.reindex(index=temperature_data.index[::-1])

    # inserting new value of 0.04 at 09.34 22/11/2015
    new_start = pd.to_datetime("22/11/2015 09:34", dayfirst=True).tz_localize("UTC")
    temperature_data.loc[new_start] = [0.04]

    # rename column name to 'consumption'
    temperature_data.rename(columns={"value": "consumption"}, inplace=True)

    temperature_data_reformatted = format_temperature_data_for_caltrack(
        temperature_data
    )

    assert isinstance(temperature_data_reformatted, pd.Series)
    assert (
        temperature_data_reformatted.index[0] < temperature_data_reformatted.index[-1]
    )
    assert temperature_data_reformatted.index.freq == "h"
    assert temperature_data_reformatted.index.tzinfo is not None


def test_format_energy_data_for_caltrack_hourly(il_electricity_cdd_hdd_hourly):
    df = il_electricity_cdd_hdd_hourly["meter_data"]
    # flipping df
    df = df.reindex(index=df.index[::-1])

    # inserting new value of 0.04 at 09.34 22/11/2015
    new_start = pd.to_datetime("22/11/2015 09:34", dayfirst=True).tz_localize("UTC")
    df.loc[new_start] = [0.04]

    # rename column name to 'consumption'
    df.rename(columns={"value": "consumption"}, inplace=True)

    # df_flipped to pd.Series
    df = df.squeeze()

    df_reformatted = format_energy_data_for_caltrack(df, method="hourly")

    assert isinstance(df_reformatted, pd.DataFrame)
    assert df_reformatted.index[0] < df_reformatted.index[-1]
    assert df_reformatted.index.freq == "h"
    assert df_reformatted.columns[0] == "value"
    assert df_reformatted.index.tzinfo is not None
    assert len(df_reformatted.columns) == 1


def test_format_energy_data_for_caltrack_daily(il_electricity_cdd_hdd_daily):
    df = il_electricity_cdd_hdd_daily["meter_data"]
    # flipping df
    df = df.reindex(index=df.index[::-1])

    # inserting new value of 0.04 at 09.34 22/11/2015
    new_start = pd.to_datetime("22/11/2015 09:34", dayfirst=True).tz_localize("UTC")
    df.loc[new_start] = [0.04]

    # rename column name to 'consumption'
    df.rename(columns={"value": "consumption"}, inplace=True)

    # df_flipped to pd.Series
    df = df.squeeze()

    df_reformatted = format_energy_data_for_caltrack(df, method="daily")

    assert isinstance(df_reformatted, pd.DataFrame)
    assert df_reformatted.index[0] < df_reformatted.index[-1]
    assert df_reformatted.index.freq == "D"
    assert df_reformatted.columns[0] == "value"
    assert df_reformatted.index.tzinfo is not None
    assert len(df_reformatted.columns) == 1


def test_format_energy_data_for_caltrack_billing(il_electricity_cdd_hdd_daily):
    df = il_electricity_cdd_hdd_daily["meter_data"]
    # flipping df
    df = df.reindex(index=df.index[::-1])

    # inserting new value of 0.04 at 09.34 22/11/2015
    new_start = pd.to_datetime("22/11/2015 09:34", dayfirst=True).tz_localize("UTC")
    df.loc[new_start] = [0.04]

    # rename column name to 'consumption'
    df.rename(columns={"value": "consumption"}, inplace=True)

    # df_flipped to pd.Series
    df = df.squeeze()

    df_reformatted = format_energy_data_for_caltrack(df, method="billing")

    assert isinstance(df_reformatted, pd.DataFrame)
    assert df_reformatted.index[0] < df_reformatted.index[-1]
    assert df_reformatted.index.freq == pd.tseries.offsets.MonthEnd()
    assert df_reformatted.columns[0] == "value"
    assert df_reformatted.index.tzinfo is not None
    assert len(df_reformatted.columns) == 1
