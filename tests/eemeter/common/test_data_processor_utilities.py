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

import pandas as pd
import pytest

from pandas.tseries.offsets import Day

from opendsm.eemeter.common.data_processor_utilities import (
    DataReport,
    as_freq,
    clean_billing_daily_data,
    clean_billing_data,
    compute_data_frequency,
    downsample_and_clean_daily_data,
    frequency_duration,
    trim_edge_rows,
)
from opendsm.eemeter.common.data_settings import (
    BillingDataSufficiencySettings,
    DailyDataSufficiencySettings,
    DailyObservedSufficiencySettings,
)



# day_counts, remove_duplicates and as_freq are tested in test_transform.py.

@pytest.mark.parametrize(
    "freq,periods,expected",
    [
        ("h", 48, "hourly"),
        ("D", 60, "daily"),
        ("MS", 12, "billing_monthly"),
        ("2MS", 6, "billing_bimonthly"),
    ],
)
def test_compute_minimum_data_frequency_from_regular_index(freq, periods, expected):
    """A regularly-spaced index resolves to the matching data frequency label."""
    index = pd.date_range("2020-01-01", periods=periods, freq=freq)

    assert compute_data_frequency(index.copy(), default_data_frequency="daily") == expected


def test_compute_minimum_data_frequency_single_point_returns_default():
    """A length-1 index has no spacing, so the default data frequency is returned."""
    index = pd.date_range("2020-01-01", periods=1, freq="D")

    assert compute_data_frequency(index, default_data_frequency="billing_monthly") == "billing_monthly"


def test_compute_minimum_data_frequency_irregular_index_uses_median_spacing():
    """An index with no inferrable frequency falls back to the median day spacing."""
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-05"])

    assert compute_data_frequency(index, default_data_frequency="daily") == "daily"


def test_compute_minimum_data_frequency_irregular_monthly_spacing():
    """Irregular reads averaging ~monthly spacing resolve to billing_monthly."""
    index = pd.to_datetime(
        ["2020-01-01", "2020-02-03", "2020-03-01", "2020-04-05", "2020-04-28"]
    )

    assert compute_data_frequency(index, default_data_frequency="daily") == "billing_monthly"


def test_compute_minimum_data_frequency_uses_configured_monthly_threshold():
    """A median spacing past the default monthly cutoff reclassifies once the cutoff is widened."""
    index = pd.to_datetime(["2020-01-01", "2020-02-10", "2020-03-21", "2020-04-25"])
    # median day-count is 40, past the default 35-day monthly cutoff
    assert compute_data_frequency(index.copy(), default_data_frequency="daily") == "billing_bimonthly"
    assert (
        compute_data_frequency(
            index.copy(), default_data_frequency="daily", max_days_in_monthly_period=45
        )
        == "billing_monthly"
    )


@pytest.mark.parametrize(
    "freq,expected",
    [
        ("15min", pd.Timedelta(minutes=15)),
        ("h", pd.Timedelta(hours=1)),
        ("D", pd.Timedelta(days=1)),
        ("2D", pd.Timedelta(days=2)),
        (Day(3), pd.Timedelta(days=3)),
    ],
)
def test_frequency_duration_of_fixed_frequencies(freq, expected):
    assert frequency_duration(freq) == expected, f"{freq!r} -> {frequency_duration(freq)}"


@pytest.mark.parametrize("freq", ["MS", "ME", "2MS", "QS", "YS"])
def test_frequency_duration_rejects_calendar_offsets(freq):
    with pytest.raises(TypeError):
        frequency_duration(freq)


def test_as_freq_daily_bins_start_at_the_first_observation():
    index = pd.date_range("2024-01-01 05:00", periods=72, freq="h", tz="America/Los_Angeles")
    series = pd.Series(1.0, index=index)

    resampled = as_freq(series, "D", series_type="instantaneous")

    assert list(resampled.index.hour) == [5, 5, 5], f"bin starts: {resampled.index}"
    assert resampled.tolist() == [1.0, 1.0, 1.0]


def test_as_freq_daily_bins_follow_wall_clock_across_dst():
    """Days anchored at 02:00 stay at 02:00 wall-clock time through a daylight-saving
    transition; the one label that would fall on the nonexistent 02:00 is pushed to
    03:00, and the days around the transition are 23 and 24 hours long."""
    # 95 hourly readings from 02:00 on the day before the transition end at 01:00 on the
    # fourth day; the final reading has no following interval and contributes nothing
    index = pd.date_range("2020-03-07 02:00", periods=95, freq="h", tz="America/New_York")
    series = pd.Series(1.0, index=index)

    resampled = as_freq(series, "D", atomic_freq="1h")

    assert list(resampled.index.hour) == [2, 3, 2, 2], f"bin starts: {resampled.index}"
    assert resampled.tolist() == [24.0, 23.0, 24.0, 23.0], f"daily sums: {resampled.tolist()}"



def _edge_frame(n_periods=6):
    index = pd.date_range("2020-01-01", periods=n_periods, freq="D", tz="UTC")

    return pd.DataFrame(
        {"observed": 1.0, "temperature": 50.0, "ghi": 200.0}, index=index
    )


def test_trim_edge_rows_drops_incomplete_leading_and_trailing_rows():
    """Leading rows missing observed and trailing rows missing a trailing column are dropped."""
    df = _edge_frame()
    df.iloc[0, df.columns.get_loc("observed")] = float("nan")
    df.iloc[-1, df.columns.get_loc("ghi")] = float("nan")

    trimmed, n_leading, n_trailing = trim_edge_rows(df, ("observed", "temperature", "ghi"))

    assert (n_leading, n_trailing) == (1, 1), f"expected (1, 1), got ({n_leading}, {n_trailing})"
    assert trimmed.index.equals(df.index[1:-1]), f"unexpected trimmed index {trimmed.index}"


def test_trim_edge_rows_keeps_interior_missing_rows():
    """Interior rows missing every value survive trimming."""
    df = _edge_frame()
    df.iloc[2:4, :] = float("nan")

    trimmed, n_leading, n_trailing = trim_edge_rows(df, ("observed", "temperature", "ghi"))

    assert (n_leading, n_trailing) == (0, 0), f"expected (0, 0), got ({n_leading}, {n_trailing})"
    assert trimmed.index.equals(df.index), "interior missing rows were dropped"


def test_trim_edge_rows_returns_frame_untouched_when_no_row_qualifies():
    """A frame with no usable row is passed through for the caller's sufficiency check."""
    df = _edge_frame()
    df["observed"] = float("nan")

    trimmed, n_leading, n_trailing = trim_edge_rows(df, ("observed", "temperature", "ghi"))

    assert (n_leading, n_trailing) == (0, 0), f"expected (0, 0), got ({n_leading}, {n_trailing})"
    assert trimmed.index.equals(df.index), "an unusable frame was trimmed to an empty frame"


def test_trim_edge_rows_returns_frame_untouched_when_a_column_is_absent():
    """A frame missing a trimmable column is passed through rather than raising KeyError."""
    df = _edge_frame().drop(columns=["observed"])

    trimmed, n_leading, n_trailing = trim_edge_rows(df, ("observed", "temperature", "ghi"))

    assert (n_leading, n_trailing) == (0, 0), f"expected (0, 0), got ({n_leading}, {n_trailing})"
    assert trimmed.index.equals(df.index), "a frame missing a column was modified"



def _billing_frame(dates, values, estimated=None):
    index = pd.to_datetime(dates)
    frame = pd.DataFrame({"value": values}, index=index)
    if estimated is not None:
        frame["estimated"] = estimated

    return frame


def test_clean_billing_data_reports_offcycle_periods_it_nans():
    """Periods shorter than the minimum or longer than the monthly maximum are NaNed and
    reported with their lengths in days."""
    # 31, 9, 70 and 30 day periods: the 9 and 70 day ones fall outside 25-35 days
    data = _billing_frame(
        ["2020-01-01", "2020-02-01", "2020-02-10", "2020-04-20", "2020-05-20"],
        [1.0, 2.0, 3.0, 4.0, 5.0],
    )

    cleaned, report = clean_billing_data(
        data, "billing_monthly", BillingDataSufficiencySettings()
    )

    assert report.offcycle_periods == {
        pd.Timestamp("2020-02-01"): 9.0,
        pd.Timestamp("2020-02-10"): 70.0,
    }, f"unexpected off-cycle periods {report.offcycle_periods}"
    assert report.data_frequency == "billing_monthly"
    naned = cleaned.index[cleaned["value"].isna()]
    # the final row closes the last period and carries no length of its own
    assert list(naned) == [
        pd.Timestamp("2020-02-01"),
        pd.Timestamp("2020-02-10"),
        pd.Timestamp("2020-05-20"),
    ], f"unexpected NaNed rows {list(naned)}"
    assert cleaned["value"].tolist()[0] == 1.0
    assert cleaned["value"].tolist()[3] == 4.0


def test_clean_billing_data_bimonthly_keeps_a_seventy_day_period():
    """The bimonthly maximum admits the same 70 day period the monthly maximum rejects."""
    data = _billing_frame(
        ["2020-01-01", "2020-02-10", "2020-04-20", "2020-05-20"],
        [1.0, 2.0, 3.0, 4.0],
    )

    cleaned, report = clean_billing_data(
        data, "billing_bimonthly", BillingDataSufficiencySettings()
    )

    assert report.offcycle_periods == {}, f"unexpected off-cycle periods {report.offcycle_periods}"
    assert report.data_frequency == "billing_bimonthly"
    assert cleaned["value"].tolist()[:3] == [1.0, 2.0, 3.0]


def test_clean_billing_data_thresholds_come_from_settings():
    """Widening the monthly maximum keeps a period the default would have NaNed."""
    data = _billing_frame(
        ["2020-01-01", "2020-02-01", "2020-03-15", "2020-04-15"],
        [1.0, 2.0, 3.0, 4.0],
    )
    settings = BillingDataSufficiencySettings(max_days_in_monthly_period=45)

    cleaned, report = clean_billing_data(data, "billing_monthly", settings)

    assert report.offcycle_periods == {}, f"unexpected off-cycle periods {report.offcycle_periods}"
    assert cleaned["value"].tolist()[:3] == [1.0, 2.0, 3.0]


def test_clean_billing_data_merges_an_estimated_read_into_the_following_read():
    """An estimated read is folded into the next actual read, and the merge is reported
    against the read that absorbed it."""
    data = _billing_frame(
        ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"],
        [2.0, 5.0, 4.0, 3.0, float("nan")],
        estimated=[False, True, False, False, False],
    )

    cleaned, report = clean_billing_data(
        data, "billing_monthly", BillingDataSufficiencySettings()
    )

    assert report.estimated_merges == {
        pd.Timestamp("2020-03-01"): 9.0
    }, f"unexpected merges {report.estimated_merges}"
    assert list(cleaned.index) == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-03-01"),
        pd.Timestamp("2020-04-01"),
        pd.Timestamp("2020-05-01"),
    ], f"the estimated read was not removed: {list(cleaned.index)}"
    assert cleaned.loc[pd.Timestamp("2020-03-01"), "value"] == 9.0


def test_clean_billing_data_on_empty_values_returns_an_empty_report():
    data = _billing_frame(["2020-01-01", "2020-02-01"], [float("nan"), float("nan")])

    cleaned, report = clean_billing_data(
        data, "billing_monthly", BillingDataSufficiencySettings()
    )

    assert cleaned.empty
    assert report == DataReport(), f"unexpected report {report}"


def _sub_daily_series():
    """Three days of hourly readings whose middle day is exactly half observed."""
    index = pd.date_range("2020-01-01", periods=72, freq="h")
    series = pd.Series(1.0, index=index)
    series.iloc[24:36] = float("nan")

    return series


def test_downsample_and_clean_daily_data_drops_and_reports_low_coverage_days():
    """A day at or below the coverage threshold is dropped and reported with its coverage."""
    settings = DailyDataSufficiencySettings()

    cleaned, report = downsample_and_clean_daily_data(_sub_daily_series(), settings)

    assert report.low_coverage_days == {
        pd.Timestamp("2020-01-02"): 0.5
    }, f"unexpected low coverage days {report.low_coverage_days}"
    dropped = cleaned.index[cleaned["value"].isna()]
    assert list(dropped) == list(
        report.low_coverage_days
    ), f"reported days {list(report.low_coverage_days)} do not match dropped rows {list(dropped)}"


def test_downsample_and_clean_daily_data_divides_kept_days_by_coverage():
    """A day that clears the threshold is scaled up by its coverage fraction."""
    settings = DailyDataSufficiencySettings()

    cleaned, _ = downsample_and_clean_daily_data(_sub_daily_series(), settings)

    # day 1 is fully observed; day 3 has no reading following its final hour, which
    # therefore contributes nothing
    assert cleaned["value"].tolist()[0] == pytest.approx(24.0)
    assert cleaned["value"].tolist()[2] == pytest.approx(23.0)


def test_downsample_and_clean_daily_data_threshold_comes_from_settings():
    """Lowering the coverage threshold keeps the half-observed day the default drops."""
    settings = DailyDataSufficiencySettings(
        observed=DailyObservedSufficiencySettings(min_pct_hourly_coverage=0.25)
    )

    cleaned, report = downsample_and_clean_daily_data(_sub_daily_series(), settings)

    assert report.low_coverage_days == {}, f"unexpected low coverage days {report.low_coverage_days}"
    assert cleaned["value"].tolist()[1] == pytest.approx(24.0)


def test_clean_billing_daily_data_passes_daily_data_through_with_an_empty_report():
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    series = pd.Series(1.0, index=index)

    cleaned, report = clean_billing_daily_data(
        series, "daily", DailyDataSufficiencySettings()
    )

    assert cleaned["value"].tolist() == [1.0] * 5
    assert report == DataReport(), f"unexpected report {report}"
